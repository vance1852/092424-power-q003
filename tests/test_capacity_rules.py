"""能力上限业务规则：重叠取严、半开边界、超界拒绝、重放稳定与来源保留。"""

from __future__ import annotations

import sqlite3
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from power_dispatch.clock import FrozenClock
from power_dispatch.errors import Conflict, InvalidState, ValidationFailed
from power_dispatch.planning import CapacityConstraint, effective_capacity
from power_dispatch.service import SupplyService


UTC = timezone.utc
DAY = datetime(2026, 9, 25, tzinfo=UTC)
DAY_START = DAY
DAY_END = DAY + timedelta(days=1)
NOMINAL = Decimal("100000")


def constraint(source: str, starts: datetime | None, ends: datetime | None, percent: str) -> CapacityConstraint:
    return CapacityConstraint(source, starts, ends, Decimal(percent), source)


class EffectiveCapacityRuleTests(unittest.TestCase):
    def test_overlapping_constraints_take_strictest_not_product(self) -> None:
        # 检修降容 60% 全天；临时限电 50% 仅后半天。
        report = effective_capacity(NOMINAL, DAY_START, DAY_END, [
            constraint("outage:maintenance", DAY_START, DAY_END, "60"),
            constraint("outage:curtailment", DAY_START + timedelta(hours=12), DAY_END, "50"),
        ])
        # 前 12 小时 60%，后 12 小时取 min(60,50)=50%，绝不连乘成 30%。
        self.assertEqual(report.available, Decimal("55000.000"))
        self.assertEqual([item.source for item in report.constraints], ["outage:curtailment", "outage:maintenance"])

    def test_boundary_adjacent_constraints_do_not_overlap(self) -> None:
        report = effective_capacity(NOMINAL, DAY_START, DAY_END, [
            constraint("outage:before", DAY_START - timedelta(hours=1), DAY_START, "0"),
            constraint("outage:after", DAY_END, DAY_END + timedelta(hours=1), "0"),
            constraint("outage:morning", DAY_START, DAY_START + timedelta(hours=12), "50"),
        ])
        # 结束于窗口起点、开始于窗口终点的约束均不计入；半天 50% 得 75000。
        self.assertEqual(report.available, Decimal("75000.000"))
        self.assertEqual([item.source for item in report.constraints], ["outage:morning"])

    def test_zero_percent_full_window_is_zero(self) -> None:
        report = effective_capacity(NOMINAL, DAY_START, DAY_END, [
            constraint("outage:stop", DAY_START, DAY_END, "0"),
        ])
        self.assertEqual(report.available, Decimal("0.000"))

    def test_zero_capacity_window_is_prorated(self) -> None:
        # 10:00-14:00 共 4 小时零能力，其余 20 小时满能力。
        report = effective_capacity(NOMINAL, DAY_START, DAY_END, [
            constraint("outage:stop", DAY_START + timedelta(hours=10), DAY_START + timedelta(hours=14), "0"),
        ])
        self.assertEqual(report.available, Decimal("83333.333"))

    def test_open_ended_constraint_runs_to_window_end(self) -> None:
        report = effective_capacity(
            NOMINAL, DAY_START, DAY_END,
            [constraint("outage:open", DAY_START + timedelta(hours=6), None, "80")],
        )
        # 6 小时满能力 + 18 小时 80%。
        self.assertEqual(report.available, Decimal("85000.000"))

    def test_out_of_range_and_non_finite_percentages_are_rejected(self) -> None:
        for percent in ("-0.001", "100.001"):
            with self.assertRaises(ValueError):
                effective_capacity(NOMINAL, DAY_START, DAY_END, [
                    constraint("outage:x", DAY_START, DAY_END, percent),
                ])
        with self.assertRaises(ValueError):
            effective_capacity(NOMINAL, DAY_START, DAY_END, [
                CapacityConstraint("outage:nan", DAY_START, DAY_END, Decimal("NaN"), "nan"),
            ])


class SupplyCapacityServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:", isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.clock = FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=UTC))
        self.service = SupplyService(self.connection, self.clock)
        for user_id, role in (("plan", "planner"), ("dispatch", "dispatcher"), ("risk", "risk"), ("audit", "auditor")):
            self.service.create_user(user_id, user_id, role)
        self.service.create_facility("plan", {"facility_id": "field-a", "name": "北部电厂", "kind": "storage", "timezone": "Asia/Shanghai", "capacity_mwh": "500000"})
        self.service.create_facility("plan", {"facility_id": "terminal-b", "name": "沿海终端", "kind": "terminal", "timezone": "Asia/Shanghai", "capacity_mwh": "800000"})
        self.service.create_route("plan", {"route_id": "pipe-a-b", "origin_id": "field-a", "destination_id": "terminal-b", "product": "crude", "daily_capacity": "100000", "loss_basis_points": 25, "transit_hours": 36})

    def tearDown(self) -> None:
        self.connection.close()

    def nominate(self, number: int, requested: str = "80000", priority: int = 10, service_date: str = "2026-09-25") -> None:
        self.service.submit_nomination("dispatch", {
            "nomination_id": f"nom-{number}", "route_id": "pipe-a-b", "shipper_id": f"shipper-{number}",
            "service_date": service_date, "requested_mwh": requested, "priority": priority,
            "idempotency_key": f"key-{number}",
        })

    def test_percentage_validation_rejects_out_of_range_values(self) -> None:
        for percent in ("-1", "101"):
            with self.assertRaises(ValidationFailed):
                self.service.announce_outage("risk", "pipe-a-b", "2026-09-25T00:00:00Z", "2026-09-26T00:00:00Z", percent, "检修")
        with self.assertRaises(ValidationFailed):
            self.service.announce_outage("risk", "pipe-a-b", "2026-09-25T00:00:00Z", "2026-09-26T00:00:00Z", "NaN", "检修")

    def test_same_event_replay_is_idempotent(self) -> None:
        payload = ("2026-09-25T00:00:00Z", "2026-09-26T00:00:00Z", "60", "检修")
        first = self.service.announce_outage("risk", "pipe-a-b", *payload, idempotency_key="evt-1")
        second = self.service.announce_outage("risk", "pipe-a-b", *payload, idempotency_key="evt-1")
        self.assertFalse(first["replayed"])
        self.assertTrue(second["replayed"])
        self.assertEqual(first["outage_id"], second["outage_id"])
        rows = self.connection.execute("SELECT COUNT(*) AS n FROM route_outages").fetchone()
        self.assertEqual(rows["n"], 1)
        with self.assertRaises(Conflict):
            self.service.announce_outage(
                "risk", "pipe-a-b", "2026-09-25T00:00:00Z", "2026-09-26T00:00:00Z", "50", "限电", idempotency_key="evt-1"
            )

    def test_overlapping_maintenance_and_curtailment_never_multiply(self) -> None:
        # 检修降容 60% 全天，临时限电 50% 覆盖 08:00-20:00（12 小时）。
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-25T00:00:00Z", "2026-09-26T00:00:00Z", "60", "检修", idempotency_key="maint")
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-25T08:00:00Z", "2026-09-25T20:00:00Z", "50", "临时限电", idempotency_key="curtail")
        self.nominate(1)
        allocation = self.service.allocate("dispatch", "pipe-a-b", "2026-09-25")
        # 8h*60% + 12h*50% + 4h*60% = 55% 日均值；旧连乘逻辑会得到 30000。
        self.assertEqual(allocation["available_capacity"], "55000.000")
        sources = {item["source"]: item for item in allocation["constraints"]}
        self.assertEqual(set(sources), {"outage:1", "outage:2"})
        self.assertEqual(sources["outage:1"]["capacity_percent"], "60")
        self.assertEqual(sources["outage:2"]["reason"], "临时限电")

    def test_adjacent_boundaries_at_midnight_are_excluded(self) -> None:
        # 前一天的事件恰在当日 00:00 结束；后一天的事件恰在次日 00:00 开始。
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-24T20:00:00Z", "2026-09-25T00:00:00Z", "0", "前夜检修", idempotency_key="before")
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-26T00:00:00Z", "2026-09-26T08:00:00Z", "0", "次晨检修", idempotency_key="after")
        self.nominate(1)
        allocation = self.service.allocate("dispatch", "pipe-a-b", "2026-09-25")
        self.assertEqual(allocation["available_capacity"], "100000.000")
        self.assertEqual(allocation["constraints"], [])

    def test_cross_midnight_maintenance_spreads_across_service_days(self) -> None:
        # 2026-09-24 20:00 UTC 至 2026-09-25 04:00 UTC，跨午夜共 8 小时 50%。
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-24T20:00:00Z", "2026-09-25T04:00:00Z", "50", "跨午夜检修", idempotency_key="cross")
        for index, (service_date, expected) in enumerate((
            ("2026-09-24", "91666.667"),
            ("2026-09-25", "91666.667"),
            ("2026-09-26", "100000.000"),
        ), start=1):
            self.nominate(index, service_date=service_date)
            allocation = self.service.allocate("dispatch", "pipe-a-b", service_date)
            self.assertEqual(allocation["available_capacity"], expected, service_date)

    def test_zero_capacity_window_cancels_allocation_but_keeps_sources(self) -> None:
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-25T10:00:00Z", "2026-09-25T14:00:00Z", "0", "零能力窗口", idempotency_key="zero")
        self.nominate(1, requested="100000")
        allocation = self.service.allocate("dispatch", "pipe-a-b", "2026-09-25")
        self.assertEqual(allocation["available_capacity"], "83333.333")
        self.assertEqual(allocation["allocations"][0]["allocated_mwh"], "83333.333")
        self.assertEqual(allocation["constraints"][0]["capacity_percent"], "0")
        # 全天零能力时分配必须全部取消。
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-26T00:00:00Z", "2026-09-27T00:00:00Z", "0", "全天停运", idempotency_key="zero-day")
        self.service.submit_nomination("dispatch", {
            "nomination_id": "nom-2", "route_id": "pipe-a-b", "shipper_id": "shipper-2",
            "service_date": "2026-09-26", "requested_mwh": "10", "priority": 10, "idempotency_key": "key-2",
        })
        stopped = self.service.allocate("dispatch", "pipe-a-b", "2026-09-26")
        self.assertEqual(stopped["available_capacity"], "0.000")
        self.assertEqual(stopped["allocations"][0]["allocated_mwh"], "0.000")
        state = self.connection.execute("SELECT state FROM nominations WHERE nomination_id='nom-2'").fetchone()
        self.assertEqual(state["state"], "cancelled")

    def test_same_day_recompute_reuses_input_digest(self) -> None:
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-25T00:00:00Z", "2026-09-26T00:00:00Z", "60", "检修", idempotency_key="maint")
        self.nominate(1)
        first = self.service.allocate("dispatch", "pipe-a-b", "2026-09-25")
        second = self.service.allocate("dispatch", "pipe-a-b", "2026-09-25")
        self.assertFalse(first["replayed"])
        self.assertTrue(second["replayed"])
        self.assertEqual(first["allocation_id"], second["allocation_id"])
        self.assertEqual(first["allocations"], second["allocations"])
        row = self.connection.execute("SELECT revision FROM nominations WHERE nomination_id='nom-1'").fetchone()
        self.assertEqual(row["revision"], 2)

    def test_event_revision_after_confirmation_does_not_rewrite_plan(self) -> None:
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-25T00:00:00Z", "2026-09-26T00:00:00Z", "60", "检修", idempotency_key="maint")
        self.nominate(1, requested="80000")
        confirmed = self.service.allocate("dispatch", "pipe-a-b", "2026-09-25")
        self.assertEqual(confirmed["available_capacity"], "60000.000")
        before = self.connection.execute(
            "SELECT available_capacity,result_json FROM allocation_runs WHERE allocation_id=?",
            (confirmed["allocation_id"],),
        ).fetchone()
        # 计划确认后登记更严格的临时限电：再次计算必须拒绝，而不是静默改写。
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-25T08:00:00Z", "2026-09-25T20:00:00Z", "30", "临时限电", idempotency_key="curtail")
        with self.assertRaises(Conflict):
            self.service.allocate("dispatch", "pipe-a-b", "2026-09-25")
        nomination = self.connection.execute("SELECT allocated_mwh,revision FROM nominations WHERE nomination_id='nom-1'").fetchone()
        self.assertEqual(nomination["allocated_mwh"], "60000.000")
        self.assertEqual(nomination["revision"], 2)
        after = self.connection.execute(
            "SELECT available_capacity,result_json FROM allocation_runs WHERE allocation_id=?",
            (confirmed["allocation_id"],),
        ).fetchone()
        self.assertEqual(before["available_capacity"], after["available_capacity"])
        self.assertEqual(before["result_json"], after["result_json"])

    def test_allocate_without_nominations_still_reports_zero_input(self) -> None:
        with self.assertRaises(InvalidState):
            self.service.allocate("dispatch", "pipe-a-b", "2026-09-25")


if __name__ == "__main__":
    unittest.main()
