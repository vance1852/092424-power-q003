from __future__ import annotations

import json
import sqlite3
import unittest
from datetime import datetime, timezone
from decimal import Decimal

from power_dispatch.api import JsonApplication
from power_dispatch.clock import FrozenClock
from power_dispatch.errors import Conflict, Forbidden, ValidationFailed
from power_dispatch.planning import (
    AllocationRequest,
    CapacityLimit,
    PricePoint,
    allocate_capacity,
    capacity_profile,
    daily_effective_capacity,
    effective_capacity,
    latest_streak,
)
from power_dispatch.service import SupplyService
from power_dispatch.risk import DemandBucket, inventory_coverage, mark_to_market, supply_gap


class CapacityRuleTests(unittest.TestCase):
    @staticmethod
    def _utc(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)

    def test_overlapping_limits_take_minimum_never_multiply(self) -> None:
        self.assertEqual(effective_capacity(Decimal("100"), [Decimal("60"), Decimal("40")]), Decimal("40.000"))
        self.assertEqual(effective_capacity(Decimal("100"), [Decimal("80"), Decimal("60")]), Decimal("60.000"))

    def test_percentages_out_of_range_are_clamped_to_nearest_bound(self) -> None:
        self.assertEqual(effective_capacity(Decimal("100"), [Decimal("150")]), Decimal("100.000"))
        self.assertEqual(effective_capacity(Decimal("100"), [Decimal("-10")]), Decimal("0.000"))
        self.assertEqual(effective_capacity(Decimal("100"), [Decimal("-10"), Decimal("200")]), Decimal("0.000"))

    def test_touching_half_open_intervals_do_not_double_apply(self) -> None:
        start = self._utc("2026-09-25T10:00:00Z")
        end = self._utc("2026-09-25T14:00:00Z")
        segments = capacity_profile(
            [
                CapacityLimit("outage-1", self._utc("2026-09-25T10:00:00Z"), self._utc("2026-09-25T12:00:00Z"), Decimal("50")),
                CapacityLimit("outage-2", self._utc("2026-09-25T12:00:00Z"), self._utc("2026-09-25T14:00:00Z"), Decimal("50")),
            ],
            start,
            end,
        )
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].sources, ("outage-1",))
        self.assertEqual(segments[1].sources, ("outage-2",))
        self.assertTrue(all(segment.factor == Decimal("0.5") for segment in segments))

    def test_same_event_replayed_is_deduplicated(self) -> None:
        start = self._utc("2026-09-25T00:00:00Z")
        end = self._utc("2026-09-26T00:00:00Z")
        limit = CapacityLimit("outage-7", start, end, Decimal("50"))
        segments = capacity_profile([limit, limit], start, end)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].sources, ("outage-7",))
        self.assertEqual(daily_effective_capacity(Decimal("100"), [limit, limit], "2026-09-25").effective, Decimal("50.000"))

    def test_overlap_uses_minimum_only_inside_overlapping_window(self) -> None:
        limits = [
            CapacityLimit("outage-1", self._utc("2026-09-25T00:00:00Z"), self._utc("2026-09-26T00:00:00Z"), Decimal("60")),
            CapacityLimit("outage-2", self._utc("2026-09-25T00:00:00Z"), self._utc("2026-09-25T06:00:00Z"), Decimal("40")),
        ]
        plan = daily_effective_capacity(Decimal("100000"), limits, "2026-09-25")
        # 6 小时受 40% 约束，18 小时受 60% 约束：100000*(6*0.4+18*0.6)/24
        self.assertEqual(plan.effective, Decimal("55000.000"))
        sources = {tuple(seg.sources) for seg in plan.segments}
        self.assertIn(("outage-1", "outage-2"), sources)
        self.assertIn(("outage-1",), sources)

    def test_cross_midnight_outage_is_split_across_service_days(self) -> None:
        limits = [
            CapacityLimit(
                "outage-1",
                self._utc("2026-09-25T20:00:00Z"),
                self._utc("2026-09-26T04:00:00Z"),
                Decimal("50"),
            )
        ]
        first = daily_effective_capacity(Decimal("100000"), limits, "2026-09-25")
        second = daily_effective_capacity(Decimal("100000"), limits, "2026-09-26")
        # 每个服务日各覆盖 4 小时降容：100000*(20+4*0.5)/24
        self.assertEqual(first.effective, Decimal("91666.667"))
        self.assertEqual(second.effective, Decimal("91666.667"))
        last = first.segments[-1]
        self.assertEqual(last.starts_at, self._utc("2026-09-25T20:00:00Z"))
        self.assertEqual(last.ends_at, self._utc("2026-09-26T00:00:00Z"))

    def test_zero_capacity_window_clears_only_its_duration(self) -> None:
        limits = [
            CapacityLimit(
                "outage-1",
                self._utc("2026-09-25T10:00:00Z"),
                self._utc("2026-09-25T14:00:00Z"),
                Decimal("0"),
            )
        ]
        plan = daily_effective_capacity(Decimal("100000"), limits, "2026-09-25")
        self.assertEqual(plan.effective, Decimal("83333.333"))
        zero = [seg for seg in plan.segments if seg.factor == 0]
        self.assertEqual(len(zero), 1)

    def test_open_ended_limit_covers_remainder_of_day(self) -> None:
        limits = [
            CapacityLimit("outage-1", self._utc("2026-09-25T12:00:00Z"), None, Decimal("50"))
        ]
        plan = daily_effective_capacity(Decimal("100"), limits, "2026-09-25")
        self.assertEqual(plan.effective, Decimal("75.000"))


class PlanningTests(unittest.TestCase):
    def test_latest_down_streak_uses_first_close_as_base(self) -> None:
        streak = latest_streak([
            PricePoint("2026-09-18", Decimal("108")),
            PricePoint("2026-09-19", Decimal("105")),
            PricePoint("2026-09-20", Decimal("102")),
            PricePoint("2026-09-21", Decimal("98")),
        ])
        self.assertEqual(streak.direction, "down")
        self.assertEqual(streak.sessions, 4)
        self.assertEqual(streak.start_date, "2026-09-18")
        self.assertEqual(streak.end_close, Decimal("98"))

    def test_allocation_is_stable_and_does_not_exceed_capacity(self) -> None:
        rows = allocate_capacity(Decimal("100"), [
            AllocationRequest("later", Decimal("80"), 20, "2026-09-24T09:00:00Z"),
            AllocationRequest("first", Decimal("70"), 10, "2026-09-24T10:00:00Z"),
        ])
        self.assertEqual(rows[0]["nomination_id"], "first")
        self.assertEqual(rows[0]["allocated_mwh"], "70.000")
        self.assertEqual(rows[1]["allocated_mwh"], "30.000")

    def test_inventory_coverage_and_supply_gap(self) -> None:
        coverage = inventory_coverage(
            [{"facility_id": "terminal", "product": "gasoline-92", "available_mwh": "250"}],
            [DemandBucket("terminal", "gasoline-92", Decimal("100"), Decimal("20"))],
        )
        self.assertEqual(coverage[0]["coverage_days"], "2.30")
        self.assertTrue(coverage[0]["below_three_days"])
        gap = supply_gap(
            opening_inventory=Decimal("100"),
            confirmed_inbound=Decimal("30"),
            forecast_demand=Decimal("120"),
            protected_reserve=Decimal("40"),
        )
        self.assertEqual(gap["supply_gap"], "30.000")

    def test_mark_to_market_groups_deterministically(self) -> None:
        result = mark_to_market(
            [{"position_id": "p1", "market_index": "PEAK_VALLEY", "quantity_mwh": "100", "entry_price_cny": "105"}],
            {"PEAK_VALLEY": Decimal("98")},
        )
        self.assertEqual(result["unrealized_pnl_cny"], "-700.00")


class SupplyServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:", isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.clock = FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc))
        self.service = SupplyService(self.connection, self.clock)
        for user_id, role in (("plan", "planner"), ("dispatch", "dispatcher"), ("risk", "risk"), ("audit", "auditor")):
            self.service.create_user(user_id, user_id, role)
        self.service.create_facility("plan", {"facility_id": "field-a", "name": "北部电厂", "kind": "storage", "timezone": "Asia/Shanghai", "capacity_mwh": "500000"})
        self.service.create_facility("plan", {"facility_id": "terminal-b", "name": "沿海终端", "kind": "terminal", "timezone": "Asia/Shanghai", "capacity_mwh": "800000"})
        self.service.create_route("plan", {"route_id": "pipe-a-b", "origin_id": "field-a", "destination_id": "terminal-b", "product": "crude", "daily_capacity": "100000", "loss_basis_points": 25, "transit_hours": 36})

    def tearDown(self) -> None:
        self.connection.close()

    def quote(self, day: int, close: str) -> dict[str, object]:
        return self.service.record_quote("plan", {"market_index": "PEAK_VALLEY", "trade_date": f"2026-09-{day}", "close_cny": close, "source_revision": f"r-{day}", "observed_at": f"2026-09-{day}T21:00:00Z"})

    def test_quote_revisions_preserve_history(self) -> None:
        first = self.quote(23, "98")
        second = self.service.record_quote("plan", {"market_index": "PEAK_VALLEY", "trade_date": "2026-09-23", "close_cny": "97.8", "source_revision": "r-23-corrected", "observed_at": "2026-09-23T22:00:00Z"})
        self.assertNotEqual(first["quote_id"], second["quote_id"])
        rows = self.connection.execute("SELECT * FROM market_index_quotes ORDER BY quote_id").fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["supersedes_quote_id"], rows[0]["quote_id"])

    def test_nomination_replay_and_payload_conflict(self) -> None:
        payload = {"nomination_id": "nom-1", "route_id": "pipe-a-b", "shipper_id": "refinery", "service_date": "2026-09-25", "requested_mwh": "80000", "priority": 10, "idempotency_key": "key-1"}
        first = self.service.submit_nomination("dispatch", payload)
        self.assertEqual(first, self.service.submit_nomination("dispatch", payload))
        changed = dict(payload, requested_mwh="81000")
        with self.assertRaises(Conflict):
            self.service.submit_nomination("dispatch", changed)

    def test_outage_reduces_allocation_and_transfer_consumes_inventory(self) -> None:
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-25T00:00:00Z", "2026-09-26T00:00:00Z", "50", "检修")
        for number, requested, priority in ((1, "40000", 10), (2, "30000", 20)):
            self.service.submit_nomination("dispatch", {"nomination_id": f"nom-{number}", "route_id": "pipe-a-b", "shipper_id": f"shipper-{number}", "service_date": "2026-09-25", "requested_mwh": requested, "priority": priority, "idempotency_key": f"key-{number}"})
        allocation = self.service.allocate("dispatch", "pipe-a-b", "2026-09-25")
        self.assertEqual(allocation["available_capacity"], "50000.000")
        self.assertEqual(allocation["allocations"][1]["allocated_mwh"], "10000.000")
        self.service.add_inventory_lot("dispatch", {"lot_id": "lot-1", "facility_id": "field-a", "product": "crude", "grade": "PEAK_VALLEY", "quantity_mwh": "60000", "unit_cost_cny": "91", "received_at": "2026-09-24T06:00:00Z"})
        transfer = self.service.dispatch_transfer("dispatch", "transfer-1", "nom-1", "lot-1", 2)
        self.assertEqual(transfer["loaded_mwh"], "40000.000")
        self.assertEqual(self.service.inventory_lot("lot-1")["available_mwh"], "20000.000")

    def test_scenario_is_approved_and_replayed_by_input(self) -> None:
        self.quote(23, "98")
        self.service.add_inventory_lot("dispatch", {"lot_id": "lot-1", "facility_id": "field-a", "product": "crude", "grade": "PEAK_VALLEY", "quantity_mwh": "60000", "unit_cost_cny": "91", "received_at": "2026-09-24T06:00:00Z"})
        self.service.create_scenario("plan", {"scenario_id": "restart", "name": "机组检修恢复", "market_index_drop_percent": "9", "route_capacity_changes": {"pipe-a-b": "20"}, "demand_changes": {"field-a:crude": "-5"}})
        with self.assertRaises(Forbidden):
            self.service.approve_scenario("plan", "restart", 1)
        self.service.approve_scenario("risk", "restart", 1)
        first = self.service.run_scenario("plan", "restart", "2026-09-23")
        second = self.service.run_scenario("plan", "restart", "2026-09-23")
        self.assertFalse(first["replayed"])
        self.assertTrue(second["replayed"])
        self.assertEqual(first["run_id"], second["run_id"])

    def test_audit_chain_detects_tampering(self) -> None:
        self.assertTrue(self.service.audit_chain("audit")["valid"])
        self.connection.execute("UPDATE supply_audit_events SET payload_json='{}' WHERE event_id=1")
        self.assertFalse(self.service.audit_chain("audit")["valid"])

    def test_api_exposes_browser_free_boundary(self) -> None:
        app = JsonApplication(self.service)
        self.assertEqual(app.handle("GET", "/health").status, 200)
        response = app.handle("GET", "/quotes/summary/PEAK_VALLEY", {"X-Actor-Id": "plan"})
        self.assertEqual(response.status, 404)
        self.assertEqual(response.body["error"]["code"], "not_found")

    def _nominate(self, number: int, requested: str, priority: int = 10, day: str = "2026-09-25") -> None:
        self.service.submit_nomination("dispatch", {"nomination_id": f"nom-{number}", "route_id": "pipe-a-b", "shipper_id": f"shipper-{number}", "service_date": day, "requested_mwh": requested, "priority": priority, "idempotency_key": f"key-{number}"})

    def test_overlapping_maintenance_and_curtailment_never_compound(self) -> None:
        # 检修降容 60% 覆盖全天；临时限电 80% 只覆盖前 12 小时。
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-25T00:00:00Z", "2026-09-26T00:00:00Z", "60", "检修降容")
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-25T00:00:00Z", "2026-09-25T12:00:00Z", "80", "临时限电")
        self._nominate(1, "100000")
        allocation = self.service.allocate("dispatch", "pipe-a-b", "2026-09-25")
        # 绝不能连乘成 48000；重叠时段取最严格的 60%。
        self.assertEqual(allocation["available_capacity"], "60000.000")
        self.assertEqual(allocation["allocations"][0]["allocated_mwh"], "60000.000")
        reasons = {limit["reason"] for limit in allocation["capacity_limits"]}
        self.assertEqual(reasons, {"检修降容", "临时限电"})
        referenced = {source for segment in allocation["capacity_segments"] for source in segment["source_ids"]}
        declared = {limit["source_id"] for limit in allocation["capacity_limits"]}
        self.assertTrue(referenced <= declared)

    def test_touching_boundaries_split_day_without_extra_deduction(self) -> None:
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-25T00:00:00Z", "2026-09-25T12:00:00Z", "50", "上午检修")
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-25T12:00:00Z", "2026-09-26T00:00:00Z", "50", "下午限电")
        self._nominate(1, "100000")
        allocation = self.service.allocate("dispatch", "pipe-a-b", "2026-09-25")
        self.assertEqual(allocation["available_capacity"], "50000.000")

    def test_duplicate_event_announcement_is_rejected(self) -> None:
        payload = ("risk", "pipe-a-b", "2026-09-25T00:00:00Z", "2026-09-26T00:00:00Z", "50", "检修")
        self.service.announce_outage(*payload)
        with self.assertRaises(Conflict):
            self.service.announce_outage(*payload)

    def test_percentage_out_of_range_or_non_numeric_is_rejected(self) -> None:
        for bad in ("120", "-1", "abc"):
            with self.assertRaises(ValidationFailed):
                self.service.announce_outage("risk", "pipe-a-b", "2026-09-25T00:00:00Z", "2026-09-26T00:00:00Z", bad, "检修")

    def test_zero_capacity_window_cancels_all_nominations(self) -> None:
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-25T00:00:00Z", "2026-09-26T00:00:00Z", "0", "全停")
        self._nominate(1, "30000", priority=10)
        self._nominate(2, "20000", priority=20)
        allocation = self.service.allocate("dispatch", "pipe-a-b", "2026-09-25")
        self.assertEqual(allocation["available_capacity"], "0.000")
        self.assertTrue(all(row["allocated_mwh"] == "0.000" for row in allocation["allocations"]))
        states = {row["nomination_id"]: row["state"] for row in self.connection.execute("SELECT nomination_id,state FROM nominations")}
        self.assertEqual(set(states.values()), {"cancelled"})

    def test_cross_midnight_outage_spans_two_service_days(self) -> None:
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-25T20:00:00Z", "2026-09-26T04:00:00Z", "50", "跨夜检修")
        self._nominate(1, "100000", day="2026-09-25")
        self._nominate(2, "100000", day="2026-09-26")
        first = self.service.allocate("dispatch", "pipe-a-b", "2026-09-25")
        second = self.service.allocate("dispatch", "pipe-a-b", "2026-09-26")
        self.assertEqual(first["available_capacity"], "91666.667")
        self.assertEqual(second["available_capacity"], "91666.667")

    def test_recompute_same_service_day_reuses_confirmed_plan(self) -> None:
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-25T00:00:00Z", "2026-09-26T00:00:00Z", "50", "检修")
        self._nominate(1, "40000", priority=10)
        self._nominate(2, "30000", priority=20)
        first = self.service.allocate("dispatch", "pipe-a-b", "2026-09-25")
        second = self.service.allocate("dispatch", "pipe-a-b", "2026-09-25")
        self.assertFalse(first["replayed"])
        self.assertTrue(second["replayed"])
        self.assertEqual(first["allocation_id"], second["allocation_id"])
        self.assertEqual(second["available_capacity"], "50000.000")
        self.assertEqual(second["allocations"], first["allocations"])
        revisions = {row["nomination_id"]: row["revision"] for row in self.connection.execute("SELECT nomination_id,revision FROM nominations")}
        self.assertEqual(revisions, {"nom-1": 2, "nom-2": 2})

    def test_event_revision_after_confirmation_does_not_rewrite_plan(self) -> None:
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-25T00:00:00Z", "2026-09-26T00:00:00Z", "50", "检修")
        self._nominate(1, "40000")
        first = self.service.allocate("dispatch", "pipe-a-b", "2026-09-25")
        # 计划确认后又登记一条更严格的限电事件。
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-25T00:00:00Z", "2026-09-26T00:00:00Z", "20", "新增限电")
        with self.assertRaises(Conflict):
            self.service.allocate("dispatch", "pipe-a-b", "2026-09-25")
        stored = self.connection.execute("SELECT available_capacity FROM allocation_runs WHERE route_id='pipe-a-b' AND service_date='2026-09-25'").fetchone()
        self.assertEqual(stored["available_capacity"], first["available_capacity"])
        nomination = self.connection.execute("SELECT allocated_mwh,revision FROM nominations WHERE nomination_id='nom-1'").fetchone()
        self.assertEqual(nomination["allocated_mwh"], "40000.000")
        self.assertEqual(nomination["revision"], 2)


if __name__ == "__main__":
    unittest.main()
