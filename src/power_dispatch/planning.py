"""确定性的电价、能力与燃料库存计算。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable, Mapping, Sequence


ZERO = Decimal("0")
HUNDRED = Decimal("100")
BASIS_POINTS = Decimal("10000")


def quantize_volume(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)


def quantize_money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def decimal_text(value: Decimal) -> str:
    return format(value, "f")


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PricePoint:
    trade_date: str
    close: Decimal


@dataclass(frozen=True, slots=True)
class Streak:
    direction: str
    sessions: int
    start_date: str
    end_date: str
    start_close: Decimal
    end_close: Decimal
    percent_change: Decimal

    def as_dict(self) -> dict[str, object]:
        return {
            "direction": self.direction,
            "sessions": self.sessions,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "start_close": decimal_text(self.start_close),
            "end_close": decimal_text(self.end_close),
            "percent_change": decimal_text(self.percent_change),
        }


def latest_streak(points: Sequence[PricePoint]) -> Streak | None:
    ordered = sorted(points, key=lambda item: item.trade_date)
    if len(ordered) < 2:
        return None
    last = ordered[-1]
    previous = ordered[-2]
    if last.close == previous.close:
        return Streak("flat", 1, last.trade_date, last.trade_date, last.close, last.close, ZERO)
    direction = "down" if last.close < previous.close else "up"
    start_index = len(ordered) - 2
    while start_index > 0:
        left = ordered[start_index - 1]
        right = ordered[start_index]
        matches = right.close < left.close if direction == "down" else right.close > left.close
        if not matches:
            break
        start_index -= 1
    start = ordered[start_index]
    change = (last.close - start.close) / start.close * HUNDRED
    return Streak(
        direction=direction,
        sessions=len(ordered) - start_index,
        start_date=start.trade_date,
        end_date=last.trade_date,
        start_close=start.close,
        end_close=last.close,
        percent_change=change.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
    )


def moving_average(points: Sequence[PricePoint], sessions: int) -> Decimal | None:
    if sessions <= 0:
        raise ValueError("sessions 必须大于零")
    ordered = sorted(points, key=lambda item: item.trade_date)
    if len(ordered) < sessions:
        return None
    values = [item.close for item in ordered[-sessions:]]
    return quantize_money(sum(values, ZERO) / Decimal(len(values)))


@dataclass(frozen=True, slots=True)
class CapacityConstraint:
    """一条作用于 UTC 半开区间 [starts_at, ends_at) 的能力限制。"""

    source: str
    starts_at: datetime | None
    ends_at: datetime | None
    capacity_percent: Decimal
    reason: str

    def as_dict(self) -> dict[str, str | None]:
        return {
            "source": self.source,
            "starts_at": None if self.starts_at is None else _utc_iso(self.starts_at),
            "ends_at": None if self.ends_at is None else _utc_iso(self.ends_at),
            "capacity_percent": decimal_text(self.capacity_percent),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CapacityReport:
    """服务窗口内的有效上限及参与计算的约束来源。"""

    available: Decimal
    constraints: tuple[CapacityConstraint, ...]


def _utc_iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _checked_percent(value: Decimal) -> Decimal:
    if not value.is_finite() or value < ZERO or value > HUNDRED:
        raise ValueError("capacity_percent 必须是 0 到 100 的有限数值")
    return value


def _microseconds(delta: timedelta) -> Decimal:
    return Decimal(delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds)


def effective_capacity(
    nominal: Decimal,
    window_start: datetime,
    window_end: datetime,
    constraints: Iterable[CapacityConstraint],
) -> CapacityReport:
    """按业务规则计算窗口有效上限。

    任一时刻生效的上限取该时刻所有重叠约束中的最严格值（最小百分比），
    不做连乘；窗口结果按各时段时长加权。约束区间为半开区间，
    边界相接的约束互不重叠；超界百分比直接拒绝而不是静默钳制。
    """
    if nominal < ZERO:
        raise ValueError("名义能力不能为负数")
    if window_start.tzinfo is None or window_end.tzinfo is None:
        raise ValueError("能力窗口必须带时区")
    if window_end <= window_start:
        raise ValueError("能力窗口结束必须晚于开始")
    relevant: list[CapacityConstraint] = []
    points: set[datetime] = {window_start, window_end}
    for constraint in constraints:
        _checked_percent(constraint.capacity_percent)
        for boundary in (constraint.starts_at, constraint.ends_at):
            if boundary is not None and boundary.tzinfo is None:
                raise ValueError("约束时间必须带时区")
        if constraint.starts_at is not None and constraint.ends_at is not None:
            if constraint.ends_at <= constraint.starts_at:
                raise ValueError("约束结束必须晚于开始")
        starts = window_start if constraint.starts_at is None else max(constraint.starts_at, window_start)
        ends = window_end if constraint.ends_at is None else min(constraint.ends_at, window_end)
        if ends <= starts:
            continue
        relevant.append(constraint)
        points.add(starts)
        points.add(ends)
    ordered = sorted(points)
    span = _microseconds(window_end - window_start)
    total = ZERO
    for left, right in zip(ordered, ordered[1:]):
        if right <= left:
            continue
        midpoint = left + (right - left) / 2
        active = [
            item
            for item in relevant
            if (item.starts_at is None or item.starts_at <= midpoint)
            and (item.ends_at is None or midpoint < item.ends_at)
        ]
        percent = min((item.capacity_percent for item in active), default=HUNDRED)
        share = _microseconds(right - left) / span
        total += nominal * percent / HUNDRED * share
    ordered_constraints = tuple(sorted(relevant, key=lambda item: item.source))
    return CapacityReport(quantize_volume(total), ordered_constraints)


@dataclass(frozen=True, slots=True)
class AllocationRequest:
    nomination_id: str
    requested: Decimal
    priority: int
    submitted_at: str


def allocate_capacity(
    available: Decimal,
    requests: Iterable[AllocationRequest],
) -> list[dict[str, str]]:
    if available < ZERO:
        raise ValueError("可用能力不能为负数")
    remaining = quantize_volume(available)
    result: list[dict[str, str]] = []
    ordered = sorted(requests, key=lambda item: (item.priority, item.submitted_at, item.nomination_id))
    for request in ordered:
        allocated = min(remaining, request.requested)
        allocated = quantize_volume(max(ZERO, allocated))
        remaining = quantize_volume(remaining - allocated)
        result.append({
            "nomination_id": request.nomination_id,
            "requested_mwh": decimal_text(request.requested),
            "allocated_mwh": decimal_text(allocated),
            "unfilled_mwh": decimal_text(quantize_volume(request.requested - allocated)),
        })
    return result


def delivered_after_loss(loaded: Decimal, loss_basis_points: int) -> Decimal:
    if not 0 <= loss_basis_points <= 1000:
        raise ValueError("损耗基点超出范围")
    retained = Decimal(1) - Decimal(loss_basis_points) / BASIS_POINTS
    return quantize_volume(loaded * retained)


def weighted_inventory_cost(lots: Iterable[Mapping[str, object]]) -> dict[str, str]:
    quantity = ZERO
    value = ZERO
    for lot in lots:
        available = Decimal(str(lot["available_mwh"]))
        unit_cost = Decimal(str(lot["unit_cost_cny"]))
        if available < ZERO or unit_cost < ZERO:
            raise ValueError("燃料库存数量和成本不能为负数")
        quantity += available
        value += available * unit_cost
    average = ZERO if quantity == ZERO else value / quantity
    return {
        "available_mwh": decimal_text(quantize_volume(quantity)),
        "inventory_value_cny": decimal_text(quantize_money(value)),
        "weighted_unit_cost_cny": decimal_text(quantize_money(average)),
    }


def reconcile_inventory(
    book_quantity: Decimal,
    measured_quantity: Decimal,
    tolerance_percent: Decimal,
) -> dict[str, object]:
    if book_quantity < ZERO or measured_quantity < ZERO:
        raise ValueError("燃料库存数量不能为负数")
    if tolerance_percent < ZERO:
        raise ValueError("容差不能为负数")
    delta = quantize_volume(measured_quantity - book_quantity)
    ratio = ZERO if book_quantity == ZERO else abs(delta) / book_quantity * HUNDRED
    return {
        "book_quantity": decimal_text(quantize_volume(book_quantity)),
        "measured_quantity": decimal_text(quantize_volume(measured_quantity)),
        "delta_mwh": decimal_text(delta),
        "variance_percent": decimal_text(ratio.quantize(Decimal("0.0001"))),
        "within_tolerance": ratio <= tolerance_percent,
    }


def scenario_projection(
    *,
    current_price: Decimal,
    market_index_drop_percent: Decimal,
    routes: Iterable[Mapping[str, object]],
    inventory: Iterable[Mapping[str, object]],
    route_capacity_changes: Mapping[str, Decimal],
    demand_changes: Mapping[str, Decimal],
) -> dict[str, object]:
    projected_price = current_price * (Decimal(1) - market_index_drop_percent / HUNDRED)
    route_rows: list[dict[str, str]] = []
    total_capacity = ZERO
    for route in sorted(routes, key=lambda item: str(item["route_id"])):
        route_id = str(route["route_id"])
        nominal = Decimal(str(route["daily_capacity"]))
        change = route_capacity_changes.get(route_id, ZERO)
        projected = max(ZERO, nominal * (Decimal(1) + change / HUNDRED))
        total_capacity += projected
        route_rows.append({
            "route_id": route_id,
            "base_capacity": decimal_text(quantize_volume(nominal)),
            "change_percent": decimal_text(change),
            "projected_capacity": decimal_text(quantize_volume(projected)),
        })
    inventory_rows: list[dict[str, str]] = []
    total_inventory = ZERO
    for row in sorted(inventory, key=lambda item: (str(item["facility_id"]), str(item["product"]))):
        key = f"{row['facility_id']}:{row['product']}"
        available = Decimal(str(row["available_mwh"]))
        demand_change = demand_changes.get(key, ZERO)
        days_factor = max(Decimal("0.01"), Decimal(1) + demand_change / HUNDRED)
        adjusted = available / days_factor
        total_inventory += adjusted
        inventory_rows.append({
            "inventory_key": key,
            "base_available": decimal_text(quantize_volume(available)),
            "demand_change_percent": decimal_text(demand_change),
            "demand_adjusted_inventory": decimal_text(quantize_volume(adjusted)),
        })
    return {
        "projected_market_index_cny": decimal_text(quantize_money(projected_price)),
        "total_projected_capacity": decimal_text(quantize_volume(total_capacity)),
        "demand_adjusted_inventory": decimal_text(quantize_volume(total_inventory)),
        "routes": route_rows,
        "inventory": inventory_rows,
    }
