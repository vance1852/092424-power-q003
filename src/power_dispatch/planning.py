"""确定性的电价、能力与燃料库存计算。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable, Mapping, Sequence


ZERO = Decimal("0")
ONE = Decimal("1")
HUNDRED = Decimal("100")
BASIS_POINTS = Decimal("10000")
DAY_SECONDS = Decimal(86400)
UTC = timezone.utc


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


def clamp_percentage(percentage: Decimal) -> Decimal:
    """将能力百分比收敛到 [0,100]，超界值按最近边界处理而非反转扣减。"""
    return max(ZERO, min(HUNDRED, percentage))


def effective_capacity(
    nominal: Decimal,
    capacity_percentages: Iterable[Decimal],
) -> Decimal:
    """同一时刻并存的多条限制取最严格值（最小值），绝不连乘重复扣减。"""
    factor = ONE
    for percentage in capacity_percentages:
        factor = min(factor, clamp_percentage(percentage) / HUNDRED)
    return quantize_volume(nominal * factor)


@dataclass(frozen=True, slots=True)
class CapacityLimit:
    """一条能力限制事件在 UTC 时间轴上的半开区间 [starts_at, ends_at)。

    ends_at 为 None 表示开放式（持续生效）；capacity_percent 为该区间内
    相对额定能力的百分比上限；source_id/source_type/label 用于在分配结果中
    保留限制来源。
    """

    source_id: str
    starts_at: datetime
    ends_at: datetime | None
    capacity_percent: Decimal
    source_type: str = "outage"
    label: str = ""

    def normalized(self) -> "CapacityLimit":
        if self.ends_at is None:
            return self
        return CapacityLimit(
            source_id=self.source_id,
            starts_at=self.starts_at,
            ends_at=max(self.ends_at, self.starts_at),
            capacity_percent=clamp_percentage(self.capacity_percent),
            source_type=self.source_type,
            label=self.label,
        )


@dataclass(frozen=True, slots=True)
class CapacitySegment:
    """扫描出的均匀能力时段（半开区间）及当时生效的限制来源。"""

    starts_at: datetime
    ends_at: datetime
    factor: Decimal
    sources: tuple[str, ...]


def service_day_bounds(service_date: str) -> tuple[datetime, datetime]:
    """服务日对应的 UTC 半开区间 [当日 00:00:00Z, 次日 00:00:00Z)。"""
    day = date.fromisoformat(service_date)
    start = datetime.combine(day, time.min, tzinfo=UTC)
    return start, start + timedelta(days=1)


def capacity_profile(
    limits: Iterable[CapacityLimit],
    window_start: datetime,
    window_end: datetime,
) -> list[CapacitySegment]:
    """扫描限制事件，把窗口切分为能力均匀的时段。

    规则：
    - 区间均为半开区间，边界相接（前一事件 ends_at 等于下一事件 starts_at）
      不会产生重叠时刻；
    - 同一时刻多条限制并存时取最小百分比（最严格上限），不连乘、不重复扣减；
    - 同一限制在扫描中按 source_id 去重，重复登记/重放同一事件只计一次；
    - 百分比裁剪到 [0,100]：负值按 0（零能力窗口），超过 100 按 100（不扩容）；
    - 与窗口仅端点相接的事件不产生任何时段。
    """
    if window_end <= window_start:
        raise ValueError("能力窗口必须为正区间")

    # 按 source_id 去重（同一事件重放），再裁剪百分比并收集边界。
    unique: dict[str, CapacityLimit] = {}
    for raw in limits:
        limit = raw.normalized()
        if limit.source_id in unique:
            continue
        unique[limit.source_id] = limit

    boundaries = {window_start, window_end}
    active: list[CapacityLimit] = []
    for limit in unique.values():
        clipped_end = window_end if limit.ends_at is None else min(limit.ends_at, window_end)
        clipped_start = max(limit.starts_at, window_start)
        if clipped_start < clipped_end:  # 半开区间：端点相接不算重叠
            active.append(limit)
            boundaries.add(clipped_start)
            boundaries.add(clipped_end)

    ordered = sorted(boundaries)
    segments: list[CapacitySegment] = []
    for left, right in zip(ordered, ordered[1:]):
        if right <= left:
            continue
        factor = ONE
        sources: set[str] = set()
        midpoint = left + (right - left) / 2
        for limit in active:
            limit_end = limit.ends_at
            applies = limit.starts_at <= midpoint and (limit_end is None or midpoint < limit_end)
            if applies:
                factor = min(factor, clamp_percentage(limit.capacity_percent) / HUNDRED)
                sources.add(limit.source_id)
        segments.append(
            CapacitySegment(left, right, factor, tuple(sorted(sources)))
        )
    return segments


@dataclass(frozen=True, slots=True)
class DailyCapacity:
    """服务日内的有效能力汇总。"""

    service_date: str
    nominal: Decimal
    effective: Decimal
    segments: tuple[CapacitySegment, ...] = ()

    def source_segments(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for segment in self.segments:
            rows.append({
                "starts_at": segment.starts_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "ends_at": segment.ends_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "capacity_percent": decimal_text(segment.factor * HUNDRED),
                "source_ids": list(segment.sources),
            })
        return rows

    def binding_sources(self) -> tuple[str, ...]:
        """至少在一个时段生效（成为绑定上限）的限制来源，按编号排序去重。"""
        seen: set[str] = set()
        for segment in self.segments:
            seen.update(segment.sources)
        return tuple(sorted(seen))


def daily_effective_capacity(
    nominal: Decimal,
    limits: Iterable[CapacityLimit],
    service_date: str,
) -> DailyCapacity:
    """按 UTC 服务日扫描限制事件并按秒加权求有效能力。

    跨午夜检修被拆到各自服务日；零能力窗口（factor=0）按其时长直接清零
    对应部分能力。输入顺序不影响结果。
    """
    window_start, window_end = service_day_bounds(service_date)
    segments = tuple(capacity_profile(limits, window_start, window_end))
    weighted = ZERO
    for segment in segments:
        seconds = Decimal((segment.ends_at - segment.starts_at).total_seconds())
        weighted += nominal * segment.factor * seconds
    effective = weighted / DAY_SECONDS
    return DailyCapacity(
        service_date=service_date,
        nominal=quantize_volume(nominal),
        effective=quantize_volume(effective),
        segments=segments,
    )


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
