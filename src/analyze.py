"""工单数据校验、指标计算和可解释异常检测。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_FIELDS = {
    "ticket_id",
    "created_at",
    "category",
    "description",
    "priority",
    "resolution_time_hours",
    "satisfaction",
    "channel",
    "is_resolved",
}
VALID_PRIORITIES = {"高", "中", "低"}
VALID_CHANNELS = {"在线", "电话", "邮件"}
PRIORITY_SCORE = {"高": 3, "中": 1, "低": 0}

TOPIC_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("重复或错误扣款", ("重复扣款", "扣了两次", "两个都扣", "多扣", "扣款金额不对")),
    (
        "支付成功但订单异常",
        ("订单显示未支付", "订单没生成", "订单没成功", "订单还是待支付", "没收到订单", "订单取消"),
    ),
    ("支付页面或结算失败", ("支付提示失败", "付款页面", "结算页面", "付不了款")),
    ("退款审核或到账延迟", ("审核中", "钱还没退", "退款还在处理中", "什么时候退")),
    ("退货运费报销", ("运费", "快递费", "报销")),
    ("物流停滞或异常", ("物流更新", "没发货", "快递信息", "快递显示异常", "正在派送")),
    ("客服机器人无效回复", ("机器人",)),
]


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", force_ascii=False, date_format="iso"))


def load_tickets(path: str | Path) -> pd.DataFrame:
    """读取并严格校验工单 JSON。"""
    source = Path(path)
    if not source.exists():
        raise ValueError(f"数据文件不存在：{source}")

    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败：第 {exc.lineno} 行第 {exc.colno} 列") from exc

    if not isinstance(raw, list):
        raise ValueError("数据顶层必须是 JSON 数组")
    if not raw:
        raise ValueError("数据集为空，无法生成分析报告")

    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 条工单必须是对象")
        missing = REQUIRED_FIELDS - item.keys()
        if missing:
            raise ValueError(f"第 {index} 条工单缺少字段：{', '.join(sorted(missing))}")
        if not isinstance(item["ticket_id"], str) or not item["ticket_id"].strip():
            raise ValueError(f"第 {index} 条工单 ticket_id 必须是非空字符串")
        if not isinstance(item["description"], str) or not item["description"].strip():
            raise ValueError(f"工单 {item['ticket_id']} 的 description 必须是非空字符串")
        if item["priority"] not in VALID_PRIORITIES:
            raise ValueError(f"工单 {item['ticket_id']} 的 priority 非法：{item['priority']}")
        if item["channel"] not in VALID_CHANNELS:
            raise ValueError(f"工单 {item['ticket_id']} 的 channel 非法：{item['channel']}")
        if isinstance(item["is_resolved"], bool) is False:
            raise ValueError(f"工单 {item['ticket_id']} 的 is_resolved 必须是布尔值")
        if isinstance(item["resolution_time_hours"], bool) or not isinstance(
            item["resolution_time_hours"], (int, float)
        ):
            raise ValueError(f"工单 {item['ticket_id']} 的 resolution_time_hours 必须是数值")
        if item["resolution_time_hours"] < 0:
            raise ValueError(f"工单 {item['ticket_id']} 的 resolution_time_hours 不能为负数")
        if isinstance(item["satisfaction"], bool) or not isinstance(item["satisfaction"], (int, float)):
            raise ValueError(f"工单 {item['ticket_id']} 的 satisfaction 必须是数值")
        if not 1 <= item["satisfaction"] <= 5:
            raise ValueError(f"工单 {item['ticket_id']} 的 satisfaction 必须在 1–5 之间")

    frame = pd.DataFrame(raw)
    if frame["ticket_id"].duplicated().any():
        duplicates = frame.loc[frame["ticket_id"].duplicated(), "ticket_id"].tolist()
        raise ValueError(f"ticket_id 重复：{', '.join(duplicates)}")

    frame["created_at"] = pd.to_datetime(frame["created_at"], format="%Y-%m-%d %H:%M", errors="coerce")
    invalid_dates = frame.loc[frame["created_at"].isna(), "ticket_id"].tolist()
    if invalid_dates:
        raise ValueError(f"创建时间格式非法：{', '.join(invalid_dates)}")

    return frame.sort_values(["created_at", "ticket_id"]).reset_index(drop=True)


def classify_topic(description: str) -> str:
    """按可审计关键词将原始描述归并为细分主题。"""
    for topic, keywords in TOPIC_RULES:
        if any(keyword in description for keyword in keywords):
            return topic
    return "其他"


def _risk_score(row: pd.Series) -> int:
    return (
        (0 if row["is_resolved"] else 4)
        + PRIORITY_SCORE[row["priority"]]
        + (2 if row["satisfaction"] <= 2 else 0)
        + (2 if row["resolution_time_hours"] >= 48 else 0)
    )


def _summary_table(frame: pd.DataFrame, group: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name, group_frame in frame.groupby(group, observed=True):
        result.append(
            {
                group: str(name),
                "count": int(len(group_frame)),
                "share_pct": _round(len(group_frame) / len(frame) * 100, 1),
                "resolved_rate_pct": _round(group_frame["is_resolved"].mean() * 100, 1),
                "low_satisfaction_rate_pct": _round((group_frame["satisfaction"] <= 2).mean() * 100, 1),
                "avg_satisfaction": _round(group_frame["satisfaction"].mean()),
                "avg_resolution_hours": _round(group_frame["resolution_time_hours"].mean()),
                "median_resolution_hours": _round(group_frame["resolution_time_hours"].median()),
                "p90_resolution_hours": _round(group_frame["resolution_time_hours"].quantile(0.9)),
            }
        )
    return sorted(result, key=lambda item: (-item["count"], item[group]))


def _window_comparison(frame: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start = frame["date"].min()
    split = start + pd.Timedelta(days=5)
    end = frame["date"].max()
    baseline_days = int((split - start).days)
    recent_days = int((end - split).days) + 1
    baseline = frame[frame["date"] < split]
    recent = frame[frame["date"] >= split]
    categories = sorted(frame["category"].unique())
    rows: list[dict[str, Any]] = []

    for category in categories:
        baseline_count = int((baseline["category"] == category).sum())
        recent_count = int((recent["category"] == category).sum())
        baseline_rate = baseline_count / baseline_days
        recent_rate = recent_count / recent_days
        growth = math.inf if baseline_rate == 0 and recent_rate > 0 else (
            0.0 if baseline_rate == 0 else (recent_rate / baseline_rate - 1) * 100
        )
        rows.append(
            {
                "category": category,
                "baseline_count": baseline_count,
                "recent_count": recent_count,
                "baseline_daily_avg": _round(baseline_rate),
                "recent_daily_avg": _round(recent_rate),
                "daily_avg_growth_pct": None if math.isinf(growth) else _round(growth, 1),
                "is_surge": recent_count >= 5 and growth >= 50,
            }
        )

    metadata = {
        "baseline_start": start.isoformat(),
        "baseline_end": (split - pd.Timedelta(days=1)).isoformat(),
        "baseline_days": baseline_days,
        "recent_start": split.isoformat(),
        "recent_end": end.isoformat(),
        "recent_days": recent_days,
    }
    return metadata, rows


def _ticket_risks(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "ticket_id",
        "created_at",
        "category",
        "topic",
        "description",
        "priority",
        "resolution_time_hours",
        "satisfaction",
        "channel",
        "is_resolved",
        "risk_score",
    ]
    ordered = frame.sort_values(
        ["risk_score", "is_resolved", "priority_rank", "resolution_time_hours", "created_at"],
        ascending=[False, True, False, False, False],
    )
    return _records(ordered[columns])


def _detect_anomalies(
    frame: pd.DataFrame,
    window: dict[str, Any],
    window_rows: list[dict[str, Any]],
    category_stats: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []

    for row in window_rows:
        if row["is_surge"]:
            growth_text = (
                "从零新增" if row["daily_avg_growth_pct"] is None else f"增长 {row['daily_avg_growth_pct']}%"
            )
            anomalies.append(
                {
                    "id": f"category-surge-{row['category']}",
                    "level": "立即处理" if row["daily_avg_growth_pct"] is None or row["daily_avg_growth_pct"] >= 100 else "持续关注",
                    "title": f"{row['category']}近期日均量明显上升",
                    "evidence": (
                        f"前窗 {row['baseline_count']} 条/{window['baseline_days']} 天，"
                        f"后窗 {row['recent_count']} 条/{window['recent_days']} 天，日均量{growth_text}。"
                    ),
                    "ticket_ids": frame.loc[
                        (frame["category"] == row["category"]) & (frame["window"] == "recent"), "ticket_id"
                    ].tolist(),
                    "action": "检查近期产品或流程变更，安排专项归因并评估客服承接能力。",
                }
            )

    last_three_start = frame["date"].max() - pd.Timedelta(days=2)
    for topic, group in frame[frame["topic"] != "其他"].groupby("topic"):
        if len(group) < 2:
            continue
        recent_count = int((group["date"] >= last_three_start).sum())
        escalated = bool((~group["is_resolved"]).any() or (group["satisfaction"] <= 2).any() or recent_count >= 2)
        anomalies.append(
            {
                "id": f"repeated-topic-{topic}",
                "level": "持续关注" if escalated else "信息提示",
                "title": f"“{topic}”重复出现",
                "evidence": (
                    f"共 {len(group)} 条，未解决 {int((~group['is_resolved']).sum())} 条，"
                    f"低满意度 {int((group['satisfaction'] <= 2).sum())} 条，最近 3 天 {recent_count} 条。"
                ),
                "ticket_ids": group["ticket_id"].tolist(),
                "action": "合并排查共同根因，避免逐单处理掩盖系统性问题。",
            }
        )

    unresolved_high = frame[(frame["priority"] == "高") & (~frame["is_resolved"])]
    if not unresolved_high.empty:
        anomalies.append(
            {
                "id": "high-priority-backlog",
                "level": "立即处理",
                "title": "高优先级未解决工单积压",
                "evidence": f"共有 {len(unresolved_high)} 条高优先级工单尚未解决。",
                "ticket_ids": unresolved_high["ticket_id"].tolist(),
                "action": "逐单指定负责人和完成时限，优先处理风险分最高的工单。",
            }
        )

    long_running = frame[frame["resolution_time_hours"] >= 48]
    if not long_running.empty:
        anomalies.append(
            {
                "id": "long-resolution-tail",
                "level": "立即处理" if (~long_running["is_resolved"]).any() else "持续关注",
                "title": "处理时长长尾明显",
                "evidence": (
                    f"{len(long_running)} 条工单处理时长达到 48 小时，"
                    f"其中 {int((~long_running['is_resolved']).sum())} 条仍未解决。"
                ),
                "ticket_ids": long_running["ticket_id"].tolist(),
                "action": "复盘跨团队等待、退款审核和物流协同环节。",
            }
        )

    for row in category_stats:
        if row["count"] >= 3 and row["low_satisfaction_rate_pct"] >= 50:
            tickets = frame.loc[
                (frame["category"] == row["category"]) & (frame["satisfaction"] <= 2), "ticket_id"
            ].tolist()
            anomalies.append(
                {
                    "id": f"experience-{row['category']}",
                    "level": "持续关注",
                    "title": f"{row['category']}低满意度集中",
                    "evidence": (
                        f"{row['count']} 条工单中低分率为 {row['low_satisfaction_rate_pct']}%，"
                        f"平均满意度 {row['avg_satisfaction']} 分。"
                    ),
                    "ticket_ids": tickets,
                    "action": "抽查低分原文，区分产品故障、政策争议和服务响应问题。",
                }
            )

    level_rank = {"立即处理": 0, "持续关注": 1, "信息提示": 2}
    return sorted(anomalies, key=lambda item: (level_rank[item["level"]], item["id"]))


def analyze_tickets(frame: pd.DataFrame) -> dict[str, Any]:
    """生成可序列化的完整分析结果。"""
    data = frame.copy()
    data["date"] = data["created_at"].dt.normalize()
    data["topic"] = data["description"].map(classify_topic)
    split = data["date"].min() + pd.Timedelta(days=5)
    data["window"] = data["date"].map(lambda value: "baseline" if value < split else "recent")
    data["risk_score"] = data.apply(_risk_score, axis=1)
    data["priority_rank"] = data["priority"].map(PRIORITY_SCORE)

    all_dates = pd.date_range(data["date"].min(), data["date"].max(), freq="D")
    daily = data.groupby("date").size().reindex(all_dates, fill_value=0)
    daily_trend = [{"date": date.date().isoformat(), "count": int(count)} for date, count in daily.items()]

    category_stats = _summary_table(data, "category")
    topic_stats = _summary_table(data, "topic")
    channel_stats = _summary_table(data, "channel")
    window, window_rows = _window_comparison(data)
    anomalies = _detect_anomalies(data, window, window_rows, category_stats)

    overview = {
        "ticket_count": int(len(data)),
        "date_start": data["date"].min().date().isoformat(),
        "date_end": data["date"].max().date().isoformat(),
        "resolved_count": int(data["is_resolved"].sum()),
        "unresolved_count": int((~data["is_resolved"]).sum()),
        "resolved_rate_pct": _round(data["is_resolved"].mean() * 100, 1),
        "high_priority_count": int((data["priority"] == "高").sum()),
        "avg_satisfaction": _round(data["satisfaction"].mean()),
        "low_satisfaction_rate_pct": _round((data["satisfaction"] <= 2).mean() * 100, 1),
        "avg_resolution_hours": _round(data["resolution_time_hours"].mean()),
        "median_resolution_hours": _round(data["resolution_time_hours"].median()),
        "p90_resolution_hours": _round(data["resolution_time_hours"].quantile(0.9)),
    }

    return {
        "metadata": {
            "title": "客服工单趋势与异常分析",
            "method": "确定性统计规则与关键词主题归并",
            "limitations": [
                "样本仅覆盖 11 天和 50 条工单，趋势仅表示样本内变化。",
                "48 小时是分析观察阈值，不代表企业正式 SLA。",
                "关联关系表示共现，不代表因果。",
            ],
        },
        "overview": overview,
        "window": window,
        "daily_trend": daily_trend,
        "window_comparison": window_rows,
        "category_stats": category_stats,
        "topic_stats": topic_stats,
        "channel_stats": channel_stats,
        "satisfaction_distribution": [
            {"score": score, "count": int((data["satisfaction"] == score).sum())}
            for score in range(1, 6)
        ],
        "priority_resolution": [
            {
                "priority": priority,
                "resolved": int(((data["priority"] == priority) & data["is_resolved"]).sum()),
                "unresolved": int(((data["priority"] == priority) & (~data["is_resolved"])).sum()),
            }
            for priority in ("高", "中", "低")
        ],
        "anomalies": anomalies,
        "ticket_risks": _ticket_risks(data),
    }
