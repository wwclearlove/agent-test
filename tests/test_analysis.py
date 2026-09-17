import json
from pathlib import Path

import pytest

from src.analyze import analyze_tickets, classify_topic, load_tickets


DATA_PATH = Path(__file__).parents[1] / "data" / "tickets.json"


def test_loads_all_tickets_and_core_metrics() -> None:
    frame = load_tickets(DATA_PATH)
    result = analyze_tickets(frame)

    assert len(frame) == 50
    assert frame["ticket_id"].is_unique
    assert result["overview"]["ticket_count"] == 50
    assert result["overview"]["unresolved_count"] == 8
    assert result["overview"]["date_start"] == "2024-06-01"
    assert result["overview"]["date_end"] == "2024-06-11"


def test_window_comparison_uses_daily_average() -> None:
    result = analyze_tickets(load_tickets(DATA_PATH))
    payment = next(row for row in result["window_comparison"] if row["category"] == "支付问题")

    assert result["window"]["baseline_days"] == 5
    assert result["window"]["recent_days"] == 6
    assert payment["baseline_count"] == 3
    assert payment["recent_count"] == 13
    assert payment["baseline_daily_avg"] == 0.6
    assert payment["recent_daily_avg"] == 2.17
    assert payment["daily_avg_growth_pct"] == pytest.approx(261.1)
    assert payment["is_surge"] is True


@pytest.mark.parametrize(
    ("description", "topic"),
    [
        ("微信支付扣了两次钱", "重复或错误扣款"),
        ("银行卡扣款了但订单没成功", "支付成功但订单异常"),
        ("结算页面打不开", "支付页面或结算失败"),
        ("退款还在处理中", "退款审核或到账延迟"),
        ("退货运费什么时候报销", "退货运费报销"),
        ("快递信息四天没更新", "物流停滞或异常"),
        ("客服机器人重复回复", "客服机器人无效回复"),
        ("想咨询商品尺寸", "其他"),
    ],
)
def test_topic_classification_is_deterministic(description: str, topic: str) -> None:
    assert classify_topic(description) == topic


def test_anomalies_are_traceable_to_rules_and_tickets() -> None:
    result = analyze_tickets(load_tickets(DATA_PATH))
    anomalies = result["anomalies"]

    payment_surge = next(item for item in anomalies if item["id"] == "category-surge-支付问题")
    backlog = next(item for item in anomalies if item["id"] == "high-priority-backlog")
    assert payment_surge["level"] == "立即处理"
    assert "261.1%" in payment_surge["evidence"]
    assert set(backlog["ticket_ids"]) == {"T019", "T031", "T036", "T039", "T042", "T046", "T047"}


def test_rejects_missing_fields(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps([{"ticket_id": "T001"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="缺少字段"):
        load_tickets(invalid_path)


def test_rejects_invalid_values(tmp_path: Path) -> None:
    raw = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    raw[0]["satisfaction"] = 6
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="1–5"):
        load_tickets(invalid_path)
