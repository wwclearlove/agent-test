import json
import re
from pathlib import Path

from src.analyze import analyze_tickets, load_tickets
from src.render_report import render_dashboard


DATA_PATH = Path(__file__).parents[1] / "data" / "tickets.json"


def test_structured_result_is_json_serializable() -> None:
    result = analyze_tickets(load_tickets(DATA_PATH))

    encoded = json.dumps(result, ensure_ascii=False)

    assert "支付问题" in encoded
    assert "ticket_risks" in encoded


def test_dashboard_is_self_contained_and_has_required_sections(tmp_path: Path) -> None:
    result = analyze_tickets(load_tickets(DATA_PATH))
    output = render_dashboard(result, tmp_path / "dashboard.html")
    html = output.read_text(encoding="utf-8")

    assert output.exists()
    assert "<!doctype html>" in html
    assert "主管今日应关注" in html
    assert "趋势变化" in html
    assert "问题结构" in html
    assert "客户体验" in html
    assert "plotly.js" in html
    assert re.search(r'<script[^>]+src=["\']https?://', html, flags=re.IGNORECASE) is None
    output.unlink()
