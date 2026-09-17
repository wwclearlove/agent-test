"""一条命令生成结构化分析结果与离线 Dashboard。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.analyze import analyze_tickets, load_tickets
from src.render_report import render_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析客服工单趋势与异常")
    parser.add_argument("--input", type=Path, default=Path("data/tickets.json"), help="工单 JSON 路径")
    parser.add_argument("--output-dir", type=Path, default=Path("report"), help="报告输出目录")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        frame = load_tickets(args.input)
        result = analyze_tickets(frame)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        summary_path = args.output_dir / "analysis_summary.json"
        summary_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        dashboard_path = render_dashboard(result, args.output_dir / "dashboard.html")
    except (OSError, ValueError) as exc:
        print(f"分析失败：{exc}", file=sys.stderr)
        return 1

    overview = result["overview"]
    print(f"分析完成：{overview['ticket_count']} 条工单，发现 {len(result['anomalies'])} 个异常信号")
    print(f"结构化结果：{summary_path}")
    print(f"离线 Dashboard：{dashboard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
