"""将结构化分析结果渲染为可离线打开的 HTML Dashboard。"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


COLORS = {
    "ink": "#14213d",
    "muted": "#64748b",
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "red": "#dc2626",
    "orange": "#ea580c",
    "green": "#059669",
    "grid": "#e2e8f0",
}


def _style_figure(figure: go.Figure, height: int = 360) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=52, r=28, t=76, b=76),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family='"Microsoft YaHei", "PingFang SC", sans-serif', color=COLORS["ink"]),
        title=dict(font=dict(size=17), x=0.02, xanchor="left", y=0.96, yanchor="top"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="left",
            x=0,
            font=dict(size=11),
            title_text="",
            bgcolor="rgba(255,255,255,.88)",
        ),
        hoverlabel=dict(font_size=13),
    )
    figure.update_xaxes(gridcolor=COLORS["grid"], zeroline=False)
    figure.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    return figure


def _chart_html(figure: go.Figure, include_js: bool = False) -> str:
    return pio.to_html(
        figure,
        full_html=False,
        include_plotlyjs=True if include_js else False,
        config={"displayModeBar": False, "responsive": True},
    )


def _build_figures(result: dict[str, Any]) -> list[str]:
    daily = pd.DataFrame(result["daily_trend"])
    daily["date"] = pd.to_datetime(daily["date"])
    daily_figure = px.line(
        daily,
        x="date",
        y="count",
        markers=True,
        title="每日工单量",
        labels={"date": "日期", "count": "工单数"},
        color_discrete_sequence=[COLORS["blue"]],
    )
    daily_figure.update_traces(line=dict(width=3), marker=dict(size=8))

    window = pd.DataFrame(result["window_comparison"])
    window_long = window.melt(
        id_vars="category",
        value_vars=["baseline_daily_avg", "recent_daily_avg"],
        var_name="window",
        value_name="daily_avg",
    )
    window_long["window"] = window_long["window"].map(
        {"baseline_daily_avg": "前窗日均", "recent_daily_avg": "后窗日均"}
    )
    window_figure = px.bar(
        window_long,
        x="category",
        y="daily_avg",
        color="window",
        barmode="group",
        text_auto=".2f",
        title="类别日均量：前后时间窗对比",
        labels={"category": "类别", "daily_avg": "日均工单", "window": "时间窗"},
        color_discrete_map={"前窗日均": "#94a3b8", "后窗日均": COLORS["blue"]},
    )

    categories = pd.DataFrame(result["category_stats"]).sort_values("count")
    category_figure = px.bar(
        categories,
        x="count",
        y="category",
        orientation="h",
        text="count",
        title="工单类别分布",
        labels={"count": "工单数", "category": ""},
        color="low_satisfaction_rate_pct",
        color_continuous_scale=["#bfdbfe", "#fb923c", "#dc2626"],
    )
    category_figure.update_coloraxes(colorbar_title="低分率 %")

    topics = pd.DataFrame(result["topic_stats"])
    topics = topics[topics["topic"] != "其他"].sort_values("count")
    topic_figure = px.bar(
        topics,
        x="count",
        y="topic",
        orientation="h",
        text="count",
        title="可行动的重复问题主题",
        labels={"count": "工单数", "topic": ""},
        color_discrete_sequence=[COLORS["cyan"]],
    )

    priority = pd.DataFrame(result["priority_resolution"])
    priority_figure = go.Figure()
    priority_figure.add_bar(
        name="已解决", x=priority["priority"], y=priority["resolved"], marker_color=COLORS["green"]
    )
    priority_figure.add_bar(
        name="未解决", x=priority["priority"], y=priority["unresolved"], marker_color=COLORS["red"]
    )
    priority_figure.update_layout(
        barmode="stack",
        title="优先级与解决状态",
        xaxis_title="优先级",
        yaxis_title="工单数",
    )

    tickets = pd.DataFrame(result["ticket_risks"])
    efficiency_figure = px.box(
        tickets,
        x="category",
        y="resolution_time_hours",
        color="category",
        points="all",
        title="各类别处理时长分布",
        labels={"category": "类别", "resolution_time_hours": "处理时长（小时）"},
    )
    efficiency_figure.update_layout(showlegend=False)

    satisfaction = pd.DataFrame(result["satisfaction_distribution"])
    satisfaction_figure = px.bar(
        satisfaction,
        x="score",
        y="count",
        text="count",
        title="满意度评分分布",
        labels={"score": "满意度评分", "count": "工单数"},
        color="score",
        color_continuous_scale=["#dc2626", "#fb923c", "#fde68a", "#86efac", "#059669"],
    )
    satisfaction_figure.update_coloraxes(showscale=False)

    relation_figure = go.Figure()
    palette = px.colors.qualitative.Safe
    for index, (category, category_tickets) in enumerate(tickets.groupby("category", sort=True)):
        relation_figure.add_trace(
            go.Scatter(
                x=category_tickets["resolution_time_hours"],
                y=category_tickets["satisfaction"],
                mode="markers",
                name=category,
                marker=dict(
                    color=palette[index % len(palette)],
                    size=10,
                    opacity=0.82,
                    symbol=["circle" if resolved else "diamond-open" for resolved in category_tickets["is_resolved"]],
                    line=dict(width=1.5),
                ),
                customdata=category_tickets[["ticket_id", "priority", "description", "is_resolved"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b> · " + category + "<br>"
                    "处理时长：%{x} 小时<br>满意度：%{y}<br>"
                    "优先级：%{customdata[1]}<br>已解决：%{customdata[3]}<br>"
                    "%{customdata[2]}<extra></extra>"
                ),
            )
        )
    relation_figure.update_layout(
        title="处理时长与满意度（共现关系）",
        xaxis_title="处理时长（小时）",
        yaxis_title="满意度",
        annotations=[
            dict(
                text="● 已解决　◇ 未解决",
                x=1,
                y=1.08,
                xref="paper",
                yref="paper",
                xanchor="right",
                showarrow=False,
                font=dict(size=11, color=COLORS["muted"]),
            )
        ],
    )
    relation_figure.update_yaxes(dtick=1, range=[0.7, 5.35])

    channel = pd.DataFrame(result["channel_stats"]).sort_values("median_resolution_hours", ascending=False)
    channel_figure = px.bar(
        channel,
        x="channel",
        y=["resolved_rate_pct", "low_satisfaction_rate_pct"],
        barmode="group",
        title="渠道解决率与低分率",
        labels={"channel": "渠道", "value": "比例（%）", "variable": "指标"},
        color_discrete_sequence=[COLORS["green"], COLORS["orange"]],
    )
    channel_figure.for_each_trace(
        lambda trace: trace.update(
            name={"resolved_rate_pct": "解决率", "low_satisfaction_rate_pct": "低分率"}.get(
                trace.name, trace.name
            )
        )
    )

    figures = [
        daily_figure,
        window_figure,
        category_figure,
        topic_figure,
        priority_figure,
        efficiency_figure,
        satisfaction_figure,
        relation_figure,
        channel_figure,
    ]
    chart_html: list[str] = []
    for index, figure in enumerate(figures):
        height = 430 if index in {5, 7} else 360
        chart_html.append(_chart_html(_style_figure(figure, height), include_js=index == 0))
    return chart_html


def _kpi(label: str, value: str, note: str) -> str:
    return (
        '<div class="kpi">'
        f'<div class="kpi-label">{escape(label)}</div>'
        f'<div class="kpi-value">{escape(value)}</div>'
        f'<div class="kpi-note">{escape(note)}</div>'
        "</div>"
    )


def _anomaly_card(item: dict[str, Any]) -> str:
    css_level = {"立即处理": "critical", "持续关注": "warning", "信息提示": "info"}[item["level"]]
    ticket_tags = "".join(f"<code>{escape(ticket_id)}</code>" for ticket_id in item["ticket_ids"])
    return f"""
    <article class="alert {css_level}">
      <div class="alert-head">
        <span class="level">{escape(item["level"])}</span>
        <h3>{escape(item["title"])}</h3>
      </div>
      <p><strong>判断依据：</strong>{escape(item["evidence"])}</p>
      <p><strong>建议动作：</strong>{escape(item["action"])}</p>
      <div class="tickets">{ticket_tags}</div>
    </article>
    """


def _action_plan_html(action_plan: dict[str, Any]) -> str:
    workstreams = "".join(
        f"""
        <article class="workstream">
          <div class="workstream-title"><span>{index}</span><h3>{escape(item['workstream'])}</h3></div>
          <p class="metric">{item['count']} 条相关工单 · {item['unresolved_count']} 条未解决</p>
          <p><strong>负责团队：</strong>{escape(item['owner'])}</p>
          <p><strong>建议时限：</strong>{escape(item['deadline'])}</p>
          <p>{escape(item['action'])}</p>
          <div class="tickets">{''.join(f'<code>{escape(ticket_id)}</code>' for ticket_id in item['ticket_ids'])}</div>
        </article>
        """
        for index, item in enumerate(action_plan["payment_response"], start=1)
    )
    backlog_rows = "".join(
        f"""
        <tr>
          <td><strong>#{item['rank']}</strong></td>
          <td><code>{escape(item['ticket_id'])}</code><small>{escape(item['category'])}</small></td>
          <td>{escape(item['description'])}</td>
          <td><strong>{item['risk_score']}</strong><small>{item['current_age_hours']}h</small></td>
          <td>{escape(item['owner'])}<small>{escape(item['deadline'])}</small></td>
        </tr>
        """
        for item in action_plan["high_priority_backlog"]
    )
    return f"""
      <div class="subhead">
        <h3>支付问题：三路并行处置</h3>
        <p>先止损和补偿，再定位系统根因。</p>
      </div>
      <div class="workstreams">{workstreams}</div>
      <div class="subhead backlog-head">
        <h3>7 条高优先级未解决工单：建议跟进顺序</h3>
        <p>{escape(action_plan['notice'])}</p>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>顺序</th><th>工单</th><th>问题</th><th>风险分</th><th>建议负责人 / 时限</th></tr></thead>
          <tbody>{backlog_rows}</tbody>
        </table>
      </div>
    """


def render_dashboard(result: dict[str, Any], output_path: str | Path) -> Path:
    """生成内嵌 Plotly 资源的单文件 HTML Dashboard。"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    charts = _build_figures(result)
    overview = result["overview"]
    window = result["window"]

    kpis = "".join(
        [
            _kpi("工单总量", str(overview["ticket_count"]), f"{overview['date_start']} 至 {overview['date_end']}"),
            _kpi(
                "未解决",
                str(overview["unresolved_count"]),
                f"解决率 {overview['resolved_rate_pct']}%",
            ),
            _kpi("高优先级", str(overview["high_priority_count"]), "优先检查未解决项"),
            _kpi("平均满意度", f"{overview['avg_satisfaction']}/5", f"低分率 {overview['low_satisfaction_rate_pct']}%"),
            _kpi(
                "处理时长中位数",
                f"{overview['median_resolution_hours']}h",
                f"P90 为 {overview['p90_resolution_hours']}h",
            ),
        ]
    )
    alerts = "".join(_anomaly_card(item) for item in result["anomalies"])
    action_plan = _action_plan_html(result["action_plan"])
    limitations = "".join(f"<li>{escape(item)}</li>" for item in result["metadata"]["limitations"])
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>客服工单趋势与异常分析</title>
  <style>
    :root {{
      --ink: #14213d; --muted: #64748b; --line: #e2e8f0; --surface: #ffffff;
      --bg: #f4f7fb; --blue: #2563eb; --red: #dc2626; --orange: #ea580c; --cyan: #0891b2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; color: var(--ink); background: var(--bg);
      font-family: "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
    }}
    .hero {{
      padding: 52px max(24px, calc((100vw - 1240px) / 2)); color: white;
      background: radial-gradient(circle at 80% 0%, #2563eb 0, #14213d 52%, #0f172a 100%);
    }}
    .eyebrow {{ margin: 0 0 12px; color: #93c5fd; font-weight: 700; letter-spacing: .08em; }}
    h1 {{ margin: 0; font-size: clamp(30px, 4vw, 50px); line-height: 1.15; }}
    .subtitle {{ max-width: 760px; margin: 16px 0 0; color: #cbd5e1; line-height: 1.75; }}
    main {{ width: min(1240px, calc(100% - 40px)); margin: -24px auto 60px; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; }}
    .kpi, .panel, .alert {{
      background: var(--surface); border: 1px solid var(--line); border-radius: 16px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, .05);
    }}
    .kpi {{ padding: 22px; min-height: 132px; }}
    .kpi-label {{ color: var(--muted); font-size: 14px; font-weight: 700; }}
    .kpi-value {{ margin: 8px 0 4px; font-size: 30px; font-weight: 800; }}
    .kpi-note {{ color: var(--muted); font-size: 12px; }}
    section {{ margin-top: 34px; scroll-margin-top: 20px; }}
    .section-head {{ display: flex; align-items: end; justify-content: space-between; gap: 24px; margin-bottom: 14px; }}
    h2 {{ margin: 0; font-size: 25px; }}
    .section-head p {{ max-width: 680px; margin: 0; color: var(--muted); line-height: 1.6; }}
    .alerts {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .alert {{ padding: 20px 22px; border-left-width: 5px; }}
    .alert.critical {{ grid-column: 1 / -1; }}
    .alert.critical {{ border-left-color: var(--red); }}
    .alert.warning {{ border-left-color: var(--orange); }}
    .alert.info {{ border-left-color: var(--cyan); }}
    .alert-head {{ display: flex; align-items: center; gap: 12px; }}
    .alert h3 {{ margin: 0; font-size: 18px; }}
    .alert p {{ margin: 10px 0 0; color: #334155; line-height: 1.65; }}
    .level {{ padding: 5px 9px; border-radius: 99px; font-size: 12px; font-weight: 800; white-space: nowrap; }}
    .critical .level {{ color: #991b1b; background: #fee2e2; }}
    .warning .level {{ color: #9a3412; background: #ffedd5; }}
    .info .level {{ color: #155e75; background: #cffafe; }}
    .tickets {{ display: flex; flex-wrap: wrap; gap: 7px; margin-top: 12px; }}
    code {{ padding: 4px 7px; border-radius: 6px; color: #334155; background: #f1f5f9; }}
    .subhead {{ display: flex; align-items: end; justify-content: space-between; gap: 20px; margin: 24px 0 12px; }}
    .subhead h3, .subhead p {{ margin: 0; }}
    .subhead p {{ color: var(--muted); font-size: 13px; }}
    .workstreams {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
    .workstream {{ padding: 20px; background: white; border: 1px solid var(--line); border-radius: 16px; }}
    .workstream-title {{ display: flex; align-items: center; gap: 10px; }}
    .workstream-title span {{ display: grid; place-items: center; width: 28px; height: 28px; color: white; background: var(--blue); border-radius: 9px; font-weight: 800; }}
    .workstream h3 {{ margin: 0; font-size: 17px; }}
    .workstream p {{ margin: 9px 0 0; color: #334155; font-size: 13px; line-height: 1.55; }}
    .workstream .metric {{ color: var(--blue); font-weight: 800; }}
    .backlog-head {{ margin-top: 30px; }}
    .table-wrap {{ overflow-x: auto; background: white; border: 1px solid var(--line); border-radius: 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{ color: var(--muted); background: #f8fafc; text-align: left; }}
    th, td {{ padding: 13px 14px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    td small {{ display: block; margin-top: 6px; color: var(--muted); line-height: 1.5; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .experience-grid {{ display: grid; grid-template-columns: minmax(320px, .72fr) minmax(0, 1.45fr); gap: 16px; }}
    .panel {{ min-width: 0; padding: 12px; overflow: hidden; }}
    .method {{ padding: 24px 28px; line-height: 1.75; }}
    .method-grid {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 40px; }}
    .method h3 {{ margin-top: 0; }}
    footer {{ margin-top: 28px; color: var(--muted); font-size: 13px; text-align: right; }}
    @media (max-width: 1080px) {{
      .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
      .grid-2, .experience-grid, .method-grid, .workstreams {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 560px) {{
      main {{ width: min(100% - 24px, 1240px); }}
      .kpi-grid {{ grid-template-columns: 1fr; }}
      .alerts {{ grid-template-columns: 1fr; }}
      .section-head, .alert-head {{ align-items: flex-start; flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <p class="eyebrow">CUSTOMER SUPPORT SIGNALS · 2024-06</p>
    <h1>客服工单趋势与异常分析</h1>
    <p class="subtitle">从 50 条工单中提取可行动信号：先处理积压与复合风险，再定位近期增长和重复根因。所有结论均可追溯到明确规则与工单。</p>
  </header>
  <main>
    <div class="kpi-grid">{kpis}</div>

    <section>
      <div class="section-head">
        <h2>主管今日应关注</h2>
        <p>异常按风险等级排序；每一项都展示触发依据、关联工单与建议动作。</p>
      </div>
      <div class="alerts">{alerts}</div>
    </section>

    <section>
      <div class="section-head">
        <h2>立即行动方案</h2>
        <p>把异常信号转换为可分派任务；建议时限用于内部推进，不代表正式 SLA。</p>
      </div>
      {action_plan}
    </section>

    <section>
      <div class="section-head">
        <h2>趋势变化</h2>
        <p>前窗为 {window['baseline_start'].split('T')[0]} 至 {window['baseline_end'].split('T')[0]}，后窗为 {window['recent_start'].split('T')[0]} 至 {window['recent_end'].split('T')[0]}；使用日均量消除窗口天数差异。</p>
      </div>
      <div class="grid-2"><div class="panel">{charts[0]}</div><div class="panel">{charts[1]}</div></div>
    </section>

    <section>
      <div class="section-head">
        <h2>问题结构</h2>
        <p>分类看资源分布，细分主题看可执行根因；颜色越暖表示该类别低满意度比例越高。</p>
      </div>
      <div class="grid-2"><div class="panel">{charts[2]}</div><div class="panel">{charts[3]}</div></div>
      <div class="panel" style="margin-top:16px">{charts[4]}</div>
    </section>

    <section>
      <div class="section-head">
        <h2>效率与渠道</h2>
        <p>同时观察中位数、分布和解决率，避免均值掩盖退款及物流问题的处理长尾。</p>
      </div>
      <div class="grid-2"><div class="panel">{charts[5]}</div><div class="panel">{charts[8]}</div></div>
    </section>

    <section>
      <div class="section-head">
        <h2>客户体验</h2>
        <p>满意度与处理时长仅表示样本内共现，不作为因果关系判断。</p>
      </div>
      <div class="experience-grid"><div class="panel">{charts[6]}</div><div class="panel">{charts[7]}</div></div>
    </section>

    <section class="panel method">
      <div class="method-grid">
        <div>
          <h3>判定方法</h3>
          <p>类别激增要求后窗至少 5 条且日均量增长不低于 50%；重复主题至少出现 2 次；高优先级未解决直接列为立即处理；48 小时作为处理长尾观察阈值；1–2 分定义为低满意度。</p>
        </div>
        <div>
          <h3>阅读限制</h3>
          <ul>{limitations}</ul>
        </div>
      </div>
    </section>
    <footer>生成时间：{escape(generated_at)} · 分析逻辑：确定性规则，可离线复现</footer>
  </main>
</body>
</html>"""
    output.write_text("\n".join(line.rstrip() for line in html.splitlines()) + "\n", encoding="utf-8")
    return output
