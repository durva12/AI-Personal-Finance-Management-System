"""
pdf_report.py — FinanceAI Analysis PDF Generator
==================================================
Generates a professional multi-section PDF report from analysis data.
Uses reportlab Platypus (flowable-based layout) for clean pagination.
"""

from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String, Circle
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF

# ── Color palette ─────────────────────────────────────────────────────────────
C_BG        = colors.HexColor('#0f172a')
C_SURFACE   = colors.HexColor('#1e293b')
C_ACCENT    = colors.HexColor('#6366f1')
C_GREEN     = colors.HexColor('#22c55e')
C_RED       = colors.HexColor('#ef4444')
C_YELLOW    = colors.HexColor('#f59e0b')
C_BLUE      = colors.HexColor('#3b82f6')
C_TEXT      = colors.HexColor('#1e293b')
C_MUTED     = colors.HexColor('#64748b')
C_BORDER    = colors.HexColor('#e2e8f0')
C_HIGHLIGHT = colors.HexColor('#f8fafc')

CHART_COLORS = [
    colors.HexColor('#6366f1'), colors.HexColor('#22c55e'),
    colors.HexColor('#f59e0b'), colors.HexColor('#ef4444'),
    colors.HexColor('#3b82f6'), colors.HexColor('#a855f7'),
    colors.HexColor('#06b6d4'), colors.HexColor('#84cc16'),
]

CATEGORIES = [
    "Housing", "Food", "Transport", "Entertainment",
    "Healthcare", "Savings", "Debt Repayment", "Miscellaneous"
]

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm


def _styles():
    base = getSampleStyleSheet()
    return {
        'title': ParagraphStyle('title', fontSize=22, fontName='Helvetica-Bold',
                                textColor=C_ACCENT, alignment=TA_CENTER, spaceAfter=4),
        'subtitle': ParagraphStyle('subtitle', fontSize=10, fontName='Helvetica',
                                   textColor=C_MUTED, alignment=TA_CENTER, spaceAfter=2),
        'section': ParagraphStyle('section', fontSize=12, fontName='Helvetica-Bold',
                                  textColor=C_TEXT, spaceBefore=14, spaceAfter=6),
        'body': ParagraphStyle('body', fontSize=9, fontName='Helvetica',
                               textColor=C_TEXT, spaceAfter=4, leading=14),
        'small': ParagraphStyle('small', fontSize=8, fontName='Helvetica',
                                textColor=C_MUTED, spaceAfter=2),
        'bold': ParagraphStyle('bold', fontSize=9, fontName='Helvetica-Bold',
                               textColor=C_TEXT, spaceAfter=2),
        'green': ParagraphStyle('green', fontSize=9, fontName='Helvetica-Bold',
                                textColor=C_GREEN, spaceAfter=2),
        'red': ParagraphStyle('red', fontSize=9, fontName='Helvetica-Bold',
                              textColor=C_RED, spaceAfter=2),
        'center': ParagraphStyle('center', fontSize=9, fontName='Helvetica',
                                 textColor=C_MUTED, alignment=TA_CENTER),
    }


def _divider():
    return HRFlowable(width='100%', thickness=0.5, color=C_BORDER,
                      spaceAfter=8, spaceBefore=4)


def _score_color(score: float):
    if score >= 80: return C_GREEN
    if score >= 65: return C_BLUE
    if score >= 50: return C_YELLOW
    if score >= 35: return colors.HexColor('#f97316')
    return C_RED


def _score_label(score: float) -> str:
    if score >= 80: return 'Excellent'
    if score >= 65: return 'Good'
    if score >= 50: return 'Fair'
    if score >= 35: return 'Needs Work'
    return 'Critical'


def _priority_color(priority: str):
    return {'High': C_RED, 'Medium': C_YELLOW, 'Low': C_ACCENT}.get(priority, C_MUTED)


# ── Section builders ──────────────────────────────────────────────────────────

def _build_header(story, styles, user_name, city, generated_at):
    # Top banner
    banner_data = [[
        Paragraph('FinanceAI', ParagraphStyle('logo', fontSize=18, fontName='Helvetica-Bold',
                                               textColor=C_ACCENT)),
        Paragraph('Personal Finance Analysis Report',
                  ParagraphStyle('hdr', fontSize=11, fontName='Helvetica',
                                 textColor=C_MUTED, alignment=TA_RIGHT)),
    ]]
    banner = Table(banner_data, colWidths=['50%', '50%'])
    banner.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LINEBELOW', (0, 0), (-1, -1), 1, C_ACCENT),
    ]))
    story.append(banner)
    story.append(Spacer(1, 10))

    # User info row
    info_data = [[
        Paragraph(f'<b>Prepared for:</b> {user_name}', styles['body']),
        Paragraph(f'<b>City:</b> {city}', styles['body']),
        Paragraph(f'<b>Generated:</b> {generated_at}', styles['body']),
    ]]
    info_tbl = Table(info_data, colWidths=['34%', '33%', '33%'])
    info_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), C_HIGHLIGHT),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('ROUNDEDCORNERS', [4]),
    ]))
    story.append(info_tbl)
    story.append(Spacer(1, 12))


def _build_score_summary(story, styles, rec, income):
    story.append(Paragraph('1. Financial Health Overview', styles['section']))
    story.append(_divider())

    score     = rec.get('health_score', 0)
    projected = rec.get('projected_score', 0)
    sc_color  = _score_color(score)
    sc_label  = _score_label(score)
    surplus   = rec.get('monthly_surplus', 0)
    total_exp = rec.get('total_expenses', 0)
    potential = rec.get('total_potential_savings', 0)
    summary   = rec.get('summary', '')

    score_data = [
        ['Health Score', 'Label', 'Projected', 'Income', 'Total Expenses', 'Monthly Surplus'],
        [
            Paragraph(f'<font color="{sc_color.hexval()}" size="16"><b>{score:.1f}/100</b></font>', styles['center']),
            Paragraph(f'<font color="{sc_color.hexval()}"><b>{sc_label}</b></font>', styles['center']),
            Paragraph(f'<b>{projected:.1f}</b>', styles['center']),
            Paragraph(f'<b>Rs. {income:,.0f}</b>', styles['center']),
            Paragraph(f'Rs. {total_exp:,.0f}', styles['center']),
            Paragraph(
                f'<font color="{"#22c55e" if surplus >= 0 else "#ef4444"}"><b>Rs. {surplus:,.0f}</b></font>',
                styles['center']
            ),
        ]
    ]
    score_tbl = Table(score_data, colWidths=['17%','14%','14%','18%','18%','19%'])
    score_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_ACCENT),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, 1), C_HIGHLIGHT),
        ('ALIGN',      (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID',       (0, 0), (-1, -1), 0.3, C_BORDER),
        ('ROUNDEDCORNERS', [4]),
    ]))
    story.append(score_tbl)

    if potential > 0:
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            f'Potential monthly savings if recommendations followed: '
            f'<font color="#22c55e"><b>Rs. {potential:,.0f}/month '
            f'(Rs. {potential*12:,.0f}/year)</b></font>',
            styles['body']
        ))

    if summary:
        story.append(Spacer(1, 6))
        story.append(Paragraph(f'<i>{summary}</i>', styles['small']))

    story.append(Spacer(1, 10))


def _build_spending_breakdown(story, styles, rec, income):
    story.append(Paragraph('2. Spending Breakdown', styles['section']))
    story.append(_divider())

    budget = rec.get('budget', {})
    rows   = [['Category', 'Current Amount', '% of Income', 'Recommended', 'Action', 'Status']]

    action_labels  = {'increase': 'Increase', 'decrease': 'Reduce', 'maintain': 'OK'}
    action_colors  = {'increase': C_GREEN, 'decrease': C_RED, 'maintain': C_MUTED}

    for cat in CATEGORIES:
        info = budget.get(cat)
        if not info:
            continue
        action   = info.get('action', 'maintain')
        is_fixed = info.get('is_fixed', False)
        ac_label = 'Fixed' if is_fixed else action_labels.get(action, 'OK')
        ac_color = C_ACCENT if is_fixed else action_colors.get(action, C_MUTED)

        curr_pct = info.get('current_pct', 0)
        status = (
            'Fixed' if is_fixed else
            'Good' if action == 'maintain' else
            'Over limit'
        )
        status_color = (
            C_ACCENT if is_fixed else
            C_GREEN  if action == 'maintain' else
            C_RED
        )

        rows.append([
            Paragraph(f'<b>{cat}</b>', styles['body']),
            Paragraph(f'Rs. {info["current"]:,.0f}', styles['body']),
            Paragraph(f'{curr_pct:.1f}%', styles['body']),
            Paragraph(f'Rs. {info["recommended"]:,.0f}', styles['body']),
            Paragraph(f'<font color="{ac_color.hexval()}"><b>{ac_label}</b></font>', styles['body']),
            Paragraph(f'<font color="{status_color.hexval()}">{status}</font>', styles['small']),
        ])

    tbl = Table(rows, colWidths=['22%', '16%', '13%', '16%', '14%', '19%'])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), C_ACCENT),
        ('TEXTCOLOR',     (0, 0), (-1, 0), colors.white),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0), 8),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [C_HIGHLIGHT, colors.white]),
        ('GRID',          (0, 0), (-1, -1), 0.3, C_BORDER),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING',   (0, 0), (-1, -1), 8),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))


def _build_pie_chart(story, styles, rec, income):
    """Spending pie chart using reportlab graphics."""
    budget = rec.get('budget', {})
    labels, values = [], []
    for cat in CATEGORIES:
        info = budget.get(cat)
        if info and info.get('current', 0) > 0:
            labels.append(cat[:10])
            values.append(float(info['current']))

    if not values:
        return

    d   = Drawing(PAGE_W - 2 * MARGIN, 160)
    pie = Pie()
    pie.x      = 20
    pie.y      = 10
    pie.width  = 140
    pie.height = 140
    pie.data   = values
    pie.labels = [f'{l}\n{v/sum(values)*100:.1f}%' for l, v in zip(labels, values)]
    pie.sideLabels     = True
    pie.sideLabelsOffset = 0.05
    pie.simpleLabels   = False

    for i, col in enumerate(CHART_COLORS[:len(values)]):
        pie.slices[i].fillColor    = col
        pie.slices[i].strokeColor  = colors.white
        pie.slices[i].strokeWidth  = 0.8
        pie.slices[i].labelRadius  = 1.25

    d.add(pie)

    # Legend on the right
    lx, ly = 210, 140
    for i, (label, val) in enumerate(zip(labels, values)):
        col = CHART_COLORS[i % len(CHART_COLORS)]
        r   = Rect(lx, ly - i * 16, 10, 10, fillColor=col, strokeColor=col)
        s   = String(lx + 14, ly - i * 16 + 2,
                     f'{label}: Rs.{val:,.0f}',
                     fontSize=7, fillColor=C_TEXT.clone())
        d.add(r)
        d.add(s)

    story.append(Paragraph('3. Spending Distribution', styles['section']))
    story.append(_divider())
    story.append(d)
    story.append(Spacer(1, 10))


def _build_recommendations(story, styles, rec):
    recs = rec.get('recommendations', [])
    story.append(Paragraph('4. AI Recommendations', styles['section']))
    story.append(_divider())

    if not recs:
        story.append(Paragraph(
            'All spending categories are within healthy limits. No changes recommended.',
            styles['green']
        ))
        story.append(Spacer(1, 10))
        return

    for r in recs:
        pc   = _priority_color(r.get('priority', 'Low'))
        diff = r.get('difference', 0)
        action_word = 'Increase' if r.get('action') == 'increase' else 'Reduce'
        amt_color   = C_GREEN if r.get('action') == 'increase' else C_RED

        block = [
            # Header row
            Table([[
                Paragraph(f'{r.get("icon","")} <b>{r["category"]}</b>', styles['bold']),
                Paragraph(
                    f'<font color="{pc.hexval()}"><b>{r.get("priority","Low")} Priority</b></font>',
                    ParagraphStyle('pr', fontSize=8, fontName='Helvetica-Bold',
                                   alignment=TA_RIGHT)
                ),
            ]], colWidths=['70%', '30%'],
            style=TableStyle([
                ('BACKGROUND',    (0,0),(-1,-1), colors.HexColor('#f1f5f9')),
                ('LEFTPADDING',   (0,0),(-1,-1), 8),
                ('RIGHTPADDING',  (0,0),(-1,-1), 8),
                ('TOPPADDING',    (0,0),(-1,-1), 6),
                ('BOTTOMPADDING', (0,0),(-1,-1), 6),
            ])),
            Spacer(1, 3),
            # Detail row
            Table([[
                Paragraph(f'<b>Current:</b> Rs. {r["current"]:,.0f} ({r["current_pct"]}%)', styles['small']),
                Paragraph(f'<b>Target:</b> Rs. {r["recommended"]:,.0f} ({r["recommended_pct"]}%)', styles['small']),
                Paragraph(
                    f'<font color="{amt_color.hexval()}"><b>{action_word} by Rs. {diff:,.0f}/mo</b></font>',
                    styles['small']
                ),
            ]], colWidths=['33%','34%','33%'],
            style=TableStyle([
                ('LEFTPADDING',   (0,0),(-1,-1), 8),
                ('TOPPADDING',    (0,0),(-1,-1), 4),
                ('BOTTOMPADDING', (0,0),(-1,-1), 4),
            ])),
            Spacer(1, 2),
            Paragraph(r.get('detail', r.get('reason', '')), styles['small']),
        ]

        # Trend alert
        if r.get('trend_flag') in ('spike', 'rising'):
            block.append(Paragraph(
                f'Trend alert: {r.get("trend_detail", "")}',
                ParagraphStyle('trend', fontSize=8, fontName='Helvetica-Oblique',
                               textColor=C_RED, leftIndent=8, spaceAfter=2)
            ))

        block.append(Spacer(1, 8))
        story.append(KeepTogether(block))

    story.append(Spacer(1, 6))


def _build_healthy(story, styles, rec):
    healthy = [h for h in rec.get('healthy', []) if h.get('status') != 'fixed']
    if not healthy:
        return

    story.append(Paragraph('5. Healthy Categories', styles['section']))
    story.append(_divider())

    rows = [['Category', 'Current Amount', '% of Income', 'Status', 'Note']]
    for h in healthy:
        star = '★' if h['status'] == 'excellent' else '✓'
        col  = C_GREEN if h['status'] == 'excellent' else C_BLUE
        rows.append([
            Paragraph(f'{h.get("icon","")} <b>{h["category"]}</b>', styles['body']),
            Paragraph(f'Rs. {h["current"]:,.0f}', styles['body']),
            Paragraph(f'{h["current_pct"]}%', styles['body']),
            Paragraph(f'<font color="{col.hexval()}"><b>{star} {h["status"].title()}</b></font>',
                      styles['body']),
            Paragraph(h.get('message', '')[:60], styles['small']),
        ])

    tbl = Table(rows, colWidths=['22%', '17%', '14%', '18%', '29%'])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), C_GREEN),
        ('TEXTCOLOR',     (0, 0), (-1, 0), colors.white),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0), 8),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [C_HIGHLIGHT, colors.white]),
        ('GRID',          (0, 0), (-1, -1), 0.3, C_BORDER),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING',   (0, 0), (-1, -1), 8),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))


def _build_cluster(story, styles, cluster):
    if not cluster:
        return
    story.append(Paragraph('6. Spending Profile', styles['section']))
    story.append(_divider())

    label = cluster.get('label', 'Unknown')
    emoji = cluster.get('emoji', '')
    desc  = cluster.get('description', '')
    risk  = cluster.get('risk_level', '')
    conf  = cluster.get('primary_confidence', 0)

    profile_data = [[
        Paragraph(f'<b>Profile:</b> {emoji} {label}', styles['bold']),
        Paragraph(f'<b>Risk Level:</b> {risk}', styles['body']),
        Paragraph(f'<b>Confidence:</b> {conf:.1f}%', styles['body']),
    ]]
    profile_tbl = Table(profile_data, colWidths=['40%', '30%', '30%'])
    profile_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), C_HIGHLIGHT),
        ('TOPPADDING',    (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ('GRID',          (0, 0), (-1, -1), 0.3, C_BORDER),
    ]))
    story.append(profile_tbl)
    story.append(Spacer(1, 6))
    story.append(Paragraph(desc, styles['small']))

    advice = cluster.get('advice', [])
    if advice:
        story.append(Spacer(1, 6))
        story.append(Paragraph('<b>Personalised Advice:</b>', styles['bold']))
        for tip in advice:
            story.append(Paragraph(f'• {tip}', styles['small']))

    story.append(Spacer(1, 10))


def _build_goal(story, styles, goal_analysis):
    if not goal_analysis:
        return
    story.append(Paragraph('7. Goal Progress', styles['section']))
    story.append(_divider())

    g = goal_analysis
    on_track    = g.get('on_track', False)
    track_color = C_GREEN if on_track else C_YELLOW

    goal_data = [[
        Paragraph(f'<b>Goal:</b> {g.get("goal_label","")}', styles['bold']),
        Paragraph(f'<b>Target:</b> Rs. {g.get("target_amount",0):,.0f}', styles['body']),
        Paragraph(f'<b>Monthly target:</b> Rs. {g.get("monthly_target",0):,.0f}', styles['body']),
        Paragraph(
            f'<font color="{track_color.hexval()}"><b>{"On Track" if on_track else "Off Track"}</b></font>',
            styles['body']
        ),
    ]]
    goal_tbl = Table(goal_data, colWidths=['30%', '22%', '25%', '23%'])
    goal_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), C_HIGHLIGHT),
        ('TOPPADDING',    (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ('GRID',          (0, 0), (-1, -1), 0.3, C_BORDER),
    ]))
    story.append(goal_tbl)
    story.append(Spacer(1, 6))
    story.append(Paragraph(g.get('gap_message', ''), styles['body']))

    progress = g.get('progress_pct', 0)
    # Simple text progress bar
    filled = int(progress / 5)
    bar    = '█' * filled + '░' * (20 - filled)
    story.append(Paragraph(
        f'Progress: {bar}  {progress:.1f}%',
        ParagraphStyle('bar', fontSize=9, fontName='Courier',
                       textColor=C_ACCENT, spaceAfter=4)
    ))
    story.append(Spacer(1, 10))


def _build_timeline(story, styles, timeline):
    if not timeline:
        return
    story.append(Paragraph('8. 12-Month Score Projection', styles['section']))
    story.append(_divider())

    rows = [['Month', 'Projected Score', 'Label', 'Projected Savings', 'Debt Remaining']]
    for t in timeline:
        sc    = t.get('score', 0)
        lbl   = _score_label(sc)
        col   = _score_color(sc)
        rows.append([
            Paragraph(f'Month {t["month"]}', styles['small']),
            Paragraph(f'<font color="{col.hexval()}"><b>{sc:.1f}</b></font>', styles['center']),
            Paragraph(f'<font color="{col.hexval()}">{lbl}</font>', styles['small']),
            Paragraph(f'Rs. {t.get("savings",0):,.0f}', styles['small']),
            Paragraph(f'Rs. {t.get("debt",0):,.0f}', styles['small']),
        ])

    tbl = Table(rows, colWidths=['15%', '20%', '20%', '22%', '23%'])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), colors.HexColor('#334155')),
        ('TEXTCOLOR',     (0, 0), (-1, 0), colors.white),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0), 8),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [C_HIGHLIGHT, colors.white]),
        ('GRID',          (0, 0), (-1, -1), 0.3, C_BORDER),
        ('ALIGN',         (1, 0), (2, -1), 'CENTER'),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',   (0, 0), (-1, -1), 8),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))


def _build_footer(story, styles):
    story.append(_divider())
    story.append(Paragraph(
        'This report was generated by FinanceAI — an AI-powered personal finance advisor. '
        'Recommendations are based on algorithmic analysis and should be used as guidance only. '
        'For major financial decisions, consult a certified financial advisor.',
        ParagraphStyle('footer', fontSize=7, fontName='Helvetica-Oblique',
                       textColor=C_MUTED, alignment=TA_CENTER)
    ))


# ── Public API ────────────────────────────────────────────────────────────────

def generate_pdf_report(
    user_name:    str,
    income:       float,
    city:         str,
    rec:          dict,
    cluster:      dict,
    timeline:     list,
    goal_analysis: dict | None = None,
) -> bytes:
    """
    Generate a PDF analysis report and return raw bytes.
    Call from Flask: send_file(BytesIO(bytes), mimetype='application/pdf')
    """
    buf        = BytesIO()
    generated  = datetime.now().strftime('%d %B %Y, %I:%M %p')

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=MARGIN,
        title=f'FinanceAI Report — {user_name}',
        author='FinanceAI',
        subject='Personal Finance Analysis',
    )

    s     = _styles()
    story = []

    _build_header(story, s, user_name, city, generated)
    _build_score_summary(story, s, rec, income)
    _build_spending_breakdown(story, s, rec, income)
    _build_pie_chart(story, s, rec, income)
    _build_recommendations(story, s, rec)
    _build_healthy(story, s, rec)
    _build_cluster(story, s, cluster)
    if goal_analysis:
        _build_goal(story, s, goal_analysis)
    _build_timeline(story, s, timeline)
    _build_footer(story, s)

    doc.build(story)
    return buf.getvalue()
