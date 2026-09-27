"""
Downloadable PDF site report (ReportLab + a matplotlib SHAP chart).

Sections: header with provenance, score summary (ML siting / physical / AHP),
energy estimate, remote-sensing site characteristics, SHAP explanation and a
short methodology + limitations note.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,  # noqa: E402
                                TableStyle)

_FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", str(_FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(_FONT_DIR / "DejaVuSans-Bold.ttf")))

INK = colors.HexColor("#0b0b0b")
INK_2 = colors.HexColor("#52514e")
MUTED = colors.HexColor("#898781")
HAIR = colors.HexColor("#e1e0d9")
PAPER = colors.HexColor("#fcfcfb")
ACCENT = colors.HexColor("#eb6834")
CLASS_COLORS = {"High": colors.HexColor("#0ca30c"), "Medium": colors.HexColor("#c98500"),
                "Low": colors.HexColor("#d03b3b")}

S = {
    "title": ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=19, leading=23, textColor=INK),
    "sub": ParagraphStyle("sub", fontName="DejaVu", fontSize=9, leading=13, textColor=INK_2),
    "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=11.5, leading=15, textColor=INK, spaceBefore=10,
                         spaceAfter=5),
    "body": ParagraphStyle("body", fontName="DejaVu", fontSize=8.8, leading=12.5, textColor=INK_2, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="DejaVu", fontSize=7.6, leading=10.5, textColor=MUTED),
    "big": ParagraphStyle("big", fontName="DejaVu-Bold", fontSize=38, leading=40, textColor=INK),
    "cell": ParagraphStyle("cell", fontName="DejaVu", fontSize=8.4, leading=11, textColor=INK),
    "cellm": ParagraphStyle("cellm", fontName="DejaVu", fontSize=8.0, leading=10.5, textColor=MUTED),
}


def _fmt(v, nd=2):
    if v is None:
        return "–"
    if isinstance(v, str):
        return v
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{nd}f}"


def _shap_png(contribs: list[dict], n: int = 10) -> io.BytesIO:
    items = contribs[:n][::-1]
    fig, ax = plt.subplots(figsize=(6.6, 0.32 * len(items) + 0.8), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    vals = [c["contribution_pct"] for c in items]
    cols = ["#e34948" if v > 0 else "#2a78d6" for v in vals]
    ax.barh([c["label"] for c in items], vals, color=cols, height=0.62)
    for y, v in enumerate(vals):
        ax.text(v + (0.4 if v >= 0 else -0.4), y, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=7, color="#52514e")
    ax.axvline(0, color="#c3c2b7", lw=0.8)
    lim = max(abs(v) for v in vals) * 1.3 + 1
    ax.set_xlim(-lim, lim)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="y", length=0, labelsize=7.5, colors="#0b0b0b")
    ax.tick_params(axis="x", labelsize=7, colors="#898781")
    ax.set_xlabel("Contribution to score (percentage points)  ·  red raises, blue lowers", fontsize=7, color="#52514e")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def _table(rows, widths, header=True, zebra=True):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "DejaVu"), ("FONTSIZE", (0, 0), (-1, -1), 8.4),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, HAIR), ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        style += [("FONTNAME", (0, 0), (-1, 0), "DejaVu-Bold"), ("TEXTCOLOR", (0, 0), (-1, 0), INK_2),
                  ("FONTSIZE", (0, 0), (-1, 0), 7.8), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK_2)]
    t.setStyle(TableStyle(style))
    return t


def build_pdf(pred: dict, name: str | None = None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm,
                            bottomMargin=14 * mm, title="Solarsite site report",
                            author="Solarsite")
    W = A4[0] - 32 * mm
    lat, lon = pred["lat"], pred["lon"]
    prov = pred["provenance"]
    story = []

    # ---------------------------------------------------------------- header
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    title = name or f"Site at {abs(lat):.4f}° {ns}, {abs(lon):.4f}° {ew}"
    story += [Paragraph("SOLAR SITE SUITABILITY REPORT", ParagraphStyle("k", parent=S["small"], textColor=ACCENT,
                                                                          fontName="DejaVu-Bold")),
              Spacer(1, 2), Paragraph(title, S["title"]), Spacer(1, 3),
              Paragraph(f"{lat:.5f}, {lon:.5f} &nbsp;·&nbsp; generated "
                        f"{datetime.now(timezone.utc).strftime('%d %b %Y, %H:%M UTC')} &nbsp;·&nbsp; "
                        f"data: <b>{prov['label']}</b>", S["sub"])]
    if prov["source"] == "grid_estimate":
        story.append(Paragraph(
            f"Features were taken from the nearest cached 0.5° grid cell ({prov['cell_distance_km']} km away) because "
            f"live Earth Engine extraction was unavailable. Treat values as a regional estimate.", S["small"]))
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------------- scores
    cls = pred["class"]
    big = Table([[Paragraph(f"{pred['score']:.0f}", S["big"]),
                  Paragraph(f"<font color='{CLASS_COLORS[cls].hexval()}'><b>{cls}</b></font> suitability<br/>"
                            f"<font size=8 color='#898781'>Physical suitability (0-100) · climate, terrain and land "
                            f"cover only{' · excluded: ' + pred['excluded'] if pred.get('excluded') else ''}"
                            f"</font>", S["body"])]],
                colWidths=[30 * mm, W - 30 * mm])
    big.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(big)
    if pred.get("exclusion_message"):
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>{pred['exclusion_message']}</b> Without the exclusion mask the model would score "
                               f"{pred.get('model_score', 0):.0f}.", ParagraphStyle("ex", parent=S["body"], textColor=INK)))
    rows = [["Score", "Value", "Class", "What it measures"]]
    for s in pred.get("scores", {}).values():
        rows.append([Paragraph(s["label"], S["cell"]), f"{s['score']:.1f}", s["class"],
                     Paragraph(s["description"], S["cellm"])])
    if "tamil_nadu_local" in pred:
        t = pred["tamil_nadu_local"]
        rows.append([Paragraph(t.get("label", "High-res regional model"), S["cell"]), f"{t['score']:.1f}", t["class"],
                     Paragraph(t["note"], S["cellm"])])
    story += [Spacer(1, 6), _table(rows, [42 * mm, 16 * mm, 16 * mm, W - 74 * mm])]

    # ---------------------------------------------------------------- energy
    e = pred.get("energy")
    if e:
        story.append(Paragraph("Energy estimate", S["h2"]))
        erows = [["Quantity", "Value"],
                 ["Array area", f"{e['area_m2']:,.0f} m²  ({e['area_acres']:.2f} acre)"],
                 ["Global horizontal irradiance", f"{e['ghi_kwh_m2_day']:.2f} kWh/m²/day"],
                 ["Module efficiency × performance ratio", f"{e['efficiency']:.2f} × {e['performance_ratio']:.2f}"],
                 ["Annual energy", f"{e['annual_mwh']:,.1f} MWh/yr"],
                 ["Peak capacity (STC)", f"{e['peak_capacity_kwp']:,.0f} kWp"],
                 ["Specific yield", f"{e['specific_yield_kwh_per_kwp']:,.0f} kWh/kWp/yr"],
                 ["CO₂ avoided (0.475 kg/kWh grid)", f"{e['co2_avoided_tonnes']:,.0f} t/yr"],
                 ["Households supplied (3,500 kWh/yr)", f"{e['households_powered']:,.0f}"]]
        story.append(_table(erows, [80 * mm, W - 80 * mm]))
        story.append(Paragraph(f"Formula: {e['formula']} (A = area, GHI in kWh/m²/day, η = module efficiency, "
                               f"PR = performance ratio).", S["small"]))

    # ---------------------------------------------------------------- features
    story.append(Paragraph("Remote-sensing site characteristics", S["h2"]))
    meta = pred.get("feature_meta", {})
    frows = [["Variable", "Value", "Unit", "Source"]]
    for k, v in pred["features"].items():
        m = meta.get(k, {})
        frows.append([Paragraph(m.get("label", k), S["cell"]), _fmt(v), m.get("unit", ""),
                      Paragraph(m.get("source", ""), S["cellm"])])
    story.append(_table(frows, [52 * mm, 26 * mm, 24 * mm, W - 102 * mm]))

    # ---------------------------------------------------------------- SHAP
    contribs = pred["shap"]["contributions"]
    shap_block = [Paragraph("Why this score? (SHAP explanation)", S["h2"]),
                  Paragraph(f"Starting from the suitability model's average output of "
                            f"{pred['shap']['base_score']:.1f}, each bar shows how much a predictor moved this "
                            f"site's score (model output {pred.get('model_score', pred['score']):.1f}"
                            f"{', before the exclusion mask' if pred.get('excluded') else ''}). Exact Shapley values "
                            f"from TreeExplainer ({pred['shap']['units']}), rescaled to score points.", S["body"]),
                  Spacer(1, 4), Image(_shap_png(contribs), width=W, height=W * (0.32 * min(10, len(contribs)) + 0.8) / 6.6)]
    story.append(KeepTogether(shap_block))

    # ---------------------------------------------------------------- notes
    story += [Paragraph("Method & limitations", S["h2"]), Paragraph(
        "Features come from global Earth Engine datasets: ERA5-Land (irradiance, temperature), MODIS (NDVI, NDBI, "
        "LST, land cover, cloud fraction), Copernicus GLO-30 DEM, VIIRS night lights, GHSL population and the Malaria "
        "Atlas Project accessibility surface. The model was trained on 8,213 utility-scale PV plants from the "
        "Kruitwagen et al. (2021) global inventory and 9,000 negatives, using land conditions from <i>before</i> the "
        "plants were built (2008-10) to avoid label leakage. Suitability uses only physical predictors, region-balanced "
        "weights and positive-unlabelled learning so that sunny, undeveloped land is not penalised for lacking plants. "
        "Development likelihood reflects where plants have historically been built - near infrastructure and demand - "
        "and is not a measure of physical suitability. "
        "This is a screening tool: it does not replace a site survey, grid-capacity study or land-title check.",
        S["body"])]
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DejaVu", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(16 * mm, 9 * mm, "Solarsite · solar site suitability from remote sensing and machine learning")
    canvas.drawRightString(A4[0] - 16 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()
