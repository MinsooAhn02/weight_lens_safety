"""Build the FLMSec weight-by-lens interaction figure from the audited summary."""

import argparse
import io
import sys
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from audit_common import cell, load  # noqa: E402


OUTPUT = HERE / "figures" / "fig1_weight_by_lens.pdf"
# The plotted band starts at Y_LOW, with an explicit break mark on the axis below it.
# Data floor is 0.3642 (arm1 / paraphrase, seed 42). 0.30 leaves ~0.06 of margin and
# keeps the break mark clear of the 0.0 tick; values() asserts nothing falls through,
# so a data change that would silently clip the axis fails the build instead.
Y_LOW = 0.30
LENSES = (
    ("none", "Standard"),
    ("agreeable", "Compliant"),
    ("principled", "Principled"),
    ("agreeable_para1", "Held-out paraphrase"),
)
ARMS = (
    ("arm0", "arm 0: untrained", "#333333", (4, 3), "circle"),
    ("arm1", "arm 1: compliant persona", "#0072B2", (), "circle"),
    ("arm2", "arm 2: generic control", "#D55E00", (2, 2), "square"),
    ("arm3", "arm 3: principled persona", "#009E73", (), "diamond"),
)
SEED_COLORS = {"arm1": "#7FB8D8", "arm2": "#EA9E7F", "arm3": "#7FCEB9"}


def values():
    analysis = load("_env76")["analyses"]["truncation_included"]
    grid = {(row["arm"], row["lens"]): row for row in analysis["grid"]}
    rows = {
        arm: [cell(analysis, arm, lens, "corrected_refusal_rate") for lens, _ in LENSES]
        for arm, *_ in ARMS
    }
    seeds = {
        arm: [
            [run["corrected_refusal_rate"] for run in sorted(grid[(arm, lens)]["runs"], key=lambda run: run["seed"])]
            for lens, _ in LENSES
        ]
        for arm in SEED_COLORS
    }
    assert set(rows) == {"arm0", "arm1", "arm2", "arm3"}
    assert all(len(row) == 4 and all(0 <= value <= 1 for value in row) for row in rows.values())
    assert all(len(run_values) == 3 for arm in seeds.values() for run_values in arm)
    drawn = [v for row in rows.values() for v in row]
    drawn += [v for arm in seeds.values() for run_values in arm for v in run_values]
    assert min(drawn) >= Y_LOW, (
        f"a plotted value ({min(drawn):.4f}) falls below the broken axis floor "
        f"{Y_LOW} -- lower Y_LOW rather than let the axis clip it silently"
    )
    return rows, seeds


def marker(pdf, kind, x, y, size=2.7):
    if kind == "circle":
        pdf.circle(x, y, size, stroke=1, fill=1)
    elif kind == "square":
        pdf.rect(x - size, y - size, 2 * size, 2 * size, stroke=1, fill=1)
    else:
        path = pdf.beginPath()
        path.moveTo(x, y + size + 0.5)
        path.lineTo(x + size + 0.5, y)
        path.lineTo(x, y - size - 0.5)
        path.lineTo(x - size - 0.5, y)
        path.close()
        pdf.drawPath(path, stroke=1, fill=1)


def axis_break(pdf, x, y, w=3.8, amp=1.9):
    """Two stacked squiggles across the axis: the conventional break mark.

    The mask stays narrow so it cannot clip the 0.0 tick label, which is set
    right-aligned just left of the axis.
    """
    pdf.setFillColor(HexColor("#FFFFFF"))
    pdf.rect(x - w, y - 5, 2 * w, 10, stroke=0, fill=1)
    pdf.setStrokeColor(HexColor("#333333"))
    pdf.setLineWidth(0.75)
    for dy in (-2.3, 2.3):
        path = pdf.beginPath()
        path.moveTo(x - w, y + dy)
        path.curveTo(x - w / 2, y + dy + amp, x - w / 2, y + dy - amp, x, y + dy)
        path.curveTo(x + w / 2, y + dy + amp, x + w / 2, y + dy - amp, x + w, y + dy)
        pdf.drawPath(path, stroke=1, fill=0)


def render():
    width, height = 396, 133
    left, right, bottom, top = 42, 104, 24, 8
    plot_width = width - left - right
    plot_height = height - bottom - top
    x_positions = [left + plot_width * i / 3 for i in range(4)]
    # Strip reserved below the plotted band for the zero tick and the break mark.
    break_gap = 13
    band = plot_height - break_gap

    def y_position(value):
        return bottom + break_gap + (value - Y_LOW) / (1 - Y_LOW) * band

    stream = io.BytesIO()
    pdf = canvas.Canvas(stream, pagesize=(width, height), invariant=1, pageCompression=1)
    pdf.setTitle("Weight-by-lens corrected refusal rate")

    pdf.setFont("Helvetica", 7.5)
    for tick in (0.4, 0.6, 0.8, 1.0):
        y = y_position(tick)
        pdf.setStrokeColor(HexColor("#D8D8D8"))
        pdf.setLineWidth(0.45)
        pdf.line(left, y, left + plot_width, y)
        pdf.setFillColor(HexColor("#444444"))
        pdf.drawRightString(left - 5, y - 2.5, f"{tick:.1f}")
    pdf.setFillColor(HexColor("#444444"))
    pdf.drawRightString(left - 5, bottom - 2.5, "0.0")

    pdf.setStrokeColor(HexColor("#333333"))
    pdf.setLineWidth(0.75)
    pdf.line(left, bottom, left, bottom + plot_height)
    pdf.line(left, bottom + break_gap, left + plot_width, bottom + break_gap)
    axis_break(pdf, left, bottom + break_gap * 0.62)
    for x, (_, label) in zip(x_positions, LENSES):
        pdf.setFillColor(HexColor("#222222"))
        pdf.setFont("Helvetica", 7.5)
        pdf.drawCentredString(x, bottom - 13, label)

    pdf.saveState()
    pdf.translate(12, bottom + plot_height / 2)
    pdf.rotate(90)
    pdf.setFont("Helvetica", 8)
    pdf.setFillColor(HexColor("#222222"))
    pdf.drawCentredString(0, 0, "Corrected refusal rate")
    pdf.restoreState()

    rows, seeds = values()
    for arm, label, color, dash, marker_kind in ARMS:
        points = [(x, y_position(value)) for x, value in zip(x_positions, rows[arm])]
        pdf.setStrokeColor(HexColor(color))
        pdf.setFillColor(HexColor(color))
        pdf.setLineWidth(1.65)
        pdf.setDash(dash)
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            pdf.line(x1, y1, x2, y2)
        pdf.setDash()
        for x, y in points:
            marker(pdf, marker_kind, x, y)
        if arm in seeds:
            pdf.setStrokeColor(HexColor(SEED_COLORS[arm]))
            pdf.setFillColor(HexColor(SEED_COLORS[arm]))
            for x, run_values in zip(x_positions, seeds[arm]):
                for offset, value in zip((-4, 0, 4), run_values):
                    pdf.circle(x + offset, y_position(value), 1.25, stroke=0, fill=1)
            pdf.setFillColor(HexColor(color))
        pdf.setFont("Helvetica", 8.2)
        label_y = points[-1][1] + {"arm0": -8, "arm1": -1, "arm2": 5, "arm3": -1}[arm]
        pdf.drawString(points[-1][0] + 7, label_y - 2.5, label)

    pdf.showPage()
    pdf.save()
    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_bytes() != expected:
            raise SystemExit(f"STALE: {OUTPUT}")
        print(f"PASS: {OUTPUT}")
        return
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_bytes(expected)
    print(OUTPUT)


if __name__ == "__main__":
    main()
