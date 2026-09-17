"""Inline SVG for the report.

Hand-rolled rather than a charting library because the report has to be a single file you can attach to an
email: no JavaScript, no external assets, no fonts to fetch. Two charts earn their place.
"""

from __future__ import annotations

from html import escape


def _scale(lo: float, hi: float, size: int, pad: int, invert: bool = False):
    span = (hi - lo) or 1.0

    def to_px(v: float) -> float:
        frac = (v - lo) / span
        return (size - pad - frac * (size - 2 * pad)) if invert else (pad + frac * (size - 2 * pad))

    return to_px


def cost_success_chart(
    analytic: list[dict],
    verified: list[dict],
    teacher: tuple[float, float] | None = None,
    width: int = 560,
    height: int = 320,
    pad: int = 52,
) -> str:
    """Cost per task against success.

    The analytic curve is faint and the harness-verified points are bold, because that is the honest hierarchy:
    the analytic model assumes an escalated turn is as good as the teacher's, and it is not.
    """
    points = [
        (p.get("cost_per_turn") or p.get("cost_per_task") or 0.0,
         p.get("cascade_success") or p.get("success") or 0.0)
        for p in analytic
    ]
    marks = [(p.get("cost_per_task") or p.get("cost_per_turn") or 0.0, p.get("success") or 0.0) for p in verified]
    everything = points + marks + ([teacher] if teacher else [])
    if not everything:
        return _empty(width, height, "no cascade points to plot")

    xs = [p[0] for p in everything]
    ys = [p[1] for p in everything]
    x = _scale(min(xs), max(xs), width, pad)
    y = _scale(min(ys), max(ys), height, pad, invert=True)

    body = [_axes(width, height, pad)]
    if len(points) > 1:
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{x(cx):.1f},{y(cy):.1f}"
            for i, (cx, cy) in enumerate(sorted(points))
        )
        body.append(f'<path d="{path}" fill="none" stroke="#bbb" stroke-width="1.5"/>')
        body.append(f'<text x="{pad + 6}" y="{pad - 8}" font-size="11" fill="#999">analytic estimate</text>')
    for cx, cy in marks:
        body.append(f'<circle cx="{x(cx):.1f}" cy="{y(cy):.1f}" r="5" fill="#1a1a1a"/>')
    if marks:
        body.append(f'<text x="{pad + 6}" y="{pad + 8}" font-size="11" fill="#1a1a1a">measured</text>')
    if teacher:
        body.append(
            f'<circle cx="{x(teacher[0]):.1f}" cy="{y(teacher[1]):.1f}" r="5" fill="none" '
            f'stroke="#c0392b" stroke-width="2"/>'
            f'<text x="{x(teacher[0]) + 8:.1f}" y="{y(teacher[1]) + 4:.1f}" font-size="11" fill="#c0392b">'
            f'teacher</text>'
        )
    body.append(_axis_labels(width, height, pad, "cost per task ($)", "success"))
    return _svg(width, height, "".join(body))


def reliability_chart(bins: list[dict], width: int = 420, height: int = 320, pad: int = 52) -> str:
    """Predicted confidence against observed frequency, with the diagonal a perfect gate would follow."""
    filled = [b for b in bins if b.get("n")]
    if not filled:
        return _empty(width, height, "no calibration bins")

    body = [_axes(width, height, pad)]
    x = _scale(0.0, 1.0, width, pad)
    y = _scale(0.0, 1.0, height, pad, invert=True)
    body.append(
        f'<line x1="{x(0):.1f}" y1="{y(0):.1f}" x2="{x(1):.1f}" y2="{y(1):.1f}" '
        f'stroke="#ccc" stroke-dasharray="4 3"/>'
    )
    bar_width = max(4.0, (width - 2 * pad) / max(len(bins), 1) - 3)
    for b in filled:
        centre = (b["lo"] + b["hi"]) / 2
        observed = b.get("observed") or 0.0
        left = x(centre) - bar_width / 2
        top = y(observed)
        body.append(
            f'<rect x="{left:.1f}" y="{top:.1f}" width="{bar_width:.1f}" '
            f'height="{y(0) - top:.1f}" fill="#4a6fa5" opacity="0.75"/>'
        )
    body.append(_axis_labels(width, height, pad, "predicted", "observed"))
    return _svg(width, height, "".join(body))


def _axes(width: int, height: int, pad: int) -> str:
    return (
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#444"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#444"/>'
    )


def _axis_labels(width: int, height: int, pad: int, x_label: str, y_label: str) -> str:
    return (
        f'<text x="{width / 2:.0f}" y="{height - 12}" font-size="12" fill="#444" '
        f'text-anchor="middle">{escape(x_label)}</text>'
        f'<text x="14" y="{height / 2:.0f}" font-size="12" fill="#444" text-anchor="middle" '
        f'transform="rotate(-90 14 {height / 2:.0f})">{escape(y_label)}</text>'
    )


def _svg(width: int, height: int, body: str) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img">{body}</svg>'
    )


def _empty(width: int, height: int, message: str) -> str:
    return _svg(
        width, height,
        f'<text x="{width / 2:.0f}" y="{height / 2:.0f}" font-size="13" fill="#999" '
        f'text-anchor="middle">{escape(message)}</text>',
    )
