"""Plot train and validation loss of finished runs side by side (docs/figures/loss-curves.png).

Usage:
    uv run python scripts/plot_loss_curves.py \\
        --run "syrinx-10m · TinyStories · done=runs/10m-a/metrics.jsonl" \\
        --run "syrinx-50m · FineWeb-Edu · done=runs/50m-a/metrics.jsonl" \\
        --out docs/figures/loss-curves.png

Draws an SVG page from the ``metrics.jsonl`` files and renders it to PNG with headless Chrome, so
no plotting library is needed. A resumed run logs some steps twice; the last record of each step
(the one the final model descends from) is kept. Validation points are placed at
``step x tokens per step``, with tokens per step taken from the run's last train record.
"""

import argparse
import html
import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

from syrinx.utils.log import setup_logging
from syrinx.utils.metrics import read_jsonl

logger = logging.getLogger(__name__)

DEFAULT_CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
DEFAULT_TITLE = "Syrinx: language models trained from scratch on a single local device"
DEFAULT_NOTE = (
    "Losses are on different datasets, so the panels are not directly comparable. Train loss is one "
    "batch per logged step; validation is the mean over the run's eval batches."
)
BLUE, ORANGE, INK, MUTED, GRID, PAPER = (
    "#2a72d4", "#e8643a", "#1d1d1f", "#5f6368", "#e3e3e3", "#fafaf9"
)  # fmt: skip
PANEL_SLOT, PANEL_W, PANEL_H, HEIGHT, TOP, Y_MAX = 592, 470, 310, 540, 86, 10.0


@dataclass(frozen=True)
class Curves:
    """One run's losses: ``train`` and ``val`` as ``(tokens_millions, loss)`` in step order."""

    label: str
    train: list[tuple[float, float]]
    val: list[tuple[float, float]]


def load_curves(label: str, path: Path) -> Curves:
    """Read a run's ``metrics.jsonl``, keeping the last record of each step.

    Raises:
        ValueError: If the file has no train records or no val records.
    """
    train: dict[int, dict] = {}
    val: dict[int, float] = {}
    for record in read_jsonl(path):
        if record["kind"] == "train":
            train[record["step"]] = record
        elif record["kind"] == "val":
            val[record["step"]] = record["val_loss"]
    if not train or not val:
        raise ValueError(f"{path}: needs train and val records (found {len(train)}, {len(val)})")
    last = train[max(train)]
    tokens_per_step = last["tokens_seen"] / last["step"]
    return Curves(
        label=label,
        train=[(train[s]["tokens_seen"] / 1e6, train[s]["loss"]) for s in sorted(train)],
        val=[(s * tokens_per_step / 1e6, val[s]) for s in sorted(val)],
    )


def axis_top(x_max: float) -> tuple[float, float]:
    """``(tick step, axis end)`` for an x range of ``0..x_max``: at most 6 ticks. The end may fall
    short of ``x_max`` by under 0.5% of a tick step (a few hundred thousand tokens past a round
    number), so a run of 1.0B tokens ends its axis at 1000, not 1200."""
    step = next((s for s in (50, 100, 200, 250, 500, 1000, 2000, 5000) if x_max / s <= 6), 10000)
    return step, step * math.ceil(x_max / step - 0.005)


def _panel(i: int, curves: Curves, legend: bool) -> list[str]:
    left = 58 + i * PANEL_SLOT
    x0, y0 = left + 50, TOP + 34
    step, x_end = axis_top(max(x for x, _ in curves.train))

    def sx(x: float) -> float:
        return x0 + x / x_end * PANEL_W

    def sy(y: float) -> float:
        return y0 + PANEL_H - min(y, Y_MAX) / Y_MAX * PANEL_H

    out = [f'<text x="{left}" y="{TOP}" class="sub">{html.escape(curves.label)}</text>']
    for y in range(0, int(Y_MAX) + 1, 2):
        out.append(
            f'<line x1="{x0}" x2="{x0 + PANEL_W}" y1="{sy(y)}" y2="{sy(y)}" stroke="{GRID}"/>'
        )
        out.append(f'<text x="{x0 - 10}" y="{sy(y) + 5}" class="tick" text-anchor="end">{y}</text>')
    for k in range(round(x_end / step) + 1):
        out.append(
            f'<text x="{sx(k * step)}" y="{y0 + PANEL_H + 24}" class="tick" '
            f'text-anchor="middle">{k * step:g}</text>'
        )
    out.append(
        f'<text x="{x0 + PANEL_W / 2}" y="{y0 + PANEL_H + 52}" class="axis" '
        f'text-anchor="middle">tokens seen (M)</text>'
    )
    out.append(
        f'<text transform="translate({left + 4},{y0 + PANEL_H / 2}) rotate(-90)" class="axis" '
        f'text-anchor="middle">loss (nats / token)</text>'
    )
    for points, colour, width in ((curves.train, BLUE, 1.6), (curves.val, ORANGE, 2.2)):
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        out.append(
            f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="{width}" '
            f'stroke-linejoin="round"/>'
        )
    for x, y in curves.val:
        out.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.2" fill="{ORANGE}" '
            f'stroke="{PAPER}" stroke-width="1"/>'
        )
    lx, ly = curves.val[-1]
    out.append(
        f'<text x="{sx(lx) - 2}" y="{sy(ly) - 14}" class="end" text-anchor="end">val {ly:.2f}</text>'
    )
    if legend:
        gx, gy = x0 + PANEL_W - 190, y0 + 18
        out += [
            f'<line x1="{gx}" x2="{gx + 30}" y1="{gy}" y2="{gy}" stroke="{BLUE}" stroke-width="2"/>',
            f'<text x="{gx + 40}" y="{gy + 5}" class="legend">train loss</text>',
            f'<line x1="{gx}" x2="{gx + 30}" y1="{gy + 28}" y2="{gy + 28}" stroke="{ORANGE}" stroke-width="2.2"/>',
            f'<circle cx="{gx + 15}" cy="{gy + 28}" r="3.2" fill="{ORANGE}"/>',
            f'<text x="{gx + 40}" y="{gy + 33}" class="legend">validation loss</text>',
        ]
    return out


def render_page(runs: list[Curves], title: str, note: str) -> tuple[str, int, int]:
    """The figure as an HTML page with one inline SVG; returns ``(html, width, height)`` in CSS px."""
    width = 16 + PANEL_SLOT * len(runs)
    body = "\n".join(line for i, c in enumerate(runs) for line in _panel(i, c, legend=i == 0))
    page = f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;background:{PAPER}}}
text{{font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;fill:{INK}}}
.title{{font-size:21px}} .sub{{font-size:17px}} .tick{{font-size:14px;fill:{MUTED}}}
.axis{{font-size:15px;fill:{MUTED}}} .legend,.end{{font-size:15px}} .note{{font-size:13px;fill:{MUTED}}}
</style></head><body>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{HEIGHT}" viewBox="0 0 {width} {HEIGHT}">
<rect width="{width}" height="{HEIGHT}" fill="{PAPER}"/>
<text x="12" y="34" class="title">{html.escape(title)}</text>
{body}
<text x="12" y="{HEIGHT - 14}" class="note">{html.escape(note)}</text>
</svg></body></html>"""
    return page, width, HEIGHT


def render_png(page: Path, out: Path, width: int, height: int, chrome: Path) -> None:
    """Screenshot ``page`` at 2x scale with headless Chrome.

    Raises:
        FileNotFoundError: If ``chrome`` does not exist.
        RuntimeError: If Chrome exits non-zero or writes no file.
    """
    if not chrome.exists():
        raise FileNotFoundError(f"Chrome not found at {chrome}; pass --chrome")
    out.unlink(missing_ok=True)
    cmd = [
        str(chrome), "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--force-device-scale-factor=2", f"--window-size={width},{height}",
        f"--screenshot={out.resolve()}", page.resolve().as_uri(),
    ]  # fmt: skip
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not out.exists():
        raise RuntimeError(f"Chrome failed (exit {result.returncode}): {result.stderr[-500:]}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Options: ``--run LABEL=METRICS`` (repeatable, required, one panel each, in order),
    ``--out`` (PNG path, required), ``--title``, ``--note``, ``--chrome``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=METRICS")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--note", default=DEFAULT_NOTE)
    parser.add_argument("--chrome", type=Path, default=DEFAULT_CHROME)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load each run, write ``<out>.html`` beside the PNG source, render the PNG; return 0."""
    setup_logging()
    args = parse_args(argv)
    runs = []
    for spec in args.run:
        label, sep, path = spec.rpartition("=")
        if not sep or not label:
            raise ValueError(f"--run {spec!r}: expected LABEL=METRICS")
        runs.append(load_curves(label, Path(path)))
        logger.info("%s: %d train, %d val points", label, len(runs[-1].train), len(runs[-1].val))
    page, width, height = render_page(runs, args.title, args.note)
    html_path = args.out.with_suffix(".html")
    html_path.write_text(page, encoding="utf-8")
    try:
        render_png(html_path, args.out, width, height, args.chrome)
    finally:
        html_path.unlink(missing_ok=True)
    logger.info("wrote %s (%dx%d at 2x)", args.out, width, height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
