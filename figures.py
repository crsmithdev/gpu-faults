"""The checkpoint-interval numbers and figure, from the logs in results/interval.

    uv run python figures.py <out_dir>

Each run trains 150 steps of FSDP2 124M on 4×L40S. Rank 3 crashes at steps
37, 81 and 128, and the run saves a checkpoint every k steps. `base` has no
checkpoints and no faults. The overhead of a run is its wall time from the
start of step 1 to the end of step 150, less 150 times its median step.

The SVG takes its colours from the page's CSS variables (--text, --muted,
--accent, --head), so it follows the page's light and dark themes.
"""

import pathlib
import re
import statistics
import sys

HERE = pathlib.Path(__file__).parent
STEPS, FAULTS = 150, 3
STEP = re.compile(r" step (\d+)/150 .* (\d+) ms ")
SAVE = re.compile(r"ckpt save step \d+ blocked (\d+) ms")


def run(name: str) -> dict:
    """The overhead of one run, split into saving, redoing steps and restarting."""
    lines = (HERE / "results" / "interval" / f"{name}.log").read_text().splitlines()
    steps = [(float(line.split()[0]), int(m[1]), float(m[2]) / 1e3) for line in lines if (m := STEP.search(line))]
    step = statistics.median(s for _, n, s in steps if n > 5)
    # A log line comes at the end of its step, so the wall time adds the first step's own time.
    wall = [t for t, n, _ in steps if n == STEPS][-1] - steps[0][0] + steps[0][2]
    saving = sum(float(m[1]) for line in lines if (m := SAVE.search(line))) / 1e3
    redoing = (len(steps) - len({n for _, n, _ in steps})) * step
    overhead = wall - STEPS * step
    return dict(step=step, overhead=overhead, saving=saving, redoing=redoing,
                restarting=overhead - saving - redoing, saves=sum(bool(SAVE.search(line)) for line in lines))


def model(k: int, save: float, restart: float, step: float) -> float:
    """The expected overhead: each save, and for each fault a restart and half an interval of lost steps."""
    return STEPS / k * save + FAULTS * (restart + k / 2 * step)


def figure(rows: list[tuple[int, dict, float]]) -> str:
    w, left, top, bar, gap = 640, 70, 34, 18, 10
    scale = (w - left - 70) / max(max(r["overhead"], m) for _, r, m in rows)
    h = top + len(rows) * (bar + gap) + 6
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Overhead of 3 crashes in 150 steps, by checkpoint interval" '
             'xmlns="http://www.w3.org/2000/svg" style="font: inherit; font-size: 13px; max-width: 100%">',
             "<title>Overhead of 3 crashes in 150 steps, by checkpoint interval</title>"]
    x = left
    for label, fill in [("Saving", "var(--accent)"), ("Redoing steps", "var(--muted)"), ("Restarting", "var(--accent); fill-opacity: .35")]:
        parts.append(f'<rect x="{x}" y="6" width="12" height="12" style="fill: {fill}"/>'
                     f'<text x="{x + 18}" y="16" style="fill: var(--text)">{label}</text>')
        x += 30 + 8 * len(label)
    parts.append(f'<line x1="{x}" y1="6" x2="{x}" y2="18" style="stroke: var(--head); stroke-width: 2"/>'
                 f'<text x="{x + 8}" y="16" style="fill: var(--text)">Model</text>')
    for i, (k, r, m) in enumerate(rows):
        y = top + i * (bar + gap)
        parts.append(f'<text x="{left - 10}" y="{y + 13}" text-anchor="end" style="fill: var(--text)">{k} steps</text>')
        x = left
        for key, fill in [("saving", "var(--accent)"), ("redoing", "var(--muted)"), ("restarting", "var(--accent); fill-opacity: .35")]:
            width = r[key] * scale
            parts.append(f'<rect x="{x:.1f}" y="{y}" width="{width:.1f}" height="{bar}" style="fill: {fill}"/>')
            x += width
        mx = left + m * scale
        parts.append(f'<text x="{max(x, mx) + 6:.1f}" y="{y + 13}" style="fill: var(--muted)">{r["overhead"]:.0f} s</text>')
        parts.append(f'<line x1="{mx:.1f}" y1="{y - 3}" x2="{mx:.1f}" y2="{y + bar + 3}" style="stroke: var(--head); stroke-width: 2"/>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    runs = {k: run(f"k{k}") for k in (2, 5, 10, 20, 40)}
    step = statistics.median(r["step"] for r in runs.values())
    save = sum(r["saving"] for r in runs.values()) / sum(r["saves"] for r in runs.values())
    restart = statistics.median(r["restarting"] for r in runs.values()) / FAULTS
    mtbf = STEPS * step / FAULTS
    print(f"base overhead {run('base')['overhead']:.1f} s; step {step:.3f} s, save {save:.3f} s, "
          f"restart {restart:.1f} s, MTBF {mtbf:.1f} s, Young {(2 * save * mtbf) ** 0.5:.1f} s")
    rows = []
    for k, r in runs.items():
        m = model(k, save, restart, step)
        rows.append((k, r, m))
        print(f"k={k:2} overhead {r['overhead']:5.1f} s (saving {r['saving']:4.1f}, redoing {r['redoing']:4.1f}, "
              f"restarting {r['restarting']:4.1f}); model {m:5.1f} s")
    if len(sys.argv) > 1:
        (pathlib.Path(sys.argv[1]) / "interval.svg").write_text(figure(rows))


if __name__ == "__main__":
    main()
