#!/usr/bin/env python3
"""Extract StarVLA loss logs into CSV and a dependency-free SVG curve."""

import ast
import csv
import re
import sys
from pathlib import Path


STEP_LOSS = re.compile(r"Step\s+(\d+),\s+Loss:\s+(\{.*\})")


def read_points(log_path):
    points = []
    for line in log_path.read_text(errors="replace").splitlines():
        match = STEP_LOSS.search(line)
        if not match:
            continue
        metrics = ast.literal_eval(match.group(2))
        if "action_dit_loss" in metrics:
            points.append((int(match.group(1)), float(metrics["action_dit_loss"])))
    return points


def write_csv(path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "action_dit_loss"])
        writer.writerows(points)


def write_svg(path, points):
    width, height, margin = 1000, 560, 70
    plot_w, plot_h = width - 2 * margin, height - 2 * margin
    min_step, max_step = points[0][0], points[-1][0]
    losses = [loss for _, loss in points]
    min_loss, max_loss = min(losses), max(losses)
    step_span = max(max_step - min_step, 1)
    loss_span = max(max_loss - min_loss, 1e-12)

    def xy(point):
        step, loss = point
        x = margin + (step - min_step) / step_span * plot_w
        y = margin + (max_loss - loss) / loss_span * plot_h
        return f"{x:.2f},{y:.2f}"

    polyline = " ".join(xy(point) for point in points)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="22">StarVLA pilot action loss</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333"/>
<polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="2"/>
<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="sans-serif">step ({min_step}–{max_step})</text>
<text x="18" y="{height / 2}" text-anchor="middle" font-family="sans-serif" transform="rotate(-90 18 {height / 2})">action_dit_loss ({min_loss:.6g}–{max_loss:.6g})</text>
<text x="{margin}" y="{height - 42}" font-family="sans-serif" font-size="13">first={losses[0]:.6g}</text>
<text x="{width - margin}" y="{margin - 12}" text-anchor="end" font-family="sans-serif" font-size="13">last={losses[-1]:.6g}</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg)


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: plot_loss.py LOG_FILE LOSS_SVG LOSS_CSV")
    log_path, svg_path, csv_path = map(Path, sys.argv[1:])
    points = read_points(log_path)
    if not points:
        raise SystemExit(f"no action_dit_loss entries found in {log_path}")
    write_csv(csv_path, points)
    write_svg(svg_path, points)
    print(f"loss_points={len(points)} loss_csv={csv_path} loss_curve={svg_path}")


if __name__ == "__main__":
    main()
