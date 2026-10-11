"""
Optional helper: draws each map with the path found by each algorithm and
saves PNG images (for the report). Requires matplotlib:  pip install matplotlib

    python make_screenshots.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from search_robot import MAP_A, MAP_B, MAP_C, ALGORITHMS

COLORS = {".": "#f4f4f2", "~": "#e8c98a", "#": "#3b3b3b", "S": "#4caf7a", "G": "#d9534f"}


def draw(ax, grid, res):
    rows, cols = len(grid), len(grid[0])
    path = set(res["path"] or [])
    for r in range(rows):
        for c in range(cols):
            ch = grid[r][c]
            ax.add_patch(Rectangle((c, rows - 1 - r), 1, 1, facecolor=COLORS[ch], edgecolor="#bbbbbb"))
            if ch in "SG":
                ax.text(c + .5, rows - .5 - r, ch, ha="center", va="center", color="white", fontsize=12, weight="bold")
    if res["path"]:
        xs = [c + .5 for r, c in res["path"]]
        ys = [rows - .5 - r for r, c in res["path"]]
        ax.plot(xs, ys, color="#2a6fdb", linewidth=3, marker="o", markersize=5)
        title = f"{res['algorithm']}  |  cost {res['cost']}, {res['length']} moves, {res['expanded']} expanded"
    else:
        title = f"{res['algorithm']}  |  No path found ({res['expanded']} expanded)"
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, cols); ax.set_ylim(0, rows); ax.set_aspect("equal"); ax.axis("off")


for name, grid in {"MAP_A": MAP_A, "MAP_B": MAP_B, "MAP_C": MAP_C}.items():
    fig, axes = plt.subplots(len(ALGORITHMS), 1, figsize=(6.5, 3.2 * len(ALGORITHMS)))
    for ax, algo in zip(axes.flat, ALGORITHMS):
        draw(ax, grid, algo(grid))
    fig.suptitle(f"{name}  (tan = rough floor ~, dark = wall #)", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(f"{name}_paths.png", dpi=130)
    print("saved", f"{name}_paths.png")