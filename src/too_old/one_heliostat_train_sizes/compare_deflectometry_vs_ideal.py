"""
Compare deflectometry vs ideal-scenario training across all 5 heliostats.

Produces two PNG tables:
  table1_learning_curves.png  — per-heliostat pair of rows (ideal / defl) for all train sizes
  table2_delta_n{N}.png       — snapshot at n=DELTA_N: ideal vs defl + delta + improvement %

Usage
-----
    python one_heliostat_train_sizes/compare_deflectometry_vs_ideal.py
    python one_heliostat_train_sizes/compare_deflectometry_vs_ideal.py \\
        --defl   outputs/one_hel_train_sizes_with_deflectometry \\
        --ideal  outputs/one_hel_train_sizes_without_deflectometry \\
        --out    outputs/deflectometry_comparison \\
        --delta-n 20
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import math

_HERE = pathlib.Path(__file__).resolve().parent
_BASE = _HERE.parents[1]

HELIOSTATS  = ["AC36", "AG33", "AO34", "AW36", "BE35"]
TRAIN_SIZES = [1, 5, 10, 20, 25, 50, 75, 100]
DELTA_N_DEFAULT = 20

_HDR_BG    = "#2c3e50"
_HDR_TEXT  = "white"
_IDEAL_BG  = "#d6eaf8"   # light blue  — ideal rows
_DEFL_BG   = "#d5f5e3"   # light green — defl rows
_SEP_BG    = "#ecf0f1"   # light grey  — heliostat label cells
_GREEN_CELL = "#1a7a3a"
_RED_CELL   = "#922b21"
_NEUTRAL    = "#ecf0f1"


def _load(base: pathlib.Path, hid: str) -> dict:
    with open(base / hid / "summary.json") as f:
        return json.load(f)


def _values(summary: dict, sizes: list[int]) -> list[float]:
    return [summary["results"][str(n)]["post_training"]["test"]["mean_mrad"] for n in sizes]


# ---------------------------------------------------------------------------
# Table 1 — learning curves side by side
# ---------------------------------------------------------------------------

def _table1(ideal_data: dict, defl_data: dict, out_dir: pathlib.Path) -> None:
    sizes = TRAIN_SIZES
    size_labels = [str(n) for n in sizes]
    n_cols = 1 + len(sizes)   # label + one col per train size

    # Two rows per heliostat: "Ideal" and "Defl"
    cell_text   = []
    cell_colors = []

    for hid in HELIOSTATS:
        iv = ideal_data[hid]
        dv = defl_data[hid]

        # Ideal row
        cell_text.append([f"{hid}\nIdeal"] + [f"{v:.3f}" for v in iv])
        cell_colors.append([_SEP_BG] + [_IDEAL_BG] * len(sizes))

        # Defl row
        cell_text.append(["Defl"] + [f"{v:.3f}" for v in dv])
        cell_colors.append([_SEP_BG] + [_DEFL_BG] * len(sizes))

    header_text   = [["Heliostat /\nScenario"] + size_labels]
    header_colors = [[_HDR_BG] * n_cols]

    all_text   = header_text + cell_text
    all_colors = header_colors + cell_colors

    n_data_rows = len(cell_text)
    row_h  = 0.38
    hdr_h  = 0.50
    fig_w  = max(10, 0.9 * n_cols)
    fig_h  = hdr_h + row_h * n_data_rows + 1.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    tbl = ax.table(cellText=all_text, cellColours=all_colors,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    for c in range(n_cols):
        tbl[0, c].set_height(hdr_h / fig_h)
        tbl[0, c].set_text_props(color=_HDR_TEXT, fontweight="bold")
    for r in range(1, 1 + n_data_rows):
        for c in range(n_cols):
            tbl[r, c].set_height(row_h / fig_h)

    # Bold heliostat label on every "Ideal" row
    for i in range(len(HELIOSTATS)):
        tbl[1 + i * 2, 0].set_text_props(fontweight="bold")

    ax.set_title(
        "Test focal-spot error (mrad) — all training sizes\n"
        "Ideal scenario (no deflectometry)  vs  Deflectometry scenario",
        fontsize=10, fontweight="bold", pad=10,
    )
    fig.tight_layout()
    out_path = out_dir / "table1_learning_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Table 2 — snapshot at delta_n
# ---------------------------------------------------------------------------

def _table2(ideal_data: dict, defl_data: dict, delta_n: int, out_dir: pathlib.Path) -> None:
    idx = TRAIN_SIZES.index(delta_n)

    header_text   = [[f"Heliostat", f"Ideal  (mrad)", f"Deflectometry  (mrad)",
                       f"Δ  (ideal − defl)"]]
    header_colors = [[_HDR_BG] * 4]

    cell_text   = []
    cell_colors = []

    for i, hid in enumerate(HELIOSTATS):
        iv    = ideal_data[hid][idx]
        dv    = defl_data[hid][idx]
        delta = iv - dv

        base = "white" if i % 2 == 0 else _NEUTRAL
        if delta > 0.01:
            delta_bg = _GREEN_CELL
        elif delta < -0.01:
            delta_bg = _RED_CELL
        else:
            delta_bg = base

        cell_text.append([hid, f"{iv:.4f}", f"{dv:.4f}", f"{delta:+.4f}"])
        cell_colors.append([base, _IDEAL_BG, _DEFL_BG, delta_bg])

    n_data_rows = len(cell_text)
    row_h = 0.42
    hdr_h = 0.50
    fig_h = hdr_h + row_h * n_data_rows + 1.0

    fig, ax = plt.subplots(figsize=(7, fig_h))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    all_text   = header_text + cell_text
    all_colors = header_colors + cell_colors

    tbl = ax.table(cellText=all_text, cellColours=all_colors,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    for c in range(4):
        tbl[0, c].set_height(hdr_h / fig_h)
        tbl[0, c].set_text_props(color=_HDR_TEXT, fontweight="bold")
    for r in range(1, 1 + n_data_rows):
        for c in range(4):
            tbl[r, c].set_height(row_h / fig_h)
        tbl[r, 0].set_text_props(fontweight="bold")

    # White text on coloured delta cells
    for i, hid in enumerate(HELIOSTATS):
        iv    = ideal_data[hid][idx]
        dv    = defl_data[hid][idx]
        delta = iv - dv
        if abs(delta) > 0.01:
            tbl[1 + i, 3].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        f"Deflectometry vs Ideal — snapshot at n={delta_n} training samples\n"
        f"Δ = ideal − deflectometry  (green = deflectometry better,  red = ideal better)",
        fontsize=10, fontweight="bold", pad=10,
    )
    fig.tight_layout()
    out_path = out_dir / f"table2_delta_n{delta_n}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Field position plot
# ---------------------------------------------------------------------------

_HIGHLIGHT_COLORS = {
    "AC36": "#e74c3c",
    "AG33": "#e67e22",
    "AO34": "#2ecc71",
    "AW36": "#3498db",
    "BE35": "#9b59b6",
}


def _field_plot(field_positions_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    with open(field_positions_path) as f:
        data = json.load(f)

    tower    = data["tower_enu"]
    all_ids  = data["heliostat_ids"]
    all_pos  = data["positions_enu"]

    # Positions relative to tower (East, North)
    rel = {hid: (p[0] - tower[0], p[1] - tower[1]) for hid, p in zip(all_ids, all_pos)}

    fig, ax = plt.subplots(figsize=(7, 10))
    fig.patch.set_facecolor("white")

    # All 63 heliostats — grey background dots
    for hid, (e, n) in rel.items():
        if hid not in HELIOSTATS:
            ax.scatter(e, n, s=18, color="#bdc3c7", zorder=2, linewidths=0)

    # Tower / target marker at origin
    ax.scatter(0, 0, marker="^", s=220, color="#2c3e50", zorder=5, label="Tower / target")

    # Distance rings at 50, 100, 150, 200 m
    theta = np.linspace(0, 2 * np.pi, 360)
    for r in [50, 100, 150, 200]:
        ax.plot(r * np.cos(theta), r * np.sin(theta),
                color="#95a5a6", linewidth=0.6, linestyle="--", zorder=1)
        ax.text(r * np.cos(np.radians(10)), r * np.sin(np.radians(10)),
                f"{r} m", fontsize=7, color="#7f8c8d", va="bottom")

    # 5 highlighted heliostats
    for hid in HELIOSTATS:
        e, n = rel[hid]
        dist = math.sqrt(e**2 + n**2)
        color = _HIGHLIGHT_COLORS[hid]
        ax.scatter(e, n, s=90, color=color, zorder=4, edgecolors="white",
                   linewidths=0.8, label=f"{hid}  ({dist:.0f} m)")
        ax.annotate(
            hid,
            xy=(e, n),
            xytext=(e + 4, n + 3),
            fontsize=8.5, fontweight="bold", color=color,
            zorder=5,
        )
        # Dashed line to tower
        ax.plot([0, e], [0, n], color=color, linewidth=0.8,
                linestyle="--", alpha=0.5, zorder=3)

    ax.set_xlabel("East offset from tower (m)", fontsize=10)
    ax.set_ylabel("North offset from tower (m)", fontsize=10)
    ax.set_title("Heliostat field — 5 selected heliostats relative to tower",
                 fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
    fig.tight_layout()

    out_path = out_dir / "field_positions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--defl",    type=pathlib.Path,
                        default=_BASE / "outputs" / "one_hel_train_sizes_with_deflectometry")
    parser.add_argument("--ideal",   type=pathlib.Path,
                        default=_BASE / "outputs" / "one_hel_train_sizes_without_deflectometry")
    parser.add_argument("--out",     type=pathlib.Path,
                        default=_BASE / "outputs" / "deflectometry_comparison")
    parser.add_argument("--delta-n", type=int, default=DELTA_N_DEFAULT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    ideal_data = {hid: _values(_load(args.ideal, hid), TRAIN_SIZES) for hid in HELIOSTATS}
    defl_data  = {hid: _values(_load(args.defl,  hid), TRAIN_SIZES) for hid in HELIOSTATS}

    _table1(ideal_data, defl_data, args.out)
    _table2(ideal_data, defl_data, args.delta_n, args.out)

    field_pos = (
        _BASE / "outputs" / "local_runs" /
        "full_63_synthetic_focal_spot_ideal_20260526_003757" / "field_positions.json"
    )
    if field_pos.exists():
        _field_plot(field_pos, args.out)
    else:
        print(f"WARNING: field_positions.json not found at {field_pos} — skipping field plot.")


if __name__ == "__main__":
    main()
