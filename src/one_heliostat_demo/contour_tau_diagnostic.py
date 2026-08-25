"""Offline τ diagnostic for the Wortberg contour loss.

Question this answers
---------------------
The contour loss extracts the upper contour at the τ-level of the NORMALIZED
flux image. On real data the predicted (simulated) spot is narrower and more
sharply peaked than the measured one, so at a given τ the two contours sit at
different radii — a systematic size bias that the loss can only cancel by
mis-pointing the beam upward.

Spot SIZE is translation-invariant, so it can be measured without knowing the
true pointing. This script sweeps τ and compares predicted vs measured spot
size (soft-mask area and vertical extent, the quantity that actually drives the
upper-edge bias). If the curves cross at some τ, a single shared τ can
equalize the sizes and no width matching is needed; if predicted stays smaller
at every τ, the bias is irreducible by τ alone.

No training is performed: one no-grad ray-tracing pass at the parameters of a
Stage-1 checkpoint (or nominal, if none is given).

Usage
-----
  python src/one_heliostat_demo/contour_tau_diagnostic.py \
      --heliostat-id AY39 \
      --stage1-checkpoint outputs/.../AY39/stage1_checkpoint.pt \
      --output-dir outputs/new_mapping_function/contour_tau_diagnostic
"""

import argparse
import json
import logging
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

_here = pathlib.Path(__file__).resolve().parent      # one_heliostat_demo/
_src = _here.parent                                   # src/
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_here / "single_heliostat"))

import config as cfg  # noqa: E402
import train as tr    # noqa: E402

from artist.util import constants as _const, get_device, set_logger_config  # noqa: E402
from artist.util import setup_distributed_environment  # noqa: E402
from artist.scenario.scenario import Scenario  # noqa: E402

from artist_extensions.contour_loss import ContourExtractor  # noqa: E402
from utils.synth_data import _forward_pass  # noqa: E402

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline τ diagnostic for the contour loss.")
    p.add_argument("--heliostat-id", default="AY39")
    p.add_argument("--stage1-checkpoint", type=pathlib.Path, default=None,
                   help="stage1_checkpoint.pt to load before tracing (default: nominal params)")
    p.add_argument("--output-dir", type=pathlib.Path,
                   default=cfg.BASE_DIR / "outputs" / "new_mapping_function" / "contour_tau_diagnostic")
    p.add_argument("--surface-points", type=int, default=25)
    p.add_argument("--rays", type=int, default=50,
                   help="Rays per surface point for the single diagnostic trace")
    p.add_argument("--tau-min", type=float, default=0.15)
    p.add_argument("--tau-max", type=float, default=0.90)
    p.add_argument("--tau-steps", type=int, default=16)
    return p.parse_args()


def _spot_size_metrics(mask: torch.Tensor) -> dict:
    """Translation-invariant size measures of a soft mask [N, H, W].

    area          — Σ mask (soft pixel count)
    height        — vertical extent: 2·sqrt(12)·σ_u would be the box-equivalent;
                    we report 2·σ_u (mass-weighted vertical std), which is what
                    drives the upper-edge position.
    width         — same along the horizontal axis.
    """
    N, H, W = mask.shape
    u = torch.arange(H, device=mask.device, dtype=mask.dtype).view(1, H, 1)
    e = torch.arange(W, device=mask.device, dtype=mask.dtype).view(1, 1, W)
    m = mask.sum(dim=(-2, -1)).clamp(min=1e-8)
    u_bar = (mask * u).sum(dim=(-2, -1)) / m
    e_bar = (mask * e).sum(dim=(-2, -1)) / m
    var_u = (mask * (u - u_bar.view(N, 1, 1)) ** 2).sum(dim=(-2, -1)) / m
    var_e = (mask * (e - e_bar.view(N, 1, 1)) ** 2).sum(dim=(-2, -1)) / m
    return {
        "area": mask.sum(dim=(-2, -1)),
        "height": 2.0 * var_u.clamp(min=0).sqrt(),
        "width": 2.0 * var_e.clamp(min=0).sqrt(),
    }


def main() -> None:
    args = _parse_args()
    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg.DATA_MODE = "real"
    cfg.SURFACE_POINTS_PER_FACET = args.surface_points

    scenario_path = pathlib.Path(
        cfg.SCENARIO_PATH_TEMPLATE.format(heliostat_id=args.heliostat_id)
    )
    n_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(scenario_path)
    device = get_device()

    with setup_distributed_environment(
        number_of_heliostat_groups=n_groups, device=device
    ) as ddp:
        device = ddp[_const.device]

        scenario, hg, hel_dist_m = tr._load_scenario(args.heliostat_id, cfg, device)
        kinematic = hg.kinematics

        if args.stage1_checkpoint is not None:
            ck = torch.load(args.stage1_checkpoint, map_location=device)
            kinematic.translation_deviation_parameters.data.copy_(ck["translation"].to(device))
            kinematic.rotation_deviation_parameters.data.copy_(ck["rotation"].to(device))
            kinematic.actuators.optimizable_parameters.data.copy_(ck["act_angle"].to(device))
            kinematic.actuators.non_optimizable_parameters.data.copy_(ck["act_offset"].to(device))
            kinematic._base_position_deviation = (
                ck["base_pos"].clone().to(device).requires_grad_(True)
            )
            log.info(f"Loaded Stage-1 checkpoint: {args.stage1_checkpoint}")

        # Data: same loading path and split as training.
        train_data = tr._load_split_real(args.heliostat_id, cfg, "train", hg, scenario, device)
        val_data = tr._load_split_real(args.heliostat_id, cfg, "validation", hg, scenario, device)
        test_data = tr._load_split_real(args.heliostat_id, cfg, "test", hg, scenario, device)
        train_tuple, _, _, _ = tr._pool_and_split(
            args.heliostat_id, train_data, val_data, test_data,
            cfg.SPLITTER_TRAIN_SIZE, cfg, device, cfg.SWAP_VAL_TEST,
        )
        meas_flux, _, rays, motor_pos, active_mask, target_mask = train_tuple
        log.info(f"{args.heliostat_id}: {meas_flux.shape[0]} training samples")

        # One no-grad ray-tracing pass at the loaded parameters.
        scenario.set_number_of_rays(args.rays)
        base_pos = (
            kinematic._base_position_deviation.detach()
            if hasattr(kinematic, "_base_position_deviation")
            else torch.zeros(1, 3, device=device)
        )
        with torch.no_grad():
            _, pred_flux = _forward_pass(
                scenario, hg, rays, active_mask, target_mask, base_pos, device,
                motor_positions=motor_pos,
            )
        log.info(f"Traced predicted flux: {tuple(pred_flux.shape)}")

        # Drop samples whose measured or predicted spot touches the bitmap border
        # (clipped spots give meaningless size estimates).
        def _touches_border(f: torch.Tensor) -> torch.Tensor:
            n = f / (f.amax(dim=(-2, -1), keepdim=True) + 1e-12)
            edge = torch.cat([n[:, 0, :], n[:, -1, :], n[:, :, 0], n[:, :, -1]], dim=-1)
            return edge.amax(dim=-1) > 0.2

        keep = ~(_touches_border(pred_flux) | _touches_border(meas_flux.to(device)))
        n_drop = int((~keep).sum())
        if n_drop:
            log.info(f"Dropped {n_drop} samples with spots clipped at the bitmap border")
        pred_flux = pred_flux[keep]
        meas_flux = meas_flux.to(device)[keep]

        # τ sweep — identical preprocessing to the loss, only τ varies.
        taus = np.linspace(args.tau_min, args.tau_max, args.tau_steps)
        rows = []
        for tau in taus:
            ex = ContourExtractor(
                tau=float(tau), eta=cfg.CONTOUR_ETA,
                smoothing_rounds=cfg.CONTOUR_SMOOTHING_ROUNDS,
                gaussian_sigma=cfg.CONTOUR_GAUSS_SIGMA,
                gaussian_kernel_size=cfg.CONTOUR_GAUSS_KSIZE,
            ).to(device)
            with torch.no_grad():
                def _mask(f):
                    x = ex._normalize(f.unsqueeze(1))
                    x = ex._denoise(x)
                    x = ex._normalize(x)
                    return torch.sigmoid(ex.eta * (x - ex.tau)).squeeze(1)

                mp, mm = _spot_size_metrics(_mask(pred_flux)), _spot_size_metrics(_mask(meas_flux))
            row = {"tau": float(tau)}
            for k in ("area", "height", "width"):
                row[f"pred_{k}"] = float(mp[k].median())
                row[f"meas_{k}"] = float(mm[k].median())
                row[f"ratio_{k}"] = float((mp[k] / mm[k].clamp(min=1e-8)).median())
            rows.append(row)
            log.info(
                f"τ={tau:.3f}  area ratio={row['ratio_area']:.3f}  "
                f"height ratio={row['ratio_height']:.3f}  "
                f"(pred h={row['pred_height']:.1f} px, meas h={row['meas_height']:.1f} px)"
            )

    with open(args.output_dir / f"{args.heliostat_id}_tau_sweep.json", "w") as fh:
        json.dump({"heliostat_id": args.heliostat_id, "n_samples": int(keep.sum()),
                   "surface_points": args.surface_points, "rows": rows}, fh, indent=2)

    # ---------------- plot ----------------
    t = np.array([r["tau"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    for ax, key, label in [
        (axes[0], "height", "vertical extent 2σ_u  [px]"),
        (axes[1], "area", "soft-mask area  [px]"),
    ]:
        ax.plot(t, [r[f"pred_{key}"] for r in rows], "o-", label="predicted", color="tab:red")
        ax.plot(t, [r[f"meas_{key}"] for r in rows], "s-", label="measured", color="tab:green")
        ax.axvline(cfg.CONTOUR_TAU, ls="--", c="k", lw=1, label=f"current τ={cfg.CONTOUR_TAU}")
        ax.set_xlabel("τ"); ax.set_ylabel(label); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_title(f"{label.split('  ')[0]} vs τ", fontsize=10)
    axes[1].set_yscale("log")

    ax = axes[2]
    for key, c in [("height", "tab:blue"), ("area", "tab:orange"), ("width", "tab:purple")]:
        ax.plot(t, [r[f"ratio_{key}"] for r in rows], "o-", color=c, label=key)
    ax.axhline(1.0, ls="-", c="k", lw=1.2, label="parity (no size bias)")
    ax.axvline(cfg.CONTOUR_TAU, ls="--", c="k", lw=1)
    ax.set_xlabel("τ"); ax.set_ylabel("predicted / measured"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8); ax.set_title("size ratio vs τ", fontsize=10)

    fig.suptitle(
        f"{args.heliostat_id} — contour τ diagnostic  "
        f"(n={int(keep.sum())} train samples, {args.surface_points}×{args.surface_points} surface pts)\n"
        "ratio = 1 means predicted and measured contours have the same size — no upper-edge bias",
        fontsize=11,
    )
    plt.tight_layout()
    out = args.output_dir / f"{args.heliostat_id}_tau_diagnostic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out}")


if __name__ == "__main__":
    main()
