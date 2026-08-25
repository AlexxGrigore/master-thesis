"""
Evaluation for fine_error_learning runs.

Headline metric: per-heliostat centroid error [mrad] on a held-out split,
BEFORE (θ_KR warm start, Δθ = 0) vs AFTER (θ_KR + Δθ from the best model).

Synthetic-only extra: parameter recovery against the ground-truth
perturbations.json of the dataset. The remaining error after the warm start is
(θ_GT − θ_KR); the model's Δθ should move θ toward θ_GT. Reported per
parameter group as MAE. Pivot radii have no ground truth and are skipped.
(Same direct comparison convention as the recovery_*.png plots of the live
pipeline — perturbations are the deviation-parameter values applied at dataset
generation time.)

Writes to <run_dir>/:
    evaluation_<split>.json   — per-heliostat + aggregate metrics
    mrad_before_after_<split>.png — scatter + histogram, before vs after
    evaluation_<split>.md     — per-heliostat table (markdown deliverable)
"""
from __future__ import annotations

import json
import logging
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from fine_error_learning import data as fel_data
from fine_error_learning import pipeline as fel_pipeline
from fine_error_learning import warm_start as fel_warm_start
from fine_error_learning.model import FelTransformerModel

log = logging.getLogger(__name__)

EVAL_RAYS = 50          # rays per surface point for evaluation (DISPLAY_RAYS idiom)
EVAL_SEED = 123         # fixed seed → deterministic before/after comparison

# Parameter groups of the 24-D vector for recovery reporting:
# (name, slice, unit scale for the report)
_PARAM_GROUPS = (
    ("rotation [mrad]", slice(0, 4), 1e3),
    ("translation [mm]", slice(4, 13), 1e3),
    ("actuator_angle [mrad]", slice(13, 15), 1e3),
    ("actuator_stroke [mm]", slice(15, 17), 1e3),
    ("actuator_offset [mm]", slice(17, 19), 1e3),
    # pivot_radius [19:21] — no ground truth, skipped
    ("base_position [mm]", slice(21, 24), 1e3),
)


def _load_model(run_dir: pathlib.Path, device: torch.device) -> FelTransformerModel:
    """Rebuild the model from the run's config snapshot and best checkpoint."""
    with open(run_dir / "config.json") as f:
        snap = json.load(f)
    model = FelTransformerModel(
        d_model=int(snap["D_MODEL"]),
        n_heads=int(snap["N_HEADS"]),
        n_layers=int(snap["N_LAYERS"]),
        d_ff=int(snap["D_FF"]),
        dropout=float(snap["DROPOUT"]),
        d_img=int(snap["D_IMG"]),
        use_flux=bool(snap["USE_FLUX"]),
        bounded_head=bool(snap["BOUNDED_HEAD"]),
        output_gain=float(snap.get("OUTPUT_GAIN", 0.01)),
        query_decoder=bool(snap.get("QUERY_DECODER", False)),
    ).to(device)
    state_dict = torch.load(run_dir / "fel_model_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def _mrad_under_theta(
    state: fel_warm_start.WarmStartState,
    theta: torch.Tensor,
    measurements: fel_data.HeliostatMeasurements,
    mini_batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Mean/median centroid error [mrad] of one heliostat under ``theta``."""
    state.scenario.set_number_of_rays(EVAL_RAYS)
    errors: list[torch.Tensor] = []
    n = measurements.n
    for s in range(0, n, mini_batch_size):
        e = min(s + mini_batch_size, n)
        flux, bitmap_resolution, sampler_indices = fel_pipeline.predict_flux(
            state=state,
            theta_final=theta,
            incident_rays=measurements.incident_rays[s:e],
            motor_positions=measurements.motor_positions[s:e],
            target_indices=measurements.target_indices[s:e],
            device=device,
            random_seed=EVAL_SEED,
        )
        lps, _ = fel_pipeline.focal_spot_centroid_loss(
            predicted_flux=flux,
            focal_spots=measurements.focal_spots[s:e][sampler_indices],
            target_indices=measurements.target_indices[s:e][sampler_indices],
            bitmap_resolution=bitmap_resolution,
            scenario=state.scenario,
            device=device,
        )
        errors.append(torch.sqrt(lps).detach().cpu())
    err_m = torch.cat(errors)
    err_mrad = err_m / state.hel_dist_m * 1000.0
    return err_mrad.nanmean().item(), err_mrad.nanmedian().item()


@torch.no_grad()
def _mrad_query_mode(
    model: FelTransformerModel,
    model_inputs: tuple,
    state: fel_warm_start.WarmStartState,
    measurements: fel_data.HeliostatMeasurements,
    scaler_mean: torch.Tensor,
    scaler_std: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, torch.Tensor]:
    """Per-measurement centroid error under Δθ(sun position) — Experiment B.

    Each measurement is queried with its own standardized sun direction and
    ray-traced under its own θ_KR + Δθ_i. Returns (mean, median [mrad], mean Δθ
    over measurements — used for |Δθ| reporting and parameter recovery).
    """
    state.scenario.set_number_of_rays(EVAL_RAYS)
    errors: list[torch.Tensor] = []
    deltas: list[torch.Tensor] = []
    for qi in range(measurements.n):
        ray = measurements.incident_rays[qi, :3]
        query = ((ray - scaler_mean[:3].to(device)) / scaler_std[:3].to(device)).view(1, 1, 3)
        delta = model.forward_queries(*model_inputs, query).squeeze(0).squeeze(0)
        deltas.append(delta)
        flux, bitmap_resolution, sampler_indices = fel_pipeline.predict_flux(
            state=state,
            theta_final=state.theta_kr.to(device) + delta,
            incident_rays=measurements.incident_rays[qi : qi + 1],
            motor_positions=measurements.motor_positions[qi : qi + 1],
            target_indices=measurements.target_indices[qi : qi + 1],
            device=device,
            random_seed=EVAL_SEED,
        )
        lps, _ = fel_pipeline.focal_spot_centroid_loss(
            predicted_flux=flux,
            focal_spots=measurements.focal_spots[qi : qi + 1][sampler_indices],
            target_indices=measurements.target_indices[qi : qi + 1][sampler_indices],
            bitmap_resolution=bitmap_resolution,
            scenario=state.scenario,
            device=device,
        )
        errors.append(torch.sqrt(lps).detach().cpu())
    err_m = torch.cat(errors)
    err_mrad = err_m / state.hel_dist_m * 1000.0
    mean_delta = torch.stack(deltas).mean(dim=0)
    return err_mrad.nanmean().item(), err_mrad.nanmedian().item(), mean_delta


def _theta_gt_24(perturbations: dict, heliostat_id: str) -> torch.Tensor | None:
    """Ground-truth perturbation of one heliostat in the 24-D ordering.

    pivot_radius [19:21] has no ground truth — filled with NaN and masked out
    by the caller. Returns None if the heliostat is absent.
    """
    gt = perturbations.get(heliostat_id)
    if gt is None:
        return None
    return torch.tensor(
        gt["rotation_rad"]          # 4
        + gt["translation_m"]       # 9
        + gt["actuator_angle_rad"]  # 2
        + gt["actuator_stroke_m"]   # 2
        + gt["actuator_offset_m"]   # 2
        + [float("nan")] * 2        # pivot_radius — no GT
        + gt["base_position_m"],    # 3
        dtype=torch.float32,
    )


def evaluate_run(
    cfg,
    device: torch.device,
    run_dir: pathlib.Path,
    split: str = "test",
    heliostat_ids: list[str] | None = None,
) -> dict:
    run_dir = pathlib.Path(run_dir)
    model = _load_model(run_dir, device)
    scaler_mean, scaler_std = fel_data.load_scaler(run_dir / "scaler_stats.json")

    with open(run_dir / "config.json") as f:
        snap = json.load(f)
    if heliostat_ids is None:
        heliostat_ids = snap.get("HELIOSTAT_IDS") or fel_data.discover_heliostat_ids(
            cfg.STAGE1_CHECKPOINT_DIR
        )
    # Token/batch sizes come from the run snapshot (may differ from the current
    # config, e.g. smoke runs use K=8).
    k_tokens = int(snap.get("K_TOKENS", cfg.K_TOKENS))
    mini_batch_size = int(snap.get("MINI_BATCH_SIZE", cfg.MINI_BATCH_SIZE))

    perturbations_path = pathlib.Path(cfg.SYNTHETIC_DATA_DIR) / "perturbations.json"
    perturbations = (
        json.load(open(perturbations_path)) if perturbations_path.exists() else {}
    )

    rows: list[dict] = []
    for hid in heliostat_ids:
        state = fel_warm_start.load_warm_start_state(hid, cfg, device)
        measurements = fel_data.load_measurements(
            pathlib.Path(cfg.SYNTHETIC_DATA_DIR) / split, hid,
            state.heliostat_group, state.scenario, device,
        )
        if measurements is None:
            log.warning(f"  {hid}: no {split} data — skipped")
            continue

        flux_t, scalars_t, mask_t = fel_data.build_model_tokens(
            measurements, k_tokens, scaler_mean, scaler_std, generator=None
        )
        model_inputs = (
            flux_t.unsqueeze(0), scalars_t.unsqueeze(0), mask_t.unsqueeze(0),
            state.theta_kr.to(device).unsqueeze(0),
            state.heliostat_position.to(device).unsqueeze(0),
        )

        before_mean, before_median = _mrad_under_theta(
            state, state.theta_kr.to(device), measurements, mini_batch_size, device
        )

        if bool(snap.get("QUERY_DECODER", False)):
            # Experiment B: per-measurement Δθ conditioned on its sun position.
            after_mean, after_median, delta = _mrad_query_mode(
                model, model_inputs, state, measurements, scaler_mean, scaler_std, device
            )
        else:
            delta = model(*model_inputs).squeeze(0)
            after_mean, after_median = _mrad_under_theta(
                state, state.theta_kr.to(device) + delta, measurements,
                mini_batch_size, device,
            )

        row = {
            "heliostat_id": hid,
            "mrad_before_mean": before_mean,
            "mrad_before_median": before_median,
            "mrad_after_mean": after_mean,
            "mrad_after_median": after_median,
            "delta_norm": delta.norm().item(),
        }

        # Parameter recovery vs ground truth (synthetic only).
        theta_gt = _theta_gt_24(perturbations, hid)
        if theta_gt is not None:
            theta_gt = theta_gt.to(device)
            err_before = state.theta_kr.to(device) - theta_gt
            err_after = err_before + delta
            valid = torch.isfinite(theta_gt)
            row["recovery"] = {
                name: {
                    "mae_before": err_before[sl][valid[sl]].abs().mean().item() * scale,
                    "mae_after": err_after[sl][valid[sl]].abs().mean().item() * scale,
                }
                for name, sl, scale in _PARAM_GROUPS
            }
        rows.append(row)
        log.info(
            f"  {hid}: {before_mean:.3f} → {after_mean:.3f} mrad (mean, {split})"
        )

    if not rows:
        raise RuntimeError(f"No heliostats evaluated for split {split!r}.")

    aggregate = {
        "mrad_before_mean": sum(r["mrad_before_mean"] for r in rows) / len(rows),
        "mrad_after_mean": sum(r["mrad_after_mean"] for r in rows) / len(rows),
        "mrad_before_median": sorted(r["mrad_before_median"] for r in rows)[len(rows) // 2],
        "mrad_after_median": sorted(r["mrad_after_median"] for r in rows)[len(rows) // 2],
        "n_heliostats": len(rows),
        "split": split,
    }
    results = {"aggregate": aggregate, "per_heliostat": rows}

    with open(run_dir / f"evaluation_{split}.json", "w") as f:
        json.dump(results, f, indent=2)

    _plot_before_after(rows, aggregate, run_dir / f"mrad_before_after_{split}.png", split)
    _write_markdown(rows, aggregate, run_dir / f"evaluation_{split}.md", split, bool(perturbations))
    log.info(
        f"Eval ({split}): {aggregate['mrad_before_mean']:.3f} → "
        f"{aggregate['mrad_after_mean']:.3f} mrad (mean over {len(rows)} heliostats)"
    )
    return results


def _plot_before_after(rows: list[dict], aggregate: dict, path: pathlib.Path, split: str) -> None:
    before = [r["mrad_before_mean"] for r in rows]
    after = [r["mrad_after_mean"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    lim = max(before + after) * 1.05
    axes[0].scatter(before, after, s=25, alpha=0.8)
    axes[0].plot([0, lim], [0, lim], "k--", lw=1, label="no change")
    axes[0].set_xlabel("θ_KR (stage-1 warm start) [mrad]")
    axes[0].set_ylabel("θ_KR + Δθ (FEL) [mrad]")
    axes[0].set_xlim(0, lim)
    axes[0].set_ylim(0, lim)
    axes[0].legend()
    axes[0].set_title(f"Per-heliostat centroid error ({split})")

    axes[1].hist(before, bins=15, alpha=0.6, label=f"before (mean {aggregate['mrad_before_mean']:.2f})")
    axes[1].hist(after, bins=15, alpha=0.6, label=f"after (mean {aggregate['mrad_after_mean']:.2f})")
    axes[1].set_xlabel("centroid error [mrad]")
    axes[1].set_ylabel("# heliostats")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_markdown(
    rows: list[dict], aggregate: dict, path: pathlib.Path, split: str, has_gt: bool
) -> None:
    lines = [
        f"# FEL evaluation — {split} split",
        "",
        f"- Heliostats: {aggregate['n_heliostats']}",
        f"- Mean centroid error: {aggregate['mrad_before_mean']:.3f} → "
        f"{aggregate['mrad_after_mean']:.3f} mrad",
        f"- Median: {aggregate['mrad_before_median']:.3f} → "
        f"{aggregate['mrad_after_median']:.3f} mrad",
        "",
        "| heliostat | before [mrad] | after [mrad] | Δ [mrad] | |Δθ| |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: r["mrad_after_mean"] - r["mrad_before_mean"]):
        improvement = r["mrad_before_mean"] - r["mrad_after_mean"]
        lines.append(
            f"| {r['heliostat_id']} | {r['mrad_before_mean']:.3f} | "
            f"{r['mrad_after_mean']:.3f} | {improvement:+.3f} | {r['delta_norm']:.4f} |"
        )
    if has_gt:
        lines += [
            "",
            "## Parameter recovery (MAE vs ground-truth perturbations)",
            "",
            "| group | θ_KR MAE | θ_KR+Δθ MAE |",
            "|---|---|---|",
        ]
        group_names = [name for name, _, _ in _PARAM_GROUPS]
        for name in group_names:
            keyed = [r["recovery"][name] for r in rows if "recovery" in r]
            if not keyed:
                continue
            mae_b = sum(k["mae_before"] for k in keyed) / len(keyed)
            mae_a = sum(k["mae_after"] for k in keyed) / len(keyed)
            lines.append(f"| {name} | {mae_b:.3f} | {mae_a:.3f} |")
    path.write_text("\n".join(lines) + "\n")
