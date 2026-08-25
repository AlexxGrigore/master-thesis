"""
Training loop for fine_error_learning.

One shared FelTransformerModel across all heliostats. One training sample =
one heliostat = its measurement set. Per heliostat the model predicts Δθ from
K measurement tokens; θ_final = θ_KR + Δθ is written functionally into the
kinematics and the stage-2 focal-spot centroid loss is ray-traced through
ARTIST (fresh HeliostatRayTracer per mini-batch, recorded motor positions).

Saves to output_dir/:
    fel_model_best.pt    — model state_dict at the best val centroid error [mrad]
    scaler_stats.json    — train-split scalar standardization statistics
    on_target.json       — per-heliostat on-target fraction under θ_KR
    history.csv          — epoch, train_loss, val_loss, val_mrad, lr
    loss_curves.png      — train/val loss and val mrad curves
    summary.md           — short run summary
"""
from __future__ import annotations

import csv
import json
import logging
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from artist_extensions.loss_functions_ext import robust_reduce_squared
from fine_error_learning import data as fel_data
from fine_error_learning import pipeline as fel_pipeline
from fine_error_learning import warm_start as fel_warm_start
from fine_error_learning.model import FelTransformerModel

log = logging.getLogger(__name__)


def _heliostat_batches(heliostat_ids: list[str], batch_size: int):
    for start in range(0, len(heliostat_ids), batch_size):
        yield heliostat_ids[start : start + batch_size]


def _model_inputs(state, measurements, cfg, scaler, device, generator):
    """Token batch [1, K, ...] + normalized globals for one heliostat."""
    flux, scalars, mask = fel_data.build_model_tokens(
        measurements, cfg.K_TOKENS, scaler[0], scaler[1], generator=generator
    )
    return (
        flux.unsqueeze(0),
        scalars.unsqueeze(0),
        mask.unsqueeze(0),
        state.theta_kr.to(device).unsqueeze(0),
        state.heliostat_position.to(device).unsqueeze(0),
    )


def run(cfg, device: torch.device, output_dir: pathlib.Path) -> dict:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.RANDOM_SEED)

    # ------------------------------------------------------------------ #
    # Warm start + data                                                    #
    # ------------------------------------------------------------------ #
    heliostat_ids = cfg.HELIOSTAT_IDS or fel_data.discover_heliostat_ids(
        cfg.STAGE1_CHECKPOINT_DIR
    )
    log.info(f"Heliostats: {len(heliostat_ids)}  |  warm start: {cfg.WARM_START}")

    states: dict[str, fel_warm_start.WarmStartState] = {}
    train_data: dict[str, fel_data.HeliostatMeasurements] = {}
    val_data: dict[str, fel_data.HeliostatMeasurements] = {}
    for hid in heliostat_ids:
        state = fel_warm_start.load_warm_start_state(hid, cfg, device)
        train = fel_data.load_measurements(
            pathlib.Path(cfg.SYNTHETIC_DATA_DIR) / "train", hid,
            state.heliostat_group, state.scenario, device,
        )
        if train is None:
            log.warning(f"  {hid}: no train data — skipped")
            continue
        val = fel_data.load_measurements(
            pathlib.Path(cfg.SYNTHETIC_DATA_DIR) / "val", hid,
            state.heliostat_group, state.scenario, device,
        )
        max_m = getattr(cfg, "MAX_MEASUREMENTS", None)
        train = train.capped(max_m)
        if val is not None:
            val = val.capped(max_m)
        states[hid] = state
        train_data[hid] = train
        if val is not None:
            val_data[hid] = val
        state.scenario.set_number_of_rays(cfg.TRAIN_RAYS)
        log.info(f"  {hid}: {train.n} train / {val.n if val else 0} val measurements")

    heliostat_ids = [h for h in heliostat_ids if h in states]
    if not heliostat_ids:
        raise RuntimeError("No heliostats with both a warm start and train data.")

    # Scalar standardization from the train split.
    scaler = fel_data.compute_scaler(list(train_data.values()))
    fel_data.save_scaler(scaler[0], scaler[1], output_dir / "scaler_stats.json")

    # On-target fraction under θ_KR (plan deliverable — the focal-spot loss
    # has no gradient for off-target heliostats).
    log.info("On-target fraction under θ_KR (warm start):")
    for hid in heliostat_ids:
        fel_warm_start.measure_on_target_fraction(
            states[hid], train_data[hid], device,
            max_measurements=cfg.ON_TARGET_MAX_MEASUREMENTS,
            n_rays=cfg.ON_TARGET_RAYS,
        )
    with open(output_dir / "on_target.json", "w") as f:
        json.dump(
            {
                hid: {
                    "on_target_fraction": states[hid].on_target_fraction,
                    "mean_error_m": states[hid].on_target_mean_error_m,
                }
                for hid in heliostat_ids
            },
            f,
            indent=2,
        )

    # ------------------------------------------------------------------ #
    # Model / optimizer                                                    #
    # ------------------------------------------------------------------ #
    model = FelTransformerModel(
        d_model=cfg.D_MODEL,
        n_heads=cfg.N_HEADS,
        n_layers=cfg.N_LAYERS,
        d_ff=cfg.D_FF,
        dropout=cfg.DROPOUT,
        d_img=cfg.D_IMG,
        use_flux=cfg.USE_FLUX,
        bounded_head=cfg.BOUNDED_HEAD,
        output_gain=getattr(cfg, "OUTPUT_GAIN", 0.01),
        query_decoder=getattr(cfg, "QUERY_DECODER", False),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"FelTransformerModel parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.BASE_LR, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=cfg.PLATEAU_FACTOR, patience=cfg.PLATEAU_PATIENCE
    )

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #
    history: list[dict] = []
    delta_norm_history: list[float] = []
    best_val_mrad = float("inf")
    t0 = time.time()
    query_mode = getattr(cfg, "QUERY_DECODER", False)
    pixel_ratio = getattr(cfg, "PIXEL_LOSS_RATIO", 0.0)
    pixel_lambda: float | None = None  # auto-calibrated on the first mini-batch

    def _std_query(measurements, qi: int) -> torch.Tensor:
        """Standardized sun direction of measurement qi → [1, 1, 3] query."""
        ray = measurements.incident_rays[qi, :3]
        q = (ray - scaler[0][:3].to(device)) / scaler[1][:3].to(device)
        return q.view(1, 1, 3)

    for epoch in tqdm(range(1, cfg.EPOCHS + 1), desc="FEL training"):
        model.train()
        generator = torch.Generator(device=device)
        generator.manual_seed(cfg.RANDOM_SEED + epoch)
        order = torch.randperm(len(heliostat_ids), generator=torch.Generator().manual_seed(cfg.RANDOM_SEED + epoch)).tolist()
        shuffled = [heliostat_ids[i] for i in order]

        epoch_loss = 0.0
        for batch in _heliostat_batches(shuffled, cfg.HELI_BATCH_SIZE):
            optimizer.zero_grad()
            for hid in batch:
                state = states[hid]
                measurements = train_data[hid]
                inputs = _model_inputs(state, measurements, cfg, scaler, device, generator)
                theta_kr = state.theta_kr.to(device)

                if query_mode:
                    # Experiment B: per-query Δθ — each sampled measurement is
                    # ray-traced under its own sun-position-conditioned Δθ.
                    q_idx = torch.randperm(
                        measurements.n, generator=generator, device=device
                    )[: cfg.QUERIES_PER_STEP]
                    weight = 1.0 / (len(batch) * len(q_idx))
                    hel_loss = 0.0
                    for qi in q_idx.tolist():
                        delta = model.forward_queries(
                            *inputs, _std_query(measurements, qi)
                        ).squeeze(0).squeeze(0)
                        flux, bitmap_resolution, sampler_indices = fel_pipeline.predict_flux(
                            state=state,
                            theta_final=theta_kr + delta,
                            incident_rays=measurements.incident_rays[qi : qi + 1],
                            motor_positions=measurements.motor_positions[qi : qi + 1],
                            target_indices=measurements.target_indices[qi : qi + 1],
                            device=device,
                            random_seed=cfg.RANDOM_SEED + 1000 * epoch + qi,
                        )
                        lps, _ = fel_pipeline.focal_spot_centroid_loss(
                            predicted_flux=flux,
                            focal_spots=measurements.focal_spots[qi : qi + 1][sampler_indices],
                            target_indices=measurements.target_indices[qi : qi + 1][sampler_indices],
                            bitmap_resolution=bitmap_resolution,
                            scenario=state.scenario,
                            device=device,
                        )
                        delta_m = cfg.HUBER_DELTA_MRAD * state.hel_dist_m / 1000.0
                        q_loss = robust_reduce_squared(
                            lps, mode=cfg.LOSS_REDUCTION, delta=delta_m
                        )
                        (
                            (q_loss + cfg.RESIDUAL_L2_WEIGHT * delta.pow(2).mean()) * weight
                        ).backward()
                        hel_loss += q_loss.detach().item() / len(q_idx)
                    epoch_loss += hel_loss / len(batch)
                    continue

                n_mb = (measurements.n + cfg.MINI_BATCH_SIZE - 1) // cfg.MINI_BATCH_SIZE
                weight = 1.0 / (len(batch) * n_mb)
                hel_loss = 0.0
                for mb in range(n_mb):
                    s = mb * cfg.MINI_BATCH_SIZE
                    e = min(s + cfg.MINI_BATCH_SIZE, measurements.n)
                    # Recompute Δθ per mini-batch: each backward() frees the
                    # graph, and the model forward is cheap vs. ray tracing.
                    delta = model(*inputs).squeeze(0)
                    flux, bitmap_resolution, sampler_indices = fel_pipeline.predict_flux(
                        state=state,
                        theta_final=theta_kr + delta,
                        incident_rays=measurements.incident_rays[s:e],
                        motor_positions=measurements.motor_positions[s:e],
                        target_indices=measurements.target_indices[s:e],
                        device=device,
                        random_seed=cfg.RANDOM_SEED + 1000 * epoch + mb,
                    )
                    lps, _ = fel_pipeline.focal_spot_centroid_loss(
                        predicted_flux=flux,
                        focal_spots=measurements.focal_spots[s:e][sampler_indices],
                        target_indices=measurements.target_indices[s:e][sampler_indices],
                        bitmap_resolution=bitmap_resolution,
                        scenario=state.scenario,
                        device=device,
                    )
                    delta_m = cfg.HUBER_DELTA_MRAD * state.hel_dist_m / 1000.0
                    mb_loss = robust_reduce_squared(
                        lps, mode=cfg.LOSS_REDUCTION, delta=delta_m
                    )
                    if pixel_ratio > 0:
                        # Experiment A: auxiliary flux-distribution loss, weight
                        # auto-calibrated once so λ·L_pixel ≈ ratio·L_centroid.
                        lpx = fel_pipeline.pixelwise_flux_loss(
                            predicted_flux=flux,
                            gt_flux=measurements.flux[s:e][sampler_indices],
                            out_size=cfg.PIXEL_LOSS_DOWNSIZE,
                        ).mean()
                        if pixel_lambda is None:
                            pixel_lambda = (
                                pixel_ratio
                                * mb_loss.detach()
                                / lpx.detach().clamp(min=1e-12)
                            ).item()
                            log.info(
                                f"Pixel-loss weight auto-calibrated: λ = {pixel_lambda:.4g} "
                                f"(L_centroid {mb_loss.item():.4g} m², L_pixel {lpx.item():.4g})"
                            )
                        mb_loss = mb_loss + pixel_lambda * lpx
                    (
                        (mb_loss + cfg.RESIDUAL_L2_WEIGHT * delta.pow(2).mean()) * weight
                    ).backward()
                    hel_loss += mb_loss.detach().item() / n_mb
                epoch_loss += hel_loss / len(batch)

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

        # -------------------------------------------------------------- #
        # Validation (no grad, eval mode → deterministic Δθ)               #
        # -------------------------------------------------------------- #
        model.eval()
        val_loss = 0.0
        val_mrad = 0.0
        val_delta_norm = 0.0
        n_val = 0
        with torch.no_grad():
            for hid, measurements in val_data.items():
                state = states[hid]
                inputs = _model_inputs(state, measurements, cfg, scaler, device, None)
                state.scenario.set_number_of_rays(cfg.VAL_RAYS)
                if query_mode:
                    # Fixed val subset, one Δθ(query) + one trace per measurement.
                    n_qv = min(cfg.VAL_QUERY_MEASUREMENTS, measurements.n)
                    for qi in range(n_qv):
                        delta = model.forward_queries(
                            *inputs, _std_query(measurements, qi)
                        ).squeeze(0).squeeze(0)
                        flux, bitmap_resolution, sampler_indices = fel_pipeline.predict_flux(
                            state=state,
                            theta_final=state.theta_kr.to(device) + delta,
                            incident_rays=measurements.incident_rays[qi : qi + 1],
                            motor_positions=measurements.motor_positions[qi : qi + 1],
                            target_indices=measurements.target_indices[qi : qi + 1],
                            device=device,
                            random_seed=42,
                        )
                        lps, _ = fel_pipeline.focal_spot_centroid_loss(
                            predicted_flux=flux,
                            focal_spots=measurements.focal_spots[qi : qi + 1][sampler_indices],
                            target_indices=measurements.target_indices[qi : qi + 1][sampler_indices],
                            bitmap_resolution=bitmap_resolution,
                            scenario=state.scenario,
                            device=device,
                        )
                        val_loss += lps.mean().item() / n_qv
                        val_mrad += (
                            torch.sqrt(lps).mean().item() / state.hel_dist_m * 1000.0 / n_qv
                        )
                        val_delta_norm += delta.norm().item() / n_qv
                    n_val += 1
                    state.scenario.set_number_of_rays(cfg.TRAIN_RAYS)
                    continue
                delta = model(*inputs).squeeze(0)
                n_mb = (measurements.n + cfg.MINI_BATCH_SIZE - 1) // cfg.MINI_BATCH_SIZE
                for mb in range(n_mb):
                    s = mb * cfg.MINI_BATCH_SIZE
                    e = min(s + cfg.MINI_BATCH_SIZE, measurements.n)
                    flux, bitmap_resolution, sampler_indices = fel_pipeline.predict_flux(
                        state=state,
                        theta_final=state.theta_kr.to(device) + delta,
                        incident_rays=measurements.incident_rays[s:e],
                        motor_positions=measurements.motor_positions[s:e],
                        target_indices=measurements.target_indices[s:e],
                        device=device,
                        random_seed=42,
                    )
                    lps, pred_coords = fel_pipeline.focal_spot_centroid_loss(
                        predicted_flux=flux,
                        focal_spots=measurements.focal_spots[s:e][sampler_indices],
                        target_indices=measurements.target_indices[s:e][sampler_indices],
                        bitmap_resolution=bitmap_resolution,
                        scenario=state.scenario,
                        device=device,
                    )
                    val_loss += lps.mean().item() / n_mb
                    val_mrad += (
                        torch.sqrt(lps).mean().item() / state.hel_dist_m * 1000.0 / n_mb
                    )
                val_delta_norm += delta.norm().item()
                n_val += 1
                state.scenario.set_number_of_rays(cfg.TRAIN_RAYS)
        if n_val:
            val_loss /= n_val
            val_mrad /= n_val
            val_delta_norm /= n_val
        delta_norm_history.append(val_delta_norm)

        scheduler.step(val_mrad)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "val_loss": val_loss,
                "val_mrad": val_mrad,
                "lr": current_lr,
            }
        )
        log.info(
            f"Epoch {epoch:>4}: train {epoch_loss:.4f} m²  |  val {val_loss:.4f} m²  |  "
            f"val {val_mrad:.3f} mrad  |  |Δθ| {val_delta_norm:.4f}  |  lr {current_lr:.2e}"
        )

        if val_mrad < best_val_mrad:
            best_val_mrad = val_mrad
            torch.save(model.state_dict(), output_dir / "fel_model_best.pt")

    # ------------------------------------------------------------------ #
    # Outputs                                                              #
    # ------------------------------------------------------------------ #
    with open(output_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_mrad", "lr"])
        writer.writeheader()
        writer.writerows(history)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [h["epoch"] for h in history]
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    if n_val:
        axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("focal-spot loss [m²]")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[1].plot(epochs, [h["val_mrad"] for h in history], color="tab:green")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val centroid error [mrad]")
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close(fig)

    results = {
        "heliostat_ids": heliostat_ids,
        "n_heliostats": len(heliostat_ids),
        "best_val_mrad": best_val_mrad,
        "initial_train_loss": history[0]["train_loss"],
        "final_train_loss": history[-1]["train_loss"],
        "initial_val_mrad": history[0]["val_mrad"],
        "final_val_mrad": history[-1]["val_mrad"],
        "delta_norm_history": delta_norm_history,
        "total_time_min": (time.time() - t0) / 60.0,
        "on_target": {
            hid: states[hid].on_target_fraction for hid in heliostat_ids
        },
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "# Fine Error Learning — run summary",
        "",
        f"- Heliostats: {len(heliostat_ids)} ({', '.join(heliostat_ids)})",
        f"- Warm start: `{cfg.WARM_START}` from `{cfg.STAGE1_CHECKPOINT_DIR}`",
        f"- Data: `{cfg.SYNTHETIC_DATA_DIR}`",
        f"- Model: d_model={cfg.D_MODEL}, layers={cfg.N_LAYERS}, K={cfg.K_TOKENS}, "
        f"use_flux={cfg.USE_FLUX}, bounded_head={cfg.BOUNDED_HEAD}",
        f"- Epochs: {cfg.EPOCHS}  |  train rays: {cfg.TRAIN_RAYS}  |  "
        f"total time: {results['total_time_min']:.1f} min",
        "",
        "## On-target fraction under θ_KR",
        "",
        "| heliostat | on-target | mean error [m] |",
        "|---|---|---|",
    ]
    for hid in heliostat_ids:
        lines.append(
            f"| {hid} | {states[hid].on_target_fraction:.0%} | "
            f"{states[hid].on_target_mean_error_m:.3f} |"
        )
    lines += [
        "",
        "## Training",
        "",
        f"- Initial train loss: {results['initial_train_loss']:.4f} m²  →  "
        f"final: {results['final_train_loss']:.4f} m²",
        f"- Initial val error: {results['initial_val_mrad']:.3f} mrad  →  "
        f"best: {best_val_mrad:.3f} mrad",
        f"- |Δθ| (val, eval mode) per epoch: "
        + ", ".join(f"{d:.4f}" for d in delta_norm_history),
        "",
        "See `loss_curves.png`, `history.csv`, `on_target.json`, `fel_model_best.pt`.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")

    log.info(f"Best val centroid error: {best_val_mrad:.3f} mrad")
    return results
