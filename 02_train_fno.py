"""Train a PhysicsNeMo FNO surrogate from OpenFOAM cases.

Voxelizes cell-center CFD data to a regular cubic grid, then trains a 3D
Fourier Neural Operator. Architecture preferred by roadmap for its ability
to interpolate fluid dynamics across varying RPM accurately.

The trained model can be exported via:
    python 03_export_model.py --arch fno --out-dir out_fno

Usage:
    python 02_train_fno.py
    python 02_train_fno.py --grid-res 64 --epochs 3000
    python 02_train_fno.py --pi-weight 1e-3 --out out_fno
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import physicsnemo
import pyvista as pv
import torch
from physicsnemo.models.fno import FNO
from scipy.spatial import cKDTree
from torch import nn

from cfd_cases import discover_cases, resolve_input_dir

DEFAULT_CONFIG = {
    "input": "input",
    "out": "out_fno",
    "epochs": 3000,
    "save_every": 20,
    "lr": 1e-3,
    "lr_min": 1e-5,
    "seed": 0,
    "grid_res": 64,
    "infer_frames": 21,
    "validation_omega": None,
    "device": "auto",
    "fno_latent": 32,
    "fno_layers": 4,
    "fno_modes": 8,
    "resume": None,
    "make_zip": True,
    "pi_weight": 0.0,
}


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional JSON config file.")
    parser.add_argument(
        "--write-default-config",
        type=Path,
        help="Write the default JSON config to this path and exit.",
    )
    parser.add_argument("--input", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr-min", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--grid-res", type=int, help="Voxel grid resolution per axis (default 64).")
    parser.add_argument("--infer-frames", type=int)
    parser.add_argument("--validation-omega", type=float)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--fno-latent", type=int, help="FNO latent channel width.")
    parser.add_argument("--fno-layers", type=int, help="Number of FNO spectral layers.")
    parser.add_argument("--fno-modes", type=int, help="Fourier modes per spatial axis.")
    parser.add_argument("--resume", type=str, help="Checkpoint .pt to resume from.")
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument(
        "--pi-weight",
        type=float,
        help="Weight for finite-difference continuity loss (∇·U=0).",
    )

    args = parser.parse_args()

    if args.write_default_config:
        args.write_default_config.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
        print(f"Wrote default config: {args.write_default_config}")
        raise SystemExit(0)

    cfg = dict(DEFAULT_CONFIG)
    if args.config:
        loaded = json.loads(args.config.read_text())
        unknown = sorted(set(loaded) - set(DEFAULT_CONFIG))
        if unknown:
            raise ValueError(f"unknown config key(s): {unknown}")
        cfg.update(loaded)

    for key, cli_attr in [
        ("input", "input"), ("out", "out"), ("epochs", "epochs"),
        ("save_every", "save_every"), ("lr", "lr"), ("lr_min", "lr_min"),
        ("seed", "seed"), ("grid_res", "grid_res"), ("infer_frames", "infer_frames"),
        ("validation_omega", "validation_omega"), ("device", "device"),
        ("fno_latent", "fno_latent"), ("fno_layers", "fno_layers"),
        ("fno_modes", "fno_modes"), ("resume", "resume"), ("pi_weight", "pi_weight"),
    ]:
        val = getattr(args, cli_attr, None)
        if val is not None:
            cfg[key] = val
    if args.no_zip:
        cfg["make_zip"] = False

    return cfg


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return requested


def extract(path: Path):
    r = pv.POpenFOAMReader(str(path / "case.foam"))
    r.set_active_time_value(r.time_values[-1])
    r.enable_all_cell_arrays()
    r.cell_to_point_creation = False
    m = r.read()["internalMesh"]
    cc = np.asarray(m.cell_centers().points, dtype=np.float32)
    f = {k: np.asarray(m.cell_data[k], dtype=np.float32) for k in ["U", "p", "k", "nut", "omega"]}
    f["omega_turb"] = f.pop("omega")
    return cc, f


def voxelize(
    cc: np.ndarray,
    fields: dict,
    grid_res: int,
    bmin: np.ndarray,
    bmax: np.ndarray,
) -> dict:
    """Nearest-neighbor mapping from cell-center cloud to regular cubic grid."""
    axes = [np.linspace(bmin[i], bmax[i], grid_res, dtype=np.float32) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    query = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    _, idx = cKDTree(cc).query(query)
    G = grid_res
    result: dict = {}
    for k, v in fields.items():
        arr = v[idx]
        result[k] = arr.reshape(G, G, G) if arr.ndim == 1 else arr.reshape(G, G, G, arr.shape[1])
    return result


def fno_input(grid_res: int, omega: float, omega_max: float) -> np.ndarray:
    """Build (4, G, G, G) input: normalized grid coords [−1,1]³ + omega channel."""
    G = grid_res
    lin = np.linspace(-1.0, 1.0, G, dtype=np.float32)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    om_ch = np.full((G, G, G), omega / omega_max, dtype=np.float32)
    return np.stack([gx, gy, gz, om_ch], axis=0)  # (4, G, G, G)


def fno_target(vg: dict, omega: float, s: dict) -> np.ndarray:
    """Build (7, G, G, G) normalized target tensor with physics-informed scaling."""
    U = (vg["U"] / omega - np.asarray(s["U_mean"])) / np.asarray(s["U_std"])  # (G,G,G,3)
    p = (vg["p"] / omega ** 2 - s["p_mean"]) / s["p_std"]
    k = (vg["k"] / omega ** 2 - s["k_mean"]) / s["k_std"]
    nu = (vg["nut"] - s["nut_mean"]) / s["nut_std"]
    lom = (np.log(vg["omega_turb"] + 1e-6) - s["logom_mean"]) / s["logom_std"]
    Ut = np.moveaxis(U, -1, 0)  # (3, G, G, G)
    return np.concatenate([Ut, p[None], k[None], nu[None], lom[None]], axis=0)  # (7, G, G, G)


def fno_denorm(y: np.ndarray, omega: float, s: dict) -> dict:
    """Denormalize (7, G, G, G) output to physical fields."""
    U = (y[0:3] * np.asarray(s["U_std"])[:, None, None, None]
         + np.asarray(s["U_mean"])[:, None, None, None]) * omega
    return {
        "U": np.moveaxis(U, 0, -1),  # (G, G, G, 3)
        "p": (y[3] * s["p_std"] + s["p_mean"]) * omega ** 2,
        "k": (y[4] * s["k_std"] + s["k_mean"]) * omega ** 2,
        "nut": y[5] * s["nut_std"] + s["nut_mean"],
        "omega_turb": np.exp(y[6] * s["logom_std"] + s["logom_mean"]) - 1e-6,
    }


def continuity_residual_fd(
    pred: torch.Tensor,  # (1, 7, G, G, G)
    dx: float,
    dy: float,
    dz: float,
    U_std: torch.Tensor,  # (3,)
) -> torch.Tensor:
    """FD-based continuity residual ∇·U≈0 on regular grid (simpler than MLS)."""
    U = pred[0, 0:3]  # (3, G, G, G)
    dUx_dx = (U[0, 2:, 1:-1, 1:-1] - U[0, :-2, 1:-1, 1:-1]) / (2.0 * dx)
    dUy_dy = (U[1, 1:-1, 2:, 1:-1] - U[1, 1:-1, :-2, 1:-1]) / (2.0 * dy)
    dUz_dz = (U[2, 1:-1, 1:-1, 2:] - U[2, 1:-1, 1:-1, :-2]) / (2.0 * dz)
    div = U_std[0] * dUx_dx + U_std[1] * dUy_dy + U_std[2] * dUz_dz
    return (div ** 2).mean()


def load_checkpoint(model: nn.Module, path: str | None, device: str) -> None:
    if not path:
        return
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    print(f"Resumed from {path}")


def main() -> None:
    cfg = parse_args()
    print("torch", torch.__version__, "  cuda", torch.cuda.is_available())
    print("physicsnemo", physicsnemo.__version__)
    print("config:")
    print(json.dumps(cfg, indent=2))

    INPUT = resolve_input_dir(cfg["input"], base_dir=Path(__file__).parent)
    OUT = Path(cfg["out"])
    OUT.mkdir(exist_ok=True)
    G = int(cfg["grid_res"])
    DEVICE = choose_device(str(cfg["device"]))

    # 1 — Discover and extract every case
    case_records = discover_cases(INPUT)
    print(f"\nFound {len(case_records)} case(s):")

    cc_global = None
    raw_cases: dict[float, dict] = {}
    for case in case_records:
        t0 = time.time()
        cc, fields = extract(case.path)
        if cc_global is None:
            cc_global = cc
        raw_cases[case.omega] = fields
        umax = float(np.linalg.norm(fields["U"], axis=1).max())
        print(f"  {case.path.name}  omega={case.omega:.0f} rad/s  "
              f"({case.rpm:.0f} RPM)  |U|max={umax:.2f}  cells={cc.shape[0]:,}  "
              f"t={time.time()-t0:.1f}s")

    assert cc_global is not None
    OMEGAS = sorted(raw_cases.keys())

    # 2 — Train / val split
    if len(OMEGAS) < 3:
        raise ValueError("need at least 3 cases for interpolation LOO split")
    if cfg["validation_omega"] is None:
        VAL_OM = [OMEGAS[len(OMEGAS) // 2]]
    else:
        val_om = float(cfg["validation_omega"])
        if val_om not in OMEGAS:
            raise ValueError(f"validation_omega={val_om} not in {OMEGAS}")
        VAL_OM = [val_om]
    TRAIN_OM = [om for om in OMEGAS if om not in VAL_OM]
    INFER_OM = np.linspace(min(OMEGAS), max(OMEGAS), int(cfg["infer_frames"])).tolist()
    print(f"train: {TRAIN_OM}")
    print(f"val  : {VAL_OM}")
    print(f"infer: {INFER_OM[0]:.0f} -> {INFER_OM[-1]:.0f} ({len(INFER_OM)} frames)")

    # 3 — Determine grid bounds from training cases union
    bmin = cc_global.min(0)
    bmax = cc_global.max(0)
    span = bmax - bmin
    dx, dy, dz = (span / (G - 1)).tolist()
    print(f"\n--- Voxelizing to {G}³ grid ---")
    print(f"  bmin={bmin.tolist()}, bmax={bmax.tolist()}")

    t0 = time.time()
    vox: dict[float, dict] = {}
    for om, fields in raw_cases.items():
        vox[om] = voxelize(cc_global, fields, G, bmin, bmax)
    print(f"  done ({time.time()-t0:.1f}s)")

    # 4 — Normalization (same physics-informed scaling as MGN)
    print("\n--- Normalization ---")
    om_max = float(max(OMEGAS))
    us, ps, ks, nus, lom = [], [], [], [], []
    for om in TRAIN_OM:
        v = vox[om]
        us.append(v["U"].reshape(-1, 3) / om)
        ps.append(v["p"].ravel() / om ** 2)
        ks.append(v["k"].ravel() / om ** 2)
        nus.append(v["nut"].ravel())
        lom.append(np.log(v["omega_turb"].ravel() + 1e-6))
    U_cat = np.concatenate(us)
    p_cat = np.concatenate(ps)
    k_cat = np.concatenate(ks)
    nu_cat = np.concatenate(nus)
    lom_cat = np.concatenate(lom)

    # Build grid cell centers in physical coordinates (saved for export + Kit)
    axes = [np.linspace(bmin[i], bmax[i], G, dtype=np.float32) for i in range(3)]
    pgx, pgy, pgz = np.meshgrid(*axes, indexing="ij")
    grid_cc = np.column_stack([pgx.ravel(), pgy.ravel(), pgz.ravel()])  # (G^3, 3)

    stats = {
        "arch": "fno",
        "grid_res": G,
        "bmin": bmin.tolist(),
        "bmax": bmax.tolist(),
        "omega_max": om_max,
        "U_mean": U_cat.mean(0).tolist(),
        "U_std": U_cat.std(0).tolist(),
        "p_mean": float(p_cat.mean()), "p_std": float(p_cat.std()),
        "k_mean": float(k_cat.mean()), "k_std": float(k_cat.std()),
        "nut_mean": float(nu_cat.mean()), "nut_std": float(nu_cat.std()),
        "logom_mean": float(lom_cat.mean()), "logom_std": float(lom_cat.std()),
        "fno_latent": int(cfg["fno_latent"]),
        "fno_layers": int(cfg["fno_layers"]),
        "fno_modes": int(cfg["fno_modes"]),
    }
    (OUT / "norm_stats.json").write_text(json.dumps(stats, indent=2))
    np.save(OUT / "cell_centers.npy", grid_cc)
    del us, ps, ks, nus, lom, U_cat, p_cat, k_cat, nu_cat, lom_cat

    # 5 — Pre-build input / target tensors (one per omega, each is (1, 4/7, G, G, G))
    X_tr = {
        om: torch.from_numpy(fno_input(G, om, om_max)[None]).to(DEVICE)
        for om in TRAIN_OM
    }
    y_tr = {
        om: torch.from_numpy(fno_target(vox[om], om, stats)[None]).to(DEVICE)
        for om in TRAIN_OM
    }
    X_va = torch.from_numpy(fno_input(G, VAL_OM[0], om_max)[None]).to(DEVICE)
    y_va = torch.from_numpy(fno_target(vox[VAL_OM[0]], VAL_OM[0], stats)[None]).to(DEVICE)

    # 6 — FNO model
    print("\n--- Initialize FNO ---")
    model = FNO(
        in_channels=4,
        out_channels=7,
        decoder_layers=1,
        decoder_layer_size=int(cfg["fno_latent"]),
        dimension=3,
        latent_channels=int(cfg["fno_latent"]),
        num_fno_layers=int(cfg["fno_layers"]),
        num_fno_modes=int(cfg["fno_modes"]),
        padding=9,
    ).to(DEVICE)
    load_checkpoint(model, cfg["resume"], DEVICE)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 7 — Train
    print("\n--- Training ---")
    EPOCHS = int(cfg["epochs"])
    SAVE_EVERY = int(cfg["save_every"])
    LR, LR_MIN = float(cfg["lr"]), float(cfg["lr_min"])

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)

    PI_WEIGHT = float(cfg["pi_weight"])
    pi_enabled = PI_WEIGHT > 0.0
    if pi_enabled:
        U_std_t = torch.tensor(stats["U_std"], dtype=torch.float32, device=DEVICE)
        print(f"FD continuity loss enabled (weight={PI_WEIGHT}, dx={dx:.4f} dy={dy:.4f} dz={dz:.4f})")

    def save_ckpt(tag: str = "checkpoint") -> None:
        torch.save(model.state_dict(), OUT / f"{tag}.pt")
        (OUT / "training_log.json").write_text(json.dumps(log, indent=2))

    log: list = []
    best_val = float("inf")
    t0 = time.time()
    try:
        for ep in range(EPOCHS):
            model.train()
            om = TRAIN_OM[ep % len(TRAIN_OM)]
            pred = model(X_tr[om])
            data_loss = ((pred - y_tr[om]) ** 2).mean()
            if pi_enabled:
                pi_loss = continuity_residual_fd(pred, dx, dy, dz, U_std_t)
                loss = data_loss + PI_WEIGHT * pi_loss
                pl_val = pi_loss.item()
            else:
                loss = data_loss
                pl_val = 0.0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
            tl = data_loss.item()

            model.eval()
            with torch.no_grad():
                vp = model(X_va)
                vlm = ((vp - y_va) ** 2).mean().item()
            log.append({"epoch": ep, "train": tl, "val": vlm, "pi": pl_val,
                         "lr": sch.get_last_lr()[0], "t": time.time() - t0})
            if ep % 10 == 0 or ep == EPOCHS - 1:
                pi_str = f"  pi={pl_val:.4e}" if pi_enabled else ""
                print(f"ep {ep:3d}  train={tl:.4e}  val={vlm:.4e}{pi_str}  "
                      f"t={time.time()-t0:.0f}s")
            if vlm < best_val:
                best_val = vlm
                save_ckpt("checkpoint_best")
            if ep % SAVE_EVERY == 0 or ep == EPOCHS - 1:
                save_ckpt("checkpoint")
    except KeyboardInterrupt:
        print("Training stopped manually.")
    finally:
        save_ckpt("checkpoint")
        print(f"final save. best val loss = {best_val:.4e}")

    # 8 — Inference sweep
    print("\n--- Inference ---")
    model.eval()
    frames: dict = {}
    with torch.no_grad():
        for om in INFER_OM:
            x = torch.from_numpy(fno_input(G, om, om_max)[None]).to(DEVICE)
            y = model(x)[0].cpu().numpy()  # (7, G, G, G)
            frames[f"{om:.1f}"] = fno_denorm(y, om, stats)

    packed: dict = {}
    for k, d in frames.items():
        for f, a in d.items():
            packed[f"{k}/{f}"] = a.astype(np.float32)
    np.savez_compressed(OUT / "predictions.npz", **packed)
    print(f"{len(frames)} frames; predictions.npz = "
          f"{(OUT/'predictions.npz').stat().st_size / 1e6:.1f} MB")

    if cfg["make_zip"]:
        zp = OUT.with_suffix(".zip")
        shutil.make_archive(str(zp.with_suffix("")), "zip", str(OUT))
        print(f"Saved outputs to {zp}  ({zp.stat().st_size / 1e6:.1f} MB)")
    else:
        print("Skipped zip (--no-zip).")
    print("Done! Run: python 03_export_model.py --arch fno --out-dir out_fno")


if __name__ == "__main__":
    main()
