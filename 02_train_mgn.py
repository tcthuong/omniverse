import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import pyvista as pv
from scipy.spatial import cKDTree
import physicsnemo
from physicsnemo.models.meshgraphnet import MeshGraphNet
from torch_geometric.data import Data

from cfd_cases import discover_cases, resolve_input_dir

DEFAULT_CONFIG = {
    "input": "input",
    "out": "out_outputs",
    "subsample": 300_000,
    "k_neighbors": 8,
    "epochs": 3000,
    "save_every": 20,
    "lr": 1e-3,
    "lr_min": 1e-5,
    "seed": 0,
    "infer_frames": 21,
    "validation_omega": None,
    "device": "auto",
    "processor_size": 8,
    "hidden_dim": 64,
    "checkpoint_segments": 4,
    "resume": None,
    "make_zip": True,
    "pi_weight": 0.0,
    "pi_ridge": 1e-6,
}


def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="Train a PhysicsNeMo MeshGraphNet surrogate from OpenFOAM cases."
    )
    parser.add_argument("--config", type=Path, help="Optional JSON config file.")
    parser.add_argument(
        "--write-default-config",
        type=Path,
        help="Write the default JSON config to this path and exit.",
    )
    parser.add_argument("--input", dest="input", type=str, help="OpenFOAM input directory.")
    parser.add_argument("--out", dest="out", type=str, help="Training output directory.")
    parser.add_argument("--subsample", type=int, help="Maximum number of cells to sample.")
    parser.add_argument("--k-neighbors", type=int, help="KNN graph neighbors per node.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--save-every", type=int, help="Checkpoint cadence in epochs.")
    parser.add_argument("--lr", type=float, help="Initial learning rate.")
    parser.add_argument("--lr-min", type=float, help="Minimum cosine schedule LR.")
    parser.add_argument("--seed", type=int, help="Sampling seed.")
    parser.add_argument("--infer-frames", type=int, help="Number of inference sweep frames.")
    parser.add_argument("--validation-omega", type=float, help="Omega value held out for validation.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        help="Training device. 'auto' uses CUDA when available.",
    )
    parser.add_argument("--processor-size", type=int, help="MeshGraphNet processor blocks.")
    parser.add_argument("--hidden-dim", type=int, help="Hidden dimension for MGN MLPs.")
    parser.add_argument("--checkpoint-segments", type=int, help="Processor checkpoint segments.")
    parser.add_argument("--resume", type=str, help="Checkpoint .pt to load before training.")
    parser.add_argument("--no-zip", action="store_true", help="Skip final output zip archive.")
    parser.add_argument(
        "--pi-weight",
        type=float,
        help="Weight for physics-informed continuity loss (incompressible div(U)=0). 0 disables.",
    )
    parser.add_argument(
        "--pi-ridge",
        type=float,
        help="Tikhonov ridge added to MLS normal equations for stability.",
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

    overrides = vars(args)
    for key in DEFAULT_CONFIG:
        if key == "make_zip":
            continue
        value = overrides.get(key)
        if value is not None:
            cfg[key] = value
    if args.no_zip:
        cfg["make_zip"] = False

    return cfg


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return requested


def load_model_checkpoint(model: nn.Module, path: str | None, device: str) -> None:
    if not path:
        return
    ckpt_path = Path(path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        checkpoint = checkpoint["model_state"]
    model.load_state_dict(checkpoint)
    print(f"Resumed model weights from {ckpt_path.resolve()}")

def extract(path: Path):
    r = pv.POpenFOAMReader(str(path / 'case.foam'))
    r.set_active_time_value(r.time_values[-1])
    r.enable_all_cell_arrays(); r.cell_to_point_creation = False
    m = r.read()['internalMesh']
    cc = np.asarray(m.cell_centers().points, dtype=np.float32)
    f = {k: np.asarray(m.cell_data[k], dtype=np.float32) for k in ['U','p','k','nut','omega']}
    f['omega_turb'] = f.pop('omega')
    return cc, f

def main():
    cfg = parse_args()
    print('torch', torch.__version__, '  cuda', torch.cuda.is_available(),
          '  gpu', torch.cuda.get_device_name() if torch.cuda.is_available() else '-')
    print('physicsnemo', physicsnemo.__version__)
    print("config:")
    print(json.dumps(cfg, indent=2))

    # Local paths
    INPUT = resolve_input_dir(cfg["input"], base_dir=Path(__file__).parent)
    print('INPUT =', INPUT.resolve())
    
    if int(cfg["infer_frames"]) < 2:
        raise ValueError("infer_frames must be at least 2")
    OUT = Path(cfg["out"])
    OUT.mkdir(exist_ok=True)

    # 1 - Discover + extract every case in input/
    case_records = discover_cases(INPUT)
    print(f'Found {len(case_records)} case(s):')

    cc = None; cases = {}
    for case in case_records:
        cd = case.path
        om = case.omega
        t0 = time.time()
        ccx, fields = extract(cd)
        if cc is None: cc = ccx
        cases[om] = fields
        umax = float(np.linalg.norm(fields['U'], axis=1).max())
        print(f'  {cd.name}  omega={om}  |U|max={umax:.2f}  cells={ccx.shape[0]:,}  t={time.time()-t0:.1f}s')

    OMEGAS = sorted(cases.keys())
    print(f'Omegas: {OMEGAS}')

    # 2 - Train / val split (middle omega = validation, rest = training)
    if len(OMEGAS) < 3:
        raise ValueError('need at least 3 cases for interpolation-style LOO split')
    if cfg["validation_omega"] is None:
        VAL_OM = [OMEGAS[len(OMEGAS)//2]]
    else:
        val_om = float(cfg["validation_omega"])
        if val_om not in OMEGAS:
            raise ValueError(f"validation_omega={val_om} is not in available omegas {OMEGAS}")
        VAL_OM = [val_om]
    TRAIN_OM = [om for om in OMEGAS if om not in VAL_OM]
    INFER_OM = np.linspace(min(OMEGAS), max(OMEGAS), int(cfg["infer_frames"])).tolist()
    print(f'train: {TRAIN_OM}')
    print(f'val  : {VAL_OM}')
    print(f'infer: {INFER_OM[0]:.0f} -> {INFER_OM[-1]:.0f} ({len(INFER_OM)} frames)')

    # 3 - Importance-weighted subsample + KNN graph
    print("\n--- Subsampling & KNN ---")
    SUBSAMPLE = int(cfg["subsample"])
    K_NEIGHBORS = int(cfg["k_neighbors"])

    mag = np.zeros(cc.shape[0], np.float32)
    for om in TRAIN_OM:
        mag = np.maximum(mag, np.linalg.norm(cases[om]['U'], axis=1) / om)
    w = 0.3/len(mag) + 0.7*mag/mag.sum()
    w = w/w.sum()
    rng = np.random.default_rng(int(cfg["seed"]))
    N = cc.shape[0]
    sub_idx = np.sort(rng.choice(N, size=min(SUBSAMPLE, N), replace=False, p=w))
    cc_s = cc[sub_idx]
    cases_s = {om: {k: v[sub_idx] for k, v in f.items()} for om, f in cases.items()}
    print(f'Subsampled {len(sub_idx):,} / {N:,}')

    t0 = time.time()
    tree = cKDTree(cc_s)
    _, nbrs = tree.query(cc_s, k=K_NEIGHBORS+1)
    src = np.repeat(np.arange(len(cc_s)), K_NEIGHBORS)
    dst = nbrs[:, 1:].reshape(-1)
    rel = cc_s[dst] - cc_s[src]
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    edge_feat = np.concatenate([rel, dist], axis=1).astype(np.float32)
    edge_index_np = np.stack([src, dst], axis=0).astype(np.int64)
    print(f'KNN graph: {len(src):,} edges  ({time.time()-t0:.1f}s)')

    # MLS gradient operator (physical units): per-node weights W (3, K) such that
    # grad(f) at node n approximates W_n @ (f[neighbors] - f[n]).
    PI_WEIGHT = float(cfg["pi_weight"])
    PI_RIDGE = float(cfg["pi_ridge"])
    rel_per_node = rel.reshape(len(cc_s), K_NEIGHBORS, 3).astype(np.float32)
    RtR = np.einsum('nki,nkj->nij', rel_per_node, rel_per_node)
    RtR += PI_RIDGE * np.eye(3, dtype=np.float32)[None, :, :]
    RtR_inv = np.linalg.inv(RtR).astype(np.float32)
    mls_W = np.einsum('nij,nkj->nik', RtR_inv, rel_per_node).astype(np.float32)  # (N, 3, K)
    neighbor_idx_np = nbrs[:, 1:].astype(np.int64)

    # 4 - Physics-informed normalization
    print("\n--- Normalization ---")
    DEVICE = choose_device(str(cfg["device"]))

    b_min, b_max = cc_s.min(0), cc_s.max(0)
    b_cen, b_half = 0.5*(b_min+b_max), 0.5*(b_max-b_min)
    om_max = float(max(OMEGAS))

    us, ps, ks, nus, lom = [], [], [], [], []
    for om in TRAIN_OM:
        c = cases_s[om]
        us.append(c['U']/om); ps.append(c['p']/om**2); ks.append(c['k']/om**2)
        nus.append(c['nut']); lom.append(np.log(c['omega_turb']+1e-6))
    U_cat = np.concatenate(us); p_cat = np.concatenate(ps); k_cat = np.concatenate(ks)
    nu_cat = np.concatenate(nus); lom_cat = np.concatenate(lom)

    stats = {
        'arch': 'mgn',
        'coord_center': b_cen.tolist(), 'coord_half': b_half.tolist(), 'omega_max': om_max,
        'U_mean': U_cat.mean(0).tolist(), 'U_std': U_cat.std(0).tolist(),
        'p_mean': float(p_cat.mean()), 'p_std': float(p_cat.std()),
        'k_mean': float(k_cat.mean()), 'k_std': float(k_cat.std()),
        'nut_mean': float(nu_cat.mean()), 'nut_std': float(nu_cat.std()),
        'logom_mean': float(lom_cat.mean()), 'logom_std': float(lom_cat.std()),
        'edge_mean': edge_feat.mean(0).tolist(), 'edge_std': (edge_feat.std(0)+1e-8).tolist(),
    }
    (OUT/'norm_stats.json').write_text(json.dumps(stats, indent=2))
    del us, ps, ks, nus, lom, U_cat, p_cat, k_cat, nu_cat, lom_cat
    
    def node_feat(cc_s, om, s):
        xyz = (cc_s - np.asarray(s['coord_center'],np.float32)) / np.asarray(s['coord_half'],np.float32)
        o = np.full((cc_s.shape[0],1), om/s['omega_max'], np.float32)
        return np.concatenate([xyz.astype(np.float32), o], axis=1)

    def targets(f, om, s):
        U = (f['U']/om - np.asarray(s['U_mean'])) / np.asarray(s['U_std'])
        p = (f['p']/om**2 - s['p_mean']) / s['p_std']
        k = (f['k']/om**2 - s['k_mean']) / s['k_std']
        nu = (f['nut'] - s['nut_mean']) / s['nut_std']
        lom = (np.log(f['omega_turb']+1e-6) - s['logom_mean']) / s['logom_std']
        return np.stack([U[:,0],U[:,1],U[:,2],p,k,nu,lom], axis=1).astype(np.float32)

    def denorm(y, om, s):
        U = (y[:,:3]*np.asarray(s['U_std']) + np.asarray(s['U_mean'])) * om
        return {
            'U': U,
            'p': (y[:,3]*s['p_std'] + s['p_mean']) * om**2,
            'k': (y[:,4]*s['k_std'] + s['k_mean']) * om**2,
            'nut': y[:,5]*s['nut_std'] + s['nut_mean'],
            'omega_turb': np.exp(y[:,6]*s['logom_std'] + s['logom_mean']) - 1e-6,
        }

    e_mean = np.asarray(stats['edge_mean'], np.float32)
    e_std = np.asarray(stats['edge_std'], np.float32)
    edge_n = ((edge_feat - e_mean) / e_std).astype(np.float32)
    edge_tensor = torch.from_numpy(edge_n).to(DEVICE)

    # PhysicsNeMo MGN expects a PyG Data object (with edge_index), not a raw tensor.
    graph = Data(edge_index=torch.from_numpy(edge_index_np).to(DEVICE),
                 num_nodes=len(cc_s)).to(DEVICE)
    print('graph:', graph)

    # 5 - PhysicsNeMo MeshGraphNet
    print("\n--- Initialize Model ---")
    model = MeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=4,
        output_dim=7,
        processor_size=int(cfg["processor_size"]),
        hidden_dim_node_encoder=int(cfg["hidden_dim"]),
        hidden_dim_edge_encoder=int(cfg["hidden_dim"]),
        hidden_dim_processor=int(cfg["hidden_dim"]),
        hidden_dim_node_decoder=int(cfg["hidden_dim"]),
        mlp_activation_fn='silu',
        num_processor_checkpoint_segments=int(cfg["checkpoint_segments"]),
        recompute_activation=False,
    ).to(DEVICE)
    load_model_checkpoint(model, cfg["resume"], DEVICE)
    print(f'params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    # 6 - Train
    print("\n--- Training ---")
    EPOCHS = int(cfg["epochs"])
    SAVE_EVERY = int(cfg["save_every"])
    LR, LR_MIN = float(cfg["lr"]), float(cfg["lr_min"])

    X_tr = {om: torch.from_numpy(node_feat(cc_s, om, stats)).to(DEVICE) for om in TRAIN_OM}
    y_tr = {om: torch.from_numpy(targets(cases_s[om], om, stats)).to(DEVICE) for om in TRAIN_OM}
    X_va = torch.from_numpy(node_feat(cc_s, VAL_OM[0], stats)).to(DEVICE)
    y_va = torch.from_numpy(targets(cases_s[VAL_OM[0]], VAL_OM[0], stats)).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)

    # Physics-informed continuity setup (incompressible: omega * sum_i U_std_i * dU_norm_i/dx_i = 0).
    pi_enabled = PI_WEIGHT > 0.0
    if pi_enabled:
        mls_W_t = torch.from_numpy(mls_W).to(DEVICE)              # (N, 3, K)
        neighbor_idx_t = torch.from_numpy(neighbor_idx_np).to(DEVICE)  # (N, K)
        U_std_t = torch.tensor(stats['U_std'], dtype=torch.float32, device=DEVICE)  # (3,)
        print(f'physics-informed continuity loss enabled (weight={PI_WEIGHT}, ridge={PI_RIDGE})')

    def continuity_residual(pred):
        # pred: (N, 7); pred[:, :3] is U_norm. Returns scalar mean-squared divergence.
        U_norm = pred[:, :3]
        U_neighbors = U_norm[neighbor_idx_t]                    # (N, K, 3)
        delta = U_neighbors - U_norm.unsqueeze(1)               # (N, K, 3)
        # grad[n, j, i] = dU_norm_i / dx_j at node n
        grad = torch.einsum('njk,nki->nji', mls_W_t, delta)     # (N, 3, 3)
        diag = grad.diagonal(dim1=1, dim2=2)                    # (N, 3): dU_i/dx_i
        div = (diag * U_std_t).sum(dim=1)                       # (N,) up to omega scalar
        return (div ** 2).mean()

    def save_ckpt(tag='checkpoint'):
        torch.save(model.state_dict(), OUT / f'{tag}.pt')
        (OUT/'training_log.json').write_text(json.dumps(log, indent=2))

    log = []; t0 = time.time()
    best_val = float('inf')
    try:
        for ep in range(EPOCHS):
            model.train()
            om = TRAIN_OM[ep % len(TRAIN_OM)]
            pred = model(X_tr[om], edge_tensor, graph)
            data_loss = ((pred - y_tr[om])**2).mean()
            if pi_enabled:
                pi_loss = continuity_residual(pred)
                loss = data_loss + PI_WEIGHT * pi_loss
                pl_val = pi_loss.item()
            else:
                loss = data_loss
                pl_val = 0.0
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
            tl = data_loss.item()

            model.eval()
            with torch.no_grad():
                vp = model(X_va, edge_tensor, graph)
                vlm = ((vp - y_va)**2).mean().item()
            log.append({'epoch': ep, 'train': tl, 'val': vlm, 'pi': pl_val,
                        'lr': sch.get_last_lr()[0], 't': time.time()-t0})
            if ep % 10 == 0 or ep == EPOCHS-1:
                pi_str = f'  pi={pl_val:.4e}' if pi_enabled else ''
                print(f'ep {ep:3d}  train={tl:.4e}  val={vlm:.4e}{pi_str}  t={time.time()-t0:.0f}s')
            if vlm < best_val:
                best_val = vlm
                save_ckpt('checkpoint_best')
            if ep % SAVE_EVERY == 0 or ep == EPOCHS-1:
                save_ckpt('checkpoint')
    except KeyboardInterrupt:
        print("Training stopped manually.")
    finally:
        save_ckpt('checkpoint')
        print(f'final save. best val loss = {best_val:.4e}')
        print('files in OUT:')
        for f in sorted(OUT.iterdir()):
            try:
                print(f'  {f.name}  ({f.stat().st_size/1e6:.2f} MB)')
            except Exception:
                pass

    # 7 - Inference sweep + download equivalent
    print("\n--- Inference ---")
    model.eval(); frames = {}
    with torch.no_grad():
        for om in INFER_OM:
            X = torch.from_numpy(node_feat(cc_s, om, stats)).to(DEVICE)
            y = model(X, edge_tensor, graph).cpu().numpy()
            frames[f'{om:.1f}'] = denorm(y, om, stats)

    packed = {}
    for k, d in frames.items():
        for f, a in d.items(): packed[f'{k}/{f}'] = a.astype(np.float32)
    np.savez_compressed(OUT/'predictions.npz', **packed)
    np.save(OUT/'cell_centers.npy', cc_s)
    np.save(OUT/'sub_idx.npy', sub_idx)
    print(f'{len(frames)} frames; predictions.npz = {(OUT/"predictions.npz").stat().st_size/1e6:.1f} MB')

    if cfg["make_zip"]:
        zp = OUT.with_suffix('.zip')
        shutil.make_archive(str(zp.with_suffix('')), 'zip', str(OUT))
        print(f'Saved all output files to {zp}  ({zp.stat().st_size/1e6:.1f} MB)')
    else:
        print('Skipped zip archive (--no-zip).')
    print("Done! You can now use the outputs locally.")
if __name__ == '__main__':
    main()
