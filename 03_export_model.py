"""Export a trained MeshGraphNet checkpoint to TorchScript for real-time inference.

The exported module takes a single scalar `omega_rad_s` tensor and returns
a (N, 7) tensor of physical predictions in the channel order
[Ux, Uy, Uz, p, k, nut, omega_turb].

Usage:
    python 03_export_model.py
    python 03_export_model.py --arch fno --out-dir out_fno
    python 03_export_model.py --checkpoint out_outputs/checkpoint_best.pt --out out_outputs/cfd_surrogate.ts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch_geometric.data import Data

from physicsnemo.models.fno import FNO
from physicsnemo.models.meshgraphnet import MeshGraphNet


CHANNEL_ORDER = ["Ux", "Uy", "Uz", "p", "k", "nut", "omega_turb"]


class FNOSurrogate(nn.Module):
    """Wraps FNO with normalized coord-grid baked in.

    Input : scalar omega_rad_s tensor
    Output: (G^3, 7) physical fields in CHANNEL_ORDER
    """

    def __init__(
        self,
        fno: FNO,
        cell_centers: torch.Tensor,
        coord_grid: torch.Tensor,
        omega_max: torch.Tensor,
        u_mean: torch.Tensor,
        u_std: torch.Tensor,
        p_mean: torch.Tensor,
        p_std: torch.Tensor,
        k_mean: torch.Tensor,
        k_std: torch.Tensor,
        nut_mean: torch.Tensor,
        nut_std: torch.Tensor,
        logom_mean: torch.Tensor,
        logom_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.fno = fno
        self.register_buffer("cell_centers", cell_centers)  # (G^3, 3) physical coords
        self.register_buffer("coord_grid", coord_grid)      # (3, G, G, G) normalized [-1,1]
        self.register_buffer("omega_max", omega_max)
        self.register_buffer("u_mean", u_mean)
        self.register_buffer("u_std", u_std)
        self.register_buffer("p_mean", p_mean)
        self.register_buffer("p_std", p_std)
        self.register_buffer("k_mean", k_mean)
        self.register_buffer("k_std", k_std)
        self.register_buffer("nut_mean", nut_mean)
        self.register_buffer("nut_std", nut_std)
        self.register_buffer("logom_mean", logom_mean)
        self.register_buffer("logom_std", logom_std)

    def forward(self, omega_rad_s: torch.Tensor) -> torch.Tensor:
        G = self.coord_grid.shape[1]
        om_ch = (omega_rad_s / self.omega_max).view(1, 1, 1, 1).expand(1, 1, G, G, G)
        x = torch.cat([self.coord_grid.unsqueeze(0), om_ch], dim=1)  # (1, 4, G, G, G)
        y = self.fno(x)  # (1, 7, G, G, G)

        u = (y[0, 0:3] * self.u_std.view(3, 1, 1, 1)
             + self.u_mean.view(3, 1, 1, 1)) * omega_rad_s  # (3, G, G, G)
        p = (y[0, 3] * self.p_std + self.p_mean) * (omega_rad_s ** 2)
        k = (y[0, 4] * self.k_std + self.k_mean) * (omega_rad_s ** 2)
        nut = y[0, 5] * self.nut_std + self.nut_mean
        omt = torch.exp(y[0, 6] * self.logom_std + self.logom_mean) - 1e-6

        u_flat = u.reshape(3, -1).T  # (G^3, 3)
        return torch.cat(
            [u_flat, p.reshape(-1, 1), k.reshape(-1, 1),
             nut.reshape(-1, 1), omt.reshape(-1, 1)],
            dim=1,
        )  # (G^3, 7)


class CFDSurrogate(nn.Module):
    """Wraps MeshGraphNet with normalization + static graph baked in."""

    def __init__(
        self,
        mgn: MeshGraphNet,
        cell_centers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        coord_center: torch.Tensor,
        coord_half: torch.Tensor,
        omega_max: torch.Tensor,
        u_mean: torch.Tensor,
        u_std: torch.Tensor,
        p_mean: torch.Tensor,
        p_std: torch.Tensor,
        k_mean: torch.Tensor,
        k_std: torch.Tensor,
        nut_mean: torch.Tensor,
        nut_std: torch.Tensor,
        logom_mean: torch.Tensor,
        logom_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.mgn = mgn
        self.register_buffer("cell_centers", cell_centers)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_features", edge_features)
        self.register_buffer("coord_center", coord_center)
        self.register_buffer("coord_half", coord_half)
        self.register_buffer("omega_max", omega_max)
        self.register_buffer("u_mean", u_mean)
        self.register_buffer("u_std", u_std)
        self.register_buffer("p_mean", p_mean)
        self.register_buffer("p_std", p_std)
        self.register_buffer("k_mean", k_mean)
        self.register_buffer("k_std", k_std)
        self.register_buffer("nut_mean", nut_mean)
        self.register_buffer("nut_std", nut_std)
        self.register_buffer("logom_mean", logom_mean)
        self.register_buffer("logom_std", logom_std)

    def forward(self, omega_rad_s: torch.Tensor) -> torch.Tensor:
        # omega_rad_s: scalar tensor.
        n = self.cell_centers.shape[0]
        xyz = (self.cell_centers - self.coord_center) / self.coord_half
        om_n = (omega_rad_s / self.omega_max).reshape(1, 1).expand(n, 1)
        x = torch.cat([xyz, om_n], dim=1)

        # MeshGraphNet expects a graph object exposing edge_index. Build a
        # PyG Data object on the fly; both edge_index and num_nodes are static
        # so this is cheap.
        graph = Data(edge_index=self.edge_index, num_nodes=n)

        y = self.mgn(x, self.edge_features, graph)

        u = (y[:, 0:3] * self.u_std + self.u_mean) * omega_rad_s
        p = (y[:, 3] * self.p_std + self.p_mean) * (omega_rad_s ** 2)
        k = (y[:, 4] * self.k_std + self.k_mean) * (omega_rad_s ** 2)
        nut = y[:, 5] * self.nut_std + self.nut_mean
        omega_turb = torch.exp(y[:, 6] * self.logom_std + self.logom_mean) - 1e-6

        return torch.cat(
            [u, p.unsqueeze(1), k.unsqueeze(1), nut.unsqueeze(1), omega_turb.unsqueeze(1)],
            dim=1,
        )


def build_knn_graph(cell_centers: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(cell_centers)
    _, nbrs = tree.query(cell_centers, k=k + 1)
    src = np.repeat(np.arange(len(cell_centers)), k)
    dst = nbrs[:, 1:].reshape(-1)
    rel = cell_centers[dst] - cell_centers[src]
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    edge_feat = np.concatenate([rel, dist], axis=1).astype(np.float32)
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    return edge_index, edge_feat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="out_outputs",
                   help="Training output directory (contains norm_stats.json + checkpoint_best.pt).")
    p.add_argument("--arch", choices=["auto", "mgn", "fno"], default="auto",
                   help="Model architecture. 'auto' reads arch field from norm_stats.json.")
    p.add_argument("--checkpoint", default=None, help="Override checkpoint path.")
    p.add_argument("--out", default=None, help="Output TorchScript .ts path.")
    p.add_argument("--k-neighbors", type=int, default=8, help="Must match MGN training.")
    p.add_argument("--processor-size", type=int, default=8, help="MGN only.")
    p.add_argument("--hidden-dim", type=int, default=64, help="MGN only.")
    p.add_argument("--checkpoint-segments", type=int, default=4, help="MGN only.")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu",
                   help="Device for tracing. cpu is portable; cuda matches deployment.")
    p.add_argument("--test-omega", type=float, default=250.0,
                   help="Sanity inference value (rad/s).")
    return p.parse_args()


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return requested


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else out_dir / "checkpoint_best.pt"
    ts_path = Path(args.out) if args.out else out_dir / "cfd_surrogate.ts"
    stats_path = out_dir / "norm_stats.json"
    cc_path = out_dir / "cell_centers.npy"

    for path in (ckpt_path, stats_path, cc_path):
        if not path.exists():
            raise FileNotFoundError(path)

    device = choose_device(args.device)
    stats = json.loads(stats_path.read_text())
    cc = np.load(cc_path).astype(np.float32)

    arch = stats.get("arch", "mgn") if args.arch == "auto" else args.arch
    print(f"arch={arch}, cell_centers={cc.shape}, device={device}")

    def t(x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=device)

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]

    if arch == "fno":
        G = int(stats["grid_res"])
        fno = FNO(
            in_channels=4,
            out_channels=7,
            decoder_layers=1,
            decoder_layer_size=int(stats.get("fno_latent", 32)),
            dimension=3,
            latent_channels=int(stats.get("fno_latent", 32)),
            num_fno_layers=int(stats.get("fno_layers", 4)),
            num_fno_modes=int(stats.get("fno_modes", 8)),
            padding=9,
        ).to(device)
        fno.load_state_dict(state)
        fno.eval()

        bmin = np.asarray(stats["bmin"], dtype=np.float32)
        bmax = np.asarray(stats["bmax"], dtype=np.float32)
        lin = np.linspace(-1.0, 1.0, G, dtype=np.float32)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        coord_grid = np.stack([gx, gy, gz], axis=0)  # (3, G, G, G)

        surrogate = FNOSurrogate(
            fno=fno,
            cell_centers=t(cc),
            coord_grid=t(coord_grid),
            omega_max=t(stats["omega_max"]),
            u_mean=t(stats["U_mean"]),
            u_std=t(stats["U_std"]),
            p_mean=t(stats["p_mean"]),
            p_std=t(stats["p_std"]),
            k_mean=t(stats["k_mean"]),
            k_std=t(stats["k_std"]),
            nut_mean=t(stats["nut_mean"]),
            nut_std=t(stats["nut_std"]),
            logom_mean=t(stats["logom_mean"]),
            logom_std=t(stats["logom_std"]),
        ).to(device).eval()

    else:  # mgn
        edge_index_np, edge_feat_np = build_knn_graph(cc, args.k_neighbors)
        e_mean = np.asarray(stats["edge_mean"], np.float32)
        e_std = np.asarray(stats["edge_std"], np.float32)
        edge_feat_norm = ((edge_feat_np - e_mean) / e_std).astype(np.float32)
        print(f"edge_index: {edge_index_np.shape}, edge_features: {edge_feat_norm.shape}")

        mgn = MeshGraphNet(
            input_dim_nodes=4,
            input_dim_edges=4,
            output_dim=7,
            processor_size=args.processor_size,
            hidden_dim_node_encoder=args.hidden_dim,
            hidden_dim_edge_encoder=args.hidden_dim,
            hidden_dim_processor=args.hidden_dim,
            hidden_dim_node_decoder=args.hidden_dim,
            mlp_activation_fn="silu",
            num_processor_checkpoint_segments=args.checkpoint_segments,
            recompute_activation=False,
        ).to(device)
        mgn.load_state_dict(state)
        mgn.eval()

        surrogate = CFDSurrogate(
            mgn=mgn,
            cell_centers=t(cc),
            edge_index=t(edge_index_np, dtype=torch.int64),
            edge_features=t(edge_feat_norm),
            coord_center=t(stats["coord_center"]),
            coord_half=t(stats["coord_half"]),
            omega_max=t(stats["omega_max"]),
            u_mean=t(stats["U_mean"]),
            u_std=t(stats["U_std"]),
            p_mean=t(stats["p_mean"]),
            p_std=t(stats["p_std"]),
            k_mean=t(stats["k_mean"]),
            k_std=t(stats["k_std"]),
            nut_mean=t(stats["nut_mean"]),
            nut_std=t(stats["nut_std"]),
            logom_mean=t(stats["logom_mean"]),
            logom_std=t(stats["logom_std"]),
        ).to(device).eval()

    omega_test = torch.tensor(args.test_omega, dtype=torch.float32, device=device)

    print("running eager forward...")
    with torch.no_grad():
        y_eager = surrogate(omega_test)
    print(f"eager output: {tuple(y_eager.shape)}, |U|max={y_eager[:, :3].norm(dim=1).max().item():.3f}")

    print("tracing TorchScript...")
    with torch.no_grad():
        traced = torch.jit.trace(surrogate, (omega_test,), strict=False)
    traced.save(str(ts_path))
    print(f"saved TorchScript: {ts_path}  ({ts_path.stat().st_size/1e6:.1f} MB)")

    # Verify trace matches eager.
    reloaded = torch.jit.load(str(ts_path), map_location=device)
    with torch.no_grad():
        y_traced = reloaded(omega_test)
    diff = (y_traced - y_eager).abs().max().item()
    print(f"trace vs eager max abs diff: {diff:.3e}")

    # Metadata for downstream consumers (Kit extension, etc.).
    meta = {
        "arch": arch,
        "torchscript": ts_path.name,
        "channels": CHANNEL_ORDER,
        "num_nodes": int(cc.shape[0]),
        "k_neighbors": int(args.k_neighbors) if arch == "mgn" else None,
        "input": {"omega_rad_s": "scalar float tensor"},
        "output_shape": list(y_eager.shape),
        "device_traced": device,
        "checkpoint_source": str(ckpt_path),
    }
    meta_path = ts_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
