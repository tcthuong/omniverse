import numpy as np
import pyvista as pv
from pathlib import Path
from pxr import Usd, UsdGeom, Sdf, Vt
import zipfile

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def make_inlet_seed(bounds, mean_u, resolution=30, inset=0.02):
    bmin = np.asarray(bounds[::2], dtype=float)
    bmax = np.asarray(bounds[1::2], dtype=float)
    span = bmax - bmin
    axis = int(np.argmax(np.abs(mean_u)))
    sign = 1.0 if mean_u[axis] >= 0 else -1.0
    center = 0.5 * (bmin + bmax)
    center[axis] = bmin[axis] + inset * span[axis] if sign > 0 else bmax[axis] - inset * span[axis]

    direction = np.zeros(3)
    direction[axis] = 1.0
    plane_axes = [i for i in range(3) if i != axis]
    return pv.Plane(
        center=tuple(center),
        direction=tuple(direction),
        i_size=span[plane_axes[0]] * 0.9,
        j_size=span[plane_axes[1]] * 0.9,
        i_resolution=resolution,
        j_resolution=resolution,
    )


def make_center_seed(bounds, resolution=30):
    bmin = np.asarray(bounds[::2], dtype=float)
    bmax = np.asarray(bounds[1::2], dtype=float)
    return pv.Plane(
        center=tuple(0.5 * (bmin + bmax)),
        direction=(0, 1, 0),
        i_size=(bmax[0] - bmin[0]) * 0.9,
        j_size=(bmax[2] - bmin[2]) * 0.9,
        i_resolution=resolution,
        j_resolution=resolution,
    )


def bounds_from_minmax(bmin, bmax):
    return (
        bmin[0], bmax[0],
        bmin[1], bmax[1],
        bmin[2], bmax[2],
    )


def ordered_streamline_arrays(stream):
    points = stream.points
    vectors = stream["U"]
    ordered_points = []
    ordered_vectors = []
    counts = []

    lines = stream.lines
    i = 0
    while i < len(lines):
        n = int(lines[i])
        ids = lines[i + 1:i + 1 + n]
        counts.append(n)
        ordered_points.append(points[ids])
        ordered_vectors.append(vectors[ids])
        i += n + 1

    if not ordered_points:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0, 3), dtype=np.float32),
        )

    return (
        np.concatenate(ordered_points).astype(np.float32),
        np.asarray(counts, dtype=np.int32),
        np.concatenate(ordered_vectors).astype(np.float32),
    )

def find(name: str) -> Path:
    for zp in HERE.glob("*.zip"):
        tgt = HERE / zp.stem
        if not tgt.exists():
            print(f"Unzipping {zp.name} -> {tgt}")
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(tgt)
    for p in HERE.rglob(name):
        return p
    raise FileNotFoundError(name)

def load():
    cc = np.load(find("cell_centers.npy"))
    preds = np.load(find("predictions.npz"))
    omegas = sorted({float(k.split("/")[0]) for k in preds.files})
    frames = []
    for om in omegas:
        ok = f"{om:.1f}"
        frames.append({
            k.split("/")[1]: preds[k]
            for k in preds.files if k.startswith(ok + "/")
        })
    print(f"Loaded {len(cc):,} points, {len(frames)} frames")
    return cc, omegas, frames

def export_streamlines_usd(cc, omegas, frames, out_path):
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(omegas) - 1)
    stage.SetTimeCodesPerSecond(10)
    UsdGeom.Xform.Define(stage, "/World")

    curves = UsdGeom.BasisCurves.Define(stage, "/World/Streamlines")
    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    
    points_attr = curves.CreatePointsAttr()
    counts_attr = curves.CreateCurveVertexCountsAttr()
    widths_attr = curves.CreateWidthsAttr()
    
    color_pv = UsdGeom.PrimvarsAPI(curves).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, interpolation=UsdGeom.Tokens.vertex
    )

    bmin, bmax = cc.min(0), cc.max(0)
    
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("turbo")
    except:
        # Fallback if no matplotlib
        cmap = lambda x: np.stack([x, 1-x, 1-x, np.ones_like(x)], axis=-1)
    
    # Calculate global max velocity to keep colors consistent across frames
    global_vmax = 0
    for f in frames:
        m = np.max(np.linalg.norm(f["U"], axis=1))
        if m > global_vmax:
            global_vmax = m

    for t, (om, f) in enumerate(zip(omegas, frames)):
        print(f"Processing frame {t}/{len(omegas)-1} (omega={om})...")
        cloud = pv.PolyData(cc)
        cloud["U"] = f["U"]
        
        grid = pv.ImageData(
            dimensions=(120, 120, 120),
            origin=tuple(bmin),
            spacing=tuple((bmax - bmin) / 119),
        )
        vol = grid.interpolate(cloud, radius=0.08, sharpness=3.0)
        seed = make_center_seed(bounds_from_minmax(bmin, bmax))
        stream = vol.streamlines_from_source(seed, vectors="U",
                                             integration_direction="forward", max_steps=2500)
        
        if stream.n_points == 0:
            print("  Warning: no points in stream")
            points_attr.Set(Vt.Vec3fArray(), time=t)
            counts_attr.Set(Vt.IntArray(), time=t)
            widths_attr.Set(Vt.FloatArray(), time=t)
            color_pv.Set(Vt.Vec3fArray(), time=t)
            continue
            
        pts, counts, vectors = ordered_streamline_arrays(stream)
        
        points_attr.Set(Vt.Vec3fArray.FromNumpy(pts), time=t)
        counts_attr.Set(Vt.IntArray.FromNumpy(counts), time=t)
        
        widths = np.full(len(pts), 0.003, dtype=np.float32)
        widths_attr.Set(Vt.FloatArray.FromNumpy(widths), time=t)
        
        u_mag = np.linalg.norm(vectors, axis=1)
        norm_u = np.clip(u_mag / global_vmax, 0, 1)
        colors = cmap(norm_u)[:, :3].astype(np.float32)
        color_pv.Set(Vt.Vec3fArray.FromNumpy(colors), time=t)

    stage.GetRootLayer().Save()
    print(f"Saved USD: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

if __name__ == '__main__':
    cc, omegas, frames = load()
    out_file = OUT / "streamlines_anim.usda"
    if out_file.exists():
        out_file.unlink()
    export_streamlines_usd(cc, omegas, frames, out_file)
