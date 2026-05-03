"""Local viz + USD export from colab_outputs.

Usage:
    # Unzip colab_outputs.zip anywhere under d:/Work/
    python viz.py

Outputs:
    out/frame_XX_omegaXXX.png   # PyVista streamlines per frame
    out/streamlines_anim.usda   # animated USD streamlines for Omniverse
"""
from pathlib import Path

import numpy as np
import pyvista as pv

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


PRED_DIR = HERE / "out_outputs"


def load():
    cc = np.load(PRED_DIR / "cell_centers.npy")
    preds = np.load(PRED_DIR / "predictions.npz")
    omegas = sorted({float(k.split("/")[0]) for k in preds.files})
    frames = []
    for om in omegas:
        ok = f"{om:.1f}"
        frames.append({
            k.split("/")[1]: preds[k]
            for k in preds.files if k.startswith(ok + "/")
        })
    print(f"{len(cc):,} points, {len(frames)} frames  omega={omegas[0]}..{omegas[-1]}")
    return cc, omegas, frames


def render_png(cc: np.ndarray, frame: dict, omega: float, out: Path) -> None:
    cloud = pv.PolyData(cc)
    cloud["U"] = frame["U"]
    bmin, bmax = cc.min(0), cc.max(0)
    grid = pv.ImageData(
        dimensions=(120, 120, 120),
        origin=tuple(bmin),
        spacing=tuple((bmax - bmin) / 119),
    )
    vol = grid.interpolate(cloud, radius=0.08, sharpness=3.0)
    seed = make_center_seed(bounds_from_minmax(bmin, bmax), resolution=30)
    stream = vol.streamlines_from_source(seed, vectors="U",
                                         integration_direction="forward", max_steps=2500)
    p = pv.Plotter(off_screen=True)
    if stream.n_points > 0:
        stream["Umag"] = np.linalg.norm(stream["U"], axis=1)
        vmax = float(np.linalg.norm(frame["U"], axis=1).max())
        p.add_mesh(stream.tube(radius=0.003), scalars="Umag", cmap="turbo", clim=(0.0, vmax))
    p.add_mesh(cloud.outline(), color="gray")
    p.camera_position = "iso"
    p.add_text(f"omega = {omega:.1f} rad/s (predicted)", font_size=10)
    p.screenshot(str(out), window_size=(1280, 800))
    p.close()


def export_usd(cc: np.ndarray, omegas: list, frames: list, out: Path) -> None:
    try:
        import importlib
        mod = importlib.import_module("04_export_streamlines")
        export_streamlines_usd = mod.export_streamlines_usd
    except ImportError:
        print("USD export dependencies missing - skipping. `pip install usd-core` to enable.")
        return

    export_streamlines_usd(cc, omegas, frames, out)
    return

    try:
        from pxr import Usd, UsdGeom, Sdf, Vt
    except ImportError:
        print("pxr not installed — skipping USD. `pip install usd-core` to enable.")
        return

    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(omegas) - 1)
    stage.SetTimeCodesPerSecond(10)
    UsdGeom.Xform.Define(stage, "/World")

    pts = UsdGeom.Points.Define(stage, "/World/FlowPoints")
    pts.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(cc.astype(np.float32)))
    pts.CreateWidthsAttr(Vt.FloatArray.FromNumpy(np.full(cc.shape[0], 0.003, np.float32)))

    pv_api = UsdGeom.PrimvarsAPI(pts)
    umag_pv = pv_api.CreatePrimvar("Umag", Sdf.ValueTypeNames.FloatArray,
                                   interpolation=UsdGeom.Tokens.vertex)
    p_pv = pv_api.CreatePrimvar("p", Sdf.ValueTypeNames.FloatArray,
                                interpolation=UsdGeom.Tokens.vertex)
    for t, (om, f) in enumerate(zip(omegas, frames)):
        umag = np.linalg.norm(f["U"], axis=1).astype(np.float32)
        umag_pv.Set(Vt.FloatArray.FromNumpy(umag), time=t)
        p_pv.Set(Vt.FloatArray.FromNumpy(f["p"].astype(np.float32)), time=t)

    stage.GetRootLayer().Save()
    print(f"USD saved: {out}  ({out.stat().st_size/1e6:.1f} MB)")


def main() -> None:
    cc, omegas, frames = load()

    print("\nRendering PNGs...")
    for i, (om, f) in enumerate(zip(omegas, frames)):
        out = OUT / f"frame_{i:02d}_omega{om:.0f}.png"
        render_png(cc, f, om, out)
        print(f"  {out.name}")

    print("\nExporting USD...")
    export_usd(cc, omegas, frames, OUT / "streamlines_anim.usda")


if __name__ == "__main__":
    main()
