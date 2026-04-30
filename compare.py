"""Render GROUND TRUTH streamlines from OpenFOAM with same viz settings.
Compare against out/frame_*.png to tell if issue is viz or model."""
import argparse
from pathlib import Path
import numpy as np
import pyvista as pv

from cfd_cases import discover_cases, resolve_input_dir

HERE = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("input"), help="OpenFOAM input directory.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("out_truth"),
        help="Directory for rendered truth PNG files.",
    )
    return parser.parse_args()


def render_case(path: Path, omega: float, out_dir: Path) -> None:
    reader = pv.POpenFOAMReader(str(path / 'case.foam'))
    reader.set_active_time_value(reader.time_values[-1])
    reader.enable_all_cell_arrays()
    reader.cell_to_point_creation = True
    mesh = reader.read()['internalMesh']

    bmin, bmax = mesh.bounds[::2], mesh.bounds[1::2]
    bmin = np.asarray(bmin); bmax = np.asarray(bmax)
    seed = pv.Plane(
        center=tuple(0.5 * (bmin + bmax)),
        direction=(0, 1, 0),
        i_size=(bmax[0] - bmin[0]) * 0.9,
        j_size=(bmax[2] - bmin[2]) * 0.9,
        i_resolution=15, j_resolution=15,
    )
    stream = mesh.streamlines_from_source(seed, vectors='U',
                                          integration_direction='forward', max_steps=1500)
    p = pv.Plotter(off_screen=True)
    if stream.n_points > 0:
        p.add_mesh(stream.tube(radius=0.003), scalars='U', cmap='turbo')
    p.add_mesh(mesh.outline(), color='gray')
    p.camera_position = 'iso'
    p.add_text(f'omega = {omega} rad/s (TRUTH - full mesh)', font_size=10)
    out = out_dir / f'truth_omega{omega:g}.png'
    p.screenshot(str(out), window_size=(1280, 800))
    p.close()
    print(f'  {out.name}  |U|max(full)={np.linalg.norm(mesh.point_data["U"], axis=1).max():.2f}')


def main() -> None:
    args = parse_args()
    input_dir = resolve_input_dir(args.input, base_dir=HERE)
    out_dir = args.out if args.out.is_absolute() else HERE / args.out
    out_dir.mkdir(exist_ok=True)

    cases = discover_cases(input_dir)
    print(f"Found {len(cases)} case(s) in {input_dir}:")
    for case in cases:
        render_case(case.path, case.omega, out_dir)

    print(f'\nCompare: {out_dir}/  vs  {HERE / "out"}/')


if __name__ == "__main__":
    main()
