"""Render GROUND TRUTH streamlines from OpenFOAM with same viz settings.
Compare against out/frame_*.png to tell if issue is viz or model."""
from pathlib import Path
import numpy as np
import pyvista as pv

HERE = Path(__file__).parent
OUT = HERE / "out_truth"
OUT.mkdir(exist_ok=True)

CASES = {150: 'XSNNN-Incompressible-150RAD-SOLUTION_FIELDS',
         250: 'XSNNN-Incompressible-250RAD-SOLUTION_FIELDS',
         350: 'XSNNN-Incompressible-350RAD-SOLUTION_FIELDS'}

for omega, name in CASES.items():
    path = HERE / "input" / name
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
    out = OUT / f'truth_omega{omega}.png'
    p.screenshot(str(out), window_size=(1280, 800))
    p.close()
    print(f'  {out.name}  |U|max(full)={np.linalg.norm(mesh.point_data["U"], axis=1).max():.2f}')

print(f'\nCompare: out_truth/  vs  out/')
