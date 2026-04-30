# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
"""Real-time CFD inference bridge for the streaming Kit app.

Listens for `cfd.setOmega` messages from the web UI (sent via
AppStreamer.sendMessage), runs the exported TorchScript surrogate, and
updates a Points prim on the USD stage with the predicted velocity field.

Settings (carb):
    /exts/thuong.tc_extension/cfd/enabled         bool, default True
    /exts/thuong.tc_extension/cfd/model_path      str, TorchScript .ts path
    /exts/thuong.tc_extension/cfd/device          "cpu" | "cuda" | "auto"
    /exts/thuong.tc_extension/cfd/stage_root      USD root path, default "/World/CFD"
    /exts/thuong.tc_extension/cfd/point_width     float, point widget size
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import carb
import carb.events
import omni.ext
import omni.kit.app
import omni.usd
from pxr import Sdf, UsdGeom, Vt


EVENT_TYPE_SET_OMEGA = "cfd.setOmega"

DEFAULT_MODEL_PATH = "D:/work/out_outputs/cfd_surrogate.ts"
DEFAULT_STAGE_ROOT = "/World/CFD"
DEFAULT_POINT_WIDTH = 0.005


def _setting(path: str, default):
    s = carb.settings.get_settings()
    val = s.get(path)
    return default if val is None else val


class CFDInferenceBridge:
    """Owns the loaded TorchScript model and a USD Points prim that it updates."""

    def __init__(self) -> None:
        self._torch = None
        self._model = None
        self._device = "cpu"
        self._meta: dict = {}
        self._cell_centers = None
        self._u_max_global = 1.0

        self._stage_root = DEFAULT_STAGE_ROOT
        self._points_path = self._stage_root + "/Predictions"
        self._point_width = DEFAULT_POINT_WIDTH

        self._sub = None
        self._last_omega = None

    def start(self) -> None:
        if not bool(_setting("/exts/thuong.tc_extension/cfd/enabled", True)):
            carb.log_info("[cfd] disabled via settings")
            return

        model_path = str(_setting(
            "/exts/thuong.tc_extension/cfd/model_path", DEFAULT_MODEL_PATH))
        self._stage_root = str(_setting(
            "/exts/thuong.tc_extension/cfd/stage_root", DEFAULT_STAGE_ROOT))
        self._points_path = self._stage_root + "/Predictions"
        self._point_width = float(_setting(
            "/exts/thuong.tc_extension/cfd/point_width", DEFAULT_POINT_WIDTH))
        device_pref = str(_setting(
            "/exts/thuong.tc_extension/cfd/device", "auto"))

        if not self._load_model(model_path, device_pref):
            return

        self._ensure_stage_prims()
        self._subscribe()
        carb.log_info(
            f"[cfd] ready. model={model_path} device={self._device} "
            f"nodes={self._cell_centers.shape[0]}"
        )

    def stop(self) -> None:
        self._sub = None
        self._model = None
        self._cell_centers = None
        self._torch = None

    def _load_model(self, model_path: str, device_pref: str) -> bool:
        try:
            import torch
        except Exception as exc:
            carb.log_error(f"[cfd] torch not importable inside Kit: {exc}")
            return False
        self._torch = torch

        path = Path(model_path)
        if not path.exists():
            carb.log_error(f"[cfd] TorchScript model not found: {path}")
            return False

        if device_pref == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device_pref == "cuda" and not torch.cuda.is_available():
            carb.log_warn("[cfd] cuda requested but unavailable, falling back to cpu")
            self._device = "cpu"
        else:
            self._device = device_pref

        try:
            self._model = torch.jit.load(str(path), map_location=self._device).eval()
        except Exception as exc:
            carb.log_error(f"[cfd] failed to load TorchScript: {exc}")
            return False

        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            try:
                self._meta = json.loads(meta_path.read_text())
            except Exception as exc:
                carb.log_warn(f"[cfd] cannot parse metadata {meta_path}: {exc}")

        try:
            self._cell_centers = getattr(self._model, "cell_centers")
        except Exception as exc:
            carb.log_error(f"[cfd] model missing cell_centers buffer: {exc}")
            return False
        return True

    def _ensure_stage_prims(self) -> None:
        usd_ctx = omni.usd.get_context()
        stage = usd_ctx.get_stage()
        if stage is None:
            carb.log_warn("[cfd] no active USD stage; will retry on first message")
            return

        if not stage.GetPrimAtPath(self._stage_root):
            UsdGeom.Xform.Define(stage, self._stage_root)

        points_prim = UsdGeom.Points.Get(stage, self._points_path)
        if not points_prim:
            points_prim = UsdGeom.Points.Define(stage, self._points_path)

        cc = self._cell_centers.detach().cpu().numpy()
        n = cc.shape[0]
        points_prim.CreatePointsAttr(
            Vt.Vec3fArray.FromNumpy(cc.astype("float32"))
        )
        widths = [float(self._point_width)] * n
        points_prim.CreateWidthsAttr(Vt.FloatArray(widths))
        points_prim.SetWidthsInterpolation(UsdGeom.Tokens.vertex)

        primvars = UsdGeom.PrimvarsAPI(points_prim)
        if not primvars.HasPrimvar("displayColor"):
            primvars.CreatePrimvar(
                "displayColor",
                Sdf.ValueTypeNames.Color3fArray,
                interpolation=UsdGeom.Tokens.vertex,
            )

    def _subscribe(self) -> None:
        bus = omni.kit.app.get_app().get_message_bus_event_stream()
        event_type = carb.events.type_from_string(EVENT_TYPE_SET_OMEGA)
        self._sub = bus.create_subscription_to_pop_by_type(
            event_type, self._on_set_omega, name="cfd.setOmega.listener"
        )

    def _on_set_omega(self, event) -> None:
        try:
            payload = dict(event.payload) if event.payload is not None else {}
        except Exception:
            payload = {}
        omega = payload.get("omega_rad_s")
        if omega is None:
            rpm = payload.get("rpm")
            if rpm is not None:
                omega = float(rpm) * math.pi / 30.0
            else:
                carb.log_warn(f"[cfd] cfd.setOmega missing omega_rad_s and rpm: {payload}")
                return
        try:
            self._run_inference(float(omega))
        except Exception as exc:
            carb.log_error(f"[cfd] inference failed: {exc}")

    def _run_inference(self, omega_rad_s: float) -> None:
        if self._model is None or self._torch is None:
            return
        torch = self._torch
        t0 = time.perf_counter()
        with torch.no_grad():
            om = torch.tensor(omega_rad_s, dtype=torch.float32, device=self._device)
            y = self._model(om)
        u_xyz = y[:, 0:3]
        u_mag = u_xyz.norm(dim=1)
        vmax = float(u_mag.max().item())
        self._u_max_global = max(self._u_max_global, vmax, 1e-6)

        norm = (u_mag / self._u_max_global).clamp(0.0, 1.0)
        r = (2.0 * norm - 1.0).clamp(0.0, 1.0)
        g = (1.0 - 2.0 * (norm - 0.5).abs()).clamp(0.0, 1.0)
        b = (1.0 - 2.0 * norm).clamp(0.0, 1.0)
        colors = torch.stack([r, g, b], dim=1).cpu().numpy().astype("float32")

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._update_stage(colors, omega_rad_s, vmax, dt_ms)
        self._last_omega = omega_rad_s

    def _update_stage(self, colors, omega_rad_s: float, vmax: float, dt_ms: float) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        points_prim = UsdGeom.Points.Get(stage, self._points_path)
        if not points_prim:
            self._ensure_stage_prims()
            points_prim = UsdGeom.Points.Get(stage, self._points_path)
            if not points_prim:
                return
        primvars = UsdGeom.PrimvarsAPI(points_prim)
        color_pv = primvars.GetPrimvar("displayColor")
        if not color_pv:
            color_pv = primvars.CreatePrimvar(
                "displayColor",
                Sdf.ValueTypeNames.Color3fArray,
                interpolation=UsdGeom.Tokens.vertex,
            )
        color_pv.Set(Vt.Vec3fArray.FromNumpy(colors))
        rpm = omega_rad_s * 30.0 / math.pi
        carb.log_info(
            f"[cfd] {rpm:.0f} RPM ({omega_rad_s:.1f} rad/s) |U|max={vmax:.3f} infer={dt_ms:.1f}ms"
        )


_bridge: Optional[CFDInferenceBridge] = None


def start_bridge() -> None:
    global _bridge
    if _bridge is not None:
        return
    _bridge = CFDInferenceBridge()
    _bridge.start()


def stop_bridge() -> None:
    global _bridge
    if _bridge is None:
        return
    _bridge.stop()
    _bridge = None


class CFDInferenceExtension(omni.ext.IExt):
    """Auto-discovered IExt wrapper that owns the bridge lifecycle.

    Kit instantiates every IExt subclass found in the package's python
    module, so adding this class is enough to wire the bridge into the
    existing thuong.tc_extension package without modifying extension.py.
    """

    def on_startup(self, _ext_id: str) -> None:
        try:
            start_bridge()
        except Exception as exc:
            carb.log_error(f"[cfd] bridge start failed: {exc}")

    def on_shutdown(self) -> None:
        try:
            stop_bridge()
        except Exception as exc:
            carb.log_error(f"[cfd] bridge stop failed: {exc}")
