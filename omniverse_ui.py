import math

import omni.timeline
import omni.ui as ui

_OMEGA_MIN_RAD_S = 150.0
_FRAME_STEP_RAD_S = 10.0
_RPM_MIN = round(_OMEGA_MIN_RAD_S * 30.0 / math.pi)  # 1432
_RPM_MAX = round(350.0 * 30.0 / math.pi)              # 3342


class CFDControls:
    def __init__(self):
        self.window = ui.Window(
            "CFDDashboard",
            width=450,
            height=200,
            position_x=30,
            position_y=30,
            flags=ui.WINDOW_FLAGS_NO_TITLE_BAR
            | ui.WINDOW_FLAGS_NO_SCROLLBAR
            | ui.WINDOW_FLAGS_NO_BACKGROUND,
        )

        self.window.frame.style = {
            "Window": {
                "background_color": 0xDD151515,
                "border_radius": 12,
                "border_color": 0x44FFFFFF,
                "border_width": 1,
            },
            "Label": {
                "color": 0xFFE0E0E0,
                "font_size": 15,
            },
            "Label::Header": {
                "font_size": 22,
                "color": 0xFFFFFFFF,
            },
            "Label::Dim": {
                "color": 0xFF888888,
                "font_size": 12,
            },
            "Slider": {
                "background_color": 0xFF333333,
                "secondary_color": 0xFF76B900,
                "border_radius": 6,
                "height": 12,
            },
            "Slider::knob": {
                "background_color": 0xFFFFFFFF,
                "border_radius": 8,
            },
        }

        self.build_ui()

    def build_ui(self):
        with self.window.frame:
            with ui.VStack(spacing=15, margin=25):
                ui.Label("CFD Volume Trace", name="Header", height=20)
                ui.Spacer(height=5)

                with ui.HStack(height=20):
                    ui.Label("Speed Control", width=120)
                    ui.Label("(RPM)", name="Dim", width=50)
                    ui.Spacer()

                with ui.HStack(height=25, spacing=10):
                    ui.Label(f"{_RPM_MIN}", name="Dim", width=40)
                    self.slider = ui.FloatSlider(min=_RPM_MIN, max=_RPM_MAX)
                    self.slider.model.add_value_changed_fn(self.on_slider_changed)
                    ui.Label(f"{_RPM_MAX}", name="Dim", width=40, alignment=ui.Alignment.RIGHT)

                ui.Spacer(height=10)

                with ui.HStack(height=20):
                    ui.Label("Velocity Magnitude", width=150)
                    ui.Spacer()

                with ui.HStack(height=10, style={"border_radius": 5}):
                    ui.Rectangle(style={"background_color": 0xFFB03030})
                    ui.Rectangle(style={"background_color": 0xFFD0B030})
                    ui.Rectangle(style={"background_color": 0xFF40D040})
                    ui.Rectangle(style={"background_color": 0xFF30D0D0})
                    ui.Rectangle(style={"background_color": 0xFF3030D0})

                with ui.HStack(height=15):
                    ui.Label("0", name="Dim")
                    ui.Spacer()
                    ui.Label("50", name="Dim")
                    ui.Spacer()
                    ui.Label("100", name="Dim")

    def on_slider_changed(self, model):
        rpm = model.as_float
        omega_rad_s = rpm * math.pi / 30.0
        frame = (omega_rad_s - _OMEGA_MIN_RAD_S) / _FRAME_STEP_RAD_S
        omni.timeline.get_timeline_interface().set_current_time(frame)


my_cfd_ui = CFDControls()
