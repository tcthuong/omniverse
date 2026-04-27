import omni.ui as ui
import omni.timeline

class CFDControls:
    def __init__(self):
        # Tạo cửa sổ float, không có thanh tiêu đề, nền trong suốt
        self.window = ui.Window(
            "CFDDashboard", 
            width=450, height=200, 
            position_x=30, position_y=30,
            flags=ui.WINDOW_FLAGS_NO_TITLE_BAR | ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_NO_BACKGROUND
        )
        
        # Định nghĩa style cực đẹp (Dark Theme / Glassmorphism)
        self.window.frame.style = {
            "Window": {
                "background_color": 0xDD151515, # Đen mờ (Alpha=DD)
                "border_radius": 12,
                "border_color": 0x44FFFFFF,
                "border_width": 1
            },
            "Label": {
                "color": 0xFFE0E0E0,
                "font_size": 15
            },
            "Label::Header": {
                "font_size": 22,
                "color": 0xFFFFFFFF
            },
            "Label::Dim": {
                "color": 0xFF888888,
                "font_size": 12
            },
            "Slider": {
                "background_color": 0xFF333333,
                "secondary_color": 0xFF76B900, # NVIDIA Green
                "border_radius": 6,
                "height": 12
            },
            "Slider::knob": {
                "background_color": 0xFFFFFFFF,
                "border_radius": 8
            }
        }
        
        self.build_ui()

    def build_ui(self):
        with self.window.frame:
            with ui.VStack(spacing=15, margin=25):
                # Header
                ui.Label("CFD Volume Trace", name="Header", height=20)
                
                ui.Spacer(height=5)
                
                # Speed Control Slider
                with ui.HStack(height=20):
                    ui.Label("Speed Control", width=120)
                    ui.Label("(RPM)", name="Dim", width=40)
                    ui.Spacer()
                
                with ui.HStack(height=25, spacing=10):
                    ui.Label("150", name="Dim", width=30)
                    self.slider = ui.FloatSlider(min=150, max=350)
                    # Gán sự kiện khi kéo slider
                    self.slider.model.add_value_changed_fn(self.on_slider_changed)
                    ui.Label("350", name="Dim", width=30, alignment=ui.Alignment.RIGHT)
                
                ui.Spacer(height=10)
                
                # Velocity Magnitude Legend
                with ui.HStack(height=20):
                    ui.Label("Velocity Magnitude", width=150)
                    ui.Spacer()
                
                # Dải màu gradient (Turbo Color Map approximation)
                with ui.HStack(height=10, style={"border_radius": 5}):
                    # Xanh đậm (A=FF, R=30, G=30, B=B0) => 0xFF3030B0
                    ui.Rectangle(style={"background_color": 0xFFB03030}) # B=B0, G=30, R=30 -> Xanh đậm (Do Omniverse dùng ABGR or ARGB, test bằng 0xFF...)
                    # Xanh lơ
                    ui.Rectangle(style={"background_color": 0xFFD0B030}) 
                    # Xanh lá
                    ui.Rectangle(style={"background_color": 0xFF40D040})
                    # Vàng
                    ui.Rectangle(style={"background_color": 0xFF30D0D0})
                    # Đỏ
                    ui.Rectangle(style={"background_color": 0xFF3030D0})
                
                with ui.HStack(height=15):
                    ui.Label("0", name="Dim")
                    ui.Spacer()
                    ui.Label("50", name="Dim")
                    ui.Spacer()
                    ui.Label("100", name="Dim")

    def on_slider_changed(self, model):
        val = model.as_float
        # Khớp với file: 150 -> frame 0, 350 -> frame 20. (Step = 10)
        frame = (val - 150.0) / 10.0
        omni.timeline.get_timeline_interface().set_current_time(frame)

# Khởi tạo UI
my_cfd_ui = CFDControls()
