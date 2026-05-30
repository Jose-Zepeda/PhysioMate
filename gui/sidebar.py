"""
PhysioMate - Panel Lateral (Sidebar View)

Responsabilidad única: construir el panel lateral con sus controles y
métricas, y cablear cada control a los callbacks que recibe.

No conoce a MainApp: recibe los handlers a través de SidebarCallbacks y
expone como atributos públicos los widgets que la app necesita manipular
(estados de botones, slider, labels) más el MetricsPresenter ya armado.

Extraído de MainApp._build_sidebar / _create_metric_card para respetar SRP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import customtkinter as ctk

from exercises.base import get_available_exercises
from gui.metrics_presenter import MetricsPresenter


@dataclass
class SidebarCallbacks:
    """Handlers que el sidebar invoca al interactuar el usuario."""

    on_exercise_change: Callable
    on_camera_change: Callable
    on_start: Callable
    on_stop: Callable
    on_reset: Callable
    on_audio_toggle: Callable
    on_load_video: Callable
    on_seek_drag: Callable
    on_seek_start: Callable
    on_seek_release: Callable
    on_video_pause_resume: Callable
    on_stop_video: Callable
    on_speed_change: Callable


class SidebarView(ctk.CTkFrame):
    """Panel lateral fijo con scroll interno, controles y métricas.

    Atributos públicos expuestos a la app:
        exercise_var, audio_var: variables de los selectores.
        start_btn, stop_btn: botones de control de cámara.
        video_name_label, video_progress, video_time_label,
        play_pause_btn, stop_video_btn: controles de video pregrabado.
        metrics: MetricsPresenter ya cableado a los labels de métricas.
        default_camera: nombre de la cámara preseleccionada.
    """

    def __init__(
        self,
        master,
        available_cameras: dict,
        speed_options: dict,
        callbacks: SidebarCallbacks,
        width: int = 280,
    ) -> None:
        """Construye el panel lateral completo.

        Args:
            master: Ventana padre donde se posiciona el sidebar.
            available_cameras: Mapa {nombre: índice} de cámaras detectadas.
            speed_options: Mapa {etiqueta: factor} de velocidades de video.
            callbacks: Handlers a cablear en cada control.
            width: Ancho fijo del panel.
        """
        super().__init__(
            master,
            width=width,
            corner_radius=0,
            fg_color=("#1a1a2e", "#1a1a2e"),
        )
        self._width = width
        self._cb = callbacks

        self.grid_propagate(False)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ── Área scrollable interna ──
        self._scroll = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            scrollbar_button_color="#2a2a4a",
            scrollbar_button_hover_color="#3a3a6a",
            corner_radius=0,
        )
        self._scroll.grid(row=0, column=0, sticky="nsew")
        # Alias conveniente: todos los widgets se crean dentro de _scroll
        _s = self._scroll

        # ── Logo / Título ──
        logo_frame = ctk.CTkFrame(_s, fg_color="transparent")
        logo_frame.pack(fill="x", padx=15, pady=(14, 4))

        title_label = ctk.CTkLabel(
            logo_frame,
            text="🏋️ PhysioMate",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color="#00d4ff",
        )
        title_label.pack(anchor="w")

        subtitle_label = ctk.CTkLabel(
            logo_frame,
            text="Asistente de Rehabilitación",
            font=ctk.CTkFont(size=11),
            text_color="#888888",
        )
        subtitle_label.pack(anchor="w")

        # ── Separador ──
        sep = ctk.CTkFrame(_s, height=2, fg_color="#2a2a4a")
        sep.pack(fill="x", padx=15, pady=6)

        # ── Selector de Ejercicio ──
        exercise_label = ctk.CTkLabel(
            _s,
            text="EJERCICIO",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#666666",
        )
        exercise_label.pack(anchor="w", padx=20)

        exercises = get_available_exercises()
        self.exercise_var = ctk.StringVar(
            value=exercises[0] if exercises else ""
        )
        self._exercise_dropdown = ctk.CTkOptionMenu(
            _s,
            values=exercises if exercises else ["Sin ejercicios"],
            variable=self.exercise_var,
            command=self._cb.on_exercise_change,
            width=width - 55,
            height=32,
            fg_color="#16213e",
            button_color="#0f3460",
            button_hover_color="#1a5276",
            dropdown_fg_color="#16213e",
            dropdown_hover_color="#0f3460",
            font=ctk.CTkFont(size=12),
        )
        self._exercise_dropdown.pack(padx=20, pady=(4, 8))

        # ── Selector de Cámara ──
        cam_label = ctk.CTkLabel(
            _s,
            text="CÁMARA",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#666666",
        )
        cam_label.pack(anchor="w", padx=20)

        cam_names = list(available_cameras.keys()) or ["Sin cámaras"]

        # Preseleccionar la primera cámara que NO sea la integrada (idx != 0)
        default_cam = next(
            (name for name, idx in available_cameras.items() if idx != 0),
            cam_names[0],
        )
        self.default_camera = default_cam

        self.camera_var = ctk.StringVar(value=default_cam)
        self._camera_dropdown = ctk.CTkOptionMenu(
            _s,
            values=cam_names,
            variable=self.camera_var,
            command=self._cb.on_camera_change,
            width=width - 55,
            height=32,
            fg_color="#16213e",
            button_color="#0f3460",
            button_hover_color="#1a5276",
            dropdown_fg_color="#16213e",
            dropdown_hover_color="#0f3460",
            font=ctk.CTkFont(size=12),
        )
        self._camera_dropdown.pack(padx=20, pady=(4, 8))

        # ── Botones de Control ──
        btn_frame = ctk.CTkFrame(_s, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20)

        self.start_btn = ctk.CTkButton(
            btn_frame,
            text="▶  Iniciar",
            command=self._cb.on_start,
            height=36,
            fg_color="#0e6c3a",
            hover_color="#12944f",
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=8,
        )
        self.start_btn.pack(fill="x", pady=(0, 6))

        self.stop_btn = ctk.CTkButton(
            btn_frame,
            text="■  Detener",
            command=self._cb.on_stop,
            height=36,
            fg_color="#8b0000",
            hover_color="#b22222",
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=8,
            state="disabled",
        )
        self.stop_btn.pack(fill="x", pady=(0, 6))

        self._reset_btn = ctk.CTkButton(
            btn_frame,
            text="↺  Reiniciar",
            command=self._cb.on_reset,
            height=30,
            fg_color="#333355",
            hover_color="#444477",
            font=ctk.CTkFont(size=11),
            corner_radius=8,
        )
        self._reset_btn.pack(fill="x")

        # ── Separador ──
        sep2 = ctk.CTkFrame(_s, height=2, fg_color="#2a2a4a")
        sep2.pack(fill="x", padx=15, pady=8)

        # ── Métricas ──
        metrics_label = ctk.CTkLabel(
            _s,
            text="MÉTRICAS",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#666666",
        )
        metrics_label.pack(anchor="w", padx=20)

        # Repeticiones
        rep_frame = self._create_metric_card("Repeticiones", "0", "#00d4ff")
        rep_frame.pack(fill="x", padx=20, pady=(6, 3))

        # Ángulo actual
        angle_frame = self._create_metric_card("Ángulo", "-- °", "#ffd700")
        angle_frame.pack(fill="x", padx=20, pady=3)

        # Estado
        state_frame = self._create_metric_card("Estado", "Inactivo", "#00ff88")
        state_frame.pack(fill="x", padx=20, pady=3)

        # ── Indicador de Forma ──
        sep3 = ctk.CTkFrame(_s, height=2, fg_color="#2a2a4a")
        sep3.pack(fill="x", padx=15, pady=8)

        form_header = ctk.CTkLabel(
            _s,
            text="FORMA",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#666666",
        )
        form_header.pack(anchor="w", padx=20)

        form_indicator = ctk.CTkLabel(
            _s,
            text="● CORRECTA",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#00ff88",
        )
        form_indicator.pack(anchor="w", padx=20, pady=(4, 6))

        # ── Feedback ──
        feedback_label = ctk.CTkLabel(
            _s,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="#aaaaaa",
            wraplength=width - 55,
        )
        feedback_label.pack(anchor="w", padx=20, pady=(0, 6))

        # ── Audio toggle ──
        sep4 = ctk.CTkFrame(_s, height=2, fg_color="#2a2a4a")
        sep4.pack(fill="x", padx=15, pady=(4, 6))

        self.audio_var = ctk.BooleanVar(value=True)
        self._audio_switch = ctk.CTkSwitch(
            _s,
            text="Audio Feedback",
            variable=self.audio_var,
            command=self._cb.on_audio_toggle,
            onvalue=True,
            offvalue=False,
            font=ctk.CTkFont(size=11),
            progress_color="#00d4ff",
        )
        self._audio_switch.pack(anchor="w", padx=20, pady=(0, 6))

        # ── Sección: Análisis de Video Pregrabado ──
        sep5 = ctk.CTkFrame(_s, height=2, fg_color="#2a2a4a")
        sep5.pack(fill="x", padx=15, pady=(4, 6))

        video_section_label = ctk.CTkLabel(
            _s,
            text="ANÁLISIS DE VIDEO",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#666666",
        )
        video_section_label.pack(anchor="w", padx=20)

        # Botón cargar video
        self._load_video_btn = ctk.CTkButton(
            _s,
            text="📂  Cargar Video",
            command=self._cb.on_load_video,
            height=34,
            fg_color="#1a3a5c",
            hover_color="#254f7a",
            font=ctk.CTkFont(size=12, weight="bold"),
            corner_radius=8,
        )
        self._load_video_btn.pack(fill="x", padx=20, pady=(5, 3))

        # Nombre del archivo cargado
        self.video_name_label = ctk.CTkLabel(
            _s,
            text="Sin video cargado",
            font=ctk.CTkFont(size=10),
            text_color="#555555",
            wraplength=width - 55,
            anchor="w",
        )
        self.video_name_label.pack(anchor="w", padx=20, pady=(0, 3))

        # Barra de progreso / Seek slider del video
        self.video_progress = ctk.CTkSlider(
            _s,
            from_=0,
            to=1,
            width=width - 55,
            height=14,
            progress_color="#00d4ff",
            fg_color="#16213e",
            button_color="#00aacc",
            button_hover_color="#00d4ff",
            corner_radius=4,
            button_corner_radius=6,
            command=self._cb.on_seek_drag,
        )
        self.video_progress.set(0)
        self.video_progress.pack(padx=20, pady=(0, 3))
        # Detectar inicio y fin del arrastre para pausar/reanudar el loop
        self.video_progress.bind("<ButtonPress-1>",   self._cb.on_seek_start)
        self.video_progress.bind("<ButtonRelease-1>", self._cb.on_seek_release)

        # Label de tiempo
        self.video_time_label = ctk.CTkLabel(
            _s,
            text="00:00 / 00:00",
            font=ctk.CTkFont(family="Consolas", size=10),
            text_color="#555555",
        )
        self.video_time_label.pack(anchor="w", padx=20, pady=(0, 3))

        # Fila de controles de video
        video_ctrl_frame = ctk.CTkFrame(_s, fg_color="transparent")
        video_ctrl_frame.pack(fill="x", padx=20, pady=(0, 3))

        self.play_pause_btn = ctk.CTkButton(
            video_ctrl_frame,
            text="⏸ Pausar",
            command=self._cb.on_video_pause_resume,
            height=30,
            fg_color="#2a2a4a",
            hover_color="#3a3a6a",
            font=ctk.CTkFont(size=11),
            corner_radius=6,
            state="disabled",
        )
        self.play_pause_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.stop_video_btn = ctk.CTkButton(
            video_ctrl_frame,
            text="⏹",
            command=self._cb.on_stop_video,
            height=30,
            width=34,
            fg_color="#4a1a1a",
            hover_color="#6a2a2a",
            font=ctk.CTkFont(size=11),
            corner_radius=6,
            state="disabled",
        )
        self.stop_video_btn.pack(side="right")

        # Velocidad de reproducción
        speed_frame = ctk.CTkFrame(_s, fg_color="transparent")
        speed_frame.pack(fill="x", padx=20, pady=(0, 3))

        ctk.CTkLabel(
            speed_frame,
            text="Velocidad:",
            font=ctk.CTkFont(size=11),
            text_color="#888888",
        ).pack(side="left", padx=(0, 6))

        self.speed_var = ctk.StringVar(value="1x")
        speed_menu = ctk.CTkOptionMenu(
            speed_frame,
            values=list(speed_options.keys()),
            variable=self.speed_var,
            command=self._cb.on_speed_change,
            width=80,
            height=26,
            fg_color="#16213e",
            button_color="#0f3460",
            button_hover_color="#1a5276",
            dropdown_fg_color="#16213e",
            font=ctk.CTkFont(size=11),
        )
        speed_menu.pack(side="left")

        # ── Presentador de métricas (delega la actualización del sidebar) ──
        self.metrics = MetricsPresenter(
            rep_label=rep_frame._value_label,        # type: ignore[attr-defined]
            angle_label=angle_frame._value_label,    # type: ignore[attr-defined]
            state_label=state_frame._value_label,    # type: ignore[attr-defined]
            form_indicator=form_indicator,
            feedback_label=feedback_label,
        )

    def _create_metric_card(
        self,
        title: str,
        value: str,
        value_color: str,
    ) -> ctk.CTkFrame:
        """Crea una tarjeta de métrica reutilizable.

        Args:
            title: Título de la métrica.
            value: Valor inicial a mostrar.
            value_color: Color del valor.

        Returns:
            Frame con la tarjeta de métrica.
        """
        card = ctk.CTkFrame(
            self._scroll,
            fg_color="#16213e",
            corner_radius=8,
            height=52,
        )
        card.pack_propagate(False)

        title_lbl = ctk.CTkLabel(
            card,
            text=title.upper(),
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#888888",
        )
        title_lbl.pack(anchor="w", padx=12, pady=(6, 0))

        value_lbl = ctk.CTkLabel(
            card,
            text=value,
            font=ctk.CTkFont(family="Consolas", size=19, weight="bold"),
            text_color=value_color,
        )
        value_lbl.pack(anchor="w", padx=12, pady=(0, 6))

        # Guardar referencia al label de valor para actualizar después
        card._value_label = value_lbl  # type: ignore[attr-defined]

        return card
