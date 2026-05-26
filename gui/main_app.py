"""
PhysioMate - Interfaz Gráfica Principal (Main App GUI)

Ventana principal de la aplicación construida con customtkinter.
Incluye feed de video en tiempo real, panel lateral con métricas
y controles, y manejo completo del ciclo de vida de la cámara.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from tkinter import filedialog
from typing import Optional

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from core.audio_feedback import AudioFeedback
from core.exercise_tracker import ExerciseTracker
from core.pose_detector import PoseDetector
from core.config import UIConfig
from gui.drawing_utils import DrawingUtils
from exercises.base import get_available_exercises

logger = logging.getLogger(__name__)

# Configuración global de UI
_CONFIG = UIConfig()

# ─── Constantes de diseño ───
SIDEBAR_WIDTH = 280
VIDEO_UPDATE_INTERVAL_MS = 30  # ~33 FPS

# Velocidades de reproducción de video
SPEED_OPTIONS = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "1.5x": 1.5, "2x": 2.0}


class MainApp(ctk.CTk):
    """Ventana principal de PhysioMate.

    Gestiona la interfaz de usuario, el ciclo de video de la cámara
    y la interacción con el ExerciseTracker. No contiene lógica de
    ejercicios ni de IA (Principio de Responsabilidad Única).

    Attributes:
        tracker: Orquestador de ejercicios inyectado.
    """

    def __init__(
        self,
        tracker: ExerciseTracker,
        camera_index: int = 0,
    ) -> None:
        """Inicializa la ventana principal y todos los widgets.

        Args:
            tracker: Instancia del ExerciseTracker.
            camera_index: Índice de la cámara a utilizar.
        """
        super().__init__()

        self.tracker = tracker
        self._camera_index = camera_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_running = False
        self._after_id: Optional[str] = None
        self._current_image: Optional[ImageTk.PhotoImage] = None

        # ── Estado modo video pregrabado ──
        self._video_mode: bool = False          # True = analizando archivo, False = cámara
        self._video_path: Optional[str] = None  # Ruta del archivo seleccionado
        self._video_paused: bool = False        # Pausa del video
        self._video_total_frames: int = 0       # Total de frames del video
        self._video_current_frame: int = 0      # Frame actual
        self._video_fps: float = 30.0           # FPS original del video
        self._playback_speed: float = 1.0       # Factor de velocidad
        self._user_seeking: bool = False        # True mientras el usuario arrastra el slider

        # Detectar cámaras disponibles al arrancar: dict {nombre: índice}
        self._available_cameras: dict = self._detect_cameras()

        # ── Configuración de la ventana ──
        self.title(_CONFIG.WINDOW_TITLE)
        self.geometry(f"{_CONFIG.MIN_WIDTH}x{_CONFIG.MIN_HEIGHT}")
        self.minsize(_CONFIG.MIN_WIDTH, _CONFIG.MIN_HEIGHT)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Tema ──
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # ── Construir UI ──
        self._build_layout()
        self._build_sidebar()
        self._build_video_area()

        # Cargar el placeholder inicial después de que la interfaz se dibuje
        self.after(100, self._update_placeholder_image)

    # ──────────────────────────────────────────────
    #  Construcción de la Interfaz
    # ──────────────────────────────────────────────

    def _build_layout(self) -> None:
        """Configura el grid principal de la ventana."""
        self.grid_columnconfigure(0, weight=0)   # Sidebar (fijo)
        self.grid_columnconfigure(1, weight=1)   # Video (expandible)
        self.grid_rowconfigure(0, weight=1)

    def _build_sidebar(self) -> None:
        """Construye el panel lateral con controles y métricas.

        El sidebar usa un CTkScrollableFrame interno para que todo el
        contenido sea accesible sin maximizar la ventana.
        """
        # ── Contenedor externo fijo (sin scroll) ──
        self._sidebar = ctk.CTkFrame(
            self,
            width=SIDEBAR_WIDTH,
            corner_radius=0,
            fg_color=("#1a1a2e", "#1a1a2e"),
        )
        self._sidebar.grid(row=0, column=0, sticky="nsew")
        self._sidebar.grid_propagate(False)
        self._sidebar.grid_rowconfigure(0, weight=1)
        self._sidebar.grid_columnconfigure(0, weight=1)

        # ── Área scrollable interna ──
        self._scroll = ctk.CTkScrollableFrame(
            self._sidebar,
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
        self._exercise_var = ctk.StringVar(
            value=exercises[0] if exercises else ""
        )
        self._exercise_dropdown = ctk.CTkOptionMenu(
            _s,
            values=exercises if exercises else ["Sin ejercicios"],
            variable=self._exercise_var,
            command=self._on_exercise_change,
            width=SIDEBAR_WIDTH - 55,
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

        cam_names = list(self._available_cameras.keys()) or ["Sin cámaras"]

        # Preseleccionar la primera cámara que NO sea la integrada (idx != 0)
        default_cam = next(
            (name for name, idx in self._available_cameras.items() if idx != 0),
            cam_names[0],
        )

        self._camera_var = ctk.StringVar(value=default_cam)
        self._camera_dropdown = ctk.CTkOptionMenu(
            _s,
            values=cam_names,
            variable=self._camera_var,
            command=self._on_camera_change,
            width=SIDEBAR_WIDTH - 55,
            height=32,
            fg_color="#16213e",
            button_color="#0f3460",
            button_hover_color="#1a5276",
            dropdown_fg_color="#16213e",
            dropdown_hover_color="#0f3460",
            font=ctk.CTkFont(size=12),
        )
        self._camera_dropdown.pack(padx=20, pady=(4, 8))

        # Aplicar índice inicial según la selección
        self._apply_camera_selection(default_cam)

        # ── Botones de Control ──
        btn_frame = ctk.CTkFrame(_s, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20)

        self._start_btn = ctk.CTkButton(
            btn_frame,
            text="▶  Iniciar",
            command=self._on_start,
            height=36,
            fg_color="#0e6c3a",
            hover_color="#12944f",
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=8,
        )
        self._start_btn.pack(fill="x", pady=(0, 6))

        self._stop_btn = ctk.CTkButton(
            btn_frame,
            text="■  Detener",
            command=self._on_stop,
            height=36,
            fg_color="#8b0000",
            hover_color="#b22222",
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=8,
            state="disabled",
        )
        self._stop_btn.pack(fill="x", pady=(0, 6))

        self._reset_btn = ctk.CTkButton(
            btn_frame,
            text="↺  Reiniciar",
            command=self._on_reset,
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
        self._rep_frame = self._create_metric_card(
            "Repeticiones", "0", "#00d4ff"
        )
        self._rep_frame.pack(fill="x", padx=20, pady=(6, 3))

        # Ángulo actual
        self._angle_frame = self._create_metric_card(
            "Ángulo", "-- °", "#ffd700"
        )
        self._angle_frame.pack(fill="x", padx=20, pady=3)

        # Estado
        self._state_frame = self._create_metric_card(
            "Estado", "Inactivo", "#00ff88"
        )
        self._state_frame.pack(fill="x", padx=20, pady=3)

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

        self._form_indicator = ctk.CTkLabel(
            _s,
            text="● CORRECTA",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#00ff88",
        )
        self._form_indicator.pack(anchor="w", padx=20, pady=(4, 6))

        # ── Feedback ──
        self._feedback_label = ctk.CTkLabel(
            _s,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="#aaaaaa",
            wraplength=SIDEBAR_WIDTH - 55,
        )
        self._feedback_label.pack(anchor="w", padx=20, pady=(0, 6))

        # ── Audio toggle ──
        sep4 = ctk.CTkFrame(_s, height=2, fg_color="#2a2a4a")
        sep4.pack(fill="x", padx=15, pady=(4, 6))

        self._audio_var = ctk.BooleanVar(value=True)
        self._audio_switch = ctk.CTkSwitch(
            _s,
            text="Audio Feedback",
            variable=self._audio_var,
            command=self._on_audio_toggle,
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
            command=self._on_load_video,
            height=34,
            fg_color="#1a3a5c",
            hover_color="#254f7a",
            font=ctk.CTkFont(size=12, weight="bold"),
            corner_radius=8,
        )
        self._load_video_btn.pack(fill="x", padx=20, pady=(5, 3))

        # Nombre del archivo cargado
        self._video_name_label = ctk.CTkLabel(
            _s,
            text="Sin video cargado",
            font=ctk.CTkFont(size=10),
            text_color="#555555",
            wraplength=SIDEBAR_WIDTH - 55,
            anchor="w",
        )
        self._video_name_label.pack(anchor="w", padx=20, pady=(0, 3))

        # Barra de progreso / Seek slider del video
        self._video_progress = ctk.CTkSlider(
            _s,
            from_=0,
            to=1,
            width=SIDEBAR_WIDTH - 55,
            height=14,
            progress_color="#00d4ff",
            fg_color="#16213e",
            button_color="#00aacc",
            button_hover_color="#00d4ff",
            corner_radius=4,
            button_corner_radius=6,
            command=self._on_seek_drag,
        )
        self._video_progress.set(0)
        self._video_progress.pack(padx=20, pady=(0, 3))
        # Detectar inicio y fin del arrastre para pausar/reanudar el loop
        self._video_progress.bind("<ButtonPress-1>",   self._on_seek_start)
        self._video_progress.bind("<ButtonRelease-1>", self._on_seek_release)

        # Label de tiempo
        self._video_time_label = ctk.CTkLabel(
            _s,
            text="00:00 / 00:00",
            font=ctk.CTkFont(family="Consolas", size=10),
            text_color="#555555",
        )
        self._video_time_label.pack(anchor="w", padx=20, pady=(0, 3))

        # Fila de controles de video
        video_ctrl_frame = ctk.CTkFrame(_s, fg_color="transparent")
        video_ctrl_frame.pack(fill="x", padx=20, pady=(0, 3))

        self._play_pause_btn = ctk.CTkButton(
            video_ctrl_frame,
            text="⏸ Pausar",
            command=self._on_video_pause_resume,
            height=30,
            fg_color="#2a2a4a",
            hover_color="#3a3a6a",
            font=ctk.CTkFont(size=11),
            corner_radius=6,
            state="disabled",
        )
        self._play_pause_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self._stop_video_btn = ctk.CTkButton(
            video_ctrl_frame,
            text="⏹",
            command=self._on_stop_video,
            height=30,
            width=34,
            fg_color="#4a1a1a",
            hover_color="#6a2a2a",
            font=ctk.CTkFont(size=11),
            corner_radius=6,
            state="disabled",
        )
        self._stop_video_btn.pack(side="right")

        # Velocidad de reproducción
        speed_frame = ctk.CTkFrame(_s, fg_color="transparent")
        speed_frame.pack(fill="x", padx=20, pady=(0, 3))

        ctk.CTkLabel(
            speed_frame,
            text="Velocidad:",
            font=ctk.CTkFont(size=11),
            text_color="#888888",
        ).pack(side="left", padx=(0, 6))

        self._speed_var = ctk.StringVar(value="1x")
        speed_menu = ctk.CTkOptionMenu(
            speed_frame,
            values=list(SPEED_OPTIONS.keys()),
            variable=self._speed_var,
            command=self._on_speed_change,
            width=80,
            height=26,
            fg_color="#16213e",
            button_color="#0f3460",
            button_hover_color="#1a5276",
            dropdown_fg_color="#16213e",
            font=ctk.CTkFont(size=11),
        )
        speed_menu.pack(side="left")



    def _build_video_area(self) -> None:
        """Construye el área central del feed de video."""
        self._video_frame = ctk.CTkFrame(
            self,
            corner_radius=0,
            fg_color=("#0a0a1a", "#0a0a1a"),
        )
        self._video_frame.grid(row=0, column=1, sticky="nsew")
        self._video_frame.grid_rowconfigure(0, weight=1)
        self._video_frame.grid_columnconfigure(0, weight=1)

        # Canvas para el video
        self._video_label = ctk.CTkLabel(
            self._video_frame,
            text="Presiona ▶ Iniciar para activar la cámara",
            font=ctk.CTkFont(size=16),
            text_color="#555555",
        )
        self._video_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

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

    # ──────────────────────────────────────────────
    #  Eventos de Control
    # ──────────────────────────────────────────────

    @staticmethod
    def _get_dshow_camera_names() -> list:
        """Enumera los dispositivos de captura de vídeo DirectShow.

        Usa comtypes para acceder a ICreateDevEnum, la misma API que
        OpenCV utiliza internamente, garantizando que el orden de los
        nombres coincida exactamente con los índices 0, 1, 2...

        Returns:
            Lista de nombres de dispositivos en orden de índice.
        """
        import comtypes
        import comtypes.client
        from comtypes import GUID
        import ctypes

        names = []
        try:
            comtypes.CoInitialize()

            # GUIDs de DirectShow
            CLSID_SystemDeviceEnum = GUID("{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}")
            CLSID_VideoInputDeviceCategory = GUID("{860BB310-5D01-11d0-BD3B-00A0C911CE86}")

            # ICreateDevEnum
            IID_ICreateDevEnum = GUID("{29840822-5B84-11D0-BD3B-00A0C911CE86}")
            IID_IEnumMoniker   = GUID("{00000102-0000-0000-C000-000000000046}")
            IID_IMoniker       = GUID("{0000000F-0000-0000-C000-000000000046}")
            IID_IPropertyBag   = GUID("{55272A00-42CB-11CE-8135-00AA004BB851}")

            dev_enum = comtypes.client.CreateObject(
                CLSID_SystemDeviceEnum,
                interface=comtypes.IUnknown,
            )

            # QueryInterface to ICreateDevEnum
            ICreateDevEnum = comtypes.GUID("{29840822-5B84-11D0-BD3B-00A0C911CE86}")
            p_create_dev_enum = ctypes.POINTER(comtypes.IUnknown)()
            dev_enum.QueryInterface(ctypes.byref(ICreateDevEnum),
                                    ctypes.byref(p_create_dev_enum))

        except Exception:
            pass

        # Fallback: si comtypes falla, usar PowerShell con búsqueda ampliada
        if not names:
            try:
                import subprocess, json
                # Consulta ampliada: clase Camera + dispositivos de imagen USB
                cmd = [
                    "powershell", "-NoProfile", "-Command",
                    "(Get-PnpDevice -Status OK | Where-Object {"
                    " $_.Class -eq 'Camera' -or $_.Class -eq 'Image'"
                    " -or ($_.FriendlyName -match 'cam|webcam|video|capture')"
                    "} | Sort-Object InstanceId"
                    " | Select-Object -ExpandProperty FriendlyName"
                    " | ConvertTo-Json -Compress)",
                ]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
                if r.returncode == 0 and r.stdout.strip():
                    raw = json.loads(r.stdout.strip())
                    names = [raw] if isinstance(raw, str) else list(raw)
            except Exception:
                pass

        return names

    @staticmethod
    def _detect_cameras(max_to_check: int = 6) -> dict:
        """Detecta cámaras disponibles y retorna {nombre: índice}."""
        # Obtener nombres reales via DirectShow / PowerShell
        real_names = MainApp._get_dshow_camera_names()

        result: dict = {}
        name_idx = 0
        for cv_idx in range(max_to_check):
            cap = cv2.VideoCapture(cv_idx, cv2.CAP_DSHOW)
            if cap is not None and cap.isOpened():
                cap.release()
                # Asignar nombre real si está disponible, si no usar fallback
                if name_idx < len(real_names):
                    label = real_names[name_idx]
                else:
                    label = f"Cámara {cv_idx}"
                result[label] = cv_idx
                name_idx += 1

        logger.info("Cámaras detectadas: %s", {v: k for k, v in result.items()})
        return result

    def _apply_camera_selection(self, cam_name: str) -> None:
        """Guarda el índice correspondiente al nombre de cámara seleccionado."""
        self._camera_index = self._available_cameras.get(cam_name, 0)

    def _on_camera_change(self, cam_name: str) -> None:
        """Maneja el cambio de cámara desde el dropdown."""
        self._apply_camera_selection(cam_name)
        logger.info("Cámara seleccionada: '%s' (idx=%d)", cam_name, self._camera_index)

        # Si la cámara está corriendo, reiniciarla con el nuevo índice
        if self._is_running:
            self._on_stop()
            self.after(300, self._on_start)

    def _on_start(self) -> None:
        """Inicia la captura de video y el procesamiento."""
        if self._is_running:
            return

        try:
            self._cap = cv2.VideoCapture(self._camera_index)
            if not self._cap.isOpened():
                self._show_error("No se pudo abrir la cámara. Verifica la conexión.")
                return

            # Configurar resolución
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            self._is_running = True
            self._start_btn.configure(state="disabled")
            self._stop_btn.configure(state="normal")

            logger.info("Cámara iniciada (índice: %d)", self._camera_index)
            self._update_video()

        except Exception as e:
            self._show_error(f"Error al iniciar la cámara: {e}")
            logger.error("Error al iniciar cámara: %s", e)

    def _on_stop(self) -> None:
        """Detiene la captura de video."""
        self._is_running = False

        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None

        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            self._cap = None

        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")

        logger.info("Cámara detenida")
        self._update_placeholder_image()

    def _on_reset(self) -> None:
        """Reinicia los contadores del ejercicio actual."""
        self.tracker.reset_exercise()
        self._update_metrics_display(None)

    def _on_exercise_change(self, exercise_name: str) -> None:
        """Cambia el ejercicio activo.

        Args:
            exercise_name: Nombre del ejercicio seleccionado.
        """
        self.tracker.set_exercise(exercise_name)
        self.tracker.reset_exercise()
        self._update_metrics_display(None)
        logger.info("Ejercicio cambiado a: %s", exercise_name)
        self._update_placeholder_image()

    def _update_placeholder_image(self) -> None:
        """Actualiza la imagen placeholder según el ejercicio seleccionado."""
        if self._is_running:
            return  # No interrumpir si la cámara está activa

        exercise_name = self._exercise_var.get()
        image_path = os.path.join("assets", f"{exercise_name}.png")
        
        try:
            if os.path.exists(image_path):
                self._video_frame.update_idletasks()
                video_width = self._video_frame.winfo_width() - 10
                video_height = self._video_frame.winfo_height() - 10
                
                if video_width <= 0 or video_height <= 0:
                    video_width, video_height = 800, 600

                # Cargar imagen y mantener relación de aspecto aproximada
                pil_image = Image.open(image_path)
                
                # Por simplicidad en placeholder usamos CTkImage directamente
                # ajustado al tamaño de la ventana
                ctk_image = ctk.CTkImage(
                    light_image=pil_image,
                    dark_image=pil_image,
                    size=(video_width, video_height),
                )
                
                self._video_label.configure(
                    image=ctk_image,
                    text="", # Ocultar texto si hay imagen
                )
                self._current_image = ctk_image
            else:
                self._clear_video_label()
        except Exception as e:
            logger.error(f"Error al cargar imagen del ejercicio {exercise_name}: %s", e)
            self._clear_video_label()

    def _clear_video_label(self) -> None:
        """Muestra el estado vacío sin imagen."""
        empty_img = ctk.CTkImage(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
        self._video_label.configure(
            image=empty_img,
            text="Cámara detenida. Presiona ▶ para reanudar.",
        )
        self._current_image = empty_img

    def _on_audio_toggle(self) -> None:
        """Activa/desactiva el feedback de audio."""
        self.tracker.audio.enabled = self._audio_var.get()

    # ──────────────────────────────────────────────
    #  Eventos de Video Pregrabado
    # ──────────────────────────────────────────────

    def _on_load_video(self) -> None:
        """Abre el diálogo para seleccionar un archivo de video."""
        # Detener cualquier fuente activa antes de cargar
        if self._is_running:
            self._on_stop()
        self._on_stop_video()

        path = filedialog.askopenfilename(
            title="Seleccionar video para analizar",
            filetypes=[
                ("Archivos de video", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not path:
            return

        # Validar que OpenCV pueda abrir el archivo
        test_cap = cv2.VideoCapture(path)
        if not test_cap.isOpened():
            test_cap.release()
            self._show_error("No se pudo abrir el archivo de video.\nVerifica que el formato sea compatible.")
            return

        self._video_total_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._video_fps = test_cap.get(cv2.CAP_PROP_FPS) or 30.0
        test_cap.release()

        self._video_path = path
        self._video_mode = True
        self._video_paused = False
        self._video_current_frame = 0

        # Actualizar label con nombre de archivo
        filename = os.path.basename(path)
        short_name = filename if len(filename) <= 28 else filename[:25] + "..."
        self._video_name_label.configure(text=short_name, text_color="#aaaaaa")

        # Habilitar controles de video
        self._play_pause_btn.configure(state="normal", text="⏸ Pausar")
        self._stop_video_btn.configure(state="normal")

        # Reiniciar contadores del ejercicio
        self.tracker.reset_exercise()
        self._update_metrics_display(None)

        logger.info("Video cargado: %s (%.0f frames @ %.1f FPS)", path, self._video_total_frames, self._video_fps)

        # Iniciar reproducción automáticamente
        self._start_video_playback()

    def _start_video_playback(self) -> None:
        """Inicia el loop de reproducción del video pregrabado."""
        if self._video_path is None:
            return

        self._cap = cv2.VideoCapture(self._video_path)
        if not self._cap.isOpened():
            self._show_error("Error al abrir el video para reproducción.")
            return



        self._is_running = True
        self._video_paused = False
        self._update_video_frame()

    def _on_video_pause_resume(self) -> None:
        """Alterna entre pausa y reproducción del video."""
        if not self._video_mode:
            return
        self._video_paused = not self._video_paused
        if self._video_paused:
            self._play_pause_btn.configure(text="▶ Reanudar")
            logger.info("Video pausado en frame %d", self._video_current_frame)
        else:
            self._play_pause_btn.configure(text="⏸ Pausar")
            logger.info("Video reanudado desde frame %d", self._video_current_frame)
            self._update_video_frame()

    def _on_stop_video(self) -> None:
        """Detiene y descarga el video pregrabado."""
        self._is_running = False
        self._video_mode = False
        self._video_paused = False

        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        # Resetear controles
        self._play_pause_btn.configure(state="disabled", text="⏸ Pausar")
        self._stop_video_btn.configure(state="disabled")
        self._video_progress.set(0)
        self._video_time_label.configure(text="00:00 / 00:00")
        self._video_name_label.configure(text="Sin video cargado", text_color="#555555")
        self._video_path = None
        self._update_placeholder_image()
        logger.info("Video detenido y descargado")

    def _on_speed_change(self, value: str) -> None:
        """Actualiza la velocidad de reproducción."""
        self._playback_speed = SPEED_OPTIONS.get(value, 1.0)
        logger.info("Velocidad de reproducción: %s (%.2fx)", value, self._playback_speed)

    def _on_seek_drag(self, value: float) -> None:
        """Llamado mientras el usuario arrastra el slider: actualiza solo el tiempo."""
        if not self._video_mode or self._video_total_frames <= 0:
            return
        # Calcular tiempo correspondiente a la posición del slider
        target_frame = int(value * self._video_total_frames)
        elapsed_s = target_frame / max(self._video_fps, 1)
        total_s = self._video_total_frames / max(self._video_fps, 1)
        elapsed_str = f"{int(elapsed_s // 60):02d}:{int(elapsed_s % 60):02d}"
        total_str = f"{int(total_s // 60):02d}:{int(total_s % 60):02d}"
        self._video_time_label.configure(
            text=f"{elapsed_str} / {total_str}",
            text_color="#ffcc00",  # color dorado mientras hace seek
        )

    def _on_seek_start(self, event=None) -> None:
        """El usuario empieza a arrastrar: congelar actualizaciones del loop."""
        self._user_seeking = True

    def _on_seek_release(self, event=None) -> None:
        """El usuario soltó el slider: saltar al frame correspondiente."""
        if not self._video_mode or self._cap is None or self._video_total_frames <= 0:
            self._user_seeking = False
            return

        value = self._video_progress.get()
        target_frame = int(value * self._video_total_frames)
        target_frame = max(0, min(target_frame, self._video_total_frames - 1))

        # Saltar al frame deseado
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        self._video_current_frame = target_frame
        logger.info("Seek a frame %d (%.1f%%)", target_frame, value * 100)

        self._user_seeking = False

        # Si estaba pausado, mostrar el frame en el que aterrizó
        if self._video_paused:
            ret, frame = self._cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                try:
                    frame, results_mp, exercise_result = self.tracker.process_frame(frame)
                    color = exercise_result.color_bgr if exercise_result else None
                    annotated = self.tracker.pose_detector.draw_landmarks(frame, results_mp, color=color)
                    if exercise_result:
                        pass # La información ya se muestra en el sidebar izquierdo
                    self._display_frame(annotated)
                    self._update_metrics_display(exercise_result)
                except Exception as e:
                    logger.error("Error previsualizando frame seek: %s", e)
                    self._display_frame(frame)
                # Retroceder 1 frame para que el loop siga desde aquí
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        else:
            # Si no estaba pausado, reanudar reproducción desde la nueva posición
            if self._is_running:
                if self._after_id is not None:
                    self.after_cancel(self._after_id)
                    self._after_id = None
                self._update_video_frame()

    def _update_video_progress(self) -> None:
        """Actualiza el slider de progreso y el label de tiempo."""
        if self._video_total_frames <= 0 or self._user_seeking:
            return  # No sobreescribir mientras el usuario arrastra
        progress = self._video_current_frame / self._video_total_frames
        self._video_progress.set(min(progress, 1.0))

        # Calcular tiempos
        elapsed_s = self._video_current_frame / self._video_fps
        total_s = self._video_total_frames / self._video_fps
        elapsed_str = f"{int(elapsed_s // 60):02d}:{int(elapsed_s % 60):02d}"
        total_str = f"{int(total_s // 60):02d}:{int(total_s % 60):02d}"
        self._video_time_label.configure(
            text=f"{elapsed_str} / {total_str}",
            text_color="#aaaaaa",
        )

    def _update_video_frame(self) -> None:
        """Loop de procesamiento de video pregrabado frame a frame."""
        if not self._is_running or not self._video_mode:
            return

        # Si está pausado, no avanzar; solo reprogramar para detectar reanudación
        if self._video_paused:
            return

        if self._cap is None or not self._cap.isOpened():
            self._on_video_finished()
            return

        ret, frame = self._cap.read()
        if not ret:
            # Fin del video
            self._on_video_finished()
            return

        self._video_current_frame = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Espejo horizontal para consistencia con modo cámara
        frame = cv2.flip(frame, 1)

        try:
            # Procesar con el tracker
            frame, results_mp, exercise_result = self.tracker.process_frame(frame)

            # Dibujar esqueleto
            color = exercise_result.color_bgr if exercise_result else None
            annotated_frame = self.tracker.pose_detector.draw_landmarks(frame, results_mp, color=color)

            # La info del ejercicio ahora solo se muestra en el sidebar

            # Guardar frame anotado si está configurado
            # Mostrar en GUI
            self._display_frame(annotated_frame)
            self._update_metrics_display(exercise_result)

        except Exception as e:
            logger.error("Error procesando frame %d: %s", self._video_current_frame, e)
            # Mostrar el frame sin anotar para no detener la reproducción
            self._display_frame(frame)

        self._update_video_progress()

        # Calcular intervalo según velocidad (ms entre frames)
        interval_ms = max(1, int((1000.0 / self._video_fps) / self._playback_speed))
        self._after_id = self.after(interval_ms, self._update_video_frame)

    def _on_video_finished(self) -> None:
        """Maneja el fin natural del video pregrabado."""
        logger.info("Video finalizado. Total frames procesados: %d", self._video_current_frame)

        # Cerrar el writer si estaba guardando
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self._is_running = False
        self._video_paused = False
        self._play_pause_btn.configure(state="disabled", text="✅ Finalizado")
        self._video_progress.set(1.0)

        # Notificar al usuario
        self._show_info("Análisis de video completado.")

    # ──────────────────────────────────────────────
    #  Loop de Video
    # ──────────────────────────────────────────────

    def _update_video(self) -> None:
        """Ciclo principal de actualización del video.

        Lee un frame de la cámara, lo procesa con el tracker,
        y lo muestra en la GUI. Se auto-programa con `after()`.
        """
        if not self._is_running:
            return

        try:
            if self._cap is not None and self._cap.isOpened():
                ret, frame = self._cap.read()

                if ret and frame is not None:
                    # Espejo horizontal para feedback natural
                    frame = cv2.flip(frame, 1)

                    # 1. Procesar frame con el tracker (Lógica y Evaluación)
                    frame, results_mp, exercise_result = self.tracker.process_frame(frame)

                    # 2. Dibujar esqueleto (Uso de PoseDetector para visualización)
                    color = exercise_result.color_bgr if exercise_result else None
                    annotated_frame = self.tracker.pose_detector.draw_landmarks(
                        frame, results_mp, color=color
                    )

                    # 3. Dibujar información del ejercicio (Uso de DrawingUtils)

                    # 4. Convertir BGR → RGB y mostrar
                    self._display_frame(annotated_frame)

                    # 5. Actualizar métricas en la sidebar
                    self._update_metrics_display(exercise_result)

                else:
                    logger.warning("No se pudo leer frame de la cámara")

        except Exception as e:
            logger.error("Error en el loop de video: %s", e)

        # Re-programar la siguiente actualización
        if self._is_running:
            self._after_id = self.after(
                VIDEO_UPDATE_INTERVAL_MS, self._update_video
            )

    def _display_frame(self, frame: np.ndarray) -> None:
        """Convierte y muestra un frame de OpenCV en el widget.

        Args:
            frame: Frame BGR de OpenCV.
        """
        try:
            # Obtener tamaño disponible del área de video
            video_width = self._video_frame.winfo_width() - 10
            video_height = self._video_frame.winfo_height() - 10

            if video_width <= 0 or video_height <= 0:
                return

            # Mantener relación de aspecto
            h, w = frame.shape[:2]
            scale = min(video_width / w, video_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            if new_w <= 0 or new_h <= 0:
                return

            # Redimensionar
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # BGR → RGB → PIL → CTkImage
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            ctk_image = ctk.CTkImage(
                light_image=pil_image,
                dark_image=pil_image,
                size=(new_w, new_h),
            )

            self._video_label.configure(image=ctk_image, text="")
            self._current_image = ctk_image

        except Exception as e:
            logger.error("Error al mostrar frame: %s", e)

    # ──────────────────────────────────────────────
    #  Actualización de Métricas
    # ──────────────────────────────────────────────

    def _update_metrics_display(self, result) -> None:
        """Actualiza los widgets de métricas con los datos del resultado.

        Args:
            result: ExerciseResult o None.
        """
        if result is None:
            self._rep_frame._value_label.configure(text="0")  # type: ignore
            self._angle_frame._value_label.configure(text="-- °")  # type: ignore
            self._state_frame._value_label.configure(text="Inactivo")  # type: ignore
            self._form_indicator.configure(
                text="● CORRECTA", text_color="#00ff88"
            )
            self._feedback_label.configure(text="")
            return

        # Repeticiones
        self._rep_frame._value_label.configure(  # type: ignore
            text=str(result.rep_count)
        )

        # Ángulo
        self._angle_frame._value_label.configure(  # type: ignore
            text=f"{result.angle:.1f}°"
        )

        # Estado
        self._state_frame._value_label.configure(  # type: ignore
            text=result.state
        )

        # Forma
        if not result.state or "detectado" in result.state.lower() or "esperando" in result.state.lower() or "visible" in result.state.lower():
            self._form_indicator.configure(
                text="● ESPERANDO",
                text_color="#AAAAAA",
            )
        elif result.form_ok:
            self._form_indicator.configure(
                text="● CORRECTA",
                text_color="#00ff88",
            )
        else:
            self._form_indicator.configure(
                text="● INCORRECTA",
                text_color="#ff4444",
            )

        # Feedback
        self._feedback_label.configure(text=result.feedback_message)

    # ──────────────────────────────────────────────
    #  Utilidades
    # ──────────────────────────────────────────────

    def _show_error(self, message: str) -> None:
        """Muestra un diálogo de error."""
        self._show_dialog("⚠️ Error", message, "#ff4444")

    def _show_info(self, message: str) -> None:
        """Muestra un diálogo informativo."""
        self._show_dialog("✅ Listo", message, "#00ff88")

    def _show_dialog(self, title: str, message: str, title_color: str) -> None:
        """Muestra un diálogo modal genérico.

        Args:
            title: Título del diálogo.
            message: Mensaje a mostrar.
            title_color: Color del texto del título.
        """
        dialog = ctk.CTkToplevel(self)
        dialog.title(title.replace("⚠️ ", "").replace("✅ ", ""))
        dialog.geometry("420x170")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        # Centrar en pantalla
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - 420) // 2
        y = (dialog.winfo_screenheight() - 170) // 2
        dialog.geometry(f"+{x}+{y}")

        ctk.CTkLabel(
            dialog,
            text=title,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=title_color,
        ).pack(pady=(15, 5))

        ctk.CTkLabel(
            dialog,
            text=message,
            font=ctk.CTkFont(size=12),
            wraplength=380,
        ).pack(pady=5)

        ctk.CTkButton(
            dialog,
            text="Aceptar",
            command=dialog.destroy,
            width=100,
        ).pack(pady=10)

    def _on_close(self) -> None:
        """Maneja el cierre de la ventana."""
        self._on_stop()
        self._on_stop_video()
        self.tracker.audio.stop()
        self.tracker.pose_detector.release()
        self.destroy()
        sys.exit(0)
