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
from tkinter import filedialog
from typing import Optional

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from core.audio_feedback import AudioFeedback
from core.camera_detector import CameraDetector
from core.exercise_tracker import ExerciseTracker
from core.pose_detector import PoseDetector
from core.config import UIConfig
from gui.dialogs import show_dialog
from gui.frame_renderer import FrameRenderer
from gui.sidebar import SidebarCallbacks, SidebarView

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
        self._available_cameras: dict = CameraDetector.detect()

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
        """Construye el panel lateral delegando en SidebarView.

        Cablea los callbacks de la app a los controles y recupera de la
        vista los widgets y el presentador de métricas que la app necesita.
        """
        callbacks = SidebarCallbacks(
            on_exercise_change=self._on_exercise_change,
            on_camera_change=self._on_camera_change,
            on_start=self._on_start,
            on_stop=self._on_stop,
            on_reset=self._on_reset,
            on_audio_toggle=self._on_audio_toggle,
            on_load_video=self._on_load_video,
            on_seek_drag=self._on_seek_drag,
            on_seek_start=self._on_seek_start,
            on_seek_release=self._on_seek_release,
            on_video_pause_resume=self._on_video_pause_resume,
            on_stop_video=self._on_stop_video,
            on_speed_change=self._on_speed_change,
        )
        self._sidebar = SidebarView(
            self,
            available_cameras=self._available_cameras,
            speed_options=SPEED_OPTIONS,
            callbacks=callbacks,
            width=SIDEBAR_WIDTH,
        )
        self._sidebar.grid(row=0, column=0, sticky="nsew")

        # Aplicar índice inicial según la cámara preseleccionada por la vista
        self._apply_camera_selection(self._sidebar.default_camera)

        # Exponer el presentador de métricas ya cableado por la vista
        self._metrics = self._sidebar.metrics

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

    # ──────────────────────────────────────────────
    #  Eventos de Control
    # ──────────────────────────────────────────────

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
            self._sidebar.start_btn.configure(state="disabled")
            self._sidebar.stop_btn.configure(state="normal")

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

        self._sidebar.start_btn.configure(state="normal")
        self._sidebar.stop_btn.configure(state="disabled")

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

        exercise_name = self._sidebar.exercise_var.get()
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
        self.tracker.audio.enabled = self._sidebar.audio_var.get()

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
        self._sidebar.video_name_label.configure(text=short_name, text_color="#aaaaaa")

        # Habilitar controles de video
        self._sidebar.play_pause_btn.configure(state="normal", text="⏸ Pausar")
        self._sidebar.stop_video_btn.configure(state="normal")

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
            self._sidebar.play_pause_btn.configure(text="▶ Reanudar")
            logger.info("Video pausado en frame %d", self._video_current_frame)
        else:
            self._sidebar.play_pause_btn.configure(text="⏸ Pausar")
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
        self._sidebar.play_pause_btn.configure(state="disabled", text="⏸ Pausar")
        self._sidebar.stop_video_btn.configure(state="disabled")
        self._sidebar.video_progress.set(0)
        self._sidebar.video_time_label.configure(text="00:00 / 00:00")
        self._sidebar.video_name_label.configure(text="Sin video cargado", text_color="#555555")
        self._video_path = None
        self._update_placeholder_image()
        logger.info("Video detenido y descargado")

    def _on_speed_change(self, value: str) -> None:
        """Actualiza la velocidad de reproducción."""
        self._playback_speed = SPEED_OPTIONS.get(value, 1.0)
        logger.info("Velocidad de reproducción: %s (%.2fx)", value, self._playback_speed)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Formatea segundos como mm:ss."""
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    def _on_seek_drag(self, value: float) -> None:
        """Llamado mientras el usuario arrastra el slider: actualiza solo el tiempo."""
        if not self._video_mode or self._video_total_frames <= 0:
            return
        # Calcular tiempo correspondiente a la posición del slider
        target_frame = int(value * self._video_total_frames)
        elapsed_s = target_frame / max(self._video_fps, 1)
        total_s = self._video_total_frames / max(self._video_fps, 1)
        self._sidebar.video_time_label.configure(
            text=f"{self._format_time(elapsed_s)} / {self._format_time(total_s)}",
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

        value = self._sidebar.video_progress.get()
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
                self._process_and_render(frame)
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
        self._sidebar.video_progress.set(min(progress, 1.0))

        # Calcular tiempos
        elapsed_s = self._video_current_frame / self._video_fps
        total_s = self._video_total_frames / self._video_fps
        self._sidebar.video_time_label.configure(
            text=f"{self._format_time(elapsed_s)} / {self._format_time(total_s)}",
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

        # Procesar y renderizar (espejo + tracker + esqueleto + display + métricas)
        self._process_and_render(frame)

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
        self._sidebar.play_pause_btn.configure(state="disabled", text="✅ Finalizado")
        self._sidebar.video_progress.set(1.0)

        # Notificar al usuario
        self._show_info("Análisis de video completado.")

    # ──────────────────────────────────────────────
    #  Loop de Video
    # ──────────────────────────────────────────────

    def _process_and_render(self, frame: np.ndarray) -> None:
        """Pipeline común de un frame: espejo, tracker, esqueleto, display y métricas.

        Unifica la secuencia que comparten el loop de cámara, el loop de
        video pregrabado y la previsualización del seek. No gestiona el
        scheduling ni el estado del loop (eso sigue en cada caller).

        Args:
            frame: Frame BGR de OpenCV recién leído de la fuente.
        """
        # Espejo horizontal para feedback natural / consistencia con cámara
        frame = cv2.flip(frame, 1)
        try:
            frame, results_mp, exercise_result = self.tracker.process_frame(frame)
            color = exercise_result.color_bgr if exercise_result else None
            annotated = self.tracker.pose_detector.draw_landmarks(
                frame, results_mp, color=color
            )
            self._display_frame(annotated)
            self._update_metrics_display(exercise_result)
        except Exception as e:
            logger.error("Error procesando frame: %s", e)
            # Mostrar el frame sin anotar para no detener la reproducción
            self._display_frame(frame)

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
                    self._process_and_render(frame)
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

        Mide el área de video disponible y delega la conversión a
        FrameRenderer; solo gestiona el widget (lo que depende de self).

        Args:
            frame: Frame BGR de OpenCV.
        """
        # Obtener tamaño disponible del área de video
        video_width = self._video_frame.winfo_width() - 10
        video_height = self._video_frame.winfo_height() - 10

        ctk_image = FrameRenderer.render(frame, video_width, video_height)
        if ctk_image is None:
            return

        self._video_label.configure(image=ctk_image, text="")
        self._current_image = ctk_image

    # ──────────────────────────────────────────────
    #  Actualización de Métricas
    # ──────────────────────────────────────────────

    def _update_metrics_display(self, result) -> None:
        """Actualiza los widgets de métricas delegando en MetricsPresenter.

        Args:
            result: ExerciseResult o None.
        """
        self._metrics.update(result)

    # ──────────────────────────────────────────────
    #  Utilidades
    # ──────────────────────────────────────────────

    def _show_error(self, message: str) -> None:
        """Muestra un diálogo de error."""
        show_dialog(self, "⚠️ Error", message, "#ff4444")

    def _show_info(self, message: str) -> None:
        """Muestra un diálogo informativo."""
        show_dialog(self, "✅ Listo", message, "#00ff88")

    def _on_close(self) -> None:
        """Maneja el cierre de la ventana."""
        self._on_stop()
        self._on_stop_video()
        self.tracker.audio.stop()
        self.tracker.pose_detector.release()
        self.destroy()
        sys.exit(0)
