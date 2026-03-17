"""
PhysioMate - Motor de Inferencia de Pose (Pose Engine)

Envuelve MediaPipe Pose para detectar landmarks corporales y dibujar
el esqueleto digital sobre el frame de video.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class PoseDetector:
    """Detector de pose corporal basado en MediaPipe Pose.

    Encapsula la configuración, detección y dibujo de landmarks.
    Aplica el Principio de Responsabilidad Única: solo se encarga
    de la inferencia de pose, sin lógica de ejercicios ni GUI.

    Attributes:
        min_detection_confidence: Confianza mínima para la detección inicial.
        min_tracking_confidence: Confianza mínima para el seguimiento.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ) -> None:
        """Inicializa el detector de pose con MediaPipe.

        Args:
            min_detection_confidence: Umbral de confianza para detección [0,1].
            min_tracking_confidence: Umbral de confianza para seguimiento [0,1].
            model_complexity: Complejidad del modelo (0, 1 o 2).

        Raises:
            RuntimeError: Si MediaPipe Pose no puede inicializarse.
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity

        self._mp_pose = mp.solutions.pose
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        try:
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        except Exception as e:
            raise RuntimeError(
                f"No se pudo inicializar MediaPipe Pose: {e}"
            ) from e

    def detect(self, frame: np.ndarray) -> Any:
        """Ejecuta la inferencia de pose sobre un frame BGR.

        Convierte el frame a RGB internamente para MediaPipe.

        Args:
            frame: Imagen BGR de OpenCV (np.ndarray).

        Returns:
            Objeto de resultados de MediaPipe con los landmarks detectados,
            o None si la detección falla.
        """
        if frame is None or frame.size == 0:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._pose.process(frame_rgb)
        frame_rgb.flags.writeable = True
        return results

    def draw_landmarks(
        self,
        frame: np.ndarray,
        results: Any,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Dibuja los landmarks y conexiones del esqueleto sobre el frame.

        Args:
            frame: Imagen BGR sobre la que dibujar.
            results: Resultados de MediaPipe Pose.
            color: Color BGR opcional para los landmarks y conexiones.
                   Si es None, usa los estilos por defecto de MediaPipe.

        Returns:
            Frame con los landmarks dibujados.
        """
        if results is None or results.pose_landmarks is None:
            return frame

        output = frame.copy()

        if color is not None:
            # Crear especificaciones de dibujo personalizadas
            landmark_spec = self._mp_drawing.DrawingSpec(
                color=color, thickness=3, circle_radius=3
            )
            connection_spec = self._mp_drawing.DrawingSpec(
                color=color, thickness=2, circle_radius=2
            )
            self._mp_drawing.draw_landmarks(
                output,
                results.pose_landmarks,
                self._mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=connection_spec,
            )
        else:
            self._mp_drawing.draw_landmarks(
                output,
                results.pose_landmarks,
                self._mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self._mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        return output

    def get_landmark_coords(
        self,
        results: Any,
        landmark_id: int,
        frame_shape: Tuple[int, int, int],
    ) -> Optional[Tuple[float, float]]:
        """Obtiene las coordenadas en píxeles de un landmark específico.

        Args:
            results: Resultados de MediaPipe Pose.
            landmark_id: Índice del landmark en el enum PoseLandmark.
            frame_shape: Forma del frame (height, width, channels).

        Returns:
            Tupla (x, y) en coordenadas de píxel, o None si no se encontró.
        """
        if results is None or results.pose_landmarks is None:
            return None

        try:
            landmark = results.pose_landmarks.landmark[landmark_id]
            h, w, _ = frame_shape
            x = landmark.x * w
            y = landmark.y * h
            return (x, y)
        except (IndexError, AttributeError):
            return None

    def get_landmark_visibility(
        self,
        results: Any,
        landmark_id: int,
    ) -> float:
        """Obtiene la visibilidad de un landmark específico.

        Args:
            results: Resultados de MediaPipe Pose.
            landmark_id: Índice del landmark.

        Returns:
            Valor de visibilidad [0, 1], o 0.0 si no se encontró.
        """
        if results is None or results.pose_landmarks is None:
            return 0.0

        try:
            return float(results.pose_landmarks.landmark[landmark_id].visibility)
        except (IndexError, AttributeError):
            return 0.0

    def release(self) -> None:
        """Libera los recursos de MediaPipe."""
        if hasattr(self, "_pose") and self._pose is not None:
            self._pose.close()

    def __del__(self) -> None:
        """Destructor: libera recursos al ser recolectado."""
        self.release()
