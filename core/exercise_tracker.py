"""
PhysioMate - Rastreador de Ejercicios (Exercise Tracker)

Orquesta la interacción entre el detector de pose, el ejercicio activo
y el sistema de audio. Actúa como intermediario (mediator) para desacoplar
los componentes siguiendo el Principio de Inversión de Dependencias.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from core.audio_feedback import AudioFeedback
from core.pose_detector import PoseDetector
from exercises.base import (
    ExerciseBase,
    ExerciseResult,
    create_exercise,
    get_available_exercises,
)

logger = logging.getLogger(__name__)


class ExerciseTracker:
    """Orquestador principal del sistema de seguimiento de ejercicios.

    Coordina la detección de pose, evaluación de ejercicios, dibujo
    de anotaciones y retroalimentación de audio. No tiene dependencia
    directa con la GUI (Inversión de Dependencias).

    Attributes:
        pose_detector: Detector de pose MediaPipe.
        audio: Sistema de feedback de audio.
        current_exercise: Ejercicio activo actual.
    """

    def __init__(
        self,
        pose_detector: PoseDetector,
        audio: AudioFeedback,
        initial_exercise: Optional[str] = None,
    ) -> None:
        """Inicializa el tracker con sus dependencias inyectadas.

        Args:
            pose_detector: Instancia del detector de pose.
            audio: Instancia del sistema de audio.
            initial_exercise: Nombre del ejercicio inicial a cargar.
        """
        self.pose_detector = pose_detector
        self.audio = audio
        self.current_exercise: Optional[ExerciseBase] = None

        self._last_result: Optional[ExerciseResult] = None

        # Cargar ejercicio inicial si se especifica
        if initial_exercise:
            self.set_exercise(initial_exercise)
        else:
            # Cargar el primero disponible
            available = get_available_exercises()
            if available:
                self.set_exercise(available[0])

    def set_exercise(self, name: str) -> bool:
        """Cambia el ejercicio activo.

        Args:
            name: Nombre del ejercicio registrado.

        Returns:
            True si el ejercicio se cambió exitosamente.
        """
        exercise = create_exercise(name)
        if exercise is None:
            logger.warning("Ejercicio '%s' no encontrado en el registro", name)
            return False

        self.current_exercise = exercise
        self._last_result = None
        logger.info("Ejercicio cambiado a: %s", name)
        return True

    def reset_exercise(self) -> None:
        """Reinicia los contadores del ejercicio actual."""
        if self.current_exercise is not None:
            self.current_exercise.reset()
            self._last_result = None

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[ExerciseResult]]:
        """Procesa un frame completo: detección, evaluación y anotación.

        Este es el método principal llamado por la GUI en cada ciclo.

        Args:
            frame: Frame BGR de OpenCV.

        Returns:
            Tupla de (frame anotado, resultado del ejercicio).
        """
        if frame is None or frame.size == 0:
            return frame, None

        # 1. Detectar pose
        results = self.pose_detector.detect(frame)

        # 2. Evaluar ejercicio
        exercise_result: Optional[ExerciseResult] = None

        if self.current_exercise is not None and results is not None:
            landmarks = results.pose_landmarks
            exercise_result = self.current_exercise.evaluate(
                landmarks, frame.shape
            )
            self._last_result = exercise_result

            # 3. Dibujar esqueleto con color según la forma
            color = exercise_result.color_bgr if exercise_result else None
            annotated_frame = self.pose_detector.draw_landmarks(
                frame, results, color=color
            )

            # 4. Dibujar información del ejercicio sobre el frame
            annotated_frame = self._draw_exercise_info(
                annotated_frame, exercise_result
            )

            # 5. Producir feedback de audio
            if exercise_result and exercise_result.feedback_message:
                self.audio.speak(exercise_result.feedback_message)

        else:
            # Sin ejercicio activo: dibujar esqueleto con color por defecto
            annotated_frame = self.pose_detector.draw_landmarks(
                frame, results
            )

        return annotated_frame, exercise_result

    def _draw_exercise_info(
        self,
        frame: np.ndarray,
        result: ExerciseResult,
    ) -> np.ndarray:
        """Dibuja información del ejercicio superpuesta en el frame.

        Incluye ángulo actual, estado, contador de repeticiones
        y mensaje de feedback.

        Args:
            frame: Frame donde dibujar.
            result: Resultado de la evaluación del ejercicio.

        Returns:
            Frame con la información dibujada.
        """
        h, w, _ = frame.shape

        # ── Fondo semitransparente para info ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (330, 150), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # ── Ángulo ──
        cv2.putText(
            frame,
            f"Angulo: {result.angle:.1f} grados",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # ── Estado ──
        state_color = result.color_bgr
        cv2.putText(
            frame,
            f"Estado: {result.state}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            state_color,
            2,
        )

        # ── Repeticiones ──
        cv2.putText(
            frame,
            f"Reps: {result.rep_count}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        # ── Feedback ──
        if result.feedback_message:
            feedback_color = (0, 200, 0) if result.form_ok else (0, 0, 255)
            cv2.putText(
                frame,
                result.feedback_message,
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                feedback_color,
                2,
            )

        return frame

    @property
    def last_result(self) -> Optional[ExerciseResult]:
        """Retorna el último resultado de evaluación."""
        return self._last_result
