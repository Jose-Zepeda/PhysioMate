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
from core.math_utils import SmoothingFilter
from core.config import DetectionConfig
from exercises.base import (
    ExerciseBase,
    ExerciseResult,
    create_exercise,
    get_available_exercises,
)

logger = logging.getLogger(__name__)


class ExerciseTracker:
    """Orquestador principal del sistema de seguimiento de ejercicios.

    Coordina la detección de pose, evaluación de ejercicios y
    retroalimentación de audio. Se ha desacoplado la lógica de dibujo
    siguiendo SRP.

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
        config: DetectionConfig = DetectionConfig(),
    ) -> None:
        """Inicializa el tracker con sus dependencias inyectadas.

        Args:
            pose_detector: Instancia del detector de pose.
            audio: Instancia del sistema de audio.
            initial_exercise: Nombre del ejercicio inicial a cargar.
            config: Configuración de detección.
        """
        self.pose_detector = pose_detector
        self.audio = audio
        self.current_exercise: Optional[ExerciseBase] = None
        self._config = config

        # Filtro de suavizado para el ángulo
        self._angle_filter = SmoothingFilter(alpha=self._config.SMOOTHING_ALPHA)

        self._last_result: Optional[ExerciseResult] = None
        self._last_rep_count: int = 0

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
        self._angle_filter.reset()
        logger.info("Ejercicio cambiado a: %s", name)
        return True

    def reset_exercise(self) -> None:
        """Reinicia los contadores del ejercicio actual."""
        if self.current_exercise is not None:
            self.current_exercise.reset()
            self._last_result = None
            self._angle_filter.reset()

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Any, Optional[ExerciseResult]]:
        """Procesa un frame: detección y evaluación.

        Ya no dibuja información del ejercicio directamente (SRP).
        Retorna los resultados de MediaPipe y el resultado del ejercicio
        para que la UI decida cómo dibujar.

        Args:
            frame: Frame BGR de OpenCV.

        Returns:
            Tupla de (frame, results_mp, exercise_result).
        """
        if frame is None or frame.size == 0:
            return frame, None, None

        # 1. Detectar pose
        results = self.pose_detector.detect(frame)

        # 2. Evaluar ejercicio
        exercise_result: Optional[ExerciseResult] = None

        if self.current_exercise is not None and results is not None:
            landmarks = results.pose_landmarks
            exercise_result = self.current_exercise.evaluate(
                landmarks, frame.shape
            )

            # 3. Aplicar suavizado al ángulo detectado
            if exercise_result:
                exercise_result.angle = self._angle_filter.update(exercise_result.angle)
            
            self._last_result = exercise_result

            # 4. Producir feedback de audio
            if exercise_result and exercise_result.feedback_message:
                state_lower = exercise_result.state.lower()
                is_positioning_msg = (
                    "no detectado" in state_lower
                    or "no visible" in state_lower
                    or "esperando" in state_lower
                    or "piernas" in state_lower
                )
                # Hablar si la postura está mal O si se necesita reposicionarse
                if not exercise_result.form_ok or is_positioning_msg:
                    self.audio.speak(exercise_result.feedback_message)
                # Anunciar el número de reps al completar una
                elif self._last_rep_count is not None and exercise_result.rep_count > self._last_rep_count:
                    self.audio.speak(str(exercise_result.rep_count))

            self._last_rep_count = exercise_result.rep_count if exercise_result else 0

        return frame, results, exercise_result

    @property
    def last_result(self) -> Optional[ExerciseResult]:
        """Retorna el último resultado de evaluación."""
        return self._last_result
