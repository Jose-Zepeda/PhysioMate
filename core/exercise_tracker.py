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

        else:
            # Sin ejercicio activo: dibujar esqueleto con color por defecto
            annotated_frame = self.pose_detector.draw_landmarks(
                frame, results
            )

        return annotated_frame, exercise_result

    @staticmethod
    def _ascii(text: str) -> str:
        """Elimina tildes y caracteres especiales que OpenCV no puede renderizar."""
        replacements = {
            "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
            "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
            "ü": "u", "Ü": "U", "ñ": "n", "Ñ": "N",
            "¡": "!", "¿": "?",
        }
        for orig, rep in replacements.items():
            text = text.replace(orig, rep)
        return text

    @staticmethod
    def _wrap_text(text: str, max_chars: int = 28) -> list:
        """Parte un texto largo en líneas de max_chars caracteres."""
        words = text.split()
        lines = []
        current = ""
        for word in words:
            if len(current) + len(word) + 1 <= max_chars:
                current = (current + " " + word).strip()
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def _draw_exercise_info(
        self,
        frame: np.ndarray,
        result: ExerciseResult,
    ) -> np.ndarray:
        """Dibuja información del ejercicio superpuesta en el frame."""
        h, w, _ = frame.shape

        # Parámetros de diseño del panel — siempre en la esquina superior izquierda
        PANEL_X = 10
        PANEL_W = 300
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SM = 0.58
        FONT_MD = 0.68
        THICK = 2
        LINE_H = 32  # altura entre líneas

        # Calcular cuántas líneas necesita el feedback
        feedback_lines = self._wrap_text(self._ascii(result.feedback_message)) if result.feedback_message else []
        panel_h = 115 + max(len(feedback_lines), 1) * LINE_H

        # Fondo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (PANEL_X, 8), (PANEL_X + PANEL_W, panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # Borde del panel
        cv2.rectangle(frame, (PANEL_X, 8), (PANEL_X + PANEL_W, panel_h), (80, 80, 80), 1)

        tx = PANEL_X + 10
        y = 38

        # ── Ángulo ──
        cv2.putText(frame, f"Angulo: {result.angle:.1f} grados", (tx, y),
                    FONT, FONT_MD, (255, 255, 255), THICK)
        y += LINE_H

        # ── Estado ──
        state_text = self._ascii(f"Estado: {result.state}")
        cv2.putText(frame, state_text, (tx, y),
                    FONT, FONT_MD, result.color_bgr, THICK)
        y += LINE_H

        # ── Repeticiones ──
        cv2.putText(frame, f"Reps: {result.rep_count}", (tx, y),
                    FONT, FONT_MD, (0, 255, 255), THICK)
        y += LINE_H

        # ── Separador ──
        cv2.line(frame, (tx, y - 8), (PANEL_X + PANEL_W - 10, y - 8), (80, 80, 80), 1)

        # ── Feedback (multi-línea) ──
        if feedback_lines:
            fb_color = (0, 200, 0) if result.form_ok else (50, 50, 255)
            # Si es mensaje de posición/warning usar naranja
            state_lower = result.state.lower()
            if "no detectado" in state_lower or "visible" in state_lower or "esperando" in state_lower:
                fb_color = (0, 165, 255)  # naranja
            for line in feedback_lines:
                cv2.putText(frame, line, (tx, y), FONT, FONT_SM, fb_color, THICK)
                y += LINE_H

        return frame

    @property
    def last_result(self) -> Optional[ExerciseResult]:
        """Retorna el último resultado de evaluación."""
        return self._last_result
