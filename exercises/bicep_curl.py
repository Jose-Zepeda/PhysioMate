"""
PhysioMate - Ejercicio: Curl de Bíceps

Implementación completa del ejercicio de curl de bíceps con:
- Detección de estados "Arriba" y "Abajo"
- Conteo de repeticiones correctas
- Detección de mala forma (movimiento del codo)
- Retroalimentación visual y auditiva
"""

from __future__ import annotations

from typing import Any, Tuple

import mediapipe as mp

from core.math_utils import MathUtils
from core.config import UIConfig
from exercises.base import ExerciseBase, ExerciseResult, register_exercise

# IDs de landmarks de MediaPipe Pose
_mp_pose = mp.solutions.pose.PoseLandmark

# ... (sin cambios en las constantes de landmarks) ...

# ─── Colores BGR desde Config ───
_UI = UIConfig()
COLOR_OK = _UI.COLOR_OK
COLOR_BAD = _UI.COLOR_BAD
COLOR_WARNING = _UI.COLOR_WARNING


@register_exercise
class BicepCurlExercise(ExerciseBase):
    """Ejercicio de Curl de Bíceps con tracking bilateral.

    Detecta el brazo más visible y evalúa:
    - Ángulo del codo (hombro → codo → muñeca)
    - Estado "Abajo" (brazo extendido > 160°) y "Arriba" (brazo flexionado < 40°)
    - Forma correcta: el codo no debe moverse hacia adelante
      (verificado comparando la posición X del codo con el hombro)
    - Cuenta repeticiones solo en transiciones válidas con buena forma
    """

    # ── Umbrales configurables ──
    ANGLE_UP_THRESHOLD: float = 40.0      # Ángulo máximo para "Arriba"
    ANGLE_DOWN_THRESHOLD: float = 160.0   # Ángulo mínimo para "Abajo"
    ELBOW_DRIFT_TOLERANCE: float = 0.04   # Tolerancia de desplazamiento del codo
    VISIBILITY_THRESHOLD: float = 0.5     # Visibilidad mínima de landmarks

    def __init__(self) -> None:
        """Inicializa el ejercicio con contadores en cero."""
        self._rep_count: int = 0
        self._state: str = "Esperando..."
        self._prev_state: str = ""
        self._form_ok: bool = True
        self._direction: str = ""  # "subiendo" o "bajando"

    @property
    def name(self) -> str:
        return "Curl de Bíceps"

    @property
    def description(self) -> str:
        return (
            "Flexión y extensión del brazo para fortalecer el bíceps. "
            "Mantén el codo pegado al costado del cuerpo."
        )

    def reset(self) -> None:
        """Reinicia todos los contadores y estados."""
        self._rep_count = 0
        self._state = "Esperando..."
        self._prev_state = ""
        self._form_ok = True
        self._direction = ""

    def evaluate(
        self,
        landmarks: Any,
        frame_shape: Tuple[int, int, int],
    ) -> ExerciseResult:
        """Evalúa un frame para el ejercicio de curl de bíceps.

        Detecta automáticamente qué brazo es más visible y evalúa
        el ángulo, estado, forma y retroalimentación.

        Args:
            landmarks: MediaPipe pose_landmarks.
            frame_shape: (height, width, channels) del frame.

        Returns:
            ExerciseResult con todos los datos de la evaluación.
        """
        if landmarks is None:
            return ExerciseResult(
                state="No detectado",
                feedback_message="Colócate frente a la cámara",
                color_bgr=COLOR_WARNING,
            )

        lm = landmarks.landmark
        h, w, _ = frame_shape

        # ── Determinar brazo más visible ──
        left_vis = min(
            lm[_LEFT_SHOULDER].visibility,
            lm[_LEFT_ELBOW].visibility,
            lm[_LEFT_WRIST].visibility,
        )
        right_vis = min(
            lm[_RIGHT_SHOULDER].visibility,
            lm[_RIGHT_ELBOW].visibility,
            lm[_RIGHT_WRIST].visibility,
        )

        if left_vis > right_vis and left_vis > self.VISIBILITY_THRESHOLD:
            shoulder = lm[_LEFT_SHOULDER]
            elbow = lm[_LEFT_ELBOW]
            wrist = lm[_LEFT_WRIST]
            hip = lm[_LEFT_HIP]
            side = "izquierdo"
            landmark_ids = [_LEFT_SHOULDER, _LEFT_ELBOW, _LEFT_WRIST]
        elif right_vis > self.VISIBILITY_THRESHOLD:
            shoulder = lm[_RIGHT_SHOULDER]
            elbow = lm[_RIGHT_ELBOW]
            wrist = lm[_RIGHT_WRIST]
            hip = lm[_RIGHT_HIP]
            side = "derecho"
            landmark_ids = [_RIGHT_SHOULDER, _RIGHT_ELBOW, _RIGHT_WRIST]
        else:
            return ExerciseResult(
                state="Brazo no visible",
                feedback_message="Muestra tu brazo a la cámara",
                color_bgr=COLOR_WARNING,
            )

        # ── Calcular ángulo del codo ──
        shoulder_pt = (shoulder.x * w, shoulder.y * h)
        elbow_pt = (elbow.x * w, elbow.y * h)
        wrist_pt = (wrist.x * w, wrist.y * h)
        hip_pt = (hip.x * w, hip.y * h)

        angle = MathUtils.calculate_angle(shoulder_pt, elbow_pt, wrist_pt)

        # ── Verificar forma: codo no debe avanzar ──
        # Comparar posición X del codo vs hombro
        # Si el codo se mueve significativamente hacia adelante del hombro,
        # la forma es incorrecta
        elbow_drift = abs(elbow.x - shoulder.x)
        # También verificar alineación vertical codo-hombro-cadera
        shoulder_hip_angle = MathUtils.calculate_angle(
            elbow_pt, shoulder_pt, hip_pt
        )

        form_ok = True
        feedback = ""

        if elbow_drift > self.ELBOW_DRIFT_TOLERANCE * 3 and shoulder_hip_angle > 40:
            form_ok = False
            feedback = "Mantén el codo fijo"

        # ── Determinar estado ──
        new_state = self._state

        if angle > self.ANGLE_DOWN_THRESHOLD:
            new_state = "Abajo"
            if self._direction == "bajando" and self._form_ok:
                # Transición completa: contar repetición
                self._rep_count += 1
                self._direction = ""
            if not feedback:
                feedback = "Sube el brazo"
        elif angle < self.ANGLE_UP_THRESHOLD:
            new_state = "Arriba"
            if not feedback:
                feedback = "¡Bien! Ahora baja"
        else:
            # En transición
            if self._state == "Abajo":
                new_state = "Subiendo"
                self._direction = "subiendo"
                if not feedback:
                    feedback = "Sube más"
            elif self._state == "Arriba":
                new_state = "Bajando"
                self._direction = "bajando"
                if not feedback:
                    feedback = "Estira el brazo"
            else:
                # Mantener estado intermedio
                if self._direction == "subiendo":
                    new_state = "Subiendo"
                    if not feedback:
                        feedback = "Sube más"
                elif self._direction == "bajando":
                    new_state = "Bajando"
                    if not feedback:
                        feedback = "Estira el brazo"

        self._prev_state = self._state
        self._state = new_state
        self._form_ok = form_ok

        # ── Determinar color ──
        if not form_ok:
            color = COLOR_BAD
        elif new_state == "Arriba":
            color = COLOR_OK
        elif new_state == "Abajo":
            color = COLOR_OK
        else:
            color = COLOR_OK

        return ExerciseResult(
            angle=angle,
            rep_count=self._rep_count,
            state=f"{new_state} ({side})",
            form_ok=form_ok,
            feedback_message=feedback,
            color_bgr=color,
            landmarks_to_draw=landmark_ids,
        )
