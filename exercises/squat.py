"""
PhysioMate - Ejercicio: Sentadillas (Squats)
"""

from __future__ import annotations

from typing import Any, Tuple

import mediapipe as mp

from core.math_utils import MathUtils
from exercises.base import ExerciseBase, ExerciseResult, register_exercise

_mp_pose = mp.solutions.pose.PoseLandmark

_LEFT_HIP = _mp_pose.LEFT_HIP
_LEFT_KNEE = _mp_pose.LEFT_KNEE
_LEFT_ANKLE = _mp_pose.LEFT_ANKLE
_RIGHT_HIP = _mp_pose.RIGHT_HIP
_RIGHT_KNEE = _mp_pose.RIGHT_KNEE
_RIGHT_ANKLE = _mp_pose.RIGHT_ANKLE
_LEFT_SHOULDER = _mp_pose.LEFT_SHOULDER
_RIGHT_SHOULDER = _mp_pose.RIGHT_SHOULDER

COLOR_OK = (0, 220, 0)
COLOR_BAD = (0, 0, 220)
COLOR_WARNING = (0, 165, 255)

@register_exercise
class SquatExercise(ExerciseBase):
    ANGLE_UP_THRESHOLD = 150.0   # Casi recto = Arriba
    ANGLE_DOWN_THRESHOLD = 110.0 # Flexionado = Abajo
    VISIBILITY_THRESHOLD = 0.5   

    def __init__(self) -> None:
        self._rep_count: int = 0
        self._state: str = "Esperando..."
        self._form_ok: bool = True
        self._direction: str = ""

    @property
    def name(self) -> str:
        return "Sentadillas"

    @property
    def description(self) -> str:
        return "Flexión y extensión de rodillas. Mantén la espalda recta."

    def reset(self) -> None:
        self._rep_count = 0
        self._state = "Esperando..."
        self._form_ok = True
        self._direction = ""

    def evaluate(self, landmarks: Any, frame_shape: Tuple[int, int, int]) -> ExerciseResult:
        if landmarks is None:
            return ExerciseResult(state="No detectado", feedback_message="Cuerpo completo en cámara", color_bgr=COLOR_WARNING)

        lm = landmarks.landmark
        h, w, _ = frame_shape

        # Determinar qué lado es más visible
        left_vis = min(lm[_LEFT_HIP].visibility, lm[_LEFT_KNEE].visibility, lm[_LEFT_ANKLE].visibility)
        right_vis = min(lm[_RIGHT_HIP].visibility, lm[_RIGHT_KNEE].visibility, lm[_RIGHT_ANKLE].visibility)

        if left_vis > right_vis and left_vis > self.VISIBILITY_THRESHOLD:
            hip = lm[_LEFT_HIP]
            knee = lm[_LEFT_KNEE]
            ankle = lm[_LEFT_ANKLE]
            shoulder = lm[_LEFT_SHOULDER]
            landmark_ids = [_LEFT_HIP, _LEFT_KNEE, _LEFT_ANKLE, _LEFT_SHOULDER]
        elif right_vis > self.VISIBILITY_THRESHOLD:
            hip = lm[_RIGHT_HIP]
            knee = lm[_RIGHT_KNEE]
            ankle = lm[_RIGHT_ANKLE]
            shoulder = lm[_RIGHT_SHOULDER]
            landmark_ids = [_RIGHT_HIP, _RIGHT_KNEE, _RIGHT_ANKLE, _RIGHT_SHOULDER]
        else:
            return ExerciseResult(state="Piernas no visibles", feedback_message="Aleja la cámara", color_bgr=COLOR_WARNING)

        hip_pt = (hip.x * w, hip.y * h)
        knee_pt = (knee.x * w, knee.y * h)
        ankle_pt = (ankle.x * w, ankle.y * h)
        shoulder_pt = (shoulder.x * w, shoulder.y * h)

        # Angulo rodilla
        angle = MathUtils.calculate_angle(hip_pt, knee_pt, ankle_pt)

        # Angulo cadera-hombro vs suelo para la forma
        torso_angle = MathUtils.calculate_angle(shoulder_pt, hip_pt, (hip_pt[0], hip_pt[1] - 100)) # vertical
        
        form_ok = True
        feedback = ""

        # Si el torso se inclina demasiado hacia adelante
        if torso_angle > 45:
            form_ok = False
            feedback = "Mantén el pecho más arriba"

        new_state = self._state

        if angle > self.ANGLE_UP_THRESHOLD:
            new_state = "Arriba"
            # Contar la repetición al volver arriba después de bajar profundamente
            if self._direction == "subiendo" and self._form_ok:
                self._rep_count += 1
                self._direction = ""
            if not feedback: feedback = "Ahora baja la cadera"
        elif angle < self.ANGLE_DOWN_THRESHOLD:
            new_state = "Abajo"
            if not feedback: feedback = "¡Sube!"
        else:
            # Transición
            if self._state == "Abajo": new_state = "Subiendo"; self._direction = "subiendo"
            elif self._state == "Arriba": new_state = "Bajando"; self._direction = "bajando"
            elif self._direction == "subiendo": new_state = "Subiendo"
            elif self._direction == "bajando": new_state = "Bajando"

        self._state = new_state
        self._form_ok = form_ok

        color = COLOR_BAD if not form_ok else COLOR_OK

        return ExerciseResult(
            angle=angle,
            rep_count=self._rep_count,
            state=new_state,
            form_ok=form_ok,
            feedback_message=feedback,
            color_bgr=color,
            landmarks_to_draw=landmark_ids,
        )
