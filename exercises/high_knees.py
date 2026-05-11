"""
PhysioMate - Ejercicio: Elevación de Rodillas (High Knees)
"""

from __future__ import annotations

from typing import Any, Tuple

import mediapipe as mp

from core.math_utils import MathUtils
from exercises.base import ExerciseBase, ExerciseResult, register_exercise

_mp_pose = mp.solutions.pose.PoseLandmark

_LEFT_SHOULDER = _mp_pose.LEFT_SHOULDER
_LEFT_HIP = _mp_pose.LEFT_HIP
_LEFT_KNEE = _mp_pose.LEFT_KNEE
_RIGHT_SHOULDER = _mp_pose.RIGHT_SHOULDER
_RIGHT_HIP = _mp_pose.RIGHT_HIP
_RIGHT_KNEE = _mp_pose.RIGHT_KNEE

COLOR_OK = (0, 220, 0)
COLOR_BAD = (0, 0, 220)
COLOR_WARNING = (0, 165, 255)

@register_exercise
class HighKneesExercise(ExerciseBase):
    ANGLE_UP_THRESHOLD = 95.0    # Rodilla alta (el ángulo hombro-cadera-rodilla disminuye)
    ANGLE_DOWN_THRESHOLD = 150.0 # Pierna bajada
    VISIBILITY_THRESHOLD = 0.5   

    def __init__(self) -> None:
        self._rep_count: int = 0
        self._steps_count: int = 0
        self._state: str = "Esperando..."
        self._form_ok: bool = True

    @property
    def name(self) -> str:
        return "Elevación de Rodillas"

    @property
    def description(self) -> str:
        return "Marcha levantando las rodillas hacia el pecho de forma alterna."

    def reset(self) -> None:
        self._rep_count = 0
        self._steps_count = 0
        self._state = "Esperando..."
        self._form_ok = True

    def evaluate(self, landmarks: Any, frame_shape: Tuple[int, int, int]) -> ExerciseResult:
        if landmarks is None:
            return ExerciseResult(state="No detectado", feedback_message="Coloca la camara apuntando hacia tu cintura y piernas", color_bgr=COLOR_WARNING)

        lm = landmarks.landmark
        h, w, _ = frame_shape

        l_sh = lm[_LEFT_SHOULDER]
        l_hip = lm[_LEFT_HIP]
        l_knee = lm[_LEFT_KNEE]

        r_sh = lm[_RIGHT_SHOULDER]
        r_hip = lm[_RIGHT_HIP]
        r_knee = lm[_RIGHT_KNEE]

        if l_hip.visibility < self.VISIBILITY_THRESHOLD or r_hip.visibility < self.VISIBILITY_THRESHOLD:
            return ExerciseResult(state="Piernas no visibles", feedback_message="Alejate de la camara para que se vean tu cadera y tus rodillas", color_bgr=COLOR_WARNING)

        # Evaluar ángulo hombro -> cadera -> rodilla para ambas piernas
        l_sh_pt = (l_sh.x*w, l_sh.y*h); l_hip_pt = (l_hip.x*w, l_hip.y*h); l_knee_pt = (l_knee.x*w, l_knee.y*h)
        r_sh_pt = (r_sh.x*w, r_sh.y*h); r_hip_pt = (r_hip.x*w, r_hip.y*h); r_knee_pt = (r_knee.x*w, r_knee.y*h)

        angle_l = MathUtils.calculate_angle(l_sh_pt, l_hip_pt, l_knee_pt)
        angle_r = MathUtils.calculate_angle(r_sh_pt, r_hip_pt, r_knee_pt)

        # La pierna activa es la que tenga el MENOR ángulo (más elevada)
        if angle_l < angle_r:
            angle = angle_l
            landmark_ids = [_LEFT_SHOULDER, _LEFT_HIP, _LEFT_KNEE]
            active_side = "Izquierda"
        else:
            angle = angle_r
            landmark_ids = [_RIGHT_SHOULDER, _RIGHT_HIP, _RIGHT_KNEE]
            active_side = "Derecha"

        form_ok = True
        feedback = ""

        new_state = self._state

        if angle > self.ANGLE_DOWN_THRESHOLD:
            new_state = "Abajo"
            if not feedback: feedback = "Sube la rodilla!"
        elif angle < self.ANGLE_UP_THRESHOLD:
            new_state = "Arriba"
            if self._state == "Abajo" and self._form_ok:
                self._steps_count += 1
                self._rep_count = self._steps_count // 2
            if not feedback: feedback = "¡Bien! Cambia de pierna"
        
        self._state = new_state

        return ExerciseResult(
            angle=angle,
            rep_count=self._rep_count,
            state=f"{new_state} ({active_side})",
            form_ok=True,
            feedback_message=feedback,
            color_bgr=COLOR_OK,
            landmarks_to_draw=landmark_ids,
        )
