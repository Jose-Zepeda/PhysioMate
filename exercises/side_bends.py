"""
PhysioMate - Ejercicio: Inclinación Lateral (Side Bends)
"""

from __future__ import annotations

from typing import Any, Tuple

import mediapipe as mp

from core.math_utils import MathUtils
from exercises.base import ExerciseBase, ExerciseResult, register_exercise

_mp_pose = mp.solutions.pose.PoseLandmark

_LEFT_SHOULDER = _mp_pose.LEFT_SHOULDER
_RIGHT_SHOULDER = _mp_pose.RIGHT_SHOULDER
_LEFT_HIP = _mp_pose.LEFT_HIP
_RIGHT_HIP = _mp_pose.RIGHT_HIP

COLOR_OK = (0, 220, 0)
COLOR_BAD = (0, 0, 220)
COLOR_WARNING = (0, 165, 255)

@register_exercise
class SideBendsExercise(ExerciseBase):
    ANGLE_CENTER_MAX = 5.0    # Grados de desviación para considerarse recto
    ANGLE_BEND_MIN = 20.0     # Grados de inclinación necesarios para contar
    VISIBILITY_THRESHOLD = 0.5   

    def __init__(self) -> None:
        self._rep_count: int = 0
        self._state: str = "Centro"
        self._form_ok: bool = True

    @property
    def name(self) -> str:
        return "Inclinación Lateral"

    @property
    def description(self) -> str:
        return "Flexión lateral del tronco para oblicuos. Mantén la cadera fija."

    def reset(self) -> None:
        self._rep_count = 0
        self._state = "Centro"
        self._form_ok = True

    def evaluate(self, landmarks: Any, frame_shape: Tuple[int, int, int]) -> ExerciseResult:
        if landmarks is None:
            return ExerciseResult(state="No detectado", feedback_message="Colócate frente a la cámara", color_bgr=COLOR_WARNING)

        lm = landmarks.landmark
        h, w, _ = frame_shape

        l_sh = lm[_LEFT_SHOULDER]; r_sh = lm[_RIGHT_SHOULDER]
        l_hip = lm[_LEFT_HIP]; r_hip = lm[_RIGHT_HIP]

        if l_sh.visibility < self.VISIBILITY_THRESHOLD or l_hip.visibility < self.VISIBILITY_THRESHOLD:
            return ExerciseResult(state="No visible", feedback_message="Torso no visible", color_bgr=COLOR_WARNING)

        # Calcular el punto medio de los hombros y de las caderas
        sh_mid = ((l_sh.x + r_sh.x) / 2 * w, (l_sh.y + r_sh.y) / 2 * h)
        hip_mid = ((l_hip.x + r_hip.x) / 2 * w, (l_hip.y + r_hip.y) / 2 * h)

        # Punto de referencia directamente arriba de la cadera (vertical perfecta)
        vertical_pt = (hip_mid[0], hip_mid[1] - 100)

        # Angulo de inclinación del torso respecto a la vertical
        angle = MathUtils.calculate_angle(sh_mid, hip_mid, vertical_pt)

        # Determinar dirección (solo usando x de los hombros y cadera)
        # Si sh_mid.x < hip_mid.x, está inclinado a la derecha (espejo)
        is_left = sh_mid[0] > hip_mid[0]

        form_ok = True
        feedback = ""

        new_state = self._state

        if angle < self.ANGLE_CENTER_MAX:
            new_state = "Centro"
            if not feedback: feedback = "Inclínate hacia un lado"
        elif angle > self.ANGLE_BEND_MIN:
            new_state = "Izquierda" if is_left else "Derecha"
            if self._state in ["Centro", "Inclinando"]:
                self._rep_count += 1
            if not feedback: feedback = "Vuelve al centro"
        else:
            new_state = "Inclinando"

        self._state = new_state

        return ExerciseResult(
            angle=angle,
            rep_count=self._rep_count,
            state=new_state,
            form_ok=True,
            feedback_message=feedback,
            color_bgr=COLOR_OK,
            landmarks_to_draw=[_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP],
        )
