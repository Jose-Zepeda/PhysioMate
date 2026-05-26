"""
PhysioMate - Ejercicio: Zancadas (Lunges)
"""

from __future__ import annotations

from typing import Any, Tuple

import mediapipe as mp

from core.math_utils import MathUtils
from core.config import UIConfig
from exercises.base import ExerciseBase, ExerciseResult, register_exercise

_mp_pose = mp.solutions.pose.PoseLandmark

# ─── Landmark IDs ───
_LEFT_HIP    = _mp_pose.LEFT_HIP
_RIGHT_HIP   = _mp_pose.RIGHT_HIP
_LEFT_KNEE   = _mp_pose.LEFT_KNEE
_RIGHT_KNEE  = _mp_pose.RIGHT_KNEE
_LEFT_ANKLE  = _mp_pose.LEFT_ANKLE
_RIGHT_ANKLE = _mp_pose.RIGHT_ANKLE

_UI = UIConfig()
COLOR_OK = _UI.COLOR_OK
COLOR_BAD = _UI.COLOR_BAD
COLOR_WARNING = _UI.COLOR_WARNING

@register_exercise
class LungeExercise(ExerciseBase):
    ANGLE_UP_THRESHOLD = 150.0   # De pie 
    ANGLE_DOWN_THRESHOLD = 100.0 # Zancada profunda
    VISIBILITY_THRESHOLD = 0.5   

    def __init__(self) -> None:
        self._rep_count: int = 0
        self._state: str = "Esperando..."
        self._form_ok: bool = True
        self._direction: str = ""

    @property
    def name(self) -> str:
        return "Zancadas"

    @property
    def description(self) -> str:
        return "Paso adelante bajando la rodilla trasera hacia el suelo."

    def reset(self) -> None:
        self._rep_count = 0
        self._state = "Esperando..."
        self._form_ok = True
        self._direction = ""

    def evaluate(self, landmarks: Any, frame_shape: Tuple[int, int, int]) -> ExerciseResult:
        if landmarks is None:
            return ExerciseResult(state="No detectado", feedback_message="Coloca tu cuerpo de perfil frente a la camara para comenzar", color_bgr=COLOR_WARNING)

        lm = landmarks.landmark
        h, w, _ = frame_shape

        # Determinar qué pierna está más adelante (usa la x para ver quién está al frente en perfil)
        # Asumimos perfil para mejor detección de zancadas
        left_knee = lm[_LEFT_KNEE].visibility
        right_knee = lm[_RIGHT_KNEE].visibility
        
        if max(left_knee, right_knee) < self.VISIBILITY_THRESHOLD:
            return ExerciseResult(state="Piernas no visibles", feedback_message="Alejate de la camara para que se vean tus dos piernas completas", color_bgr=COLOR_WARNING)

        # Detectar pierna que está más flexionada y soporta peso (la que marca el paso)
        l_hip_pt = (lm[_LEFT_HIP].x * w, lm[_LEFT_HIP].y * h)
        l_knee_pt = (lm[_LEFT_KNEE].x * w, lm[_LEFT_KNEE].y * h)
        l_ank_pt = (lm[_LEFT_ANKLE].x * w, lm[_LEFT_ANKLE].y * h)

        r_hip_pt = (lm[_RIGHT_HIP].x * w, lm[_RIGHT_HIP].y * h)
        r_knee_pt = (lm[_RIGHT_KNEE].x * w, lm[_RIGHT_KNEE].y * h)
        r_ank_pt = (lm[_RIGHT_ANKLE].x * w, lm[_RIGHT_ANKLE].y * h)

        angle_l = MathUtils.calculate_angle(l_hip_pt, l_knee_pt, l_ank_pt)
        angle_r = MathUtils.calculate_angle(r_hip_pt, r_knee_pt, r_ank_pt)

        # Usar la pierna con menor ángulo (la que flexiona para hacer la zancada al frente)
        if angle_l < angle_r:
            angle = angle_l
            landmark_ids = [_LEFT_HIP, _LEFT_KNEE, _LEFT_ANKLE]
            front_knee_x, front_ankle_x = lm[_LEFT_KNEE].x, lm[_LEFT_ANKLE].x
        else:
            angle = angle_r
            landmark_ids = [_RIGHT_HIP, _RIGHT_KNEE, _RIGHT_ANKLE]
            front_knee_x, front_ankle_x = lm[_RIGHT_KNEE].x, lm[_RIGHT_ANKLE].x

        form_ok = True
        feedback = ""

        # Mala forma: que la rodilla sobrepase mucho la punta del pie
        # Perfil: si la rodilla en x excede el tobillo x (dependiendo de dirección, simplificamos con la distancia abs)
        drift = abs(front_knee_x - front_ankle_x)
        if drift > 0.15 and angle < 120:
            form_ok = False
            feedback = "Rodilla sobrepasa el pie"

        new_state = self._state
        
        if angle > self.ANGLE_UP_THRESHOLD:
            new_state = "Arriba"
            if self._direction == "subiendo" and self._form_ok:
                self._rep_count += 1
                self._direction = ""
            if not feedback: feedback = "Da un paso y baja"
        elif angle < self.ANGLE_DOWN_THRESHOLD:
            new_state = "Abajo"
            self._direction = "subiendo" # Prep for counting on way up
            if not feedback: feedback = "Vuelve arriba"
        else:
            new_state = "Transición"

        self._state = new_state
        self._form_ok = form_ok

        return ExerciseResult(
            angle=angle,
            rep_count=self._rep_count,
            state=new_state,
            form_ok=form_ok,
            feedback_message=feedback,
            color_bgr=COLOR_BAD if not form_ok else COLOR_OK,
            landmarks_to_draw=landmark_ids,
        )
