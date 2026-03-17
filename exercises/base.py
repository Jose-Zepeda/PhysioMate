"""
PhysioMate - Base de Ejercicios y Registro (Exercise Base & Registry)

Define la clase base abstracta para todos los ejercicios y un sistema
de registro automático que permite agregar nuevos ejercicios sin
modificar el motor principal (Principio Abierto/Cerrado).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type


@dataclass
class ExerciseResult:
    """Resultado de la evaluación de un frame para un ejercicio.

    Attributes:
        angle: Ángulo principal medido (grados).
        rep_count: Número total de repeticiones completadas.
        state: Estado actual del ejercicio (ej. "Arriba", "Abajo").
        form_ok: True si la forma/postura es correcta.
        feedback_message: Mensaje de retroalimentación para el usuario.
        color_bgr: Color BGR para dibujar (verde=OK, rojo=mala forma).
        landmarks_to_draw: IDs de landmarks relevantes para resaltar.
    """
    angle: float = 0.0
    rep_count: int = 0
    state: str = ""
    form_ok: bool = True
    feedback_message: str = ""
    color_bgr: Tuple[int, int, int] = (0, 255, 0)  # Verde por defecto
    landmarks_to_draw: list[int] = field(default_factory=list)


class ExerciseBase(ABC):
    """Clase base abstracta para todos los ejercicios de rehabilitación.

    Define el contrato que todo ejercicio debe cumplir. Los ejercicios
    concretos deben implementar `evaluate()` y `reset()`.

    Principio de Sustitución de Liskov: cualquier subclase puede
    reemplazar a ExerciseBase sin alterar el comportamiento esperado
    del sistema.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre descriptivo del ejercicio para mostrar en la UI."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Descripción breve del ejercicio."""
        ...

    @abstractmethod
    def evaluate(
        self,
        landmarks: Any,
        frame_shape: Tuple[int, int, int],
    ) -> ExerciseResult:
        """Evalúa el frame actual y retorna el resultado del ejercicio.

        Args:
            landmarks: Landmarks de MediaPipe Pose (results.pose_landmarks).
            frame_shape: Forma del frame (height, width, channels).

        Returns:
            ExerciseResult con ángulo, estado, conteo y feedback.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reinicia todos los contadores y estados del ejercicio."""
        ...


# ─────────────────────────────────────────────────────────────
# Registro global de ejercicios (Open/Closed Principle)
# ─────────────────────────────────────────────────────────────

EXERCISE_REGISTRY: Dict[str, Type[ExerciseBase]] = {}


def register_exercise(cls: Type[ExerciseBase]) -> Type[ExerciseBase]:
    """Decorador para registrar automáticamente un ejercicio.

    Uso:
        @register_exercise
        class MiEjercicio(ExerciseBase):
            ...

    El ejercicio queda disponible en EXERCISE_REGISTRY bajo su `name`.

    Args:
        cls: Clase del ejercicio a registrar.

    Returns:
        La misma clase, sin modificar.
    """
    # Crear instancia temporal para obtener el nombre
    instance = cls()
    EXERCISE_REGISTRY[instance.name] = cls
    return cls


def get_available_exercises() -> list[str]:
    """Retorna la lista de nombres de ejercicios registrados.

    Returns:
        Lista de nombres de ejercicios disponibles.
    """
    return list(EXERCISE_REGISTRY.keys())


def create_exercise(name: str) -> Optional[ExerciseBase]:
    """Crea una instancia de un ejercicio registrado por nombre.

    Args:
        name: Nombre del ejercicio a instanciar.

    Returns:
        Instancia del ejercicio, o None si no se encuentra.
    """
    cls = EXERCISE_REGISTRY.get(name)
    if cls is None:
        return None
    return cls()
