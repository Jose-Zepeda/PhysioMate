"""
PhysioMate - Motor Geométrico (Math Engine)

Proporciona utilidades matemáticas para el cálculo de ángulos articulares
a partir de coordenadas de landmarks corporales.
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional


class MathUtils:
    """Utilidades matemáticas estáticas para cálculos geométricos de pose.

    Todos los métodos son estáticos ya que no requieren estado interno.
    Resuelve el cálculo de ángulos interiores entre tres puntos usando
    vectores y arcotangente de dos argumentos (atan2).
    """

    @staticmethod
    def calculate_angle(
        point_a: ArrayLike,
        point_b: ArrayLike,
        point_c: ArrayLike,
    ) -> float:
        """Calcula el ángulo interior en el punto B formado por los segmentos BA y BC.

        Utiliza np.arctan2 para obtener el ángulo con signo de cada vector
        respecto al eje X, y luego calcula la diferencia. El resultado se
        normaliza al rango [0, 180] grados.

        Args:
            point_a: Coordenadas del primer punto (ej. hombro).
            point_b: Coordenadas del vértice del ángulo (ej. codo).
            point_c: Coordenadas del tercer punto (ej. muñeca).

        Returns:
            Ángulo interior en grados, en el rango [0, 180].
        """
        a = np.array(point_a, dtype=np.float64)
        b = np.array(point_b, dtype=np.float64)
        c = np.array(point_c, dtype=np.float64)

        # Vectores desde el vértice hacia los otros dos puntos
        vector_ba = a - b
        vector_bc = c - b

        # Ángulo de cada vector respecto al eje X
        angle_ba = np.arctan2(vector_ba[1], vector_ba[0])
        angle_bc = np.arctan2(vector_bc[1], vector_bc[0])

        # Diferencia de ángulos
        angle_rad = angle_ba - angle_bc
        angle_deg = np.abs(np.degrees(angle_rad))

        # Normalizar al rango [0, 180]
        if angle_deg > 180.0:
            angle_deg = 360.0 - angle_deg

        return round(angle_deg, 2)

    @staticmethod
    def calculate_distance(
        point_a: ArrayLike,
        point_b: ArrayLike,
    ) -> float:
        """Calcula la distancia euclidiana entre dos puntos.

        Args:
            point_a: Coordenadas del primer punto.
            point_b: Coordenadas del segundo punto.

        Returns:
            Distancia euclidiana entre los dos puntos.
        """
        a = np.array(point_a, dtype=np.float64)
        b = np.array(point_b, dtype=np.float64)
        return float(np.linalg.norm(a - b))


class SmoothingFilter:
    """Filtro de suavizado basado en Media Móvil Exponencial (EMA).

    Se utiliza para reducir el ruido en los ángulos calculados, evitando
    saltos bruscos en el conteo de repeticiones y la visualización.
    """

    def __init__(self, alpha: float = 0.25) -> None:
        """Inicializa el filtro.

        Args:
            alpha: Factor de suavizado [0, 1].
                   Valores bajos = más suavizado, más retardo.
                   Valores altos = menos suavizado, más reactivo.
        """
        self.alpha = alpha
        self._last_value: Optional[float] = None

    def update(self, value: float) -> float:
        """Actualiza el filtro con un nuevo valor y retorna el valor suavizado.

        Args:
            value: Nuevo valor medido.

        Returns:
            Valor suavizado.
        """
        if self._last_value is None:
            self._last_value = value
        else:
            self._last_value = (self.alpha * value) + ((1 - self.alpha) * self._last_value)

        return self._last_value

    def reset(self) -> None:
        """Reinicia el filtro."""
        self._last_value = None
