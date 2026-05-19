"""
PhysioMate - Configuración Global

Centraliza constantes de diseño, umbrales y parámetros del sistema
para facilitar el ajuste sin modificar la lógica principal.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class UIConfig:
    """Configuración de la interfaz gráfica y dibujo."""
    WINDOW_TITLE: str = "PhysioMate — Asistente de Rehabilitación Postural"
    MIN_WIDTH: int = 960
    MIN_HEIGHT: int = 620
    
    # Colores BGR para OpenCV
    COLOR_OK: Tuple[int, int, int] = (0, 220, 0)       # Verde
    COLOR_BAD: Tuple[int, int, int] = (0, 0, 220)      # Rojo
    COLOR_WARNING: Tuple[int, int, int] = (0, 165, 255)  # Naranja
    COLOR_TEXT: Tuple[int, int, int] = (255, 255, 255) # Blanco
    COLOR_PANEL: Tuple[int, int, int] = (20, 20, 20)   # Negro azulado
    
    # Parámetros de dibujo
    PANEL_WIDTH: int = 300
    FONT_SCALE_MD: float = 0.68
    FONT_SCALE_SM: float = 0.58
    LINE_HEIGHT: int = 32


@dataclass(frozen=True)
class DetectionConfig:
    """Configuración del motor de detección y matemáticas."""
    # Filtro de suavizado (Exponential Moving Average)
    # Alpha cercano a 1 = más reactivo, menos suave
    # Alpha cercano a 0 = más suave, más lag
    SMOOTHING_ALPHA: float = 0.25
    
    # MediaPipe defaults
    MODEL_COMPLEXITY: int = 1
    MIN_DETECTION_CONFIDENCE: float = 0.5
    MIN_TRACKING_CONFIDENCE: float = 0.5


@dataclass(frozen=True)
class AudioConfig:
    """Configuración del sistema de audio."""
    COOLDOWN_SECONDS: float = 3.0
    RATE: int = 180
    VOLUME: float = 0.9
