"""
PhysioMate - Renderizado de Frames (Frame Renderer)

Responsabilidad única: convertir un frame de OpenCV (BGR) en un CTkImage
listo para mostrar, redimensionado a una caja destino manteniendo la
relación de aspecto.

Depende de customtkinter (CTkImage), por eso vive en la capa gui.
Extraído de MainApp._display_frame para respetar SRP.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class FrameRenderer:
    """Convierte frames BGR de OpenCV en CTkImage redimensionados."""

    @staticmethod
    def render(
        frame_bgr: np.ndarray,
        box_w: int,
        box_h: int,
    ) -> Optional[ctk.CTkImage]:
        """Convierte un frame BGR en un CTkImage ajustado a la caja destino.

        Args:
            frame_bgr: Frame BGR de OpenCV.
            box_w: Ancho disponible del área de video (px).
            box_h: Alto disponible del área de video (px).

        Returns:
            CTkImage listo para mostrar, o None si la caja o el resultado
            tienen dimensiones inválidas.
        """
        if box_w <= 0 or box_h <= 0:
            return None

        try:
            # Mantener relación de aspecto
            h, w = frame_bgr.shape[:2]
            scale = min(box_w / w, box_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            if new_w <= 0 or new_h <= 0:
                return None

            # Redimensionar
            resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # BGR → RGB → PIL → CTkImage
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            return ctk.CTkImage(
                light_image=pil_image,
                dark_image=pil_image,
                size=(new_w, new_h),
            )

        except Exception as e:
            logger.error("Error al renderizar frame: %s", e)
            return None
