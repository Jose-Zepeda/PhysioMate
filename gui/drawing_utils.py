"""
PhysioMate - Utilidades de Dibujo (Drawing Utils)

Encapsula la lógica de renderizado de OpenCV para desacoplar
la lógica de los ejercicios de la representación visual.
"""

import cv2
import numpy as np
from typing import List, Optional
from core.config import UIConfig
from exercises.base import ExerciseResult


class DrawingUtils:
    """Utilidades para dibujar superposiciones de información sobre los frames."""

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
    def _wrap_text(text: str, max_chars: int = 28) -> List[str]:
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

    @classmethod
    def draw_exercise_info(
        cls,
        frame: np.ndarray,
        result: ExerciseResult,
        config: UIConfig = UIConfig(),
    ) -> np.ndarray:
        """Dibuja información del ejercicio superpuesta en el frame.

        Se escala dinámicamente según la resolución del frame para mantener
        proporciones consistentes en diferentes cámaras.

        Args:
            frame: Frame BGR de OpenCV.
            result: Resultado de la evaluación del ejercicio.
            config: Configuración de UI.

        Returns:
            Frame con la información dibujada.
        """
        if result is None:
            return frame

        h, w, _ = frame.shape
        
        # Calcular factor de escala basado en una resolución de referencia (1280px de ancho)
        # Esto evita que el panel se vea gigante en cámaras de baja resolución
        scale = w / 1280.0
        
        # Ajustar parámetros según la escala
        panel_x = int(10 * scale)
        panel_y = int(10 * scale)
        panel_w = int(config.PANEL_WIDTH * scale)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Escalar fuentes y grosores
        font_scale_md = config.FONT_SCALE_MD * scale
        font_scale_sm = config.FONT_SCALE_SM * scale
        thickness = max(1, int(2 * scale))
        line_height = int(config.LINE_HEIGHT * scale)
        
        # Calcular líneas de feedback
        max_chars = int(28 * (1/scale)) if scale < 1.0 else 28
        # Un mejor wrap basado en el ancho del panel real
        feedback_lines = cls._wrap_text(cls._ascii(result.feedback_message), max_chars=28) if result.feedback_message else []
        
        panel_h = int(115 * scale) + max(len(feedback_lines), 1) * line_height

        # Fondo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), config.COLOR_PANEL, -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # Borde del panel
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (80, 80, 80), 1)

        tx = panel_x + int(10 * scale)
        y = panel_y + int(30 * scale)

        # ── Ángulo ──
        cv2.putText(frame, f"Angulo: {result.angle:.1f} grados", (tx, y),
                    font, font_scale_md, config.COLOR_TEXT, thickness)
        y += line_height

        # ── Estado ──
        state_text = cls._ascii(f"Estado: {result.state}")
        cv2.putText(frame, state_text, (tx, y),
                    font, font_scale_md, result.color_bgr, thickness)
        y += line_height

        # ── Repeticiones ──
        cv2.putText(frame, f"Reps: {result.rep_count}", (tx, y),
                    font, font_scale_md, (0, 255, 255), thickness)
        y += line_height

        # ── Separador ──
        cv2.line(frame, (tx, y - int(8 * scale)), (panel_x + panel_w - int(10 * scale), y - int(8 * scale)), (80, 80, 80), thickness // 2 or 1)

        # ── Feedback (multi-línea) ──
        if feedback_lines:
            fb_color = config.COLOR_OK if result.form_ok else config.COLOR_BAD
            state_lower = result.state.lower()
            if any(x in state_lower for x in ["no detectado", "visible", "esperando"]):
                fb_color = config.COLOR_WARNING
            
            for line in feedback_lines:
                cv2.putText(frame, line, (tx, y), font, font_scale_sm, fb_color, thickness)
                y += line_height

        return frame
