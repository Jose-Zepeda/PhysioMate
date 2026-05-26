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
    def _wrap_text_px(
        text: str,
        font: int,
        font_scale: float,
        thickness: int,
        max_width_px: int,
    ) -> List[str]:
        """Parte un texto en líneas que caben dentro de max_width_px píxeles.

        A diferencia del wrap basado en caracteres, este mide el ancho real
        de cada línea usando cv2.getTextSize, garantizando que nunca se
        desborde el panel independientemente de la fuente o resolución.

        Args:
            text: Texto a partir.
            font: Fuente de OpenCV.
            font_scale: Escala de la fuente.
            thickness: Grosor del texto.
            max_width_px: Ancho máximo permitido en píxeles.

        Returns:
            Lista de líneas que caben dentro del ancho máximo.
        """
        words = text.split()
        lines: List[str] = []
        current = ""

        for word in words:
            candidate = (current + " " + word).strip() if current else word
            (w, _), _ = cv2.getTextSize(candidate, font, font_scale, thickness)
            if w <= max_width_px:
                current = candidate
            else:
                if current:
                    lines.append(current)
                # Si la palabra sola ya es demasiado larga, forzar línea propia
                (w_word, _), _ = cv2.getTextSize(word, font, font_scale, thickness)
                if w_word > max_width_px:
                    # Truncar la palabra con "..."
                    truncated = word
                    while truncated:
                        candidate_t = truncated + "..."
                        (wt, _), _ = cv2.getTextSize(candidate_t, font, font_scale, thickness)
                        if wt <= max_width_px:
                            lines.append(candidate_t)
                            break
                        truncated = truncated[:-1]
                    current = ""
                else:
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

        Usa posicionamiento basado en porcentaje del frame y wrap de texto
        basado en píxeles reales, garantizando que el panel siempre quede
        correctamente posicionado sin importar la resolución del video.

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
        font = cv2.FONT_HERSHEY_SIMPLEX

        # ── Posicionamiento: 1.5% del frame desde la esquina superior izquierda ──
        margin_x = max(8, int(w * 0.015))
        margin_y = max(8, int(h * 0.015))
        panel_x = margin_x
        panel_y = margin_y

        # ── Ancho del panel: 26% del frame, mínimo 180px, máximo 420px ──
        panel_w = int(min(max(int(w * 0.26), 180), 420))

        # ── Escala de fuente: proporcional a la altura del frame
        #    Clamped para que sea siempre legible (0.40 – 0.75) ──
        font_scale_md = max(0.40, min(h / 900.0 * 0.68, 0.75))
        font_scale_sm = max(0.35, min(h / 900.0 * 0.56, 0.62))
        thickness = max(1, int(h / 720.0 * 1.5))

        # Altura de línea basada en el tamaño real de la fuente
        (_, lh_md), baseline_md = cv2.getTextSize("Ag", font, font_scale_md, thickness)
        (_, lh_sm), _ = cv2.getTextSize("Ag", font, font_scale_sm, thickness)
        line_height_md = lh_md + baseline_md + max(4, int(h * 0.006))
        line_height_sm = lh_sm + max(2, int(h * 0.004))

        # ── Área de texto disponible (con padding interno de 10px por lado) ──
        text_max_w = panel_w - 20

        # ── Wrap del feedback basado en píxeles reales ──
        feedback_lines: List[str] = []
        if result.feedback_message:
            feedback_lines = cls._wrap_text_px(
                cls._ascii(result.feedback_message),
                font, font_scale_sm, thickness, text_max_w,
            )

        # ── Altura total del panel ──
        header_h = line_height_md * 3 + max(6, int(h * 0.008))   # ángulo + estado + reps
        separator_h = max(4, int(h * 0.005))
        feedback_h = len(feedback_lines) * line_height_sm if feedback_lines else 0
        padding_top = max(10, int(h * 0.014))
        padding_bot = max(8, int(h * 0.010))
        panel_h = padding_top + header_h + separator_h + feedback_h + padding_bot

        # Asegurar que el panel no se salga del frame
        panel_x = max(0, min(panel_x, w - panel_w - 2))
        panel_y = max(0, min(panel_y, h - panel_h - 2))

        # ── Fondo semitransparente ──
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            config.COLOR_PANEL,
            -1,
        )
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # ── Borde del panel ──
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (80, 80, 80),
            1,
        )

        # ── Posición inicial del texto ──
        tx = panel_x + 10
        y = panel_y + padding_top + lh_md  # baseline de la primera línea

        # ── Ángulo ──
        cv2.putText(
            frame,
            f"Angulo: {result.angle:.1f} grados",
            (tx, y),
            font, font_scale_md, config.COLOR_TEXT, thickness,
        )
        y += line_height_md

        # ── Estado ──
        state_text = cls._ascii(f"Estado: {result.state}")
        cv2.putText(frame, state_text, (tx, y), font, font_scale_md, result.color_bgr, thickness)
        y += line_height_md

        # ── Repeticiones ──
        cv2.putText(frame, f"Reps: {result.rep_count}", (tx, y), font, font_scale_md, (0, 255, 255), thickness)
        y += line_height_md

        # ── Separador ──
        sep_y = y - line_height_md // 3
        cv2.line(
            frame,
            (tx, sep_y),
            (panel_x + panel_w - 10, sep_y),
            (80, 80, 80),
            max(1, thickness - 1),
        )
        y = sep_y + separator_h + lh_sm

        # ── Feedback (multi-línea) ──
        if feedback_lines:
            fb_color = config.COLOR_OK if result.form_ok else config.COLOR_BAD
            state_lower = result.state.lower()
            if any(x in state_lower for x in ["no detectado", "visible", "esperando"]):
                fb_color = config.COLOR_WARNING

            for line in feedback_lines:
                cv2.putText(frame, line, (tx, y), font, font_scale_sm, fb_color, thickness)
                y += line_height_sm

        return frame
