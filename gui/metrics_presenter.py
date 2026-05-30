"""
PhysioMate - Presentación de Métricas (Metrics Presenter)

Responsabilidad única: volcar los datos de un ExerciseResult en los widgets
de métricas del panel lateral (repeticiones, ángulo, estado, indicador de
forma y mensaje de feedback).

Es lógica de presentación: mapea el resultado del tracker a la vista, sin
contener lógica de ejercicios. Extraído de MainApp._update_metrics_display
para respetar SRP.
"""

from __future__ import annotations

import customtkinter as ctk


class MetricsPresenter:
    """Actualiza los widgets de métricas del sidebar desde un ExerciseResult."""

    def __init__(
        self,
        rep_label: ctk.CTkLabel,
        angle_label: ctk.CTkLabel,
        state_label: ctk.CTkLabel,
        form_indicator: ctk.CTkLabel,
        feedback_label: ctk.CTkLabel,
    ) -> None:
        """Recibe las referencias a los widgets que va a actualizar.

        Args:
            rep_label: Label del valor de repeticiones.
            angle_label: Label del valor del ángulo.
            state_label: Label del valor del estado.
            form_indicator: Label del indicador de forma.
            feedback_label: Label del mensaje de feedback.
        """
        self._rep_label = rep_label
        self._angle_label = angle_label
        self._state_label = state_label
        self._form_indicator = form_indicator
        self._feedback_label = feedback_label

    def update(self, result) -> None:
        """Actualiza los widgets de métricas con los datos del resultado.

        Args:
            result: ExerciseResult o None.
        """
        if result is None:
            self._rep_label.configure(text="0")
            self._angle_label.configure(text="-- °")
            self._state_label.configure(text="Inactivo")
            self._form_indicator.configure(
                text="● CORRECTA", text_color="#00ff88"
            )
            self._feedback_label.configure(text="")
            return

        # Repeticiones
        self._rep_label.configure(text=str(result.rep_count))

        # Ángulo
        self._angle_label.configure(text=f"{result.angle:.1f}°")

        # Estado
        self._state_label.configure(text=result.state)

        # Forma
        if not result.state or "detectado" in result.state.lower() or "esperando" in result.state.lower() or "visible" in result.state.lower():
            self._form_indicator.configure(
                text="● ESPERANDO",
                text_color="#AAAAAA",
            )
        elif result.form_ok:
            self._form_indicator.configure(
                text="● CORRECTA",
                text_color="#00ff88",
            )
        else:
            self._form_indicator.configure(
                text="● INCORRECTA",
                text_color="#ff4444",
            )

        # Feedback
        self._feedback_label.configure(text=result.feedback_message)
