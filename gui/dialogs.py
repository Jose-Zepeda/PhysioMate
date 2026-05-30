"""
PhysioMate - Diálogos Modales (Modal Dialogs)

Responsabilidad única: construir diálogos modales centrados (error / info)
sobre una ventana padre.

Función autocontenida, sin estado: recibe la ventana padre y los textos.
Extraído de MainApp._show_dialog para respetar SRP.
"""

from __future__ import annotations

import customtkinter as ctk


def show_dialog(parent, title: str, message: str, title_color: str) -> None:
    """Muestra un diálogo modal genérico centrado en pantalla.

    Args:
        parent: Ventana padre (la app principal).
        title: Título del diálogo.
        message: Mensaje a mostrar.
        title_color: Color del texto del título.
    """
    dialog = ctk.CTkToplevel(parent)
    dialog.title(title.replace("⚠️ ", "").replace("✅ ", ""))
    dialog.geometry("420x170")
    dialog.resizable(False, False)
    dialog.transient(parent)
    dialog.grab_set()

    # Centrar en pantalla
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() - 420) // 2
    y = (dialog.winfo_screenheight() - 170) // 2
    dialog.geometry(f"+{x}+{y}")

    ctk.CTkLabel(
        dialog,
        text=title,
        font=ctk.CTkFont(size=16, weight="bold"),
        text_color=title_color,
    ).pack(pady=(15, 5))

    ctk.CTkLabel(
        dialog,
        text=message,
        font=ctk.CTkFont(size=12),
        wraplength=380,
    ).pack(pady=5)

    ctk.CTkButton(
        dialog,
        text="Aceptar",
        command=dialog.destroy,
        width=100,
    ).pack(pady=10)
