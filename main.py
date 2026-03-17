"""
PhysioMate — Asistente de Rehabilitación Postural basado en IA

Punto de entrada principal de la aplicación.
Inicializa todos los componentes e inyecta dependencias en la GUI.

Uso:
    python main.py
"""

from __future__ import annotations

import logging
import sys

# ─── Configurar logging ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-25s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PhysioMate")


def main() -> None:
    """Punto de entrada principal de PhysioMate.

    Crea e inyecta todas las dependencias siguiendo el patrón
    de Inversión de Dependencias (DIP):
      1. PoseDetector  (Motor de inferencia)
      2. AudioFeedback (Sistema de audio)
      3. ExerciseTracker (Orquestador)
      4. MainApp (GUI)
    """
    logger.info("=" * 50)
    logger.info("  PhysioMate - Iniciando aplicación")
    logger.info("=" * 50)

    # ── 1. Importar ejercicios (activa el registro automático) ──
    try:
        import exercises.bicep_curl  # noqa: F401

        logger.info("Módulos de ejercicios cargados correctamente")
    except ImportError as e:
        logger.error("Error al importar módulos de ejercicios: %s", e)
        sys.exit(1)

    # ── 2. Crear componentes ──
    from core.audio_feedback import AudioFeedback
    from core.exercise_tracker import ExerciseTracker
    from core.pose_detector import PoseDetector
    from exercises.base import get_available_exercises
    from gui.main_app import MainApp

    logger.info("Ejercicios disponibles: %s", get_available_exercises())

    try:
        # Motor de Inferencia
        pose_detector = PoseDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )
        logger.info("PoseDetector inicializado")

        # Sistema de Audio
        audio = AudioFeedback(
            cooldown_seconds=3.0,
            rate=180,
            volume=0.9,
        )
        logger.info("AudioFeedback inicializado")

        # Orquestador
        tracker = ExerciseTracker(
            pose_detector=pose_detector,
            audio=audio,
        )
        logger.info("ExerciseTracker inicializado")

        # GUI
        app = MainApp(
            tracker=tracker,
            camera_index=0,
        )
        logger.info("Interfaz gráfica lista — iniciando mainloop")
        logger.info("=" * 50)

        app.mainloop()

    except RuntimeError as e:
        logger.critical("Error fatal al inicializar: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.critical("Error inesperado: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
