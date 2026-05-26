"""
PhysioMate - Sistema de Feedback de Audio (Audio Feedback Manager)

Proporciona retroalimentación auditiva mediante Text-to-Speech (pyttsx3)
de forma no bloqueante usando un hilo daemon y una cola de mensajes.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

from core.config import AudioConfig

logger = logging.getLogger(__name__)


class AudioFeedback:
    """Sistema de retroalimentación auditiva no bloqueante.

    Utiliza pyttsx3 para síntesis de voz offline. Los mensajes se encolan
    y un hilo daemon los procesa secuencialmente, evitando congelar
    el hilo principal del video/GUI.

    Implementa un mecanismo de cooldown para no repetir mensajes
    demasiado rápido.

    Attributes:
        cooldown_seconds: Tiempo mínimo entre mensajes idénticos.
        enabled: Si el audio está habilitado.
    """

    def __init__(
        self,
        cooldown_seconds: float = AudioConfig().COOLDOWN_SECONDS,
        rate: int = AudioConfig().RATE,
        volume: float = AudioConfig().VOLUME,
    ) -> None:
        """Inicializa el sistema de audio con un hilo worker.

        Args:
            cooldown_seconds: Segundos mínimos entre repeticiones del mismo mensaje.
            rate: Velocidad de habla (palabras por minuto).
            volume: Volumen de la voz [0.0, 1.0].
        """
        self.cooldown_seconds = cooldown_seconds
        self.enabled = True

        self._rate = rate
        self._volume = volume
        self._last_messages: dict[str, float] = {}
        self._lock = threading.Lock()
        self._tts_lock = threading.Lock()

    def _run_tts(self, message: str) -> None:
        """Función interna que ejecuta pyttsx3 en un hilo nuevo. Evita superposición."""
        if not self._tts_lock.acquire(blocking=False):
            return  # Si ya está hablando, ignoramos este mensaje para no solapar audios
            
        try:
            import pythoncom
            pythoncom.CoInitialize()
            import pyttsx3
            
            logger.info("TTS Hablando: %s", message)
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)

            # Intentar configurar voz en español si está disponible
            voices = engine.getProperty("voices")
            for voice in voices:
                if "spanish" in voice.name.lower() or "español" in voice.name.lower():
                    engine.setProperty("voice", voice.id)
                    break

            engine.say(message)
            engine.runAndWait()
            logger.info("TTS Completado")
        except Exception as e:
            logger.warning("Error al reproducir TTS: %s", e)
        finally:
            self._tts_lock.release()

    @staticmethod
    def _sanitize(text: str) -> str:
        """Elimina tildes para evitar que pyttsx3 se salte el texto silenciosamente en Windows."""
        replacements = {
            "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
            "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
            "ü": "u", "Ü": "U", "ñ": "n", "Ñ": "N"
        }
        for orig, rep in replacements.items():
            text = text.replace(orig, rep)
        return text

    def speak(self, message: str) -> None:
        """Encola un mensaje para ser hablado.

        Aplica cooldown para evitar repetir el mismo mensaje
        demasiado frecuentemente. No bloquea el hilo llamante.

        Args:
            message: Texto a sintetizar como voz.
        """
        if not self.enabled or not message:
            return

        message = self._sanitize(message)

        current_time = time.time()

        with self._lock:
            # Verificar cooldown
            last_time = self._last_messages.get(message, 0.0)
            if current_time - last_time < self.cooldown_seconds:
                return

            self._last_messages[message] = current_time

        # Ejecutar en un hilo completamente nuevo para evitar bugs de COM de pyttsx3
        t = threading.Thread(target=self._run_tts, args=(message,), daemon=True)
        t.start()

    def stop(self) -> None:
        """Detiene el sistema de audio (ahora no hace nada ya que los hilos son efímeros)."""
        pass

    def __del__(self) -> None:
        """Destructor."""
        pass
