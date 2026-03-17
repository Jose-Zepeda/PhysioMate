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
        cooldown_seconds: float = 3.0,
        rate: int = 180,
        volume: float = 0.9,
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
        self._message_queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=5)
        self._last_messages: dict[str, float] = {}
        self._lock = threading.Lock()
        self._running = True

        # Hilo daemon para procesar mensajes TTS
        self._worker_thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="AudioFeedbackWorker",
        )
        self._worker_thread.start()

    def _init_engine(self):
        """Inicializa el motor pyttsx3 dentro del hilo worker.

        pyttsx3 requiere que el motor se cree en el mismo hilo
        donde se ejecuta runAndWait().

        Returns:
            Motor pyttsx3 inicializado o None si falla.
        """
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)

            # Intentar configurar voz en español si está disponible
            voices = engine.getProperty("voices")
            for voice in voices:
                if "spanish" in voice.name.lower() or "español" in voice.name.lower():
                    engine.setProperty("voice", voice.id)
                    break

            return engine
        except Exception as e:
            logger.error("No se pudo inicializar pyttsx3: %s", e)
            return None

    def _worker(self) -> None:
        """Hilo worker que procesa la cola de mensajes TTS.

        Se ejecuta como daemon thread y se detiene cuando _running es False
        o cuando recibe None en la cola.
        """
        engine = self._init_engine()

        while self._running:
            try:
                message = self._message_queue.get(timeout=0.5)

                if message is None:
                    # Señal de parada
                    break

                if engine is not None:
                    try:
                        engine.say(message)
                        engine.runAndWait()
                    except Exception as e:
                        logger.warning("Error al reproducir TTS: %s", e)
                        # Reintentar con nuevo motor
                        engine = self._init_engine()

                self._message_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Error en worker de audio: %s", e)

        # Limpiar el motor al salir
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    def speak(self, message: str) -> None:
        """Encola un mensaje para ser hablado.

        Aplica cooldown para evitar repetir el mismo mensaje
        demasiado frecuentemente. No bloquea el hilo llamante.

        Args:
            message: Texto a sintetizar como voz.
        """
        if not self.enabled or not message:
            return

        current_time = time.time()

        with self._lock:
            # Verificar cooldown
            last_time = self._last_messages.get(message, 0.0)
            if current_time - last_time < self.cooldown_seconds:
                return

            self._last_messages[message] = current_time

        # Encolar sin bloquear (descarta si la cola está llena)
        try:
            self._message_queue.put_nowait(message)
        except queue.Full:
            pass

    def stop(self) -> None:
        """Detiene el sistema de audio y libera recursos."""
        self._running = False

        # Enviar señal de parada
        try:
            self._message_queue.put_nowait(None)
        except queue.Full:
            pass

        # Esperar a que el hilo termine (con timeout)
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

    def __del__(self) -> None:
        """Destructor: detiene el worker al ser recolectado."""
        self.stop()
