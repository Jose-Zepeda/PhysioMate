# PhysioMate — Asistente de Rehabilitación Postural basado en IA

Sistema de escritorio que utiliza visión computarizada para rastrear la pose del usuario en tiempo real, evaluar ángulos articulares durante ejercicios de rehabilitación y proporcionar retroalimentación visual y auditiva.

## Características

- 🤖 **Detección de pose en tiempo real** con MediaPipe Pose
- 📐 **Cálculo de ángulos articulares** con NumPy
- 🏋️ **Ejercicio incluido:** Curl de Bíceps con conteo de repeticiones
- 🔊 **Retroalimentación auditiva** no bloqueante con pyttsx3
- 🎨 **Interfaz moderna** con customtkinter (modo oscuro)
- ✅ **Detección de mala forma** con alertas en rojo
- 🔌 **Arquitectura extensible** — agrega ejercicios sin modificar el motor

## Requisitos

- **Python 3.9+**
- Cámara web conectada

## Instalación

```bash
# 1. Clonar o descargar el proyecto
cd PhysioMate

# 2. Crear entorno virtual (recomendado)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Ejecución

```bash
python main.py
```

## Uso

1. Presiona **▶ Iniciar** para activar la cámara
2. Colócate frente a la cámara de cuerpo completo (o al menos el torso y brazo)
3. Selecciona el ejercicio en el menú desplegable
4. Realiza el ejercicio — el sistema detectará tus movimientos automáticamente
5. Observa las métricas en el panel lateral (repeticiones, ángulo, estado)
6. Escucha las indicaciones de audio para corregir tu forma
7. Presiona **■ Detener** cuando termines

## Estructura del Proyecto

```
PhysioMate/
├── main.py                     # Punto de entrada
├── requirements.txt            # Dependencias
├── README.md                   # Este archivo
├── core/
│   ├── math_utils.py           # Cálculo de ángulos (Motor Geométrico)
│   ├── pose_detector.py        # Wrapper de MediaPipe (Motor de Inferencia)
│   ├── audio_feedback.py       # TTS no bloqueante (Feedback Manager)
│   └── exercise_tracker.py     # Orquestador de ejercicios
├── exercises/
│   ├── base.py                 # Clase base + registro automático
│   └── bicep_curl.py           # Curl de Bíceps
└── gui/
    └── main_app.py             # Interfaz gráfica (customtkinter)
```

## Agregar un Nuevo Ejercicio

Crea un archivo en `exercises/` siguiendo esta plantilla:

```python
from exercises.base import ExerciseBase, ExerciseResult, register_exercise

@register_exercise
class MiEjercicio(ExerciseBase):
    @property
    def name(self) -> str:
        return "Mi Ejercicio"

    @property
    def description(self) -> str:
        return "Descripción del ejercicio"

    def evaluate(self, landmarks, frame_shape):
        # Tu lógica aquí
        return ExerciseResult(...)

    def reset(self):
        # Reiniciar contadores
        pass
```

Luego importa el módulo en `main.py`:
```python
import exercises.mi_ejercicio  # noqa: F401
```

## Stack Tecnológico

| Componente       | Tecnología      |
|------------------|-----------------|
| Visión por IA    | MediaPipe Pose  |
| Video            | OpenCV          |
| Matemáticas      | NumPy           |
| GUI              | customtkinter   |
| Audio (TTS)      | pyttsx3         |

## Licencia

Proyecto privado — Uso educativo.
