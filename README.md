# PhysioMate — Asistente de Rehabilitación Postural basado en IA

Sistema de escritorio que utiliza visión computarizada para rastrear la pose del usuario en tiempo real, evaluar ángulos articulares durante ejercicios de rehabilitación y proporcionar retroalimentación visual y auditiva.

## Características

- 🤖 **Detección de pose en tiempo real** con MediaPipe Pose
- 📐 **Cálculo de ángulos articulares** con NumPy
- 🏋️ **5 ejercicios incluidos:** Curl de Bíceps, Sentadillas, Zancadas, Elevación de Rodillas e Inclinación Lateral
- 🎞️ **Análisis de video pregrabado** con controles de reproducción y velocidad
- 🔊 **Retroalimentación auditiva** no bloqueante con pyttsx3
- 🎨 **Interfaz moderna** con customtkinter (modo oscuro)
- ✅ **Detección de mala forma** con alertas en rojo
- 🧱 **Arquitectura SOLID por capas** (`core` / `exercises` / `gui`) — cada módulo, una responsabilidad
- 🔌 **Extensible** — agrega ejercicios sin modificar el motor

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

## Pruebas

```bash
python -m unittest discover tests/
```

Las pruebas cubren el motor geométrico, la sanitización de audio, el presentador de métricas, el renderizado de frames y el contrato del panel lateral. Las que dependen de OpenCV se omiten automáticamente si la librería no está instalada.

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
├── main.py                     # Punto de entrada e inyección de dependencias
├── requirements.txt            # Dependencias
├── README.md                   # Este archivo
├── core/                       # Lógica de negocio e infraestructura
│   ├── config.py               # Configuración (dataclasses congeladas)
│   ├── math_utils.py           # Cálculo de ángulos + filtro de suavizado
│   ├── pose_detector.py        # Wrapper de MediaPipe (Motor de Inferencia)
│   ├── audio_feedback.py       # TTS no bloqueante (Feedback Manager)
│   ├── camera_detector.py      # Detección de cámaras (DirectShow/PowerShell)
│   └── exercise_tracker.py     # Orquestador de ejercicios
├── exercises/                  # Ejercicios (registro automático, extensible)
│   ├── base.py                 # Clase base + registro + ExerciseResult
│   ├── bicep_curl.py           # Curl de Bíceps
│   ├── squat.py                # Sentadillas
│   ├── lunge.py                # Zancadas
│   ├── high_knees.py           # Elevación de Rodillas
│   └── side_bends.py           # Inclinación Lateral
├── gui/                        # Capa de presentación (customtkinter)
│   ├── main_app.py             # Ventana principal + ciclo de video
│   ├── sidebar.py              # Panel lateral (SidebarView + callbacks)
│   ├── metrics_presenter.py    # Actualización de métricas del panel
│   ├── frame_renderer.py       # Conversión de frame OpenCV → CTkImage
│   ├── dialogs.py              # Diálogos modales (error/info)
│   └── drawing_utils.py        # Dibujo de overlays sobre el frame
└── tests/                      # Pruebas unitarias
    ├── test_math_utils.py      # Ángulos y filtro de suavizado
    ├── test_audio.py           # Sanitización de texto para TTS
    ├── test_metrics_presenter.py  # Mapeo de estado/forma del panel
    ├── test_frame_renderer.py  # Render y relación de aspecto
    └── test_sidebar.py         # Contrato y cableado del panel lateral
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
