"""
PhysioMate - Detección de Cámaras (Camera Detection)

Responsabilidad única: enumerar las cámaras de captura de vídeo disponibles
en el sistema y mapear cada nombre legible a su índice de OpenCV.

No depende de la interfaz gráfica; es infraestructura pura sobre OpenCV /
DirectShow (Windows). Extraído de MainApp para respetar SRP.
"""

from __future__ import annotations

import logging

import cv2

logger = logging.getLogger(__name__)


class CameraDetector:
    """Detecta cámaras disponibles y resuelve sus nombres legibles."""

    @staticmethod
    def _dshow_names() -> list:
        """Enumera los dispositivos de captura de vídeo DirectShow.

        Usa comtypes para acceder a ICreateDevEnum, la misma API que
        OpenCV utiliza internamente, garantizando que el orden de los
        nombres coincida exactamente con los índices 0, 1, 2...

        Returns:
            Lista de nombres de dispositivos en orden de índice.
        """
        import comtypes
        import comtypes.client
        from comtypes import GUID
        import ctypes

        names = []
        try:
            comtypes.CoInitialize()

            # GUIDs de DirectShow
            CLSID_SystemDeviceEnum = GUID("{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}")
            CLSID_VideoInputDeviceCategory = GUID("{860BB310-5D01-11d0-BD3B-00A0C911CE86}")

            # ICreateDevEnum
            IID_ICreateDevEnum = GUID("{29840822-5B84-11D0-BD3B-00A0C911CE86}")
            IID_IEnumMoniker   = GUID("{00000102-0000-0000-C000-000000000046}")
            IID_IMoniker       = GUID("{0000000F-0000-0000-C000-000000000046}")
            IID_IPropertyBag   = GUID("{55272A00-42CB-11CE-8135-00AA004BB851}")

            dev_enum = comtypes.client.CreateObject(
                CLSID_SystemDeviceEnum,
                interface=comtypes.IUnknown,
            )

            # QueryInterface to ICreateDevEnum
            ICreateDevEnum = comtypes.GUID("{29840822-5B84-11D0-BD3B-00A0C911CE86}")
            p_create_dev_enum = ctypes.POINTER(comtypes.IUnknown)()
            dev_enum.QueryInterface(ctypes.byref(ICreateDevEnum),
                                    ctypes.byref(p_create_dev_enum))

        except Exception:
            pass

        # Fallback: si comtypes falla, usar PowerShell con búsqueda ampliada
        if not names:
            try:
                import subprocess, json
                # Consulta ampliada: clase Camera + dispositivos de imagen USB
                cmd = [
                    "powershell", "-NoProfile", "-Command",
                    "(Get-PnpDevice -Status OK | Where-Object {"
                    " $_.Class -eq 'Camera' -or $_.Class -eq 'Image'"
                    " -or ($_.FriendlyName -match 'cam|webcam|video|capture')"
                    "} | Sort-Object InstanceId"
                    " | Select-Object -ExpandProperty FriendlyName"
                    " | ConvertTo-Json -Compress)",
                ]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
                if r.returncode == 0 and r.stdout.strip():
                    raw = json.loads(r.stdout.strip())
                    names = [raw] if isinstance(raw, str) else list(raw)
            except Exception:
                pass

        return names

    @staticmethod
    def detect(max_to_check: int = 6) -> dict:
        """Detecta cámaras disponibles y retorna {nombre: índice}."""
        # Obtener nombres reales via DirectShow / PowerShell
        real_names = CameraDetector._dshow_names()

        result: dict = {}
        name_idx = 0
        for cv_idx in range(max_to_check):
            cap = cv2.VideoCapture(cv_idx, cv2.CAP_DSHOW)
            if cap is not None and cap.isOpened():
                cap.release()
                # Asignar nombre real si está disponible, si no usar fallback
                if name_idx < len(real_names):
                    label = real_names[name_idx]
                else:
                    label = f"Cámara {cv_idx}"
                result[label] = cv_idx
                name_idx += 1

        logger.info("Cámaras detectadas: %s", {v: k for k, v in result.items()})
        return result
