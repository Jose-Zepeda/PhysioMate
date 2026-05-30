import sys
import unittest
from unittest.mock import MagicMock

# La cadena de importación de gui.sidebar pasa por exercises/__init__, que
# importa los ejercicios y arrastra mediapipe/cv2. En entornos sin esas libs
# (p.ej. CI ligero) insertamos stubs SOLO si faltan: setdefault no pisa los
# módulos reales cuando sí están instalados.
for _mod in ("mediapipe", "cv2"):
    sys.modules.setdefault(_mod, MagicMock())

try:
    import customtkinter as ctk
    from gui.sidebar import SidebarView, SidebarCallbacks
    _IMPORT_OK = True
except Exception:
    _IMPORT_OK = False


@unittest.skipUnless(_IMPORT_OK, "customtkinter/gui.sidebar no importables")
class TestSidebarView(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Un único root para toda la clase: evita el ruido de "invalid command
        # name ... (after script)" que aparece al destruir el root mientras
        # customtkinter aún tiene callbacks after() pendientes (DPI, titlebar).
        try:
            cls.root = ctk.CTk()
            cls.root.withdraw()
            cls.root.update()  # deja correr los after() iniciales mientras es válido
        except Exception as e:  # pragma: no cover - entornos sin display
            raise unittest.SkipTest(f"No se pudo crear root Tk: {e}")

    @classmethod
    def _cancel_pending_afters(cls):
        # customtkinter reprograma after() en varios widgets (DPI, titlebar,
        # scroll). Los cancelamos para que no disparen sobre widgets ya
        # destruidos y ensucien stderr con "invalid command name".
        try:
            for after_id in cls.root.tk.eval("after info").split():
                try:
                    cls.root.after_cancel(after_id)
                except Exception:
                    pass
        except Exception:
            pass

    @classmethod
    def tearDownClass(cls):
        try:
            cls._cancel_pending_afters()
            cls.root.update_idletasks()
            cls.root.destroy()
        except Exception:
            pass

    def setUp(self):
        self.calls = []
        mk = lambda name: (lambda *a, **k: self.calls.append(name))
        self.cb = SidebarCallbacks(
            on_exercise_change=mk("exercise"),
            on_camera_change=mk("camera"),
            on_start=mk("start"),
            on_stop=mk("stop"),
            on_reset=mk("reset"),
            on_audio_toggle=mk("audio"),
            on_load_video=mk("load"),
            on_seek_drag=mk("drag"),
            on_seek_start=mk("seekstart"),
            on_seek_release=mk("seekrel"),
            on_video_pause_resume=mk("pause"),
            on_stop_video=mk("stopvideo"),
            on_speed_change=mk("speed"),
        )
        self.sv = SidebarView(
            self.root,
            available_cameras={"Integrada": 0, "USB": 1},
            speed_options={"1x": 1.0, "2x": 2.0},
            callbacks=self.cb,
            width=280,
        )

    def tearDown(self):
        try:
            self.sv.destroy()  # destruye solo el sidebar, el root persiste
            self._cancel_pending_afters()
            self.root.update_idletasks()
        except Exception:
            pass

    def test_exposes_public_contract(self):
        # Atributos que MainApp consume; si falta uno, sería AttributeError en runtime.
        for attr in (
            "exercise_var", "audio_var", "start_btn", "stop_btn",
            "video_name_label", "video_progress", "video_time_label",
            "play_pause_btn", "stop_video_btn", "metrics", "default_camera",
        ):
            self.assertTrue(hasattr(self.sv, attr), f"falta atributo público: {attr}")

    def test_default_camera_skips_integrated(self):
        # Preselecciona la primera cámara con índice != 0.
        self.assertEqual(self.sv.default_camera, "USB")

    def test_buttons_invoke_callbacks(self):
        for btn, name in (
            (self.sv.start_btn, "start"),
            (self.sv.stop_btn, "stop"),
            (self.sv.play_pause_btn, "pause"),
            (self.sv.stop_video_btn, "stopvideo"),
        ):
            btn.configure(state="normal")  # los deshabilitados ignoran invoke()
            btn.invoke()
        self.assertEqual(self.calls, ["start", "stop", "pause", "stopvideo"])

    def test_metrics_presenter_is_wired(self):
        # El presentador expuesto debe poder actualizar sin error.
        self.sv.metrics.update(None)


if __name__ == "__main__":
    unittest.main()
