import unittest

try:
    import numpy as np
    import customtkinter as ctk
    from gui.frame_renderer import FrameRenderer  # importa cv2
    _DEPS_OK = True
except Exception:
    _DEPS_OK = False


@unittest.skipUnless(_DEPS_OK, "cv2/customtkinter no disponibles en este entorno")
class TestFrameRenderer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # CTkImage puede requerir un root Tk por defecto; lo creamos oculto.
        try:
            cls.root = ctk.CTk()
            cls.root.withdraw()
        except Exception as e:  # pragma: no cover - entornos sin display
            raise unittest.SkipTest(f"No se pudo crear root Tk: {e}")

    @classmethod
    def tearDownClass(cls):
        try:
            cls.root.destroy()
        except Exception:
            pass

    @staticmethod
    def _frame(h=480, w=640):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_returns_none_for_zero_box(self):
        self.assertIsNone(FrameRenderer.render(self._frame(), 0, 100))
        self.assertIsNone(FrameRenderer.render(self._frame(), 100, 0))

    def test_returns_none_for_negative_box(self):
        self.assertIsNone(FrameRenderer.render(self._frame(), -5, 100))

    def test_returns_ctkimage_for_valid_box(self):
        img = FrameRenderer.render(self._frame(480, 640), 320, 240)
        self.assertIsInstance(img, ctk.CTkImage)

    def test_preserves_aspect_ratio(self):
        # frame 640x480 (ancho x alto) en caja 320x240 -> escala 0.5 -> 320x240
        img = FrameRenderer.render(self._frame(480, 640), 320, 240)
        self.assertEqual(img.cget("size"), (320, 240))

    def test_fits_within_box_when_aspect_differs(self):
        # frame cuadrado 480x480 en caja ancha 640x240 -> limita el alto
        img = FrameRenderer.render(self._frame(480, 480), 640, 240)
        self.assertEqual(img.cget("size"), (240, 240))


if __name__ == "__main__":
    unittest.main()
