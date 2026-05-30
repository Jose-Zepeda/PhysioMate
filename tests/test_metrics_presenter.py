import unittest
from types import SimpleNamespace

from gui.metrics_presenter import MetricsPresenter


class _FakeLabel:
    """Doble de prueba: captura los kwargs de cada configure()."""

    def __init__(self):
        self.kw = {}

    def configure(self, **kw):
        self.kw.update(kw)


def _result(state="Arriba", form_ok=True, rep_count=0, angle=0.0, feedback_message=""):
    return SimpleNamespace(
        state=state,
        form_ok=form_ok,
        rep_count=rep_count,
        angle=angle,
        feedback_message=feedback_message,
    )


class TestMetricsPresenter(unittest.TestCase):
    def setUp(self):
        self.rep = _FakeLabel()
        self.angle = _FakeLabel()
        self.state = _FakeLabel()
        self.form = _FakeLabel()
        self.feedback = _FakeLabel()
        self.p = MetricsPresenter(
            self.rep, self.angle, self.state, self.form, self.feedback
        )

    def test_none_resets_all_widgets(self):
        self.p.update(None)
        self.assertEqual(self.rep.kw["text"], "0")
        self.assertEqual(self.angle.kw["text"], "-- °")
        self.assertEqual(self.state.kw["text"], "Inactivo")
        self.assertEqual(self.form.kw["text"], "● CORRECTA")
        self.assertEqual(self.form.kw["text_color"], "#00ff88")
        self.assertEqual(self.feedback.kw["text"], "")

    def test_values_formatting(self):
        self.p.update(_result(rep_count=7, angle=91.234, state="Abajo", feedback_message="baja mas"))
        self.assertEqual(self.rep.kw["text"], "7")
        self.assertEqual(self.angle.kw["text"], "91.2°")  # un decimal
        self.assertEqual(self.state.kw["text"], "Abajo")
        self.assertEqual(self.feedback.kw["text"], "baja mas")

    def test_form_ok_shows_correcta(self):
        self.p.update(_result(state="Arriba", form_ok=True))
        self.assertEqual(self.form.kw["text"], "● CORRECTA")
        self.assertEqual(self.form.kw["text_color"], "#00ff88")

    def test_form_not_ok_shows_incorrecta(self):
        self.p.update(_result(state="Abajo", form_ok=False))
        self.assertEqual(self.form.kw["text"], "● INCORRECTA")
        self.assertEqual(self.form.kw["text_color"], "#ff4444")

    def test_waiting_states_override_form_ok(self):
        # Aunque form_ok sea True, un estado de espera muestra ESPERANDO.
        for state in ["Esperando deteccion", "Brazo no visible", "Cuerpo detectado", ""]:
            with self.subTest(state=state):
                form = _FakeLabel()
                p = MetricsPresenter(_FakeLabel(), _FakeLabel(), _FakeLabel(), form, _FakeLabel())
                p.update(_result(state=state, form_ok=True))
                self.assertEqual(form.kw["text"], "● ESPERANDO")
                self.assertEqual(form.kw["text_color"], "#AAAAAA")

    def test_waiting_detection_is_case_insensitive(self):
        self.p.update(_result(state="DETECTADO parcialmente", form_ok=False))
        self.assertEqual(self.form.kw["text"], "● ESPERANDO")


if __name__ == "__main__":
    unittest.main()
