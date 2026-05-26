import unittest
from core.audio_feedback import AudioFeedback

class TestAudioFeedback(unittest.TestCase):
    def test_sanitize_accents(self):
        text = "¡Atención! Sube el talón izquierdo y baja el fémur más allá del límite."
        clean = AudioFeedback._sanitize(text)
        self.assertEqual(clean, "¡Atencion! Sube el talon izquierdo y baja el femur mas alla del limite.")

    def test_sanitize_no_changes(self):
        text = "Hello World"
        clean = AudioFeedback._sanitize(text)
        self.assertEqual(clean, text)

    def test_sanitize_special_chars(self):
        text = "Pingüino ñandú"
        clean = AudioFeedback._sanitize(text)
        self.assertEqual(clean, "Pinguino nandu")

if __name__ == '__main__':
    unittest.main()
