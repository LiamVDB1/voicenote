"""Unit tests for the Voxtral runaway-repetition detector and the
silence-aligned boundary picker. Both are pure, fast, and standalone —
no Mistral API, no ffmpeg, no model files needed."""
from __future__ import annotations

import unittest

from voicenote.engines.voxtral import VoxtralEngine


class RepetitionDetectorTests(unittest.TestCase):
    detect = staticmethod(VoxtralEngine._detect_repetition_loop)

    def test_clean_text_is_not_a_loop(self):
        text = (
            "Goedemorgen iedereen, vandaag bespreken we de begroting. "
            "Patrick heeft de cijfers voorbereid en we lopen de fiches door. "
            "Daarna stemmen we over het nieuwe bestuur."
        )
        is_loop, _ = self.detect(text)
        self.assertFalse(is_loop)

    def test_short_text_below_threshold_is_not_a_loop(self):
        # Even with exact repeats, too few words → not flagged.
        is_loop, _ = self.detect("ja ja ja ja ja")
        self.assertFalse(is_loop)

    def test_natural_repeated_short_words_not_flagged(self):
        # Real speech often has small clusters like "Ja. Ja. Ja." — should
        # not trip the detector at default thresholds.
        text = " ".join(["Ja."] * 6) + " dat klopt zeker en vast hoor."
        is_loop, _ = self.detect(text)
        self.assertFalse(is_loop)

    def test_runaway_loop_is_flagged(self):
        # The exact phrase that ate ~20 minutes of meeting audio in prod
        # (real prod data had 890 back-to-back repeats; 50 is plenty).
        phrase = "Ik heb het niet opgesteld, maar ik heb het niet opgesteld."
        text = "Voorafgaande zin hier. " + (phrase + " ") * 50
        is_loop, sample = self.detect(text)
        self.assertTrue(is_loop)
        self.assertIsNotNone(sample)
        self.assertIn("opgesteld", (sample or "").lower())

    def test_loop_in_middle_is_flagged(self):
        before = "Begin van het gesprek met inhoudelijke discussie. " * 3
        loop = "ik heb het niet opgesteld maar ik heb het niet opgesteld " * 30
        after = "Hier gaat het gesprek verder met nieuwe onderwerpen. " * 3
        is_loop, _ = self.detect(before + loop + after)
        self.assertTrue(is_loop)

    def test_punctuation_variation_does_not_hide_loop(self):
        # Same phrase, slight punctuation drift between repeats — detector
        # normalizes trailing punctuation so this should still flag.
        phrase_a = "ik heb het niet opgesteld maar ik heb het niet opgesteld."
        phrase_b = "ik heb het niet opgesteld, maar ik heb het niet opgesteld!"
        text = ((phrase_a + " ") * 5 + (phrase_b + " ") * 5) * 3
        is_loop, _ = self.detect(text)
        self.assertTrue(is_loop)

    def test_empty_text(self):
        is_loop, sample = self.detect("")
        self.assertFalse(is_loop)
        self.assertIsNone(sample)


class BoundaryPickerTests(unittest.TestCase):
    pick = staticmethod(VoxtralEngine._pick_boundaries)

    def test_no_silences_falls_back_to_fixed_times(self):
        cuts = self.pick(
            target_chunk_sec=600.0,
            total_duration=1800.0,
            silences=[],
            search_window_sec=60.0,
        )
        self.assertEqual(cuts, [600.0, 1200.0])

    def test_silence_within_window_snaps_to_silence_centre(self):
        # A 4s silence centred at 610s is within ±60s of the 600s target.
        cuts = self.pick(
            target_chunk_sec=600.0,
            total_duration=1800.0,
            silences=[(608.0, 612.0)],
            search_window_sec=60.0,
        )
        self.assertAlmostEqual(cuts[0], 610.0, places=3)

    def test_longest_silence_within_window_wins(self):
        cuts = self.pick(
            target_chunk_sec=600.0,
            total_duration=1800.0,
            silences=[(595.0, 596.0), (605.0, 612.0), (640.0, 640.5)],
            search_window_sec=60.0,
        )
        # The 7s silence should win over the 1s and 0.5s ones.
        self.assertAlmostEqual(cuts[0], 608.5, places=3)

    def test_silence_outside_window_is_ignored(self):
        # 700s silence is 100s away from target 600s, outside the 60s window.
        cuts = self.pick(
            target_chunk_sec=600.0,
            total_duration=1800.0,
            silences=[(700.0, 705.0)],
            search_window_sec=60.0,
        )
        self.assertEqual(cuts[0], 600.0)

    def test_short_recording_returns_no_internal_cuts(self):
        cuts = self.pick(
            target_chunk_sec=600.0,
            total_duration=400.0,
            silences=[],
            search_window_sec=60.0,
        )
        self.assertEqual(cuts, [])


if __name__ == "__main__":
    unittest.main()
