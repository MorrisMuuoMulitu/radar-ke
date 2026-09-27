"""Tests for cine clip rendering and encoding."""
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cine


def synthetic_volume():
    image = np.full((10, 12, 14), 0.5, dtype=np.float32)
    mask = np.zeros((10, 12, 14), dtype=np.uint8)
    mask[2:6, 3:7, 4:8] = 21
    return image, mask


class RenderTests(unittest.TestCase):
    def test_render_slice_returns_uint8_rgb(self):
        image, mask = synthetic_volume()
        frame = cine.render_slice(image, mask, axis=0, index=5)
        self.assertEqual(frame.shape, (12, 14, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_render_slice_applies_hu_window(self):
        image = np.array([[-1000.0, 40.0, 400.0]], dtype=np.float32).reshape(1, 1, 3)
        frame = cine.render_slice(image, None, axis=0, index=0, display_mode='hu',
                                  window_level=40, window_width=400)
        self.assertEqual(int(frame[0, 0, 0]), 0)
        self.assertEqual(int(frame[0, 2, 0]), 255)

    def test_render_slice_overlay_marks_only_selected_organ(self):
        image, mask = synthetic_volume()
        plain = cine.render_slice(image, mask, axis=0, index=4, overlay=False)
        overlaid = cine.render_slice(image, mask, axis=0, index=4, overlay=True, selected=21)
        self.assertFalse(np.array_equal(plain, overlaid))
        self.assertGreater(int(np.abs(overlaid.astype(int) - plain.astype(int)).sum()), 0)
        # a different organ selection leaves pixels untouched
        other = cine.render_slice(image, mask, axis=0, index=4, overlay=True, selected=20)
        self.assertTrue(np.array_equal(plain, other))

    def test_render_frames_counts_and_downscales(self):
        image, mask = synthetic_volume()
        frames = cine.render_frames(image, mask, axis=0, start=2, end=6, scale=0.5)
        self.assertEqual(len(frames), 5)
        self.assertEqual(frames[0].shape, (6, 7, 3))

    def test_sweep_range_centres_and_clamps(self):
        self.assertEqual(cine.sweep_range(100, center=50, span=48)[:2], (26, 73))
        self.assertEqual(cine.sweep_range(10, center=5, span=48)[:2], (0, 9))
        start, end, step = cine.sweep_range(0)
        self.assertEqual((start, end, step), (0, 0, 1))


class EncodingTests(unittest.TestCase):
    def test_encode_gif_round_trip_frame_count(self):
        frames = [np.full((8, 8, 3), value, dtype=np.uint8) for value in (0, 120, 255)]
        data = cine.encode_gif(frames, fps=10)
        self.assertTrue(data.startswith(b'GIF'))
        with Image.open(io.BytesIO(data)) as img:
            self.assertEqual(getattr(img, 'n_frames', 1), 3)

    def test_encode_gif_rejects_empty(self):
        with self.assertRaises(ValueError):
            cine.encode_gif([])

    def test_encode_mp4_requires_ffmpeg(self):
        frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
        with mock.patch.object(cine.shutil, 'which', return_value=None):
            with self.assertRaises(RuntimeError):
                cine.encode_mp4(frames)

    def test_build_clip_falls_back_to_gif_without_ffmpeg(self):
        frames = [np.zeros((8, 8, 3), dtype=np.uint8), np.full((8, 8, 3), 200, dtype=np.uint8)]
        with mock.patch.object(cine.shutil, 'which', return_value=None):
            data, extension, mime = cine.build_clip(frames, fps=8)
        self.assertEqual((extension, mime), ('gif', 'image/gif'))
        self.assertTrue(data.startswith(b'GIF'))

    @unittest.skipUnless(cine.ffmpeg_available(), 'ffmpeg not installed')
    def test_encode_mp4_produces_h264_container(self):
        frames = [np.full((16, 16, 3), value, dtype=np.uint8) for value in (0, 60, 120, 180)]
        data = cine.encode_mp4(frames, fps=6)
        self.assertGreater(len(data), 100)
        self.assertIn(b'ftyp', data[:32])


if __name__ == '__main__':
    unittest.main()
