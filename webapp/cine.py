"""Cine-loop rendering and encoding for shareable CT review clips.

Streamlit-free so it can be unit-tested and reused by the CLI. Frames are
rendered with the same window/overlay logic as the in-app viewer, then encoded
to H.264 MP4 (via system ffmpeg) or animated GIF (PIL) as a fallback.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile

import numpy as np
from matplotlib import colormaps
from PIL import Image

from review import apply_window

DEFAULT_FPS = 12
DEFAULT_SPAN = 48          # slices per clip when not sweeping the whole volume
MAX_FRAMES = 400           # safety cap for generated clips
GIF_MS_MIN = 20


def render_slice(image, mask, axis, index, display_mode='normalized', window_level=60,
                 window_width=400, overlay=False, opacity=0.45, selected=0):
    """Render one plane as a uint8 RGB frame (viewer-identical look)."""
    gray = np.take(image, index, axis=axis)
    if display_mode == 'hu':
        gray = apply_window(gray, window_level, window_width)
    else:
        gray = np.clip(gray, 0, 1)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    if overlay and mask is not None:
        labels = np.take(mask, index, axis=axis)
        keep = labels > 0 if selected == 0 else labels == selected
        colors = colormaps['turbo'](labels / 36)[..., :3]
        rgb[keep] = rgb[keep] * (1 - opacity) + colors[keep] * opacity
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def downscale(frame, scale):
    """Resize a frame by `scale` (1.0 = unchanged)."""
    if scale is None or scale >= 0.999:
        return frame
    height, width = frame.shape[:2]
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return np.asarray(Image.fromarray(frame).resize(size, Image.BILINEAR))


def sweep_range(length, center=None, span=DEFAULT_SPAN):
    """Slice range centred on `center` (default: middle), clamped to the volume."""
    if length <= 0:
        return 0, 0, 1
    span = max(1, min(int(span), length))
    center = length // 2 if center is None else int(center)
    start = max(0, min(center - span // 2, length - span))
    end = min(length - 1, start + span - 1)
    return start, end, 1


def render_frames(image, mask=None, axis=0, start=0, end=None, step=1, *,
                  display_mode='normalized', window_level=60, window_width=400,
                  overlay=False, opacity=0.45, selected=0, scale=0.5):
    """Render a slice range into a list of RGB frames."""
    last = (image.shape[axis] - 1) if end is None else int(end)
    start = max(0, int(start))
    last = min(int(last), image.shape[axis] - 1)
    frames = []
    for index in range(start, last + 1, max(1, int(step))):
        frame = render_slice(image, mask, axis, index, display_mode, window_level,
                             window_width, overlay, opacity, selected)
        frames.append(downscale(frame, scale))
        if len(frames) >= MAX_FRAMES:
            break
    return frames


def ffmpeg_available():
    """True when a system ffmpeg is on PATH (needed for H.264 MP4)."""
    return shutil.which('ffmpeg') is not None


def encode_gif(frames, fps=DEFAULT_FPS, loop=0):
    """Encode frames as an animated GIF (works everywhere, no extra binaries)."""
    if not frames:
        raise ValueError('no frames to encode')
    images = [Image.fromarray(frame) for frame in frames]
    buffer = io.BytesIO()
    images[0].save(
        buffer, format='GIF', save_all=True, append_images=images[1:],
        duration=max(GIF_MS_MIN, int(1000 / max(1, int(fps)))), loop=loop, optimize=True,
    )
    return buffer.getvalue()


def encode_mp4(frames, fps=DEFAULT_FPS):
    """Encode frames as H.264 MP4 via system ffmpeg (browser + WhatsApp friendly)."""
    if not frames:
        raise ValueError('no frames to encode')
    if not ffmpeg_available():
        raise RuntimeError('ffmpeg is not available for MP4 encoding')
    # yuv420p needs even dimensions
    height, width = frames[0].shape[:2]
    height, width = height - height % 2, width - width % 2
    payload = b''.join(np.ascontiguousarray(frame[:height, :width]).tobytes() for frame in frames)
    # Encode to a real file: `+faststart` seeks back to relocate the moov atom,
    # which is impossible when piping to stdout.
    with tempfile.TemporaryDirectory(prefix='radar_clip_') as tmp:
        output = os.path.join(tmp, 'clip.mp4')
        command = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
            '-r', str(max(1, int(fps))), '-i', '-', '-an',
            '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            output,
        ]
        proc = subprocess.run(command, input=payload, capture_output=True)
        if proc.returncode or not os.path.exists(output) or os.path.getsize(output) == 0:
            raise RuntimeError(f'ffmpeg encode failed: {proc.stderr.decode(errors="replace")[:300]}')
        with open(output, 'rb') as handle:
            return handle.read()


def build_clip(frames, fps=DEFAULT_FPS, prefer='mp4'):
    """Encode frames and return (data, extension, mime).

    Prefers H.264 MP4 when ffmpeg is present; falls back to an animated GIF.
    """
    if not frames:
        raise ValueError('no frames to encode')
    if prefer == 'mp4' and ffmpeg_available():
        try:
            return encode_mp4(frames, fps), 'mp4', 'video/mp4'
        except Exception:
            pass  # fall through to GIF
    return encode_gif(frames, fps), 'gif', 'image/gif'
