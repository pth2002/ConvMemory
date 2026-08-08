"""Render the README demo GIF from real captured output.

    python experiments/make_demo_gif.py

Every line of text below is copied from an actual run of
`examples/demo_locomo.py` on the held-out conversations of the released
checkpoint's split (seed 23). Nothing here is illustrative: the question, the
two rankings, and the aggregate numbers are what that script printed.

Reproduce the source of these lines with:

    python examples/demo_locomo.py --data data/locomo10.json --device cuda
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "docs" / "assets" / "demo.gif"

FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
FONT_REGULAR = FONT_DIR / "DejaVuSansMono.ttf"
FONT_BOLD = FONT_DIR / "DejaVuSansMono-Bold.ttf"

WIDTH = 812
PADDING = 26
LINE_HEIGHT = 25
FONT_SIZE = 15

BACKGROUND = "#14161a"
CHROME = "#20242b"
TEXT = "#d7dae0"
DIM = "#7d8492"
PROMPT = "#57b6ff"
GOOD = "#3ddc97"
BAD = "#ff6b6b"
ACCENT = "#ffd479"

# --- real output, transcribed from examples/demo_locomo.py -------------------

QUESTION = "When did John take a trip to the Rocky Mountains?"

DENSE_ROWS = [
    "John: Wow, that sounds awesome! How challenging was the trek",
    "John: We went camping in the mountains and it was stunning!",
    "Tim: The book mentioned that the trek was tough but worth it",
    "John: I stumbled across this spot while hiking. The sound of",
    "John: Wow, great view! Have you visited any other places?",
]

CONV_ROWS = [
    ("John: ...This was my Rocky Mountains trip last year and it was stunning.", True),
    ("John: We went camping in the mountains and it was stunning!", False),
    ("Tim: I snapped that pic on my trip to the Smoky Mountains", False),
    ("John: I loved just chilling and taking in the beauty of nature", False),
    ("John: We're planning to take a team trip next month", False),
]

SUMMARY = [
    ("937 questions, candidate pools of 369-680 memories", DIM),
    ("hit@5    dense 49.5%   ->   dense + ConvMemory 72.8%", GOOD),
    ("hit@10   dense 60.7%   ->   dense + ConvMemory 82.1%", GOOD),
    ("+18.5 ms per query", ACCENT),
]


def load_font(path, size):
    return ImageFont.truetype(str(path), size)


class Frame:
    """One terminal frame; lines are (text, colour, bold) tuples."""

    def __init__(self):
        self.lines: list[tuple[str, str, bool]] = []

    def add(self, text="", colour=TEXT, bold=False):
        self.lines.append((text, colour, bold))
        return self

    def copy(self):
        clone = Frame()
        clone.lines = list(self.lines)
        return clone


def render(frame: Frame, height: int, regular, bold) -> Image.Image:
    image = Image.new("RGB", (WIDTH, height), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.rectangle([0, 0, WIDTH, 34], fill=CHROME)
    for index, colour in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        x = 18 + index * 18
        draw.ellipse([x, 12, x + 10, 22], fill=colour)
    draw.text((WIDTH // 2 - 72, 9), "convmemory - demo", font=regular, fill=DIM)

    y = 34 + PADDING
    for text, colour, is_bold in frame.lines:
        draw.text((PADDING, y), text, font=bold if is_bold else regular, fill=colour)
        y += LINE_HEIGHT
    return image


def build_frames(regular, bold):
    """Returns [(frame, duration_ms), ...]."""
    frames: list[tuple[Frame, int]] = []

    base = Frame()
    base.add("$ python examples/demo_locomo.py --data data/locomo10.json", PROMPT)
    base.add()
    base.add("Held-out conversations (seed 23): conv-26, conv-30, conv-41,", DIM)
    base.add("                                  conv-43, conv-50", DIM)
    base.add("Candidate pool: 680 memories", DIM)
    base.add()
    frames.append((base.copy(), 1400))

    # type the question out
    typing = base.copy()
    typing.add("Q: ", ACCENT, True)
    for index in range(1, len(QUESTION) + 1, 3):
        step = typing.copy()
        step.lines[-1] = (f"Q: {QUESTION[:index]}", ACCENT, True)
        frames.append((step, 40))
    settled = typing.copy()
    settled.lines[-1] = (f"Q: {QUESTION}", ACCENT, True)
    frames.append((settled.copy(), 900))

    # dense results, one line at a time
    dense = settled.copy()
    dense.add()
    dense.add("dense retrieval, top 5", DIM, True)
    frames.append((dense.copy(), 350))
    for position, row in enumerate(DENSE_ROWS, start=1):
        dense.add(f"  {position}.  x  {row}", TEXT)
        frames.append((dense.copy(), 220))
    dense.add("      the memory that answers the question is not here", DIM)
    frames.append((dense.copy(), 1500))

    # convmemory results
    conv = dense.copy()
    conv.add()
    conv.add("dense + ConvMemory, top 5", GOOD, True)
    frames.append((conv.copy(), 350))
    for position, (row, is_gold) in enumerate(CONV_ROWS, start=1):
        marker = "OK" if is_gold else "  "
        conv.add(f"  {position}. {marker}  {row}", GOOD if is_gold else TEXT, is_gold)
        frames.append((conv.copy(), 260))
    frames.append((conv.copy(), 1600))

    summary = conv.copy()
    summary.add()
    for text, colour in SUMMARY:
        summary.add(text, colour, colour is not DIM)
        frames.append((summary.copy(), 320))
    frames.append((summary.copy(), 3600))

    return frames


def main():
    regular = load_font(FONT_REGULAR, FONT_SIZE)
    bold = load_font(FONT_BOLD, FONT_SIZE)

    frames = build_frames(regular, bold)
    height = 34 + PADDING * 2 + LINE_HEIGHT * max(len(f.lines) for f, _ in frames)

    images = [render(frame, height, regular, bold) for frame, _ in frames]
    durations = [duration for _, duration in frames]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        OUT_PATH,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"wrote {OUT_PATH}  ({len(images)} frames, {size_kb:.0f} KB, {height}px tall)")


if __name__ == "__main__":
    main()
