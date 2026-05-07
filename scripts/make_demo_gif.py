from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "assets" / "hardware-prototype.jpg"
OUTPUT = ROOT / "assets" / "operation-demo.gif"

W, H = 760, 428
PANEL_W = 280
PHOTO_W = W - PANEL_W
FRAMES_PER_STEP = 7

STEPS = [
    {
        "title": "1. Capture screen",
        "body": "ADB grabs the phone screen and sends pixels into the Python pipeline.",
        "cmd": "STATUS",
        "pos": (0, 0, 0),
        "dot": (0.18, 0.82),
        "accent": "#38bdf8",
    },
    {
        "title": "2. OCR + vision",
        "body": "OpenCV improves the frame. OCR and templates locate target UI elements.",
        "cmd": "ANALYZE",
        "pos": (10, 20, 5),
        "dot": (0.34, 0.68),
        "accent": "#22c55e",
    },
    {
        "title": "3. Plan action",
        "body": "Task rules or AI choose a physical click, press, or swipe action.",
        "cmd": "G X10 Y20 Z5 M1",
        "pos": (10, 20, 5),
        "dot": (0.48, 0.55),
        "accent": "#f59e0b",
    },
    {
        "title": "4. Move gantry",
        "body": "The TCP controller converts millimeters to TB6600 step pulses.",
        "cmd": "X 15  |  Y -5  |  Z 10",
        "pos": (25, 15, 15),
        "dot": (0.66, 0.46),
        "accent": "#a78bfa",
    },
    {
        "title": "5. Touch target",
        "body": "The stylus reaches the target. The loop reads the next screen state.",
        "cmd": "OK G X25 Y15 Z15",
        "pos": (25, 15, 15),
        "dot": (0.72, 0.36),
        "accent": "#fb7185",
    },
]


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def rounded_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int, fill: str, outline: str | None = None) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline)


def fit_photo() -> Image.Image:
    img = Image.open(PHOTO).convert("RGB")
    scale = max(PHOTO_W / img.width, H / img.height)
    resized = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - PHOTO_W) // 2
    top = (resized.height - H) // 2
    return resized.crop((left, top, left + PHOTO_W, top + H))


def blend_hex(a: str, b: str, t: float) -> tuple[int, int, int]:
    def parse(color: str) -> tuple[int, int, int]:
        color = color.lstrip("#")
        return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)

    ar, ag, ab = parse(a)
    br, bg, bb = parse(b)
    return (
        int(ar + (br - ar) * t),
        int(ag + (bg - ag) * t),
        int(ab + (bb - ab) * t),
    )


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    probe = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(probe)
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_frame(step_index: int, progress: float, photo: Image.Image) -> Image.Image:
    step = STEPS[step_index]
    next_step = STEPS[(step_index + 1) % len(STEPS)]
    accent = "#%02x%02x%02x" % blend_hex(step["accent"], next_step["accent"], progress * 0.35)

    frame = Image.new("RGB", (W, H), "#0f172a")
    frame.paste(photo, (0, 0))

    overlay = Image.new("RGBA", (PHOTO_W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle((0, 0, PHOTO_W, H), fill=(8, 15, 28, 48))
    frame.paste(Image.alpha_composite(Image.new("RGBA", (PHOTO_W, H), (0, 0, 0, 0)), overlay).convert("RGB"), (0, 0), overlay)

    draw = ImageDraw.Draw(frame)
    title_font = load_font(23, True)
    body_font = load_font(14)
    small_font = load_font(12)
    mono_font = load_font(12)
    big_font = load_font(30, True)

    dot_x = int(step["dot"][0] * PHOTO_W)
    dot_y = int(step["dot"][1] * H)
    next_x = int(next_step["dot"][0] * PHOTO_W)
    next_y = int(next_step["dot"][1] * H)
    cur_x = int(dot_x + (next_x - dot_x) * progress)
    cur_y = int(dot_y + (next_y - dot_y) * progress)

    draw.line((70, H - 86, cur_x, cur_y), fill=accent, width=4)
    for r, alpha_color in [(36, "#ffffff"), (24, accent), (9, "#ffffff")]:
        draw.ellipse((cur_x - r, cur_y - r, cur_x + r, cur_y + r), outline=alpha_color, width=3)
    draw.ellipse((cur_x - 8, cur_y - 8, cur_x + 8, cur_y + 8), fill=accent)

    rail_y = H - 70
    draw.rounded_rectangle((58, rail_y - 10, PHOTO_W - 58, rail_y + 10), radius=8, fill="#1e293b", outline="#475569")
    gantry_x = max(78, min(PHOTO_W - 78, cur_x))
    draw.rounded_rectangle((gantry_x - 36, rail_y - 32, gantry_x + 36, rail_y + 32), radius=10, fill=accent)
    draw.line((gantry_x, rail_y - 34, gantry_x, cur_y), fill=accent, width=3)

    panel_x = PHOTO_W
    draw.rectangle((panel_x, 0, W, H), fill="#0b1120")
    draw.rectangle((panel_x, 0, panel_x + 4, H), fill=accent)

    x0 = panel_x + 22
    y = 26
    draw.text((x0, y), "OCR Three-Axis Automation", font=small_font, fill="#94a3b8")
    y += 28
    draw.text((x0, y), step["title"], font=title_font, fill="#f8fafc")
    y += 42
    for line in wrap_text(step["body"], body_font, PANEL_W - 58):
        draw.text((x0, y), line, font=body_font, fill="#cbd5e1")
        y += 21

    y += 24
    rounded_rect(draw, (x0, y, W - 22, y + 62), 9, "#111827", "#334155")
    draw.text((x0 + 12, y + 10), "Controller command", font=small_font, fill="#94a3b8")
    draw.text((x0 + 12, y + 33), step["cmd"], font=mono_font, fill="#e2e8f0")

    y += 82
    pos = step["pos"]
    draw.text((x0, y), "Machine position", font=small_font, fill="#94a3b8")
    y += 22
    draw.text((x0, y), f"X {pos[0]:>3} mm", font=big_font, fill="#f8fafc")
    y += 40
    draw.text((x0, y), f"Y {pos[1]:>3} mm", font=big_font, fill="#f8fafc")
    y += 40
    draw.text((x0, y), f"Z {pos[2]:>3} mm", font=big_font, fill="#f8fafc")

    y = H - 74
    for i, _ in enumerate(STEPS):
        cx = x0 + i * 34
        fill = accent if i == step_index else "#334155"
        draw.ellipse((cx, y, cx + 14, y + 14), fill=fill)
    draw.text((x0, H - 40), "Simulated end-to-end control loop", font=small_font, fill="#94a3b8")

    return frame


def main() -> None:
    photo = fit_photo()
    frames = []
    for idx in range(len(STEPS)):
        for j in range(FRAMES_PER_STEP):
            progress = j / FRAMES_PER_STEP
            frames.append(draw_frame(idx, progress, photo))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=110,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
