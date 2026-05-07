# OCR-Driven Three-Axis Automation Tool

This is an early Python desktop automation project for controlling a physical device through a three-axis machine.

The project combines screen capture, image processing, OCR, AI-assisted decision making, task scheduling, and ESP-based hardware control. It was built for situations where a target device cannot be controlled through a normal software API, so the automation system observes the screen and performs physical interactions instead.

## What It Does

The basic workflow is:

1. Capture a screen image from an Android device through ADB.
2. Improve the screenshot with OpenCV image processing.
3. Use OCR and template matching to locate text and UI elements.
4. Use either predefined task steps or an AI model to decide the next action.
5. Send commands to an ESP-based controller that drives a three-axis mechanism.
6. Repeat until the task is complete or requires human intervention.

## Features

- PyQt5 desktop interface for device, task, OCR, AI, and controller settings.
- ADB screenshot capture with configurable device ID and resolution.
- OCR integration through a local OCR HTTP service.
- OpenCV-based image enhancement and template matching.
- AI-assisted action planning through a configurable chat-completions API.
- Task scheduling, retry handling, and state tracking.
- ESP controller integration for physical click, long-press, and movement commands.
- Local configuration through `config.json`.

## Privacy and Configuration

Do not commit your real runtime configuration.

This repository includes `config.example.json` as a safe template. Copy it locally when you need to run the project:

```bash
cp config.example.json config.json
```

Then edit `config.json` with your own local values.

Sensitive local files are ignored by Git:

- `config.json`
- `config.json.bak`
- `*.log`
- `screenshots/`
- `templates/`
- `.env`

For AI credentials, prefer using an environment variable:

```bash
set SMARTPHONE_AUTOMATION_AI_API_KEY=your_key_here
```

On macOS/Linux:

```bash
export SMARTPHONE_AUTOMATION_AI_API_KEY=your_key_here
```

## Example Configuration

```json
{
  "ESP_IP": "192.0.2.10",
  "ESP_PORT": 8080,
  "OCR_API_URL": "http://127.0.0.1:1224/api/ocr",
  "AI_API_URL": "https://api.example.com/v1/chat/completions",
  "AI_API_KEY": "",
  "AI_MODEL": "example-model",
  "ADB_PATH": "adb",
  "CAMERA_DEVICE_ID": "",
  "DEVICE_CONFIGS": {
    "Device_Example": {
      "SCREENSHOT_RESOLUTION": [1080, 1920],
      "CROPPED_RESOLUTION": [1080, 1440],
      "HOME_SCREEN_ANCHOR_TEXTS": ["Phone", "Messages", "Settings", "Camera", "Gallery"],
      "HOME_SCREEN_MIN_ANCHORS": 3,
      "machine_origin_x": 0,
      "machine_origin_y": 0
    }
  },
  "USER_TASKS": []
}
```

`192.0.2.10` is a documentation-only example address. Replace it with your local controller address in your private `config.json`.

## Installation

Requirements:

- Python 3.8+
- ADB
- A local OCR service such as Umi-OCR
- An ESP-based controller and compatible three-axis hardware

Install Python dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

## Project Status

This is an early prototype. It is useful as a record of the architecture and experimentation, but it still needs refactoring before it should be treated as production software.

The current implementation is mostly contained in a single large `main.py`, so future work should prioritize splitting the project into modules, improving tests, and making configuration safer by design.

## Security Notes

See `SECURITY.md` before publishing screenshots, logs, templates, or real device configuration.
