# Security Notes

This project can store device identifiers, controller IP addresses, screenshots, OCR output, AI prompts, and API keys in local runtime files.

Before publishing changes:

- Keep `config.json` and `config.json.bak` private.
- Keep `screenshots/` and `templates/` private unless every image has been reviewed.
- Prefer the `SMARTPHONE_AUTOMATION_AI_API_KEY` environment variable for AI credentials.
- Do not commit real ADB device IDs, physical controller IPs, UI templates from personal devices, or generated logs.
- Rotate any API key that was ever committed or shared publicly.

