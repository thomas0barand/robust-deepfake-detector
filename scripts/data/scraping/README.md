# Suno Scraping & Download Pipeline

## Requirements

```bash
pip install aiohttp tqdm
```

## 1. Extract metadata from HTML pages

Parse saved Suno HTML pages to extract song metadata (uuid, title, styles, artist) and build download URLs.

```bash
# Extract from default directory (src/dataset/v5)
python scripts/data/scraping/extracturls.py -o songs.json

# Extract from a custom directory
python scripts/data/scraping/extracturls.py -i path/to/html_dir -o songs.json

# Only output URLs
python scripts/data/scraping/extracturls.py --urls-only
```

Output JSON format:
```json
{
  "uuid": "da977af0-963c-40ca-8834-aeeec4b2b885",
  "url": "https://cdn1.suno.ai/da977af0-963c-40ca-8834-aeeec4b2b885.mp3",
  "title": "AA Little Things",
  "styles": ["smooth jazz", "pop"],
  "artist": "YanTone",
  "username": "yanto5150",
  "source_file": "aa.html"
}
```

## 2. Download MP3s

Async concurrent downloader. Skips already downloaded files.

```bash
# Download all songs from JSON
python scripts/data/scraping/download.py -i songs.json -o data/suno_v5

# Limit to first 100 songs
python scripts/data/scraping/download.py -i songs.json -o data/suno_v5 --limit 100

# Adjust concurrency (default: 20)
python scripts/data/scraping/download.py -i songs.json -o data/suno_v5 -c 50
```

## 3. Auto-scroll helper

Utility to auto-scroll Suno pages for saving HTML. Click positions on screen, then it scrolls each in round-robin.

```bash
# Default: 4 positions, scroll 50 units every 0.5s
python scripts/data/scraping/click_scroll.py

# Custom
python scripts/data/scraping/click_scroll.py -p 6 -n 100 -t 0.3
```

Press Esc to stop. Requires `pyautogui` and `pynput`.
