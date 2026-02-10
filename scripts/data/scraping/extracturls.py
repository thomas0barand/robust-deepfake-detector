"""Extract song metadata and URLs from Suno HTML pages."""

import re
import json
import argparse
from pathlib import Path
from urllib.parse import unquote
from tqdm import tqdm

UUID_PATTERN = re.compile(
    r'data-key="([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"'
)


def extract_metadata(html_path: Path) -> list[dict]:
    """Extract title, styles, artist and URL for each song in an HTML file."""
    content = html_path.read_text(encoding="utf-8")
    keys = list(dict.fromkeys(UUID_PATTERN.findall(content)))
    songs = []

    for uuid in keys:
        idx = content.find(f'data-key="{uuid}"')
        next_idx = content.find('data-key="', idx + 10)
        chunk = content[idx:next_idx] if next_idx > 0 else content[idx:idx + 8000]

        title_match = re.search(r'aria-label="([^"]+)"', chunk)
        styles = [unquote(s) for s in re.findall(r'href="https://suno\.com/style/([^"]+)"', chunk)]
        artist_match = re.search(r'<a[^>]*title="([^"]+)"[^>]*href="https://suno\.com/@([^"]+)"', chunk)

        songs.append({
            "uuid": uuid,
            "url": f"https://cdn1.suno.ai/{uuid}.mp3",
            "title": title_match.group(1) if title_match else "",
            "styles": styles,
            "artist": artist_match.group(1) if artist_match else "",
            "username": artist_match.group(2) if artist_match else "",
            "source_file": html_path.name,
        })

    return songs


def extract_all(input_path: str) -> list[dict]:
    """Extract metadata from all HTML files in a directory."""
    all_songs = []
    seen_uuids = set()
    for html_path in tqdm(sorted(Path(input_path).glob("*.html"))):
        for song in extract_metadata(html_path):
            if song["uuid"] not in seen_uuids:
                seen_uuids.add(song["uuid"])
                all_songs.append(song)
    return all_songs


def main():
    parser = argparse.ArgumentParser(description="Extract song metadata from Suno HTML pages")
    parser.add_argument("-i", "--input", type=str, default="src/dataset/v5",
                        help="Path to directory with HTML files")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output JSON file for metadata")
    parser.add_argument("--urls-only", action="store_true",
                        help="Only print URLs (one per line)")
    args = parser.parse_args()

    songs = extract_all(args.input)

    if args.urls_only:
        for s in songs:
            print(s["url"])
    else:
        print(f"Found {len(songs)} unique songs:\n")
        for s in songs:
            print(f"  {s['uuid']} | {s['title']} | {s['styles']} | {s['artist']} (@{s['username']})")

    if args.output:
        Path(args.output).write_text(json.dumps(songs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
