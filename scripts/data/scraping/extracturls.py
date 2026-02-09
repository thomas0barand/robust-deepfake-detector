"""Extract unique data-key UUIDs from a Suno HTML page."""

import re
import argparse
from pathlib import Path

UUID_PATTERN = re.compile(
    r'data-key="([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"'
)

def extract_data_keys(input_path: str) -> list[str]:
    urls = []
    for html_path in Path(input_path).glob("*.html"):
        content = html_path.read_text(encoding="utf-8")
        keys = UUID_PATTERN.findall(content)
        urls.extend(f"https://cdn1.suno.ai/{key}.mp3" for key in keys)
    # Deduplicate while preserving order
    return list(dict.fromkeys(urls))


def main():
    parser = argparse.ArgumentParser(description="Extract data-key UUIDs from Suno HTML page")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="src/dataset/v5",
        help="Path to the HTML files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional output file to save keys (one per line)",
    )
    args = parser.parse_args()

    keys = extract_data_keys(args.input)
    print(f"Found {len(keys)} unique data-keys:\n")
    for key in keys:
        print(key)

    if args.output:
        Path(args.output).write_text("\n".join(keys) + "\n", encoding="utf-8")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
