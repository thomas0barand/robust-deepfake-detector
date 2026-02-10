"""Download MP3s from Suno JSON metadata using async concurrency."""

import json
import asyncio
import argparse
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm_asyncio

MAX_CONCURRENT = 20
TIMEOUT = aiohttp.ClientTimeout(total=120)


async def download_one(session: aiohttp.ClientSession, song: dict, output_dir: Path, semaphore: asyncio.Semaphore):
    dest = output_dir / f"{song['uuid']}.mp3"
    if dest.exists() and dest.stat().st_size > 0:
        return True  # already downloaded

    async with semaphore:
        try:
            async with session.get(song["url"]) as resp:
                if resp.status != 200:
                    print(f"  FAIL ({resp.status}): {song['uuid']}")
                    return False
                data = await resp.read()
                dest.write_bytes(data)
                return True
        except Exception as e:
            print(f"  ERROR: {song['uuid']} — {e}")
            return False


async def download_all(songs: list[dict], output_dir: Path, max_concurrent: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        tasks = [download_one(session, s, output_dir, semaphore) for s in songs]
        results = await tqdm_asyncio.gather(*tasks, desc="Downloading")

    ok = sum(1 for r in results if r)
    fail = len(results) - ok
    print(f"\nDone: {ok} downloaded, {fail} failed, out of {len(results)} total")


def main():
    parser = argparse.ArgumentParser(description="Download Suno MP3s from JSON metadata")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to JSON metadata file")
    parser.add_argument("-o", "--output", type=str, default="data/suno_v5",
                        help="Output directory for MP3 files")
    parser.add_argument("-c", "--concurrent", type=int, default=MAX_CONCURRENT,
                        help=f"Max concurrent downloads (default: {MAX_CONCURRENT})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only download first N songs")
    args = parser.parse_args()

    songs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.limit:
        songs = songs[:args.limit]

    print(f"Downloading {len(songs)} songs to {args.output} (concurrency: {args.concurrent})")
    asyncio.run(download_all(songs, Path(args.output), args.concurrent))


if __name__ == "__main__":
    main()
