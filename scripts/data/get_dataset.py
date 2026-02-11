"""Download Suno v5 dataset from a shared Google Drive folder (async)."""

import argparse
import asyncio
import os
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

FOLDER_ID = "1jMrO05xSY4q9vDRHjjXh2iJcl8-H7g15"
API_URL = "https://www.googleapis.com/drive/v3/files"
DOWNLOAD_URL = "https://drive.google.com/uc?id={id}&export=download"
MAX_CONCURRENT = 20
TIMEOUT = aiohttp.ClientTimeout(total=300)


async def list_files(session: aiohttp.ClientSession, folder_id: str, api_key: str) -> list[dict]:
    """List all files in a public Google Drive folder."""
    files = []
    page_token = None
    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "key": api_key,
            "fields": "nextPageToken, files(id, name, size)",
            "orderBy": "name",
            "pageSize": 1000,
        }
        if page_token:
            params["pageToken"] = page_token
        async with session.get(API_URL, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()
        files.extend(data.get("files", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return files


async def download_one(
    session: aiohttp.ClientSession, file: dict, output_dir: Path,
    semaphore: asyncio.Semaphore,
) -> str:
    """Download a single file. Returns 'ok', 'skipped', or 'failed'."""
    dest = output_dir / file["name"]
    if dest.exists() and dest.stat().st_size > 0:
        return "skipped"

    url = DOWNLOAD_URL.format(id=file["id"])
    async with semaphore:
        try:
            async with session.get(url) as resp:
                # Handle large file virus scan confirmation
                if resp.status == 200 and "text/html" in resp.headers.get("content-type", ""):
                    text = await resp.text()
                    if "confirm=" in text:
                        confirm_url = url + "&confirm=t"
                        async with session.get(confirm_url) as resp2:
                            resp2.raise_for_status()
                            with open(dest, "wb") as f:
                                async for chunk in resp2.content.iter_chunked(8192):
                                    f.write(chunk)
                            return "ok"

                resp.raise_for_status()
                with open(dest, "wb") as f:
                    async for chunk in resp.content.iter_chunked(8192):
                        f.write(chunk)
            return "ok"
        except Exception as e:
            print(f"\n  ERROR: {file['name']} — {e}")
            if dest.exists():
                dest.unlink()
            return "failed"


async def run(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        print(f"Listing files in folder {args.folder_id}...")
        files = await list_files(session, args.folder_id, args.api_key)
        files.sort(key=lambda f: f["name"])
        print(f"Found {len(files)} files")

        if args.limit:
            files = files[:args.limit]
            print(f"Downloading first {args.limit} files")

        semaphore = asyncio.Semaphore(args.concurrent)
        tasks = [download_one(session, f, output_dir, semaphore) for f in files]
        results = await tqdm_asyncio.gather(*tasks, desc="Downloading")

    ok = sum(1 for r in results if r == "ok")
    skipped = sum(1 for r in results if r == "skipped")
    failed = sum(1 for r in results if r == "failed")
    print(f"\nDone: {ok} downloaded, {skipped} skipped, {failed} failed")


def main():
    parser = argparse.ArgumentParser(description="Download Suno v5 dataset from Google Drive")
    parser.add_argument("-o", "--output", type=str, default="data/suno_v5",
                        help="Output directory (default: data/suno_v5)")
    parser.add_argument("-n", "--limit", type=int, default=None,
                        help="Download only first N files (default: all)")
    parser.add_argument("-c", "--concurrent", type=int, default=MAX_CONCURRENT,
                        help=f"Max concurrent downloads (default: {MAX_CONCURRENT})")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Google Drive API key (or set GDRIVE_API_KEY env var)")
    parser.add_argument("--folder-id", type=str, default=FOLDER_ID,
                        help="Google Drive folder ID")
    args = parser.parse_args()
    args.api_key = args.api_key or os.environ.get("GDRIVE_API_KEY")
    if not args.api_key:
        parser.error("API key required: use --api-key or set GDRIVE_API_KEY env var")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
