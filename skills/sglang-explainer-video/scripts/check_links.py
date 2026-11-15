"""Open every link a video's ending and its publish texts show, so none is invented or out of date.

    python check_links.py --project DIR [--text desc.txt body.txt ...]

Reads the github.com and sglang.io links on the project's last frame (the ending links card, the follow
strip, the PR link) and in any --text files, requests each one with curl, following redirects, and fails
unless it answers 200. Links are shown without a scheme on screen; https:// is assumed.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time

LINK = re.compile(
    r"(?:https?://)?((?:github\.com|(?:[\w-]+\.)*sglang\.io)(?:/[\w\-./%#?=&+~]*)?)"
)


def find_links(text):
    """The github.com / sglang.io links in text, without scheme, in order of first appearance."""
    seen = []
    for m in LINK.finditer(text):
        url = m.group(1).rstrip("./")
        if url not in seen:
            seen.append(url)
    return seen


def last_frame(project):
    frames = sorted(
        glob.glob(os.path.join(project, "compositions", "frames", "*.html"))
    )
    if not frames:
        sys.exit(f"no frames under {project}/compositions/frames")
    return frames[-1]


def status(url):
    for attempt in range(2):
        r = subprocess.run(
            ["curl", "-s", "-o", os.devnull, "-L", "--max-time", "30"]
            + ["-w", "%{http_code}", "https://" + url],
            capture_output=True,
            text=True,
        )
        if r.stdout.strip() == "200":
            break
        time.sleep(2 * (1 - attempt))
    return r.stdout.strip() or "no answer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", help="check the links on the project's last frame")
    ap.add_argument("--text", nargs="*", default=[], help="publish texts to check too")
    a = ap.parse_args()
    sources = ([last_frame(a.project)] if a.project else []) + a.text
    if not sources:
        ap.error("give --project and/or --text")
    bad = 0
    for src in sources:
        links = find_links(open(src, encoding="utf-8").read())
        print(f"{src}: {len(links)} link(s)")
        for url in links:
            code = status(url)
            bad += code != "200"
            print(f"  {'✓' if code == '200' else '✗'} {code}  {url}")
    print(f"{'✓' if not bad else '✗'} check_links: {bad} link(s) did not open")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
