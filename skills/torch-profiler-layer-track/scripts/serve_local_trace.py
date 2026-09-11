"""Serve one local trace to an embedded Perfetto UI over loopback."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def viewer_html(title):
    safe_title = html.escape(title)
    script_title = json.dumps(title).replace("<", "\\u003c")
    return f"""<!doctype html>
<meta charset="utf-8"><title>{safe_title}</title>
<style>body{{margin:0;font-family:system-ui}}header{{padding:10px 16px}}
iframe{{border:0;width:100%;height:85vh}}p{{margin:4px}}</style>
<header><b>{safe_title}</b>
<p>Pin “Layer guide” next to the GPU streams. L0 is the first layer.
Anchor intervals are navigation guides, not full layer timings.</p>
<p id="status">Loading Perfetto UI…</p></header>
<iframe id="viewer" src="https://ui.perfetto.dev/#!/?mode=embedded"></iframe>
<script>
const frame = document.getElementById('viewer');
const status = document.getElementById('status');
const origin = 'https://ui.perfetto.dev';
let sent = false;
const timer = setInterval(() => frame.contentWindow.postMessage('PING', origin), 200);
const timeout = setTimeout(() => {{
  clearInterval(timer);
  status.textContent = 'Perfetto UI did not respond. Check network access or open the trace manually.';
}}, 60000);
window.addEventListener('message', async event => {{
  if (event.origin !== origin || event.source !== frame.contentWindow || event.data !== 'PONG' || sent) return;
  sent = true; clearInterval(timer); clearTimeout(timeout);
  try {{
    status.textContent = 'Reading local trace…';
    const response = await fetch('/trace');
    if (!response.ok) throw new Error('Trace fetch failed: ' + response.status);
    const buffer = await response.arrayBuffer();
    frame.contentWindow.postMessage({{perfetto: {{buffer, title: {script_title}, keepApiOpen: true}}}}, origin);
    status.textContent = 'Local trace sent. Expand the GPU/process and pin the Layer guide track.';
  }} catch (error) {{ status.textContent = error.message; }}
}});
</script>
"""


def make_handler(trace):
    page = viewer_html(trace.name).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
            elif self.path == "/trace":
                with trace.open("rb") as handle:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(trace.stat().st_size))
                    self.end_headers()
                    shutil.copyfileobj(handle, self.wfile)
            else:
                self.send_error(404)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--port", type=int, default=0, help="Default: a free loopback port"
    )
    args = parser.parse_args()
    if not args.trace.is_file():
        parser.error("Trace file does not exist")
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), make_handler(args.trace.resolve())
    )
    print(
        f"Open http://127.0.0.1:{server.server_port}/ in your browser. Ctrl-C stops the server.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
