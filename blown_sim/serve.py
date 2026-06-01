from __future__ import annotations

import argparse
import http.server
import socketserver
import webbrowser
import os
from pathlib import Path


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the BlownWing Cesium playback app locally.")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    handler = http.server.SimpleHTTPRequestHandler

    last_error = None
    for port in range(args.port, args.port + 20):
        try:
            with ReusableTCPServer(("", port), handler) as httpd:
                print(f"Serving blown_sim at http://localhost:{port}")
                print(f"Root: {root}")
                if port != args.port:
                    print(f"Port {args.port} was unavailable, so I moved to {port}.")
                if not args.no_browser:
                    webbrowser.open(f"http://localhost:{port}/index.html")
                httpd.serve_forever()
                return
        except OSError as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"Unable to bind any port in the range {args.port} to {args.port + 19}."
    ) from last_error


if __name__ == "__main__":
    main()
