"""
Launch local demo server for the Caspian Maritime Delay-Risk project.

Run:
    python launch_demo.py

Then open:
    http://localhost:8000/demo/
"""

from __future__ import annotations

import http.server
import socketserver
import webbrowser
from pathlib import Path


PORT = 8000
DEMO_URL = f"http://localhost:{PORT}/demo/"


def main() -> None:
    repo_root = Path(__file__).resolve().parent

    demo_dir = repo_root / "demo"
    predictions_dir = repo_root / "predictions"

    if not demo_dir.exists():
        print("ERROR: demo/ folder not found.")
        print(f"Expected: {demo_dir}")
        return

    if not predictions_dir.exists():
        print("WARNING: predictions/ folder not found.")
        print("The website may open, but it will not load prediction data.")

    print("=" * 70)
    print("Caspian Maritime Delay-Risk Demo")
    print("=" * 70)
    print(f"Serving project root:")
    print(f"  {repo_root}")
    print()
    print("Open this link:")
    print(f"  {DEMO_URL}")
    print()
    print("Press Ctrl+C to stop the server.")
    print("=" * 70)

    webbrowser.open(DEMO_URL)

    handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("", PORT), handler) as httpd:
        # Serve from project root so demo/ can access ../predictions/
        import os
        os.chdir(repo_root)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()