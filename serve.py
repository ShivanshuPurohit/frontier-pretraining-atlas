#!/usr/bin/env python3
"""Dev server for the atlas site. Port comes from $PORT (default 8734)."""
import http.server
import os
import socketserver

SITE = os.path.dirname(os.path.abspath(__file__))
os.chdir(SITE)

port = int(os.environ.get("PORT", "8734"))
handler = http.server.SimpleHTTPRequestHandler


class Server(socketserver.TCPServer):
    allow_reuse_address = True


with Server(("127.0.0.1", port), handler) as httpd:
    print(f"serving {SITE} on http://127.0.0.1:{port}", flush=True)
    httpd.serve_forever()
