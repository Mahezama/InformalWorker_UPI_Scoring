"""
Lightweight local HTTP server for the UPI Credit Scoring Web App.
Run: python serve.py
"""

import http.server
import socketserver
import webbrowser
import socket
import sys
from pathlib import Path

# Change to project root directory
ROOT_DIR = Path(__file__).parent.resolve()

def find_open_port(start_port=8080, max_attempts=20):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    return start_port

def main():
    port = find_open_port(8080)
    handler = http.server.SimpleHTTPRequestHandler
    
    # Enable address reuse
    socketserver.TCPServer.allow_reuse_address = True
    
    print("=" * 65)
    print("  UPI Credit Scoring — Interactive Web Dashboard & Simulator")
    print("  IEEE Research Simulation Study Demo")
    print("=" * 65)
    print(f"\n  Serving from: {ROOT_DIR}")
    print(f"  URL         : http://localhost:{port}")
    print("\n  Press Ctrl+C in terminal to stop server.\n")
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            # Open browser automatically
            webbrowser.open(f"http://localhost:{port}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()
