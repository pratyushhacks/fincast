"""
Probe script for FoundryLocal. Tries to detect a running HTTP endpoint or falls back to checking CLI availability.

Usage:
  .venv\Scripts\python tools\probe_foundry.py --url http://127.0.0.1:11434 --timeout 5

It will:
- Attempt a GET to {url}/models or {url}/health
- If HTTP probe fails, it will try to run 'FoundryLocal' CLI (winget installed) with --help to see if available
"""
import argparse
import subprocess
import sys
import httpx


def http_probe(url, timeout=5):
    client = httpx.Client(timeout=timeout)
    paths = ["/models", "/v1/models", "/health", "/ready", "/"]
    for p in paths:
        try:
            r = client.get(url.rstrip('/') + p)
            print(f"GET {url+p} -> {r.status_code}")
            try:
                print(r.text[:1000])
            except Exception:
                pass
        except Exception as e:
            print(f"Probe {url+p} failed: {e}")


def cli_probe():
    cmds = ["FoundryLocal", "foundrylocal", "foundry-local", "foundry"]
    for c in cmds:
        try:
            p = subprocess.run([c, "--help"], capture_output=True, text=True, timeout=5)
            print(f"CLI {c} exists: return {p.returncode}")
            print(p.stdout[:1000])
            return
        except Exception as e:
            print(f"CLI probe {c} failed: {e}")
    print("No FoundryLocal CLI detected in PATH.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:11434', help='FoundryLocal HTTP endpoint')
    parser.add_argument('--timeout', type=int, default=5)
    args = parser.parse_args()

    print("HTTP probe:")
    http_probe(args.url, timeout=args.timeout)
    print("\nCLI probe:")
    cli_probe()
