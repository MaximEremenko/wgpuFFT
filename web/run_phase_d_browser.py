#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class HarnessHandler(SimpleHTTPRequestHandler):
    result_event = threading.Event()
    result_payload: dict | None = None

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if urllib.parse.urlparse(self.path).path != "/__phase_d_result":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("content-length", "0"))
            HarnessHandler.result_payload = json.loads(self.rfile.read(size).decode("utf-8"))
        except Exception as error:
            HarnessHandler.result_payload = {"ok": False, "error": f"invalid callback: {error}"}
        HarnessHandler.result_event.set()
        self.send_response(204)
        self.end_headers()


def git_revision(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def git_is_dirty(repo: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=True
    )
    return bool(result.stdout.strip())


def chrome_path(hint: str) -> Path | None:
    candidates = [
        Path(hint) if hint else None,
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path.home() / "AppData/Local/Google/Chrome/Application/chrome.exe",
    ]
    return next((path for path in candidates if path and path.is_file()), None)


def chrome_version(chrome: Path) -> str:
    escaped = str(chrome).replace("'", "''")
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            f"(Get-Item -LiteralPath '{escaped}').VersionInfo.ProductVersion",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def common_chrome_args(profile: str) -> list[str]:
    return [
        "--no-sandbox",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-breakpad",
        "--disable-crash-reporter",
        "--enable-unsafe-webgpu",
        "--disable-gpu-sandbox",
        f"--user-data-dir={profile}",
    ]


def build_wasm(repo: Path) -> None:
    subprocess.run(
        ["cargo", "build", "-p", "wgpu-web", "--target", "wasm32-unknown-unknown", "--release"],
        cwd=repo,
        check=True,
    )
    output = repo / "wgpu-web/pkg"
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "wasm-bindgen",
            "--target", "web",
            "--out-dir", str(output),
            str(repo / "target/wasm32-unknown-unknown/release/wgpu_web.wasm"),
        ],
        cwd=repo,
        check=True,
    )


def parse_dom(dom: str) -> dict | None:
    match = re.search(r'data-wgpu-nufft-phase-d="([^"]+)"', dom)
    if not match:
        return None
    try:
        return json.loads(urllib.parse.unquote(match.group(1)))
    except Exception:
        return None


def wait_for_server(url: str) -> None:
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError(f"browser harness server did not start at {url}")


def terminate_process_tree(process: subprocess.Popen) -> tuple[str, str]:
    if process.poll() is None:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, text=True
        )
    try:
        return process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.communicate(timeout=5)


def run_headless(chrome: Path, url: str, timeout: int, temp_root: Path) -> dict | None:
    with tempfile.TemporaryDirectory(
        prefix="wgpu_nufft_phase_d_", dir=temp_root, ignore_cleanup_errors=True
    ) as profile:
        process = subprocess.Popen(
            [
                str(chrome),
                *common_chrome_args(profile),
                "--headless=new",
                f"--virtual-time-budget={timeout * 1000}",
                "--dump-dom",
                url,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.time() + min(timeout, 30)
        while time.time() < deadline:
            if HarnessHandler.result_event.wait(0.25):
                payload = HarnessHandler.result_payload
                terminate_process_tree(process)
                return payload
            if process.poll() is not None:
                stdout, _ = process.communicate(timeout=5)
                return HarnessHandler.result_payload or parse_dom(stdout)
        stdout, _ = terminate_process_tree(process)
        return HarnessHandler.result_payload or parse_dom(stdout)


def run_headed(chrome: Path, url: str, timeout: int, temp_root: Path) -> dict | None:
    with tempfile.TemporaryDirectory(
        prefix="wgpu_nufft_phase_d_", dir=temp_root, ignore_cleanup_errors=True
    ) as profile:
        process = subprocess.Popen(
            [str(chrome), *common_chrome_args(profile), f"--app={url}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            return HarnessHandler.result_payload if HarnessHandler.result_event.wait(timeout) else None
        finally:
            terminate_process_tree(process)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the browser NUFFT Phase D matrix")
    parser.add_argument("--port", type=int, default=8026)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--chrome-path", default="")
    parser.add_argument("--headed-only", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    revision = git_revision(repo)
    dirty = git_is_dirty(repo)
    chrome = chrome_path(args.chrome_path)
    if not chrome:
        print("ERROR: Chrome not found; pass --chrome-path", file=sys.stderr)
        return 2
    chrome_build = chrome_version(chrome)
    rustc_version = subprocess.run(
        ["rustc", "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    wasm_bindgen_version = subprocess.run(
        ["wasm-bindgen", "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    print(f"Chrome: {chrome_build}")
    print(f"Chrome flags: {' '.join(common_chrome_args('<temporary-profile>')[:-1])}")
    print(f"Rust: {rustc_version}")
    print(f"wasm-bindgen: {wasm_bindgen_version}")
    print(f"wgpuFFT revision: {revision}{' + uncommitted Phase D changes' if dirty else ''}")

    if not args.skip_build:
        build_wasm(repo)
    wasm_path = repo / "wgpu-web/pkg/wgpu_web_bg.wasm"
    if not wasm_path.is_file():
        print(f"ERROR: generated Wasm not found at {wasm_path}", file=sys.stderr)
        return 2
    wasm_sha256 = hashlib.sha256(wasm_path.read_bytes()).hexdigest()

    temp_root = repo / ".tmp_browser_runner"
    temp_root.mkdir(exist_ok=True)
    handler = lambda *a, **kw: HarnessHandler(*a, directory=str(repo), **kw)  # noqa: E731
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = (
        f"http://127.0.0.1:{args.port}/web/phase_d.html?"
        + urllib.parse.urlencode(
            {"autorun": "1", "rust_revision": revision, "rust_dirty": "1" if dirty else "0"}
        )
    )
    try:
        wait_for_server(url)
        HarnessHandler.result_event.clear()
        HarnessHandler.result_payload = None
        if args.headed_only:
            result = None
            browser_mode = "headed"
        else:
            result = run_headless(chrome, url, args.timeout, temp_root)
            browser_mode = "headless"
        if result is None:
            if not args.headed_only:
                print("Headless Chrome produced no result; using the headed-app fallback.")
                browser_mode = "headed-fallback"
            HarnessHandler.result_event.clear()
            HarnessHandler.result_payload = None
            result = run_headed(chrome, url, args.timeout, temp_root)
        if result is None:
            print("ERROR: browser produced no result", file=sys.stderr)
            return 2
        result["environment"] = {
            "chromeProductVersion": chrome_build,
            "chromeFlags": common_chrome_args("<temporary-profile>")[:-1],
            "rustc": rustc_version,
            "wasmBindgen": wasm_bindgen_version,
            "wasmSha256": wasm_sha256,
            "browserMode": browser_mode,
            "skipBuild": args.skip_build,
            "wgpuFftRevision": revision,
            "wgpuFftWorkingTreeDirty": dirty,
        }
        if args.output:
            output = Path(args.output)
            if not output.is_absolute():
                output = repo / output
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"Wrote result JSON: {output}")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ok") else 1
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
