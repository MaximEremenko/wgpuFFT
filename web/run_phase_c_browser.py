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


JS_EXPECTED_REVISION = "fa45c93f524a69a96c9f55acfad865226bfccd29"


class HarnessHandler(SimpleHTTPRequestHandler):
    result_event = threading.Event()
    result_payload: dict | None = None

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if urllib.parse.urlparse(self.path).path != "/__phase_c_result":
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
    # On Windows, `chrome.exe --version` can attach to an existing browser and
    # print only "Opening in existing browser session." Read the executable's
    # product-version resource instead so the archive receives the pinned build.
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
    # Kept in sync with WebGPU-FFT/web/run_browser_tests.py.
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
    match = re.search(r'data-wgpu-fft-phase-c="([^"]+)"', dom)
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
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
        )
    try:
        return process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.communicate(timeout=5)


def run_headless(chrome: Path, url: str, timeout: int, temp_root: Path) -> dict | None:
    with tempfile.TemporaryDirectory(prefix="wgpu_fft_phase_c_", dir=temp_root, ignore_cleanup_errors=True) as profile:
        command = [
            str(chrome), *common_chrome_args(profile), "--headless=new",
            f"--virtual-time-budget={timeout * 1000}", "--dump-dom", url,
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # WebGPU either initializes quickly in headless Chrome or not at all on
        # this machine. Bound the probe so a failed headless attempt cannot add
        # minutes before the known-good headed fallback.
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
    with tempfile.TemporaryDirectory(prefix="wgpu_fft_phase_c_", dir=temp_root, ignore_cleanup_errors=True) as profile:
        process = subprocess.Popen(
            [str(chrome), *common_chrome_args(profile), f"--app={url}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            if HarnessHandler.result_event.wait(timeout):
                return HarnessHandler.result_payload
            return None
        finally:
            terminate_process_tree(process)


def print_result(result: dict) -> None:
    print(json.dumps(result, indent=2, sort_keys=True))
    benchmark = result.get("benchmark") or {}
    if benchmark:
        rust = benchmark["rust"]
        js = benchmark["js"]
        print(
            f"RESULT Rust/Wasm {rust['avgMs']:.4f} +/- {rust['stderrMs']:.4f} ms; "
            f"JavaScript {js['avgMs']:.4f} +/- {js['stderrMs']:.4f} ms; "
            f"Rust/JS {benchmark['rustOverJs']:.3f}x"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run wgpuFFT Phase C persistence and Rust-vs-JS browser checks")
    parser.add_argument("--mode", choices=("snapshot", "bench", "all"), default="all")
    parser.add_argument("--port", type=int, default=8025)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--chrome-path", default="")
    parser.add_argument("--headed-only", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--output", default="", help="Write the complete machine-readable result JSON")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    vault = repo.parent
    js_repo = vault / "WebGPU-FFT"
    if not js_repo.is_dir():
        print(f"ERROR: expected JS reference at {js_repo}", file=sys.stderr)
        return 2
    js_revision = git_revision(js_repo)
    if js_revision != JS_EXPECTED_REVISION:
        print(f"ERROR: WebGPU-FFT is {js_revision}; expected pinned {JS_EXPECTED_REVISION}", file=sys.stderr)
        return 2
    if git_is_dirty(js_repo):
        print("ERROR: WebGPU-FFT has uncommitted changes; the pinned revision would not describe the code run", file=sys.stderr)
        return 2
    rust_revision = git_revision(repo)
    rust_dirty = git_is_dirty(repo)
    chrome = chrome_path(args.chrome_path)
    if not chrome:
        print("ERROR: Chrome not found; pass --chrome-path", file=sys.stderr)
        return 2
    chrome_build = chrome_version(chrome)
    rustc_version = subprocess.run(["rustc", "--version"], capture_output=True, text=True, check=True).stdout.strip()
    wasm_bindgen_version = subprocess.run(
        ["wasm-bindgen", "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    print(f"Chrome: {chrome_build}")
    print(f"Chrome flags: {' '.join(common_chrome_args('<temporary-profile>')[:-1])}")
    print(f"Rust: {rustc_version}")
    print(f"wasm-bindgen: {wasm_bindgen_version}")
    print(f"wgpuFFT revision: {rust_revision}{' + uncommitted Phase C changes' if rust_dirty else ''}")
    print(f"WebGPU-FFT revision: {js_revision}")
    if not args.skip_build:
        build_wasm(repo)
    wasm_path = repo / "wgpu-web/pkg/wgpu_web_bg.wasm"
    if not wasm_path.is_file():
        print(f"ERROR: generated Wasm not found at {wasm_path}", file=sys.stderr)
        return 2
    wasm_sha256 = hashlib.sha256(wasm_path.read_bytes()).hexdigest()

    temp_root = repo / ".tmp_browser_runner"
    temp_root.mkdir(exist_ok=True)
    handler = lambda *a, **kw: HarnessHandler(*a, directory=str(vault), **kw)  # noqa: E731
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = (
        f"http://127.0.0.1:{args.port}/wgpuFFT/web/phase_c.html?"
        + urllib.parse.urlencode(
            {"autorun": args.mode, "rust_revision": rust_revision, "rust_dirty": "1" if rust_dirty else "0"}
        )
    )
    try:
        wait_for_server(url)
        HarnessHandler.result_event.clear()
        HarnessHandler.result_payload = None
        if args.headed_only:
            print("Using the proven headed Chrome app-window path (--headed-only).")
            result = None
            browser_mode = "headed"
        else:
            result = run_headless(chrome, url, args.timeout, temp_root)
            browser_mode = "headless"
        if result is None:
            if not args.headed_only:
                print("Headless Chrome produced no result; trying the proven headed-app fallback.")
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
            "wgpuFftRevision": rust_revision,
            "wgpuFftWorkingTreeDirty": rust_dirty,
            "webGpuFftRevision": js_revision,
        }
        if args.output:
            output = Path(args.output)
            if not output.is_absolute():
                output = repo / output
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"Wrote result JSON: {output}")
        print_result(result)
        return 0 if result.get("ok") else 1
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
