#!/usr/bin/env python3
"""Builds dist/wgpu_fft_web.js, wgpu-fft-web for classic <script> tags.

The file holds wasm-bindgen's no-modules glue and the gzip-compressed
WebAssembly module, so a page (or a worker) loads wgpu-fft-web without ES
modules or a fetch of a .wasm file, and works when opened from disk.

It needs the wasm32-unknown-unknown target and a wasm-bindgen CLI whose
version matches the wasm-bindgen crate in Cargo.lock:

    python wgpu-fft-web/build_standalone.py [--wasm-bindgen PATH]
"""

from __future__ import annotations

import argparse
import base64
import gzip
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUTPUT = HERE / "dist" / "wgpu_fft_web.js"
# The global that wasm-bindgen's glue defines; `wgpuFftWeb.load()` wraps it.
BINDINGS = "wgpuFftWebBindings"

LOADER = """
var wgpuFftWeb = (() => {
  "use strict";
  // The WebAssembly module, gzip-compressed and base64-encoded.
  const compressed = `
%(payload)s`;
  let exports;
  return {
    /**
     * Instantiates the WebAssembly module on the first call and resolves to
     * the wgpu-fft-web exports: `WgpuFft`, `cpuFft`, `WebFftPrecision`, ...
     */
    load() {
      exports ??= (async () => {
        const bytes = Uint8Array.from(atob(compressed), (c) => c.charCodeAt(0));
        const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
        const module = await new Response(stream).arrayBuffer();
        await %(bindings)s({ module_or_path: module });
        return %(bindings)s;
      })().catch((error) => {
        exports = undefined;
        throw error;
      });
      return exports;
    },
  };
})();
"""


def locked_version(name: str) -> str:
    lock = (REPO / "Cargo.lock").read_text(encoding="utf-8")
    match = re.search(rf'name = "{re.escape(name)}"\r?\nversion = "([^"]+)"', lock)
    if not match:
        sys.exit(f"{name} is not in Cargo.lock")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wasm-bindgen", default="wasm-bindgen", help="the wasm-bindgen CLI to run")
    args = parser.parse_args()

    expected = locked_version("wasm-bindgen")
    found = subprocess.run(
        [args.wasm_bindgen, "--version"], capture_output=True, text=True, check=True
    ).stdout.split()[-1]
    if found != expected:
        sys.exit(
            f"wasm-bindgen {found} does not match {expected} in Cargo.lock; install it with\n"
            f"  cargo install wasm-bindgen-cli --version {expected} --locked"
        )

    subprocess.run(
        ["cargo", "build", "--locked", "-p", "wgpu-fft-web", "--target", "wasm32-unknown-unknown",
         "--release"],
        cwd=REPO,
        check=True,
    )
    with tempfile.TemporaryDirectory() as out:
        subprocess.run(
            [args.wasm_bindgen, "--target", "no-modules", "--no-modules-global", BINDINGS,
             "--out-dir", out, str(REPO / "target/wasm32-unknown-unknown/release/wgpu_fft_web.wasm")],
            check=True,
        )
        glue = (Path(out) / "wgpu_fft_web.js").read_text(encoding="utf-8")
        wasm = (Path(out) / "wgpu_fft_web_bg.wasm").read_bytes()

    payload = base64.b64encode(gzip.compress(wasm, compresslevel=9, mtime=0)).decode("ascii")
    header = textwrap.dedent(f"""\
        // wgpu-fft-web {locked_version("wgpu-fft-web")}, built by build_standalone.py with
        // wasm-bindgen {expected}. Do not edit; rerun the script instead.
        //
        // A classic script, so pages can load it from disk (file:// URLs) too:
        //
        //   <script src="wgpu_fft_web.js"></script>
        //   <script>
        //     wgpuFftWeb.load().then(async ({{ WgpuFft, cpuFft }}) => {{
        //       const gpu = await WgpuFft.init();
        //     }});
        //   </script>
        """)
    # atob skips the line breaks.
    lines = "\n".join(payload[start:start + 100] for start in range(0, len(payload), 100))
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(
        header + "\n" + glue.rstrip() + "\n" + LOADER % {"payload": lines, "bindings": BINDINGS},
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.2f} MB; module {len(wasm) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
