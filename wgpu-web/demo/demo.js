import initWasm, {
  WebFftDirection,
  WebFftNormalization,
  WebFftPrecision,
  WgpuFft,
} from "../pkg/wgpu_web.js";

const output = document.querySelector("#output");
document.querySelector("#run").addEventListener("click", async () => {
  try {
    output.textContent = "Loading WebAssembly and WebGPU…";
    await initWasm();
    const fft = await WgpuFft.init();
    const cachedSnapshot = localStorage.getItem("wgpu-fft.pipeline-cache.v1");
    if (cachedSnapshot) {
      await fft.importSnapshot(cachedSnapshot);
    }
    const plan = await fft.createPlan(
      4,
      1,
      WebFftDirection.Forward,
      WebFftPrecision.F32,
      WebFftNormalization.None,
    );
    const input = new Float32Array([1, 0, 2, 0, 3, 0, 4, 0]);
    const gpuInput = fft.upload(input);
    const gpuOutput = fft.createBuffer(plan.outputBytes);
    await plan.execute(gpuInput, gpuOutput);
    localStorage.setItem("wgpu-fft.pipeline-cache.v1", fft.exportSnapshot());
    const downloaded = await fft.download(gpuOutput);
    const values = new Float32Array(
      downloaded.buffer,
      downloaded.byteOffset,
      downloaded.byteLength / Float32Array.BYTES_PER_ELEMENT,
    );
    output.textContent = JSON.stringify(
      {
        adapter: fft.adapterName,
        backend: fft.backend,
        df64Available: fft.df64Available,
        df64CanaryWords: fft.df64CanaryWords,
        df64CanaryError: fft.df64CanaryError,
        route: plan.route,
        output: Array.from(values),
      },
      null,
      2,
    );
  } catch (error) {
    output.textContent = error?.stack ?? String(error);
  }
});
