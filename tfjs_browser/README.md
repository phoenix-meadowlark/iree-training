# TF.js Latency Benchmarks

## Experiemntal Parameters

The experimental parameters are controlled by constants at the top of
`script.js`. `TFJS_BACKEND` and `OPTIMIZER` are the only parameters modified for
the experiments in the paper.

## Chrome Flags for WASM

WASM multithreading can be disabled using the `MULTI_THREAD` variable at the top
of `script.js`, but must be enabled in
`chrome://flags#enable-webassembly-threads` for it to work.

WASM SIMD support can be toggled by `chrome://flags#enable-webassembly-simd`.

## Benchmarking Locally

The benchmarks can be executed locally by opening `index.html` in your browser.
Results are logged via `console.log` (see `results/` for the logs from the
experiments ran for the paper).

## Benchmarking on Android

To run the benchmarks on Android, connect to a device via `adb` and enable port
forwarding via

```sh
adb reverse tcp:8000 tcp:8000
```

Start a local host server, e.g. via Python:

```python
python -m http.server 8000
```

Open `chrome://inspect/#devices` in desktop Chrome, open
`http://localhost:8000` in your phone's Chrome browser, and use desktop Chrome
to view the `console.log` calls. The logs can be saved by right clicking and
selecting `Save as...`, or the results can be copied directly for analysis.
