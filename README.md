# Without_Noise_There_Is_Nothing
Noise–Temperature Landscape Explorer (WNIN)

Quickstart (Windows PowerShell):

1) Install uv: see `https://docs.astral.sh/uv/`.
2) Run trainer streaming metrics over WebSocket:

```powershell
scripts\Run-WNIN.ps1 -Ws ws://127.0.0.1:8765 -Optimizer adamw -Steps 1000 -NoiseMode input_jitter -NoiseMag 0.1 -NoiseSchedule cosine -NoisePeriod 500
```

CLI directly:

```powershell
uv run wnin-train --ws ws://127.0.0.1:8765 --optimizer muon --steps 800 --noise-mode none
```

Extending optimizers:
- Register a factory with `OptimizerRegistry.register("your_name", factory)`.
- Built-ins: `adamw`, `sgd`, `rmsprop`, `muon` (placeholder), `normalized_dynamics` (basic normalized-gradient; replace with your spec).

Next:
- Zig renderer to visualize loss slices and optimizer trajectories in real time.

Optimizer plugins (modular):
- Place Python files in `optimizers/`. Each module should call `OptimizerRegistry.register(name, factory)` at import time.
- The loader imports all modules in `optimizers/` automatically at runtime.
- Built-ins provided as separate files: `adamw.py`, `sgd.py`, `rmsprop.py`, `muon.py`, `normalized_dynamics.py`.
- Start from `optimizers/template.py` for new optimizers.

Zig viewer (skeleton):
- Build (Windows PowerShell): `scripts\Build-ZigViz.ps1`
- Run: `cd zigviz && zig build run`
