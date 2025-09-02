# Without_Noise_There_Is_Nothing
Noise–Temperature Landscape Explorer (WNIN)

What & why:
- Build an interactive, research-grade tool to test the claim that “noise is a resource” in optimization and learning.
- Show direct, visual cause→effect between noise/temperature, optimizer dynamics, exploration/escape, and generalization.

How it works (architecture):
- Python trainer emits typed `step_metrics` events over WebSockets (uv-run). Small models/datasets keep iteration fast.
- Modular optimizer plugins live in `optimizers/` and self-register; compare AdamW/SGD/RMSprop/NormalizedDynamics/Muon.
- Zig viewer (GPU-first) will visualize trajectories and (next) loss-landscape slices in real time.

What you can test now:
- Optimizer comparisons under identical noise schedules.
- Noise ablations (`none`, `input_jitter`) and schedules (constant/cosine/cyclical/inverse_anneal).
- Early exploration vs late generalization trade-offs via temperature control.

## Thesis (detailed and falsifiable)
- **Claim**: Appropriately coupled noise/temperature improves exploration, enables escape from sharp basins, and yields flatter minima and better generalization compared to zero-noise training at matched training loss.
- **Operationalization**:
  - Temperature/noise is controlled via schedules and modes (e.g., `input_jitter`, later `sgld`/`gradient_gaussian`).
  - We log per-step metrics: train/val loss, perplexity, grad norm, sharpness proxies (Hutchinson trace), and schedule state.
- **Predictions** (falsifiable):
  - At matched train loss, runs with moderate noise have lower sharpness proxies and lower val loss/perplexity than no-noise baselines.
  - Cyclical/annealed temperature increases early escape events from sharp basins (detected via curvature/trajectory changes) and stabilizes late training.
  - Stochastic resonance: under weak supervision or label noise, an intermediate noise band outperforms both zero and high noise.
  - Dithered quantization at inference improves calibration (ECE) and perplexity relative to undithered at equal bit-depth.
  - Langevin-like dynamics (future `sgld`) bias toward flatter minima versus AdamW/SGD, measurable by lower sharpness at similar train loss.
- **Caveats**:
  - Excess noise destabilizes training; gradients must be clipped/normalized; schedules matter.
  - Effects are dataset- and scale-dependent; results should be replicated across seeds and tasks.

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
