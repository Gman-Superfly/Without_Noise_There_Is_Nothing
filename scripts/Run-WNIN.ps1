param(
    [string]$Ws = "ws://127.0.0.1:8765",
    [string]$Optimizer = "adamw",
    [int]$Steps = 1000,
    [string]$NoiseMode = "none",
    [double]$NoiseMag = 0.0,
    [string]$NoiseSchedule = "constant",
    [int]$NoisePeriod = 0
)

# Ensure uv is available
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is not installed. Install from https://docs.astral.sh/uv/"
    exit 1
}

# Install dependencies (idempotent)
uv sync --frozen-if-present

# Run the trainer
uv run wnin-train --ws $Ws --optimizer $Optimizer --steps $Steps --noise-mode $NoiseMode --noise-mag $NoiseMag --noise-schedule $NoiseSchedule --noise-period $NoisePeriod


