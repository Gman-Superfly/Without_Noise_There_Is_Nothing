param()

if (-not (Get-Command zig -ErrorAction SilentlyContinue)) {
    Write-Error "Zig is not installed. Get it from https://ziglang.org/download/"
    exit 1
}

Push-Location zigviz
try {
    zig build -Doptimize=ReleaseSafe
} finally {
    Pop-Location
}


