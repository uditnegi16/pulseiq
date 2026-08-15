<#
.SYNOPSIS
    Start the API and the Streamlit dashboard.

.DESCRIPTION
    Two processes: uvicorn serving the API, Streamlit serving the UI. The
    dashboard talks to the API over HTTP rather than importing models, so the
    model loads once in one process instead of once per Streamlit session.

.EXAMPLE
    .\scripts\run_local.ps1
#>
param(
    [int]$ApiPort = 8000,
    [switch]$ApiOnly
)

if (-not $env:VIRTUAL_ENV) {
    Write-Host "!! virtualenv not active" -ForegroundColor Red
    Write-Host "   run: .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host ">> starting API on port $ApiPort" -ForegroundColor Cyan
$api = Start-Process -PassThru -NoNewWindow powershell -ArgumentList @(
    "-NoProfile", "-Command",
    "uvicorn pulseiq.api.main:app --reload --port $ApiPort"
)

# Poll rather than sleep once. The API loads DistilBERT at startup, which takes
# roughly 8 seconds on CPU, so a fixed 4-second wait reported "not responding"
# for a service that was simply still booting.
Write-Host "   waiting for the API (it loads the sentiment model on startup)..." -ForegroundColor DarkGray
$health = $null
foreach ($attempt in 1..25) {
    Start-Sleep -Seconds 1
    try {
        $health = Invoke-RestMethod "http://localhost:$ApiPort/health" -TimeoutSec 3
        break
    } catch {
        continue
    }
}

if ($health) {
    Write-Host "   API status: $($health.status)" -ForegroundColor Green
    foreach ($c in $health.components) {
        $colour = if ($c.status -eq "ok") { "Green" } else { "Yellow" }
        Write-Host "     $($c.name): $($c.status)" -ForegroundColor $colour
    }
} else {
    Write-Host "   API did not respond within 25s -- check the output above" -ForegroundColor Yellow
}

Write-Host "`n   docs: http://localhost:$ApiPort/docs" -ForegroundColor Cyan

if ($ApiOnly) {
    Write-Host "`nAPI running. Ctrl+C to stop." -ForegroundColor Cyan
    Wait-Process -Id $api.Id
    exit 0
}

Write-Host ">> starting Streamlit dashboard" -ForegroundColor Cyan
$env:API_BASE_URL = "http://localhost:$ApiPort"
streamlit run app/streamlit_app.py

Write-Host "`n>> stopping API" -ForegroundColor Cyan
Stop-Process -Id $api.Id -ErrorAction SilentlyContinue
