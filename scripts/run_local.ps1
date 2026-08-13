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

Start-Sleep -Seconds 4

try {
    $health = Invoke-RestMethod "http://localhost:$ApiPort/health" -TimeoutSec 10
    Write-Host "   API status: $($health.status)" -ForegroundColor Green
    foreach ($c in $health.components) {
        $colour = if ($c.status -eq "ok") { "Green" } else { "Yellow" }
        Write-Host "     $($c.name): $($c.status)" -ForegroundColor $colour
    }
} catch {
    Write-Host "   API not responding yet -- check the output above" -ForegroundColor Yellow
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
