<#
.SYNOPSIS
    Stage, commit and push, handling pre-commit hooks that rewrite files.

.DESCRIPTION
    Formatter hooks (end-of-file-fixer, ruff-format) modify files during the
    hook run, which aborts the commit BY DESIGN -- the modified files need
    re-staging. `git push` then reports "Everything up-to-date", which reads
    like success and is not.

    This cost a full phase of work before it was noticed. See docs/error-log.md
    E-009.

    This script retries once after the hooks rewrite, then verifies the commit
    actually landed rather than trusting the exit code.

.EXAMPLE
    .\scripts\commit.ps1 "Phase 4: CI metric regression gate"

.EXAMPLE
    .\scripts\commit.ps1 "WIP" -NoPush
#>
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Message,

    [switch]$NoPush,
    [switch]$SkipTests
)

$ErrorActionPreference = "Continue"

function Write-Step($text) { Write-Host "`n>> $text" -ForegroundColor Cyan }

# --- tests ------------------------------------------------------------------
if (-not $SkipTests) {
    Write-Step "running unit tests"
    pytest tests\unit -q
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n!! tests failed -- nothing committed" -ForegroundColor Red
        Write-Host "   use -SkipTests to commit anyway (work in progress)" -ForegroundColor Yellow
        exit 1
    }
}

$before = git rev-parse HEAD 2>$null

# --- commit, retrying once after hooks rewrite files ------------------------
foreach ($attempt in 1..2) {
    Write-Step "commit attempt $attempt"
    git add -A
    git commit -m $Message

    $after = git rev-parse HEAD 2>$null
    if ($after -ne $before) {
        Write-Host "committed: $($after.Substring(0,8))" -ForegroundColor Green
        break
    }

    if ($attempt -eq 1) {
        Write-Host "hooks rewrote files -- re-staging and retrying" -ForegroundColor Yellow
    }
}

# --- verify, do not assume --------------------------------------------------
$after = git rev-parse HEAD 2>$null
if ($after -eq $before) {
    Write-Host "`n!! no commit was created." -ForegroundColor Red
    Write-Host "   Either there was nothing to commit, or a hook is failing for" -ForegroundColor Yellow
    Write-Host "   a real reason. Run 'pre-commit run --all-files' to see which." -ForegroundColor Yellow
    git status --short
    exit 1
}

# --- push -------------------------------------------------------------------
if (-not $NoPush) {
    Write-Step "pushing"
    git push
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n!! push failed -- the commit exists locally" -ForegroundColor Red
        exit 1
    }
}

Write-Step "done"
git log --oneline -1
git status -sb
