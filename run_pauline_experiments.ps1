$ErrorActionPreference = "Stop"

$python = ".venv311\Scripts\python.exe"
if (!(Test-Path $python)) {
    throw "Python executable not found at $python"
}

$member = "Pauline"
$experiments = @(
    @{ id = 1; args = "--policy CnnPolicy --lr 2e-4 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.10 --timesteps 15000" },
    @{ id = 2; args = "--policy CnnPolicy --lr 4e-4 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.03 --epsilon-decay 0.10 --timesteps 15000" },
    @{ id = 3; args = "--policy CnnPolicy --lr 8e-4 --gamma 0.98 --batch-size 64 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.12 --timesteps 15000" },
    @{ id = 4; args = "--policy CnnPolicy --lr 5e-4 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.10 --timesteps 15000" },
    @{ id = 5; args = "--policy CnnPolicy --lr 6e-5 --gamma 0.995 --batch-size 64 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.20 --timesteps 15000" },
    @{ id = 6; args = "--policy CnnPolicy --lr 3e-4 --gamma 0.96 --batch-size 128 --epsilon-start 1.0 --epsilon-end 0.10 --epsilon-decay 0.10 --timesteps 15000" },
    @{ id = 7; args = "--policy MlpPolicy --lr 9e-5 --gamma 0.99 --batch-size 48 --epsilon-start 1.0 --epsilon-end 0.02 --epsilon-decay 0.08 --timesteps 15000" },
    @{ id = 8; args = "--policy MlpPolicy --lr 5e-4 --gamma 0.985 --batch-size 96 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.06 --timesteps 15000" },
    @{ id = 9; args = "--policy CnnPolicy --lr 2.5e-4 --gamma 0.975 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.07 --epsilon-decay 0.18 --timesteps 15000" },
    @{ id = 10; args = "--policy MlpPolicy --lr 7e-5 --gamma 0.999 --batch-size 64 --epsilon-start 1.0 --epsilon-end 0.04 --epsilon-decay 0.14 --timesteps 15000" }
)

function Test-ExperimentComplete {
    param(
        [string]$Member,
        [int]$ExperimentId
    )

    $configPath = Join-Path "results\$Member\logs" "exp$ExperimentId\config.json"
    $modelPath = Join-Path "results\$Member\models" "exp${ExperimentId}_model.zip"

    if (!(Test-Path $configPath) -or !(Test-Path $modelPath)) {
        return $false
    }

    try {
        $config = Get-Content $configPath -Raw | ConvertFrom-Json
    } catch {
        return $false
    }

    return ($null -ne $config.mean_reward -and $null -ne $config.std_reward)
}

Write-Host "Running Pauline experiments with resume support..."

foreach ($exp in $experiments) {
    $id = [int]$exp.id

    if (Test-ExperimentComplete -Member $member -ExperimentId $id) {
        Write-Host "Skipping exp$id (already complete)."
        continue
    }

    $fullArgs = "train.py --member $member --experiment $id $($exp.args)"
    Write-Host "`n=== Running exp$id ==="
    Write-Host $fullArgs

    $argList = @("train.py", "--member", $member, "--experiment", "$id") + ($exp.args -split "\s+")
    & $python @argList
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment $id failed."
    }
}

Write-Host " complete."
