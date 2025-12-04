<#
.SYNOPSIS
    Automated training and evaluation script for DreamerV3 on Highway-Env environments.

.DESCRIPTION
    This script:
    1. Runs training for a specified duration
    2. Automatically finds the latest checkpoint
    3. Runs evaluation with video recording

.PARAMETER Environment
    Highway-env environment to use (default: merge)
    - merge: Highway merge negotiation
    - highway: High-speed highway driving with traffic
    - roundabout: Navigate through a roundabout
    - parking: Parallel parking task (continuous control)
    - intersection: Navigate through an intersection
    - racetrack: Drive on a racetrack (continuous control)

.PARAMETER TrainDuration
    Training duration in seconds (default: 3600 = 1 hour)

.PARAMETER ModelSize
    Model size config: size1m, size12m, size25m, size50m, size100m, size200m (default: size1m)

.PARAMETER EvalSteps
    Number of evaluation steps (default: 2000)

.PARAMETER EvalEnvs
    Number of parallel eval environments (default: 1 for video recording)

.PARAMETER TrainEnvs
    Number of parallel training environments (default: 4)

.PARAMETER Platform
    JAX platform: cpu or cuda (default: cpu)

.PARAMETER TrainRatio
    Training ratio - number of gradient steps per environment step (default: 32)

.PARAMETER BatchSize
    Batch size for training (default: 8 for cpu, 16 for cuda)

.PARAMETER Checkpoint
    Path to a specific checkpoint to resume training from or evaluate

.PARAMETER RunName
    Custom name for this run (default: auto-generated timestamp)

.PARAMETER LogDir
    Base log directory (default: $env:USERPROFILE\logdir\dreamer)

.PARAMETER SkipTraining
    Skip training and only run evaluation on the latest checkpoint

.PARAMETER SkipEval
    Skip evaluation after training

.PARAMETER OpenVideoFolder
    Open the video folder in Explorer after evaluation

.PARAMETER SpeedConfig
    Speed configuration: default, fast, veryfast (default: default)
    - default: 30 m/s = 108 km/h
    - fast: 35 m/s = 126 km/h (~17% faster)
    - veryfast: 42 m/s = 151 km/h (~40% faster)

.PARAMETER ShowVisual
    Show visual display window during training/evaluation (slower but visible).
    When enabled, forces TrainEnvs=1 to avoid multiple windows.

.EXAMPLE
    .\train_and_eval.ps1
    # Runs 1 hour training with size1m model, then evaluates with video

.EXAMPLE
    .\train_and_eval.ps1 -SpeedConfig fast
    # Train with faster vehicle speeds (35 m/s)

.EXAMPLE
    .\train_and_eval.ps1 -SpeedConfig veryfast -TrainDuration 1800
    # Train 30 min with very fast speeds (42 m/s)

.EXAMPLE
    .\train_and_eval.ps1 -TrainDuration 7200 -ModelSize size12m
    # Runs 2 hour training with size12m model

.EXAMPLE
    .\train_and_eval.ps1 -SkipTraining
    # Only runs evaluation on the latest existing checkpoint

.EXAMPLE
    .\train_and_eval.ps1 -Checkpoint "C:\path\to\ckpt" -SkipTraining
    # Evaluate a specific checkpoint

.EXAMPLE
    .\train_and_eval.ps1 -TrainDuration 1800 -SkipEval
    # Train for 30 minutes without evaluation

.EXAMPLE
    .\train_and_eval.ps1 -SkipTraining -OpenVideoFolder
    # Evaluate and open video folder when done

.EXAMPLE
    .\train_and_eval.ps1 -ShowVisual -TrainDuration 300
    # Train for 5 min with visual display (slower)

.EXAMPLE
    .\train_and_eval.ps1 -Environment highway -TrainDuration 3600
    # Train on highway environment for 1 hour

.EXAMPLE
    .\train_and_eval.ps1 -Environment roundabout -ShowVisual
    # Train on roundabout with visual display

.EXAMPLE
    .\train_and_eval.ps1 -Environment parking -TrainDuration 7200
    # Train parking for 2 hours (harder task, needs more time)
#>

param(
    [ValidateSet("merge", "highway", "roundabout", "parking", "intersection", "racetrack")]
    [string]$Environment = "merge",
    [int]$TrainDuration = 3600,
    [string]$ModelSize = "size1m",
    [int]$EvalSteps = 2000,
    [int]$EvalEnvs = 1,
    [int]$TrainEnvs = 4,
    [string]$Platform = "cpu",
    [float]$TrainRatio = 32.0,
    [int]$BatchSize = 0,
    [string]$Checkpoint = "",
    [string]$RunName = "",
    [string]$LogDir = "",
    [ValidateSet("default", "fast", "veryfast")]
    [string]$SpeedConfig = "default",
    [switch]$SkipTraining,
    [switch]$SkipEval,
    [switch]$OpenVideoFolder,
    [switch]$ShowVisual
)

# Configuration
$ErrorActionPreference = "Stop"
$BaseLogDir = if ($LogDir) { $LogDir } else { "$env:USERPROFILE\logdir\dreamer" }
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunName = if ($RunName) { $RunName } else { "train_${Timestamp}" }
$RunLogDir = "$BaseLogDir\$RunName"

# Auto-set batch size based on platform if not specified
if ($BatchSize -eq 0) {
    $BatchSize = if ($Platform -eq "cuda") { 16 } else { 8 }
}

# Set environment config based on Environment and SpeedConfig
$EnvConfig = switch ($Environment) {
    "merge" {
        switch ($SpeedConfig) {
            "fast"     { "highway_merge_fast" }
            "veryfast" { "highway_merge_veryfast" }
            default    { "highway_merge_rgb" }
        }
    }
    "highway" {
        if ($SpeedConfig -eq "fast") { "highway_fast" } else { "highway" }
    }
    "roundabout"    { "roundabout" }
    "parking"       { "parking" }
    "intersection"  { "intersection" }
    "racetrack"     { "racetrack" }
    default         { "highway_merge_rgb" }
}

$EnvDescription = switch ($Environment) {
    "merge"        { "Merge - Highway merge negotiation" }
    "highway"      { "Highway - High-speed driving with traffic" }
    "roundabout"   { "Roundabout - Navigate through a roundabout" }
    "parking"      { "Parking - Parallel parking task" }
    "intersection" { "Intersection - Navigate through intersection" }
    "racetrack"    { "Racetrack - Drive on a racetrack" }
    default        { "Merge - Highway merge negotiation" }
}

$SpeedDescription = switch ($SpeedConfig) {
    "fast"     { "Fast (35 m/s = 126 km/h)" }
    "veryfast" { "Very Fast (42 m/s = 151 km/h)" }
    default    { "Default (30 m/s = 108 km/h)" }
}

# Note: Speed config only applies to merge and highway environments
if ($SpeedConfig -ne "default" -and $Environment -notin @("merge", "highway")) {
    Write-Host "Note: SpeedConfig only affects 'merge' and 'highway' environments" -ForegroundColor Yellow
}

# Set render mode based on ShowVisual
$RenderMode = if ($ShowVisual) { "human" } else { "rgb_array" }

# Force single env when showing visual to avoid multiple windows
if ($ShowVisual -and $TrainEnvs -gt 1) {
    Write-Host "Note: ShowVisual enabled, setting TrainEnvs=1 to avoid multiple windows" -ForegroundColor Yellow
    $TrainEnvs = 1
}

# Activate virtual environment if it exists
$VenvActivate = ".\\.venv\\Scripts\\Activate.ps1"
if (Test-Path $VenvActivate) {
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    & $VenvActivate
}

function Find-LatestCheckpoint {
    param([string]$SearchDir = $BaseLogDir)
    
    Write-Host "`nSearching for latest checkpoint in: $SearchDir" -ForegroundColor Yellow
    
    # Find all checkpoint directories
    $ckptDirs = Get-ChildItem -Path $SearchDir -Recurse -Directory -Filter "ckpt" -ErrorAction SilentlyContinue
    
    if (-not $ckptDirs) {
        Write-Host "No checkpoint directories found!" -ForegroundColor Red
        return $null
    }
    
    # Find the latest checkpoint file across all ckpt directories
    $latestCkpt = $null
    $latestTime = [DateTime]::MinValue
    $latestRunDir = $null
    
    foreach ($ckptDir in $ckptDirs) {
        $ckptFiles = Get-ChildItem -Path $ckptDir.FullName -Directory -ErrorAction SilentlyContinue
        foreach ($ckpt in $ckptFiles) {
            if ($ckpt.LastWriteTime -gt $latestTime) {
                $latestTime = $ckpt.LastWriteTime
                $latestCkpt = $ckpt.FullName
                $latestRunDir = $ckptDir.Parent.FullName
            }
        }
    }
    
    if ($latestCkpt) {
        Write-Host "Found latest checkpoint: $latestCkpt" -ForegroundColor Green
        Write-Host "  From run: $latestRunDir" -ForegroundColor Green
        Write-Host "  Created: $latestTime" -ForegroundColor Green
        return @{
            Checkpoint = $latestCkpt
            RunDir = $latestRunDir
        }
    }
    
    Write-Host "No checkpoints found!" -ForegroundColor Red
    return $null
}

function Start-Training {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  STARTING TRAINING" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Environment:   $EnvDescription" -ForegroundColor Green
    Write-Host "Run Name:      $RunName"
    Write-Host "Log Directory: $RunLogDir"
    Write-Host "Duration:      $TrainDuration seconds ($([math]::Round($TrainDuration/3600, 2)) hours)"
    Write-Host "Model Size:    $ModelSize"
    Write-Host "Platform:      $Platform"
    Write-Host "Batch Size:    $BatchSize"
    Write-Host "Train Ratio:   $TrainRatio"
    Write-Host "Train Envs:    $TrainEnvs"
    if ($Environment -in @("merge", "highway")) {
        Write-Host "Speed:         $SpeedDescription" -ForegroundColor Yellow
    }
    Write-Host "Render Mode:   $RenderMode" -ForegroundColor $(if ($ShowVisual) { "Green" } else { "Gray" })
    if ($Checkpoint) {
        Write-Host "Resume From:   $Checkpoint" -ForegroundColor Yellow
    }
    Write-Host "========================================`n" -ForegroundColor Cyan
    
    $trainArgs = @(
        "dreamerv3/main.py",
        "--logdir", $RunLogDir,
        "--configs", $EnvConfig, $ModelSize,
        "--run.duration", $TrainDuration,
        "--run.envs", $TrainEnvs,
        "--run.train_ratio", $TrainRatio,
        "--batch_size", $BatchSize,
        "--jax.platform", $Platform,
        "--env.gym.render_mode", $RenderMode,
        "--logger.timer", "False"
    )
    
    # Add checkpoint resume if specified
    if ($Checkpoint) {
        $trainArgs += @("--run.from_checkpoint", $Checkpoint)
    }
    
    Write-Host "Command: python $($trainArgs -join ' ')`n" -ForegroundColor DarkGray
    
    $startTime = Get-Date
    python @trainArgs
    $exitCode = $LASTEXITCODE
    $elapsed = (Get-Date) - $startTime
    
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  TRAINING COMPLETED" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Exit Code:    $exitCode"
    Write-Host "Elapsed Time: $($elapsed.ToString('hh\:mm\:ss'))"
    Write-Host "========================================`n" -ForegroundColor Cyan
    
    return $exitCode -eq 0
}

function Start-Evaluation {
    param(
        [string]$CheckpointPath,
        [string]$RunDir
    )
    
    $VideoDir = "$RunDir\videos_eval_$Timestamp"
    $EvalLogDir = "$RunDir\eval_$Timestamp"
    
    Write-Host "`n========================================" -ForegroundColor Magenta
    Write-Host "  STARTING EVALUATION" -ForegroundColor Magenta
    Write-Host "========================================" -ForegroundColor Magenta
    Write-Host "Environment:   $EnvDescription" -ForegroundColor Green
    Write-Host "Checkpoint:    $CheckpointPath"
    Write-Host "Video Output:  $VideoDir"
    if ($Environment -in @("merge", "highway")) {
        Write-Host "Speed:         $SpeedDescription" -ForegroundColor Yellow
    }
    Write-Host "Render Mode:   $RenderMode" -ForegroundColor $(if ($ShowVisual) { "Green" } else { "Gray" })
    Write-Host "Eval Steps:    $EvalSteps"
    Write-Host "Eval Envs:     $EvalEnvs"
    Write-Host "========================================`n" -ForegroundColor Magenta
    
    $evalArgs = @(
        "dreamerv3/main.py",
        "--logdir", $EvalLogDir,
        "--configs", $EnvConfig, $ModelSize,
        "--script", "eval_only",
        "--run.from_checkpoint", $CheckpointPath,
        "--run.steps", $EvalSteps,
        "--run.envs", $EvalEnvs,
        "--jax.platform", $Platform,
        "--env.gym.render_mode", "rgb_array",
        "--env.gym.record_video", "True",
        "--env.gym.video_dir", $VideoDir
    )
    
    Write-Host "Command: python $($evalArgs -join ' ')`n" -ForegroundColor DarkGray
    
    $startTime = Get-Date
    python @evalArgs
    $exitCode = $LASTEXITCODE
    $elapsed = (Get-Date) - $startTime
    
    Write-Host "`n========================================" -ForegroundColor Magenta
    Write-Host "  EVALUATION COMPLETED" -ForegroundColor Magenta
    Write-Host "========================================" -ForegroundColor Magenta
    Write-Host "Exit Code:    $exitCode"
    Write-Host "Elapsed Time: $($elapsed.ToString('hh\:mm\:ss'))"
    
    # List generated videos
    if (Test-Path $VideoDir) {
        $videos = Get-ChildItem -Path $VideoDir -Filter "*.mp4" -ErrorAction SilentlyContinue
        if ($videos) {
            Write-Host "`nGenerated Videos:" -ForegroundColor Green
            foreach ($video in $videos) {
                $sizeKB = [math]::Round($video.Length / 1KB, 1)
                Write-Host "  - $($video.Name) (${sizeKB} KB)"
            }
            Write-Host "`nVideo folder: $VideoDir" -ForegroundColor Green
            
            # Open video folder if requested
            if ($OpenVideoFolder) {
                Write-Host "Opening video folder..." -ForegroundColor Cyan
                Start-Process explorer.exe -ArgumentList $VideoDir
            }
        }
    }
    Write-Host "========================================`n" -ForegroundColor Magenta
    
    return @{
        Success = ($exitCode -eq 0)
        VideoDir = $VideoDir
    }
}

# Main execution
Write-Host "`n"
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor White
Write-Host "║     DreamerV3 - Highway Merge Training & Evaluation          ║" -ForegroundColor White
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor White
Write-Host ""

$checkpointInfo = $null

if (-not $SkipTraining) {
    # Run training
    $trainSuccess = Start-Training
    
    if (-not $trainSuccess) {
        Write-Host "Training failed! Check the logs for errors." -ForegroundColor Red
        exit 1
    }
    
    # Find checkpoint from this training run
    $checkpointInfo = Find-LatestCheckpoint -SearchDir $RunLogDir
} else {
    Write-Host "Skipping training, searching for existing checkpoints..." -ForegroundColor Yellow
    
    # Use specified checkpoint or find latest
    if ($Checkpoint) {
        $checkpointInfo = @{
            Checkpoint = $Checkpoint
            RunDir = Split-Path (Split-Path $Checkpoint -Parent) -Parent
        }
        Write-Host "Using specified checkpoint: $Checkpoint" -ForegroundColor Green
    } else {
        $checkpointInfo = Find-LatestCheckpoint -SearchDir $BaseLogDir
    }
}

if (-not $checkpointInfo) {
    Write-Host "No checkpoint found for evaluation!" -ForegroundColor Red
    exit 1
}

# Run evaluation unless skipped
if (-not $SkipEval) {
    $evalResult = Start-Evaluation -CheckpointPath $checkpointInfo.Checkpoint -RunDir $checkpointInfo.RunDir

    if ($evalResult.Success) {
        Write-Host "`n✓ Training and evaluation completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "`n✗ Evaluation failed! Check the logs for errors." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n✓ Training completed (evaluation skipped)!" -ForegroundColor Green
    Write-Host "Checkpoint saved at: $($checkpointInfo.Checkpoint)" -ForegroundColor Cyan
}
