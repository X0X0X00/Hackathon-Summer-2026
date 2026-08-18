param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$V1Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$SourceRoot = Join-Path $V1Root "src"
$CacheRoot = Join-Path $V1Root "cache"
$ArtifactRoot = Join-Path $V1Root "artifacts"
$ExternalFile = Join-Path $CacheRoot "external_MERFISH_spinal_cord.h5ad"

New-Item -ItemType Directory -Force -Path $CacheRoot, $ArtifactRoot | Out-Null

if (-not (Test-Path -LiteralPath $ExternalFile)) {
    throw "Missing external reference: $ExternalFile. See v1/README.md for source and checksum."
}

$ExistingPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = if ($ExistingPythonPath) {
    "$SourceRoot;$ExistingPythonPath"
} else {
    $SourceRoot
}
$env:HACKATHON_USE_EXTERNAL_REFERENCE = "1"

& $Python (Join-Path $SourceRoot "train_model.py")
if ($LASTEXITCODE -ne 0) { throw "train_model.py failed" }

& $Python (Join-Path $SourceRoot "multiseed_ensemble.py")
if ($LASTEXITCODE -ne 0) { throw "multiseed_ensemble.py failed" }

& $Python (Join-Path $SourceRoot "prior_postprocess.py")
if ($LASTEXITCODE -ne 0) { throw "prior_postprocess.py failed" }

& $Python (Join-Path $SourceRoot "prepare_external_reference.py")
if ($LASTEXITCODE -ne 0) { throw "prepare_external_reference.py failed" }

& $Python (Join-Path $SourceRoot "hierarchical_glia.py")
if ($LASTEXITCODE -ne 0) { throw "hierarchical_glia.py failed" }

& $Python (Join-Path $SourceRoot "targeted_80.py")
if ($LASTEXITCODE -ne 0) { throw "targeted_80.py failed" }

& $Python (Join-Path $SourceRoot "prepare_neuronal_reference.py")
if ($LASTEXITCODE -ne 0) { throw "prepare_neuronal_reference.py failed" }

& $Python (Join-Path $SourceRoot "neuronal_experts.py")
if ($LASTEXITCODE -ne 0) { throw "neuronal_experts.py failed" }

& $Python (Join-Path $SourceRoot "pair_experts_80.py")
if ($LASTEXITCODE -ne 0) { throw "pair_experts_80.py failed" }

& $Python (Join-Path $SourceRoot "ventral_experts_80.py")
if ($LASTEXITCODE -ne 0) { throw "ventral_experts_80.py failed" }

& $Python (Join-Path $SourceRoot "neural_pair_experts_80.py")
if ($LASTEXITCODE -ne 0) { throw "neural_pair_experts_80.py failed" }

$Candidate = Join-Path $ArtifactRoot "prediction_neural_pair_experts_80.csv"
$Submission = Join-Path $V1Root "submission\prediction.csv"
Copy-Item -LiteralPath $Candidate -Destination $Submission -Force

& $Python (Join-Path $SourceRoot "validate_submission.py") $Submission
if ($LASTEXITCODE -ne 0) { throw "submission validation failed" }

Write-Host "v1 pipeline complete: $Submission"
