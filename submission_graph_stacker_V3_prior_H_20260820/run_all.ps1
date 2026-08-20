param(
    [string]$ProjectRoot = "C:\Users\lizhi\Documents\ChatGPT\hackathon",
    [string]$CacheDir = "C:\Users\lizhi\Hackathon-Summer-2026\Hackathon-Summer-2026\other_model\Hackathon-Summer-2026\work\cache_ext\gene_token",
    [string]$Python = "C:\conda\envs\d2l\python.exe"
)

$ErrorActionPreference = "Stop"
$BundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:HACKATHON_PROJECT_ROOT = $ProjectRoot
$env:CUDA_VISIBLE_DEVICES = "0"

& $Python "$BundleRoot\code\prepare_final_prior_h_inputs.py" `
    --project-root $ProjectRoot `
    --cache-dir $CacheDir `
    --output-dir "$BundleRoot\model\inputs" `
    --depth-threshold 14 `
    --h-weight 0.21
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python "$BundleRoot\code\train_final_prior_h_graph_stacker.py" `
    --class-graph "$ProjectRoot\outputs\reference_class_similarity_graph\reference_class_graph.npz" `
    --output-dir "$BundleRoot\model\trained_stacker" `
    --graph-lambda 0.05 `
    --epochs 800 `
    --learning-rate 0.03 `
    --gate-mode fixed `
    --seed 42
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python "$BundleRoot\code\finalize_bundle.py" `
    --project-root $ProjectRoot `
    --bundle-root $BundleRoot
exit $LASTEXITCODE
