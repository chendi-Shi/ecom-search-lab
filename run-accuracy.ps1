param([int]$Port = 18866)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$taskPython = $env:ECOM_SEARCH_PYTHON
if (-not $taskPython) {
    $taskPython = (Get-Command python -ErrorAction Stop).Source
}
if ($taskPython -like '*WindowsApps*') { throw 'Please install Python 3.10+ or set ECOM_SEARCH_PYTHON to an existing Python interpreter.' }
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
Push-Location -LiteralPath $projectRoot
try {
    $requiredArtifacts = @(
        'outputs/accuracy-v1/model.json',
        'outputs/accuracy-v1/report.json',
        'outputs/accuracy-v1/retrieval/products.sqlite',
        'outputs/accuracy-v1/retrieval/pool_strategy.json'
    )
    foreach ($artifact in $requiredArtifacts) {
        if (-not (Test-Path -LiteralPath $artifact -PathType Leaf)) {
            throw "Missing frozen accuracy artifact: $artifact. Finish training and final evaluation before launching this configuration."
        }
    }
    & $taskPython -m ecom_search serve --index outputs/accuracy-v1/retrieval/products.sqlite --accuracy-model outputs/accuracy-v1/model.json --retrieval-strategy outputs/accuracy-v1/retrieval/pool_strategy.json --profiles outputs/local-profiles-accuracy.sqlite --multimodal-index outputs/multimodal/selected --beauty-experiment outputs/personalization/experiment --port $Port
    exit $LASTEXITCODE
} finally { Pop-Location }
