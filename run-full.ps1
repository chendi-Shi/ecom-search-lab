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
    & $taskPython -m ecom_search serve --model outputs/ranking/model.json --disk-index outputs/disk-vectors-v1/int8 --generative-index outputs/generative-v1 --multimodal-index outputs/multimodal/selected --beauty-experiment outputs/personalization/experiment --port $Port
    exit $LASTEXITCODE
} finally { Pop-Location }
