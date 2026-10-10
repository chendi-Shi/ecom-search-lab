param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CommandArgs)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$taskPython = $env:ECOM_SEARCH_PYTHON
if (-not $taskPython) {
    $taskPython = (Get-Command python -ErrorAction Stop).Source
}
if ($taskPython -like '*WindowsApps*') { throw 'Please install Python 3.10+ or set ECOM_SEARCH_PYTHON to an existing Python interpreter.' }
Push-Location -LiteralPath $projectRoot
try {
    & $taskPython -m ecom_search @CommandArgs
    exit $LASTEXITCODE
} finally { Pop-Location }
