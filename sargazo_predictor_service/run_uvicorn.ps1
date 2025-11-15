param(
    [switch]$Reload
)

$module = 'sargazo_predictor_service.app.main:app'
$hostAddr = '0.0.0.0'
$port = 8000

if ($Reload) { $reloadFlag = '--reload' } else { $reloadFlag = '' }

Write-Host "Starting uvicorn $module on ${hostAddr}:$port (reload=$Reload)"

uvicorn $module --host $hostAddr --port $port $reloadFlag
