<#
Avvia un comando staccato da qualunque terminale o sessione (via WMI): continua a girare anche se
chi l'ha lanciato si chiude. L'output va in coda al file di log indicato.

Esempio:
  .\scripts\avvia_detached.ps1 -Command '.venv\Scripts\python.exe -u -I generator\train_generator.py --run prova' -Log 'runs\prova\console.log'
#>
param(
    [Parameter(Mandatory = $true)][string]$Command,
    [Parameter(Mandatory = $true)][string]$Log,
    [string]$WorkDir = (Split-Path -Parent $PSScriptRoot)
)
$logFull = if ([System.IO.Path]::IsPathRooted($Log)) { $Log } else { Join-Path $WorkDir $Log }
New-Item -ItemType Directory -Force (Split-Path -Parent $logFull) | Out-Null
$line = "cmd.exe /d /c `"set PYTHONIOENCODING=utf-8&& $Command >> `"$logFull`" 2>&1`""
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $line; CurrentDirectory = $WorkDir; ProcessStartupInformation = $startup }
if ($r.ReturnValue -ne 0) { Write-Error "avvio fallito, codice WMI $($r.ReturnValue)"; exit 1 }
Write-Output "avviato: pid $($r.ProcessId) (cmd), log $logFull"
