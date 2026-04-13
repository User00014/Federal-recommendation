$ErrorActionPreference = 'Stop'

$projectRoot = 'C:\Users\13488\Desktop\本科毕业项目\Federated_Privacy_Project'
$sshKey = Join-Path $projectRoot '毕设实验.pem'
$sshExe = 'C:\Windows\System32\OpenSSH\ssh.exe'
$scpExe = 'C:\Windows\System32\OpenSSH\scp.exe'
$remoteHost = 'root@8.146.232.113'
$remoteRoot = '/root/Federated_Privacy_Project'
$mainPid = 4185

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$localOutDir = Join-Path $projectRoot "cloud_results\final_pull_$stamp"
$logFile = Join-Path $projectRoot "cloud_results\fetch_then_shutdown_$stamp.log"

New-Item -ItemType Directory -Force -Path $localOutDir | Out-Null

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Add-Content -Path $logFile -Value $line -Encoding UTF8
}

function Invoke-Ssh {
    param([string]$Command)
    & $sshExe -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $sshKey $remoteHost $Command
}

try {
    Write-Log "Watcher started. Waiting for remote PID $mainPid to finish."

    while ($true) {
        $state = Invoke-Ssh "if kill -0 $mainPid 2>/dev/null; then echo RUNNING; else echo DONE; fi" 2>$null
        if ($state -match 'DONE') {
            break
        }
        Start-Sleep -Seconds 60
    }

    Write-Log "Main batch finished. Pulling results to $localOutDir"

    & $scpExe -r -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $sshKey `
        "${remoteHost}:${remoteRoot}/logs/full_15_seed42_v2" $localOutDir | Out-Null

    & $scpExe -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $sshKey `
        "${remoteHost}:${remoteRoot}/logs/nohup_full_15_seed42_v2.out" $localOutDir | Out-Null

    Write-Log "Result pull finished. Sending shutdown."
    Invoke-Ssh "sync; /sbin/shutdown -h now" | Out-Null
    Write-Log "Shutdown command sent."
}
catch {
    Write-Log ("FAILED: " + $_.Exception.Message)
    throw
}
