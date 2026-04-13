param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteHost,

    [Parameter(Mandatory = $true)]
    [string]$User,

    [Parameter(Mandatory = $true)]
    [string]$RemotePath,

    [string]$KeyPath = "",

    [string]$LocalRoot = ".\\cloud_results",

    [switch]$Watch,

    [int]$IntervalSec = 300
)

$ErrorActionPreference = "Stop"

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$targetDir = Join-Path $LocalRoot ("batch_" + $stamp)
New-Item -ItemType Directory -Path $targetDir -Force | Out-Null

Write-Host "[INFO] 本地接收目录: $targetDir"
Write-Host "[INFO] 远端目录: ${User}@${RemoteHost}:$RemotePath"
if ($KeyPath) {
    Write-Host "[INFO] 使用密钥: $KeyPath"
}

function Invoke-Scp {
    param(
        [string]$Source,
        [string]$Target,
        [string]$Pem
    )

    if ($Pem) {
        scp -i "$Pem" -o StrictHostKeyChecking=accept-new -r "$Source" "$Target" | Out-Null
    } else {
        scp -o StrictHostKeyChecking=accept-new -r "$Source" "$Target" | Out-Null
    }
}

function Sync-Once {
    param(
        [string]$H,
        [string]$U,
        [string]$R,
        [string]$T,
        [string]$Pem
    )

    $items = @(
        "batch_state.json",
        "batch_summary.csv",
        "logs"
    )

    foreach ($item in $items) {
        $remoteItem = "${U}@${H}:$R/$item"
        Invoke-Scp -Source $remoteItem -Target $T -Pem $Pem
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "本次同步中有文件拉取失败（通常是远端暂未生成），稍后会继续重试。"
    }

    $statePath = Join-Path $T "batch_state.json"
    if (Test-Path $statePath) {
        try {
            $state = Get-Content -Path $statePath -Raw | ConvertFrom-Json
            $runs = @($state.runs)
            $done = ($runs | Where-Object { $_.status -eq 'done' }).Count
            $running = ($runs | Where-Object { $_.status -eq 'running' }).Count
            $pending = ($runs | Where-Object { $_.status -eq 'pending' }).Count
            $failed = ($runs | Where-Object { $_.status -eq 'failed' }).Count
            Write-Host "[STATE] done=$done running=$running pending=$pending failed=$failed total=$($runs.Count)"

            if ($pending -eq 0 -and $running -eq 0) {
                return $true
            }
        } catch {
            Write-Warning "状态文件解析失败，将继续同步。"
        }
    }

    return $false
}

if (-not $Watch) {
    Write-Host "[INFO] 执行一次性拉取..."
    [void](Sync-Once -H $RemoteHost -U $User -R $RemotePath -T $targetDir -Pem $KeyPath)
    Write-Host "[OK] 拉取完成: $targetDir"
    exit 0
}

Write-Host "[INFO] 进入持续同步模式，每 $IntervalSec 秒拉取一次。"
while ($true) {
    $finished = Sync-Once -H $RemoteHost -U $User -R $RemotePath -T $targetDir -Pem $KeyPath
    if ($finished) {
        Write-Host "[OK] 检测到批量实验已结束，停止同步。"
        break
    }
    Start-Sleep -Seconds $IntervalSec
}

Write-Host "[OK] 最终结果目录: $targetDir"
