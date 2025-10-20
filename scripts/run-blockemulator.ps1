# BlockEmulator 2分片4节点系统启动脚本 (PowerShell版本)
# 等效于 windows_exe_run_IpAddr=127_0_0_1.bat

param(
    [switch]$Stop,
    [switch]$Status
)

$ExePath = if (Test-Path ".\blockEmulator_Windows_UTF8.exe") { 
    ".\blockEmulator_Windows_UTF8.exe" 
} elseif (Test-Path ".\blockEmulator.exe") {
    ".\blockEmulator.exe"
} else {
    ".\blockEmulator_Windows_Precompile.exe"
}

function Start-BlockEmulatorSystem {
    Write-Host "启动BlockEmulator EvolveGCN系统 (2分片, 每分片4节点)" -ForegroundColor Green
    Write-Host "=" * 60 -ForegroundColor Yellow
    
    # 检查可执行文件是否存在
    if (-not (Test-Path $ExePath)) {
        Write-Host "错误: 找不到 $ExePath" -ForegroundColor Red
        Write-Host "请先运行预编译脚本生成可执行文件" -ForegroundColor Yellow
        return
    }
    
    $jobs = @()
    
    # 启动分片0的4个节点
    Write-Host "启动分片0节点..." -ForegroundColor Cyan
    for ($nodeId = 0; $nodeId -lt 4; $nodeId++) {
        $job = Start-Process -FilePath $ExePath -ArgumentList @("-n", $nodeId, "-N", "4", "-s", "0", "-S", "2") -PassThru
        $jobs += $job
        Write-Host "  节点 S0-N$nodeId 启动中... (PID: $($job.Id))" -ForegroundColor Gray
        Start-Sleep -Milliseconds 500
    }
    
    # 启动分片1的4个节点  
    Write-Host "启动分片1节点..." -ForegroundColor Cyan
    for ($nodeId = 0; $nodeId -lt 4; $nodeId++) {
        $job = Start-Process -FilePath $ExePath -ArgumentList @("-n", $nodeId, "-N", "4", "-s", "1", "-S", "2") -PassThru
        $jobs += $job
        Write-Host "  节点 S1-N$nodeId 启动中... (PID: $($job.Id))" -ForegroundColor Gray
        Start-Sleep -Milliseconds 500
    }
    
    # 启动Supervisor
    Write-Host "启动Supervisor..." -ForegroundColor Cyan
    $supervisorJob = Start-Process -FilePath $ExePath -ArgumentList @("-c", "-N", "4", "-S", "2") -PassThru
    $jobs += $supervisorJob
    Write-Host "  Supervisor 启动中... (PID: $($supervisorJob.Id))" -ForegroundColor Gray
    
    # 保存进程ID到文件
    $jobs | ForEach-Object { $_.Id } | Out-File -FilePath "blockemulator_pids.txt" -Encoding UTF8
    
    Write-Host "" -ForegroundColor Yellow
    Write-Host "系统启动完成!" -ForegroundColor Green
    Write-Host "总共启动了 $($jobs.Count) 个进程" -ForegroundColor Green
    Write-Host "进程ID已保存到: blockemulator_pids.txt" -ForegroundColor Yellow
    Write-Host "" -ForegroundColor Yellow
    Write-Host "管理命令:" -ForegroundColor Cyan
    Write-Host "  查看状态: .\run-blockemulator.ps1 -Status" -ForegroundColor Gray
    Write-Host "  停止系统: .\run-blockemulator.ps1 -Stop" -ForegroundColor Gray
}

function Stop-BlockEmulatorSystem {
    Write-Host "停止BlockEmulator系统..." -ForegroundColor Yellow
    
    if (Test-Path "blockemulator_pids.txt") {
        $pids = Get-Content "blockemulator_pids.txt"
        $stoppedCount = 0
        
        foreach ($processId in $pids) {
            try {
                $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
                if ($process) {
                    $process.Kill()
                    $stoppedCount++
                    Write-Host "  停止进程 PID: $processId" -ForegroundColor Gray
                }
            }
            catch {
                Write-Host "  进程 PID: $processId 已停止或不存在" -ForegroundColor DarkGray
            }
        }
        
        Remove-Item "blockemulator_pids.txt" -ErrorAction SilentlyContinue
        Write-Host "系统已停止 (共停止 $stoppedCount 个进程)" -ForegroundColor Green
    }
    else {
        Write-Host "没有找到运行中的系统" -ForegroundColor Yellow
        
        # 尝试通过进程名停止
        $processes = Get-Process -Name "blockEmulator_Windows_Precompile" -ErrorAction SilentlyContinue
        if ($processes) {
            $processes | ForEach-Object { $_.Kill() }
            Write-Host "通过进程名停止了 $($processes.Count) 个进程" -ForegroundColor Green
        }
    }
}

function Show-SystemStatus {
    Write-Host "BlockEmulator系统状态:" -ForegroundColor Cyan
    Write-Host "=" * 40 -ForegroundColor Yellow
    
    if (Test-Path "blockemulator_pids.txt") {
        $pids = Get-Content "blockemulator_pids.txt"
        $runningCount = 0
        
        foreach ($processId in $pids) {
            $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
            if ($process) {
                $runningCount++
                Write-Host "  [运行中] PID: $processId - $($process.ProcessName)" -ForegroundColor Green
            }
            else {
                Write-Host "  [已停止] PID: $processId" -ForegroundColor Red
            }
        }
        
        Write-Host "" -ForegroundColor Yellow
        Write-Host "总计: $runningCount/$($pids.Count) 个进程运行中" -ForegroundColor Cyan
    }
    else {
        Write-Host "没有找到系统进程记录" -ForegroundColor Yellow
        
        # 检查是否有相关进程在运行
        $processes = Get-Process -Name "blockEmulator_Windows_Precompile" -ErrorAction SilentlyContinue
        if ($processes) {
            Write-Host "发现 $($processes.Count) 个相关进程:" -ForegroundColor Yellow
            $processes | ForEach-Object {
                Write-Host "  PID: $($_.Id) - $($_.ProcessName)" -ForegroundColor Gray
            }
        }
        else {
            Write-Host "没有相关进程在运行" -ForegroundColor Gray
        }
    }
}

# 主逻辑
if ($Stop) {
    Stop-BlockEmulatorSystem
}
elseif ($Status) {
    Show-SystemStatus
}
else {
    Start-BlockEmulatorSystem
}
