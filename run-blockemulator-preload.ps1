# BlockEmulator EvolveGCN 预启动系统脚本
# 实现分片系统预热，减少EvolveGCN处理延迟

param(
    [switch]$Stop,
    [switch]$Status,
    [switch]$Warmup
)

$ExePath = if (Test-Path ".\blockEmulator_Windows_UTF8.exe") { 
    ".\blockEmulator_Windows_UTF8.exe" 
} elseif (Test-Path ".\blockEmulator.exe") {
    ".\blockEmulator.exe"
} else {
    ".\blockEmulator_Windows_Precompile.exe"
}

function Start-PythonWarmup {
    Write-Host "🔥 启动Python EvolveGCN预热..." -ForegroundColor Yellow
    
    $pythonExe = "E:\Codefield\BlockEmulator\.venv\Scripts\python.exe"
    if (-not (Test-Path $pythonExe)) {
        Write-Host "⚠️  Python虚拟环境未找到，跳过预热" -ForegroundColor Yellow
        return $false
    }
    
    if (-not (Test-Path ".\evolvegcn_preload_service.py")) {
        Write-Host "⚠️  预加载服务脚本未找到，跳过预热" -ForegroundColor Yellow
        return $false
    }
    
    try {
        # 创建虚拟输入文件用于预热
        $dummyInput = @{
            node_features = @(
                @{ node_id = "warmup_node_1"; features = @(1.0, 2.0, 3.0) }
                @{ node_id = "warmup_node_2"; features = @(4.0, 5.0, 6.0) }
            )
            edges = @(@("warmup_node_1", "warmup_node_2", 1.0))
        } | ConvertTo-Json -Depth 10
        
        $dummyInput | Out-File -FilePath "warmup_input.json" -Encoding UTF8
        
        # 执行预热
        $warmupStart = Get-Date
        $process = Start-Process -FilePath $pythonExe -ArgumentList @(
            "evolvegcn_preload_service.py", 
            "--input", "warmup_input.json", 
            "--output", "warmup_output.json", 
            "--warmup"
        ) -Wait -PassThru -NoNewWindow -RedirectStandardOutput "warmup_stdout.log" -RedirectStandardError "warmup_stderr.log"
        
        $warmupTime = ((Get-Date) - $warmupStart).TotalSeconds
        
        if ($process.ExitCode -eq 0) {
            Write-Host "✅ Python预热完成，耗时: $([math]::Round($warmupTime, 2))秒" -ForegroundColor Green
            
            # 清理临时文件
            Remove-Item "warmup_input.json" -ErrorAction SilentlyContinue
            Remove-Item "warmup_output.json" -ErrorAction SilentlyContinue
            Remove-Item "warmup_stdout.log" -ErrorAction SilentlyContinue
            Remove-Item "warmup_stderr.log" -ErrorAction SilentlyContinue
            
            return $true
        } else {
            Write-Host "❌ Python预热失败，退出码: $($process.ExitCode)" -ForegroundColor Red
            if (Test-Path "warmup_stderr.log") {
                $errorContent = Get-Content "warmup_stderr.log" -Raw
                Write-Host "错误详情: $errorContent" -ForegroundColor Red
            }
            return $false
        }
    } catch {
        Write-Host "❌ Python预热异常: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Start-BlockEmulatorSystemWithWarmup {
    Write-Host "🚀 启动BlockEmulator EvolveGCN预启动系统" -ForegroundColor Green
    Write-Host "=" * 70 -ForegroundColor Yellow
    
    # 检查可执行文件
    if (-not (Test-Path $ExePath)) {
        Write-Host "❌ 错误: 找不到 $ExePath" -ForegroundColor Red
        Write-Host "请先运行预编译脚本生成可执行文件" -ForegroundColor Yellow
        return
    }
    
    # 第一步：Python预热（并行进行）
    $warmupJob = Start-Job -ScriptBlock ${function:Start-PythonWarmup}
    
    $jobs = @()
    
    # 第二步：启动分片节点（并行）
    Write-Host "🔧 启动区块链节点..." -ForegroundColor Cyan
    
    # 启动分片0的4个节点
    Write-Host "  启动分片0节点..." -ForegroundColor Cyan
    for ($nodeId = 0; $nodeId -lt 4; $nodeId++) {
        $job = Start-Process -FilePath $ExePath -ArgumentList @("-n", $nodeId, "-N", "4", "-s", "0", "-S", "2") -PassThru
        $jobs += $job
        Write-Host "    节点 S0-N$nodeId 启动中... (PID: $($job.Id))" -ForegroundColor Gray
        Start-Sleep -Milliseconds 300  # 稍微缩短间隔
    }
    
    # 启动分片1的4个节点
    Write-Host "  启动分片1节点..." -ForegroundColor Cyan
    for ($nodeId = 0; $nodeId -lt 4; $nodeId++) {
        $job = Start-Process -FilePath $ExePath -ArgumentList @("-n", $nodeId, "-N", "4", "-s", "1", "-S", "2") -PassThru
        $jobs += $job
        Write-Host "    节点 S1-N$nodeId 启动中... (PID: $($job.Id))" -ForegroundColor Gray
        Start-Sleep -Milliseconds 300
    }
    
    # 第三步：等待Python预热完成
    Write-Host "⏳ 等待Python预热完成..." -ForegroundColor Yellow
    $warmupResult = Receive-Job -Job $warmupJob -Wait
    Remove-Job -Job $warmupJob
    
    if ($warmupResult) {
        Write-Host "✅ Python预热成功，继续启动Supervisor" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Python预热失败，但继续启动系统" -ForegroundColor Yellow
    }
    
    # 第四步：启动Supervisor
    Write-Host "🎯 启动Supervisor..." -ForegroundColor Cyan
    $supervisorJob = Start-Process -FilePath $ExePath -ArgumentList @("-c", "-N", "4", "-S", "2") -PassThru
    $jobs += $supervisorJob
    Write-Host "  Supervisor 启动中... (PID: $($supervisorJob.Id))" -ForegroundColor Gray
    
    # 保存进程ID
    $jobs | ForEach-Object { $_.Id } | Out-File -FilePath "blockemulator_pids.txt" -Encoding UTF8
    
    Write-Host "" -ForegroundColor Yellow
    Write-Host "🎉 预启动系统启动完成!" -ForegroundColor Green
    Write-Host "=" * 70 -ForegroundColor Yellow
    Write-Host "📊 系统状态:" -ForegroundColor Cyan
    Write-Host "  总共启动了 $($jobs.Count) 个进程" -ForegroundColor Green
    Write-Host "  Python服务预热: $(if ($warmupResult) { '✅ 完成' } else { '⚠️  跳过' })" -ForegroundColor $(if ($warmupResult) { 'Green' } else { 'Yellow' })
    Write-Host "  进程ID已保存到: blockemulator_pids.txt" -ForegroundColor Yellow
    Write-Host "" -ForegroundColor Yellow
    Write-Host "💡 预期效果:" -ForegroundColor Cyan
    Write-Host "  - EvolveGCN处理时间将显著缩短 (预期 < 1秒)" -ForegroundColor Green
    Write-Host "  - Python模型已预加载，无需重复初始化" -ForegroundColor Green
    Write-Host "  - 系统整体响应性能提升" -ForegroundColor Green
    Write-Host "" -ForegroundColor Yellow
    Write-Host "🎛️  管理命令:" -ForegroundColor Cyan
    Write-Host "  查看状态: .\run-blockemulator-preload.ps1 -Status" -ForegroundColor Gray
    Write-Host "  停止系统: .\run-blockemulator-preload.ps1 -Stop" -ForegroundColor Gray
    Write-Host "  仅预热测试: .\run-blockemulator-preload.ps1 -Warmup" -ForegroundColor Gray
}

function Stop-BlockEmulatorSystem {
    Write-Host "🛑 停止BlockEmulator系统..." -ForegroundColor Yellow
    
    if (Test-Path "blockemulator_pids.txt") {
        $pids = Get-Content "blockemulator_pids.txt"
        $stoppedCount = 0
        
        foreach ($processId in $pids) {
            try {
                $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
                if ($process) {
                    $process.Kill()
                    $stoppedCount++
                    Write-Host "  ✅ 停止进程 PID: $processId" -ForegroundColor Gray
                }
            }
            catch {
                Write-Host "  ℹ️  进程 PID: $processId 已停止或不存在" -ForegroundColor DarkGray
            }
        }
        
        Remove-Item "blockemulator_pids.txt" -ErrorAction SilentlyContinue
        Write-Host "✅ 系统已停止 (共停止 $stoppedCount 个进程)" -ForegroundColor Green
    }
    else {
        Write-Host "ℹ️  没有找到运行中的系统" -ForegroundColor Yellow
        
        # 尝试通过进程名停止
        $processes = Get-Process -Name "blockEmulator_Windows*" -ErrorAction SilentlyContinue
        if ($processes) {
            $processes | ForEach-Object { $_.Kill() }
            Write-Host "✅ 通过进程名停止了 $($processes.Count) 个进程" -ForegroundColor Green
        }
    }
}

function Show-SystemStatus {
    Write-Host "📊 BlockEmulator预启动系统状态:" -ForegroundColor Cyan
    Write-Host "=" * 50 -ForegroundColor Yellow
    
    if (Test-Path "blockemulator_pids.txt") {
        $pids = Get-Content "blockemulator_pids.txt"
        $runningCount = 0
        
        foreach ($processId in $pids) {
            $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
            if ($process) {
                $runningCount++
                Write-Host "  ✅ [运行中] PID: $processId - $($process.ProcessName)" -ForegroundColor Green
            }
            else {
                Write-Host "  ❌ [已停止] PID: $processId" -ForegroundColor Red
            }
        }
        
        Write-Host "" -ForegroundColor Yellow
        Write-Host "📈 总计: $runningCount/$($pids.Count) 个进程运行中" -ForegroundColor Cyan
        
        # 检查Python预热服务是否可用
        if (Test-Path ".\evolvegcn_preload_service.py") {
            Write-Host "🐍 Python预加载服务: ✅ 可用" -ForegroundColor Green
        } else {
            Write-Host "🐍 Python预加载服务: ❌ 不可用" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "ℹ️  没有找到系统进程记录" -ForegroundColor Yellow
        
        # 检查是否有相关进程在运行
        $processes = Get-Process -Name "blockEmulator_Windows*" -ErrorAction SilentlyContinue
        if ($processes) {
            Write-Host "📍 发现 $($processes.Count) 个相关进程:" -ForegroundColor Yellow
            $processes | ForEach-Object {
                Write-Host "  PID: $($_.Id) - $($_.ProcessName)" -ForegroundColor Gray
            }
        }
        else {
            Write-Host "ℹ️  没有相关进程在运行" -ForegroundColor Gray
        }
    }
}

function Test-WarmupOnly {
    Write-Host "🧪 执行Python预热测试..." -ForegroundColor Cyan
    Write-Host "=" * 40 -ForegroundColor Yellow
    
    $result = Start-PythonWarmup
    
    if ($result) {
        Write-Host "🎉 预热测试成功!" -ForegroundColor Green
        Write-Host "💡 预启动系统已准备就绪" -ForegroundColor Cyan
    } else {
        Write-Host "❌ 预热测试失败" -ForegroundColor Red
        Write-Host "🔧 请检查Python环境和依赖" -ForegroundColor Yellow
    }
}

# 主逻辑
Write-Host "BlockEmulator EvolveGCN 预启动系统 v2.0" -ForegroundColor Magenta
Write-Host "=" * 50 -ForegroundColor Magenta

if ($Stop) {
    Stop-BlockEmulatorSystem
}
elseif ($Status) {
    Show-SystemStatus
}
elseif ($Warmup) {
    Test-WarmupOnly
}
else {
    Start-BlockEmulatorSystemWithWarmup
}
