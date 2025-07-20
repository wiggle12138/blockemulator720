# BlockEmulator EvolveGCN 预启动系统脚本
# 实现分片系统预热，减少EvolveGCN处理延迟
# 编码兼容版本 - 修复预热失败问题

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

# 使用用户指定的Python环境路径
$PythonExePath = "E:\Codefield\BlockEmulator\.venv\Scripts\python.exe"

function Start-PythonWarmup {
    Write-Host "[WARMUP] 启动Python EvolveGCN预热..." -ForegroundColor Yellow
    
    # 检查用户指定的Python环境
    if (-not (Test-Path $PythonExePath)) {
        Write-Host "[ERROR] 指定的Python环境未找到: $PythonExePath" -ForegroundColor Red
        Write-Host "[INFO] 请确认Python环境路径是否正确" -ForegroundColor Yellow
        throw "Python环境路径无效"
    }
    
    if (-not (Test-Path ".\evolvegcn_preload_service_safe.py")) {
        Write-Host "[ERROR] 预加载服务脚本未找到" -ForegroundColor Red
        throw "缺少必要的预加载服务脚本"
    }
    
    try {
        # 创建简化的虚拟输入文件用于预热（避免特殊字符）
        $dummyInputData = @"
{
  "node_features": [
    {
      "node_id": "warmup_node_1",
      "features": [1.0, 2.0, 3.0]
    },
    {
      "node_id": "warmup_node_2", 
      "features": [4.0, 5.0, 6.0]
    }
  ],
  "transaction_graph": {
    "edges": [["warmup_node_1", "warmup_node_2", 1.0]],
    "metadata": {
      "timestamp": "2025-07-20T15:30:00Z",
      "graph_type": "warmup"
    }
  },
  "config": {
    "target_shards": 2,
    "algorithm": "EvolveGCN"
  }
}
"@
        
        # 使用UTF-8编码写入文件
        $dummyInputData | Out-File -FilePath "warmup_input.json" -Encoding UTF8
        
        # 执行预热
        Write-Host "[INFO] 使用Python环境: $PythonExePath" -ForegroundColor Cyan
        $warmupStart = Get-Date
        
        $process = Start-Process -FilePath $PythonExePath -ArgumentList @(
            "evolvegcn_preload_service_safe.py", 
            "--input", "warmup_input.json", 
            "--output", "warmup_output.json", 
            "--warmup"
        ) -Wait -PassThru -NoNewWindow -RedirectStandardOutput "warmup_stdout.log" -RedirectStandardError "warmup_stderr.log"
        
        $warmupTime = ((Get-Date) - $warmupStart).TotalSeconds
        
        if ($process.ExitCode -eq 0) {
            Write-Host "[SUCCESS] Python预热完成，耗时: $([math]::Round($warmupTime, 2))秒" -ForegroundColor Green
            
            # 显示预热输出信息
            if (Test-Path "warmup_stdout.log") {
                $stdout = Get-Content "warmup_stdout.log" -Raw -Encoding UTF8
                if ($stdout -and $stdout.Trim() -ne "") {
                    Write-Host "[OUTPUT] $stdout" -ForegroundColor DarkGreen
                }
            }
            
            # 清理临时文件
            Remove-Item "warmup_input.json" -ErrorAction SilentlyContinue
            Remove-Item "warmup_output.json" -ErrorAction SilentlyContinue
            Remove-Item "warmup_stdout.log" -ErrorAction SilentlyContinue
            Remove-Item "warmup_stderr.log" -ErrorAction SilentlyContinue
            
            return $true
        } else {
            Write-Host "[ERROR] Python预热失败，退出码: $($process.ExitCode)" -ForegroundColor Red
            if (Test-Path "warmup_stderr.log") {
                $errorContent = Get-Content "warmup_stderr.log" -Raw -Encoding UTF8
                Write-Host "[ERROR] 错误详情: $errorContent" -ForegroundColor Red
            }
            throw "Python预热执行失败"
        }
    } catch {
        Write-Host "[ERROR] Python预热异常: $($_.Exception.Message)" -ForegroundColor Red
        throw $_.Exception
    }
}

function Update-PythonConfig {
    Write-Host "[CONFIG] 更新Python配置文件..." -ForegroundColor Cyan
    
    try {
        $configPath = "python_config.json"
        
        # 使用简化的JSON构建方式避免特殊字符问题
        $configJson = @"
{
  "enable_evolve_gcn": true,
  "enable_feedback": true,
  "python_path": "$($PythonExePath.Replace('\', '\\'))",
  "module_path": "./",
  "max_iterations": 10,
  "epochs_per_iteration": 8,
  "data_exchange_dir": "./data_exchange",
  "output_interval": 30,
  "continuous_mode": true,
  "log_level": "INFO",
  "evolvegcn_integration": {
    "enabled": true,
    "algorithm": "four_step_pipeline",
    "fallback_to_clpa": false,
    "auto_detect_venv": false,
    "description": "EvolveGCN integrated with specified Python environment"
  }
}
"@
        
        # 使用UTF-8编码写入配置文件
        $configJson | Out-File -FilePath $configPath -Encoding UTF8
        
        Write-Host "[SUCCESS] Python配置已更新: $configPath" -ForegroundColor Green
        Write-Host "[INFO] 已设置Python路径: $PythonExePath" -ForegroundColor Cyan
        
    } catch {
        Write-Host "[ERROR] 更新配置失败: $($_.Exception.Message)" -ForegroundColor Red
        throw $_.Exception
    }
}

function Start-BlockEmulatorSystemWithWarmup {
    Write-Host "[START] 启动BlockEmulator EvolveGCN预启动系统" -ForegroundColor Green
    Write-Host "=" * 60 -ForegroundColor Yellow
    
    # 检查可执行文件是否存在
    if (-not (Test-Path $ExePath)) {
        Write-Host "[ERROR] 找不到 $ExePath" -ForegroundColor Red
        Write-Host "[INFO] 请先运行预编译脚本生成可执行文件" -ForegroundColor Yellow
        throw "可执行文件不存在"
    }
    
    try {
        # 步骤1：更新Python配置
        Write-Host "[STEP1] 更新Python配置..." -ForegroundColor Cyan
        Update-PythonConfig
        
        # 步骤2：Python预热（必须成功才能继续）
        Write-Host "[STEP2] 执行Python预热..." -ForegroundColor Cyan
        $warmupResult = Start-PythonWarmup
        
        Write-Host "[SUCCESS] 预热验证通过，开始启动系统..." -ForegroundColor Green
        
    } catch {
        Write-Host "[CRITICAL] 预热失败，无法启动系统: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "[INFO] 请检查Python环境配置后重试" -ForegroundColor Yellow
        return
    }
    
    $jobs = @()
    
    # 步骤3：启动分片0的4个节点
    Write-Host "[STEP3] 启动分片0节点..." -ForegroundColor Cyan
    for ($nodeId = 0; $nodeId -lt 4; $nodeId++) {
        $job = Start-Process -FilePath $ExePath -ArgumentList @("-n", $nodeId, "-N", "4", "-s", "0", "-S", "2") -PassThru
        $jobs += $job
        Write-Host "  [NODE] 节点 S0-N$nodeId 启动中... (PID: $($job.Id))" -ForegroundColor Gray
        Start-Sleep -Milliseconds 500
    }
    
    # 步骤4：启动分片1的4个节点  
    Write-Host "[STEP4] 启动分片1节点..." -ForegroundColor Cyan
    for ($nodeId = 0; $nodeId -lt 4; $nodeId++) {
        $job = Start-Process -FilePath $ExePath -ArgumentList @("-n", $nodeId, "-N", "4", "-s", "1", "-S", "2") -PassThru
        $jobs += $job
        Write-Host "  [NODE] 节点 S1-N$nodeId 启动中... (PID: $($job.Id))" -ForegroundColor Gray
        Start-Sleep -Milliseconds 500
    }
    
    # 步骤5：启动Supervisor
    Write-Host "[STEP5] 启动Supervisor..." -ForegroundColor Cyan
    $supervisorJob = Start-Process -FilePath $ExePath -ArgumentList @("-c", "-N", "4", "-S", "2") -PassThru
    $jobs += $supervisorJob
    Write-Host "  [SUPERVISOR] Supervisor 启动中... (PID: $($supervisorJob.Id))" -ForegroundColor Gray
    
    # 保存进程ID到文件
    $jobs | ForEach-Object { $_.Id } | Out-File -FilePath "blockemulator_pids.txt" -Encoding UTF8
    
    Write-Host "" -ForegroundColor Yellow
    Write-Host "[SUCCESS] 预启动系统启动完成!" -ForegroundColor Green
    Write-Host "[INFO] 总共启动了 $($jobs.Count) 个进程" -ForegroundColor Green
    Write-Host "[INFO] Python环境: $PythonExePath" -ForegroundColor Cyan
    Write-Host "[INFO] Python服务预热: [COMPLETED]" -ForegroundColor Green
    Write-Host "[INFO] 进程ID已保存到: blockemulator_pids.txt" -ForegroundColor Yellow
    Write-Host "" -ForegroundColor Yellow
    Write-Host "[BENEFIT] 预期效果:" -ForegroundColor Cyan
    Write-Host "  - EvolveGCN处理时间将显著缩短 (预期 < 1秒)" -ForegroundColor Green
    Write-Host "  - Python模型已预加载，无需重复初始化" -ForegroundColor Green
    Write-Host "  - 系统整体响应性能提升" -ForegroundColor Green
    Write-Host "" -ForegroundColor Yellow
    Write-Host "[COMMANDS] 管理命令:" -ForegroundColor Cyan
    Write-Host "  查看状态: .\run-blockemulator-preload-safe.ps1 -Status" -ForegroundColor Gray
    Write-Host "  停止系统: .\run-blockemulator-preload-safe.ps1 -Stop" -ForegroundColor Gray
    Write-Host "  仅预热测试: .\run-blockemulator-preload-safe.ps1 -Warmup" -ForegroundColor Gray
}

function Stop-BlockEmulatorSystem {
    Write-Host "[STOP] 停止BlockEmulator系统..." -ForegroundColor Yellow
    
    if (Test-Path "blockemulator_pids.txt") {
        $pids = Get-Content "blockemulator_pids.txt"
        $stoppedCount = 0
        
        foreach ($processId in $pids) {
            try {
                $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
                if ($process) {
                    $process.Kill()
                    $stoppedCount++
                    Write-Host "  [SUCCESS] 停止进程 PID: $processId" -ForegroundColor Gray
                }
            }
            catch {
                Write-Host "  [INFO] 进程 PID: $processId 已停止或不存在" -ForegroundColor DarkGray
            }
        }
        
        Remove-Item "blockemulator_pids.txt" -ErrorAction SilentlyContinue
        Write-Host "[SUCCESS] 系统已停止 (共停止 $stoppedCount 个进程)" -ForegroundColor Green
    }
    else {
        Write-Host "[INFO] 没有找到运行中的系统" -ForegroundColor Yellow
        
        # 尝试通过进程名停止
        $processes = Get-Process -Name "blockEmulator_Windows*" -ErrorAction SilentlyContinue
        if ($processes) {
            $processes | ForEach-Object { $_.Kill() }
            Write-Host "[SUCCESS] 通过进程名停止了 $($processes.Count) 个进程" -ForegroundColor Green
        }
    }
}

function Show-SystemStatus {
    Write-Host "[STATUS] BlockEmulator系统状态:" -ForegroundColor Cyan
    Write-Host "=" * 40 -ForegroundColor Yellow
    
    if (Test-Path "blockemulator_pids.txt") {
        $pids = Get-Content "blockemulator_pids.txt"
        $runningCount = 0
        
        foreach ($processId in $pids) {
            $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
            if ($process) {
                $runningCount++
                Write-Host "  [RUNNING] PID: $processId - $($process.ProcessName)" -ForegroundColor Green
            }
            else {
                Write-Host "  [STOPPED] PID: $processId" -ForegroundColor Red
            }
        }
        
        Write-Host "" -ForegroundColor Yellow
        Write-Host "[SUMMARY] 总计: $runningCount/$($pids.Count) 个进程运行中" -ForegroundColor Cyan
        
        # 检查Python预热服务是否可用
        if (Test-Path ".\evolvegcn_preload_service_safe.py") {
            Write-Host "[SERVICE] Python预加载服务: [AVAILABLE]" -ForegroundColor Green
        } else {
            Write-Host "[SERVICE] Python预加载服务: [UNAVAILABLE]" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "[INFO] 没有找到系统进程记录" -ForegroundColor Yellow
        
        # 检查是否有相关进程在运行
        $processes = Get-Process -Name "blockEmulator_Windows*" -ErrorAction SilentlyContinue
        if ($processes) {
            Write-Host "[FOUND] 发现 $($processes.Count) 个相关进程:" -ForegroundColor Yellow
            $processes | ForEach-Object {
                Write-Host "  PID: $($_.Id) - $($_.ProcessName)" -ForegroundColor Gray
            }
        }
        else {
            Write-Host "[INFO] 没有相关进程在运行" -ForegroundColor Gray
        }
    }
}

function Test-WarmupOnly {
    Write-Host "[TEST] 执行Python预热测试..." -ForegroundColor Cyan
    Write-Host "=" * 40 -ForegroundColor Yellow
    
    $result = Start-PythonWarmup
    
    if ($result) {
        Write-Host "[SUCCESS] 预热测试成功!" -ForegroundColor Green
        Write-Host "[INFO] 预启动系统已准备就绪" -ForegroundColor Cyan
    } else {
        Write-Host "[ERROR] 预热测试失败" -ForegroundColor Red
        Write-Host "[INFO] 请检查Python环境和依赖" -ForegroundColor Yellow
    }
}

# 主逻辑
Write-Host "BlockEmulator EvolveGCN 预启动系统 v2.0 (编码安全版)" -ForegroundColor Magenta
Write-Host "=" * 60 -ForegroundColor Magenta

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
