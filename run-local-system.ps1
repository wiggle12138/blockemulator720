#Requires -Version 5.1
<#
.SYNOPSIS
    BlockEmulator EvolveGCN本机集成运行脚本
.DESCRIPTION
    在本机运行完整的BlockEmulator系统，模拟Docker环境的部署
    支持supervisor和多个分片节点的并发运行
.PARAMETER Action
    执行的操作：start, stop, status, logs, clean, help
.PARAMETER Shards
    分片数量，默认4个分片
.PARAMETER NodesPerShard
    每个分片的节点数，默认4个节点
.EXAMPLE
    .\run-local-system.ps1 start
    .\run-local-system.ps1 stop
    .\run-local-system.ps1 status
#>

param(
    [Parameter(Position=0)]
    [ValidateSet("start", "stop", "status", "logs", "clean", "help")]
    [string]$Action = "start",
    
    [Parameter()]
    [int]$Shards = 4,
    
    [Parameter()]
    [int]$NodesPerShard = 4
)

# 脚本配置
$ScriptName = "BlockEmulator本机运行系统"
$LogDir = "local_logs"
$PidFile = "local_system.pids"

# 颜色输出函数
function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    switch ($Color) {
        "Red" { Write-Host $Message -ForegroundColor Red }
        "Green" { Write-Host $Message -ForegroundColor Green }
        "Yellow" { Write-Host $Message -ForegroundColor Yellow }
        "Blue" { Write-Host $Message -ForegroundColor Blue }
        "Cyan" { Write-Host $Message -ForegroundColor Cyan }
        default { Write-Host $Message }
    }
}

function Write-Banner {
    Write-ColorOutput "===============================================" "Cyan"
    Write-ColorOutput "        $ScriptName" "Cyan"
    Write-ColorOutput "===============================================" "Cyan"
    Write-Host ""
}

function Test-Prerequisites {
    Write-ColorOutput "[检查] 系统环境..." "Blue"
    
    # 检查Go环境
    try {
        $goVersion = go version 2>$null
        if (-not $goVersion) {
            throw "Go未安装"
        }
        Write-ColorOutput "✅ Go环境: $goVersion" "Green"
    }
    catch {
        Write-ColorOutput "❌ Go环境检查失败: $_" "Red"
        return $false
    }
    
    # 检查Python环境
    try {
        $pythonVersion = python --version 2>$null
        if (-not $pythonVersion) {
            throw "Python未安装"
        }
        Write-ColorOutput "✅ Python环境: $pythonVersion" "Green"
    }
    catch {
        Write-ColorOutput "❌ Python环境检查失败: $_" "Red"
        return $false
    }
    
    # 检查项目文件
    $requiredFiles = @(
        "main.go",
        "paramsConfig.json",
        "ipTable.json",
        "selectedTxs_300K.csv"
    )
    
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path $file)) {
            Write-ColorOutput "❌ 缺少必要文件: $file" "Red"
            return $false
        }
    }
    Write-ColorOutput "✅ 项目文件检查通过" "Green"
    
    # 配置Python环境
    Write-ColorOutput "[配置] Python依赖检查..." "Blue"
    try {
        # 检查关键依赖
        $dependencies = @("torch", "numpy", "pandas", "networkx", "scikit-learn")
        foreach ($dep in $dependencies) {
            $result = python -c "import $dep" 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput "  ✅ $dep 已安装" "Green"
            } else {
                Write-ColorOutput "  ⚠️ 正在安装 $dep..." "Yellow"
                pip install $dep --quiet --no-warn-script-location
                if ($LASTEXITCODE -eq 0) {
                    Write-ColorOutput "  ✅ $dep 安装完成" "Green"
                } else {
                    Write-ColorOutput "  ❌ $dep 安装失败" "Red"
                }
            }
        }
        Write-ColorOutput "✅ Python环境配置完成" "Green"
    }
    catch {
        Write-ColorOutput "⚠️ Python依赖检查跳过: $_" "Yellow"
    }
    
    return $true
}

function Initialize-Environment {
    # 创建日志目录
    if (-not (Test-Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
        Write-ColorOutput "✅ 创建日志目录: $LogDir" "Green"
    }
    
    # 创建输出目录
    $outputDirs = @("expTest", "data_exchange", "outputs")
    foreach ($dir in $outputDirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-ColorOutput "✅ 创建输出目录: $dir" "Green"
        }
    }
}

function Start-Supervisor {
    Write-ColorOutput "[启动] Supervisor节点..." "Blue"
    
    $logFile = Join-Path $LogDir "supervisor.log"
    # supervisor模式：-c表示coordinator角色，-N节点数，-S分片数
    $supervisorArgs = @("-c", "-N", $NodesPerShard, "-S", $Shards)
    
    $job = Start-Job -ScriptBlock {
        param($Args, $LogFile)
        go run main.go @Args *>&1 | Tee-Object -FilePath $LogFile
    } -ArgumentList $supervisorArgs, $logFile
    
    if ($job) {
        Add-Content $PidFile "supervisor:$($job.Id)"
        Write-ColorOutput "✅ Supervisor已启动 (JobID: $($job.Id))" "Green"
        return $job.Id
    }
    else {
        Write-ColorOutput "❌ Supervisor启动失败" "Red"
        return $null
    }
}

function Start-ShardNodes {
    Write-ColorOutput "[启动] 分片节点..." "Blue"
    
    $nodeJobs = @()
    
    for ($shardId = 0; $shardId -lt $Shards; $shardId++) {
        for ($nodeId = 0; $nodeId -lt $NodesPerShard; $nodeId++) {
            $logFile = Join-Path $LogDir "shard$shardId-node$nodeId.log"
            $nodeArgs = @(
                "-n", $nodeId,
                "-N", $NodesPerShard,
                "-s", $shardId,
                "-S", $Shards,
                "-m", "4"  # EvolveGCN consensus method
            )
            
            $job = Start-Job -ScriptBlock {
                param($Args, $LogFile, $NodeName)
                Write-Host "[$NodeName] 节点启动中..."
                go run main.go @Args *>&1 | Tee-Object -FilePath $LogFile
            } -ArgumentList $nodeArgs, $logFile, "S$shardId-N$nodeId"
            
            if ($job) {
                $nodeJobs += $job.Id
                Add-Content $PidFile "shard$shardId-node${nodeId}:$($job.Id)"
                Write-ColorOutput "✅ 节点 S$shardId-N$nodeId 已启动 (JobID: $($job.Id))" "Green"
                
                # 节点间启动延迟
                Start-Sleep -Milliseconds 500
            }
            else {
                Write-ColorOutput "❌ 节点 S$shardId-N$nodeId 启动失败" "Red"
            }
        }
    }
    
    return $nodeJobs
}

function Start-System {
    Write-Banner
    
    if (-not (Test-Prerequisites)) {
        Write-ColorOutput "❌ 环境检查失败，无法启动系统" "Red"
        return
    }
    
    Initialize-Environment
    
    # 清理之前的PID文件
    if (Test-Path $PidFile) {
        Remove-Item $PidFile -Force
    }
    
    Write-ColorOutput "" ""
    Write-ColorOutput "[启动] BlockEmulator EvolveGCN集成系统" "Cyan"
    Write-ColorOutput "配置: $Shards 个分片，每分片 $NodesPerShard 个节点" "Blue"
    Write-ColorOutput "" ""
    
    # 启动Supervisor
    $supervisorId = Start-Supervisor
    if (-not $supervisorId) {
        Write-ColorOutput "❌ Supervisor启动失败，终止系统启动" "Red"
        return
    }
    
    # 等待Supervisor初始化
    Write-ColorOutput "[等待] Supervisor初始化..." "Blue"
    Start-Sleep -Seconds 3
    
    # 启动分片节点
    $nodeJobs = Start-ShardNodes
    
    Write-ColorOutput "" ""
    Write-ColorOutput "===============================================" "Green"
    Write-ColorOutput "🎉 系统启动完成！" "Green"
    Write-ColorOutput "===============================================" "Green"
    Write-ColorOutput "" ""
    Write-ColorOutput "📊 系统状态:" "Blue"
    Write-ColorOutput "  - Supervisor: 运行中" "Green"
    Write-ColorOutput "  - 分片节点: $($nodeJobs.Count) 个运行中" "Green"
    Write-ColorOutput "  - 日志目录: $LogDir" "Blue"
    Write-ColorOutput "" ""
    Write-ColorOutput "🔧 管理命令:" "Blue"
    Write-ColorOutput "  - 查看状态: .\run-local-system.ps1 status" "Yellow"
    Write-ColorOutput "  - 查看日志: .\run-local-system.ps1 logs" "Yellow"
    Write-ColorOutput "  - 停止系统: .\run-local-system.ps1 stop" "Yellow"
    Write-ColorOutput "" ""
    Write-ColorOutput "🚀 EvolveGCN分片算法已集成，请监控日志查看分片重配置过程" "Cyan"
}

function Stop-System {
    Write-Banner
    Write-ColorOutput "[停止] BlockEmulator系统..." "Yellow"
    
    if (-not (Test-Path $PidFile)) {
        Write-ColorOutput "⚠️ 未找到运行的系统进程" "Yellow"
        return
    }
    
    $stopped = 0
    $content = Get-Content $PidFile -ErrorAction SilentlyContinue
    
    foreach ($line in $content) {
        if ($line -match "^([^:]+):(\d+)$") {
            $nodeName = $matches[1]
            $jobId = $matches[2]
            
            try {
                $job = Get-Job -Id $jobId -ErrorAction SilentlyContinue
                if ($job) {
                    Stop-Job -Job $job -Force
                    Remove-Job -Job $job -Force
                    Write-ColorOutput "✅ 已停止: $nodeName (JobID: $jobId)" "Green"
                    $stopped++
                }
            }
            catch {
                Write-ColorOutput "⚠️ 清理作业失败: $nodeName" "Yellow"
            }
        }
    }
    
    # 清理PID文件
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    
    Write-ColorOutput "" ""
    Write-ColorOutput "===============================================" "Green"
    Write-ColorOutput "🛑 系统已停止 (共停止 $stopped 个进程)" "Green"
    Write-ColorOutput "===============================================" "Green"
}

function Show-Status {
    Write-Banner
    Write-ColorOutput "[状态] 系统运行状态" "Blue"
    
    if (-not (Test-Path $PidFile)) {
        Write-ColorOutput "❌ 系统未运行" "Red"
        return
    }
    
    $running = 0
    $stopped = 0
    $content = Get-Content $PidFile -ErrorAction SilentlyContinue
    
    Write-ColorOutput "" ""
    Write-ColorOutput "节点状态:" "Blue"
    Write-ColorOutput "----------------------------------------" "Blue"
    
    foreach ($line in $content) {
        if ($line -match "^([^:]+):(\d+)$") {
            $nodeName = $matches[1]
            $jobId = $matches[2]
            
            try {
                $job = Get-Job -Id $jobId -ErrorAction SilentlyContinue
                if ($job -and $job.State -eq "Running") {
                    Write-ColorOutput "✅ $nodeName (JobID: $jobId) - 运行中" "Green"
                    $running++
                }
                else {
                    Write-ColorOutput "❌ $nodeName (JobID: $jobId) - 已停止" "Red"
                    $stopped++
                }
            }
            catch {
                Write-ColorOutput "❓ $nodeName (JobID: $jobId) - 状态未知" "Yellow"
                $stopped++
            }
        }
    }
    
    Write-ColorOutput "----------------------------------------" "Blue"
    Write-ColorOutput "总计: 运行中 $running, 已停止 $stopped" "Blue"
    
    if ($running -gt 0) {
        Write-ColorOutput "" ""
        Write-ColorOutput "💡 使用 '.\run-local-system.ps1 logs' 查看系统日志" "Cyan"
    }
}

function Show-Logs {
    param([string]$NodeName = "")
    
    Write-Banner
    
    if ($NodeName) {
        $logFile = Join-Path $LogDir "$NodeName.log"
        if (Test-Path $logFile) {
            Write-ColorOutput "[日志] $NodeName" "Blue"
            Write-ColorOutput "----------------------------------------" "Blue"
            Get-Content $logFile -Tail 50
        }
        else {
            Write-ColorOutput "❌ 未找到日志文件: $logFile" "Red"
        }
    }
    else {
        Write-ColorOutput "[日志] 系统概览 (最近50行)" "Blue"
        
        # 显示supervisor日志
        $supervisorLog = Join-Path $LogDir "supervisor.log"
        if (Test-Path $supervisorLog) {
            Write-ColorOutput "" ""
            Write-ColorOutput "📋 Supervisor日志:" "Cyan"
            Write-ColorOutput "----------------------------------------" "Blue"
            Get-Content $supervisorLog -Tail 20
        }
        
        # 显示可用日志文件
        Write-ColorOutput "" ""
        Write-ColorOutput "📁 可用日志文件:" "Blue"
        Write-ColorOutput "----------------------------------------" "Blue"
        if (Test-Path $LogDir) {
            Get-ChildItem $LogDir -Name "*.log" | ForEach-Object {
                $size = (Get-Item (Join-Path $LogDir $_)).Length
                Write-ColorOutput "  - $_ ($('{0:N0}' -f $size) bytes)" "Yellow"
            }
        }
        
        Write-ColorOutput "" ""
        Write-ColorOutput "💡 查看特定节点日志: .\run-local-system.ps1 logs supervisor" "Cyan"
        Write-ColorOutput "💡 查看特定节点日志: .\run-local-system.ps1 logs shard0-node0" "Cyan"
    }
}

function Clear-Environment {
    Write-Banner
    Write-ColorOutput "[清理] 清理运行环境..." "Yellow"
    
    # 停止系统
    Stop-System
    
    # 清理日志
    if (Test-Path $LogDir) {
        Remove-Item $LogDir -Recurse -Force -ErrorAction SilentlyContinue
        Write-ColorOutput "✅ 已清理日志目录" "Green"
    }
    
    # 清理临时文件
    $tempFiles = @("expTest", "data_exchange", "outputs")
    foreach ($dir in $tempFiles) {
        if (Test-Path $dir) {
            Remove-Item $dir -Recurse -Force -ErrorAction SilentlyContinue
            Write-ColorOutput "✅ 已清理目录: $dir" "Green"
        }
    }
    
    # 清理Go缓存
    try {
        go clean -cache 2>$null
        Write-ColorOutput "✅ 已清理Go缓存" "Green"
    }
    catch {
        Write-ColorOutput "⚠️ Go缓存清理失败" "Yellow"
    }
    
    Write-ColorOutput "" ""
    Write-ColorOutput "✨ 环境清理完成" "Green"
}

function Show-Help {
    Write-Banner
    Write-ColorOutput "用法: .\run-local-system.ps1 [action] [options]" "Blue"
    Write-ColorOutput "" ""
    Write-ColorOutput "操作 (action):" "Blue"
    Write-ColorOutput "  start          启动BlockEmulator系统" "Yellow"
    Write-ColorOutput "  stop           停止运行中的系统" "Yellow"
    Write-ColorOutput "  status         查看系统运行状态" "Yellow"
    Write-ColorOutput "  logs [node]    查看系统日志" "Yellow"
    Write-ColorOutput "  clean          清理环境和临时文件" "Yellow"
    Write-ColorOutput "  help           显示此帮助信息" "Yellow"
    Write-ColorOutput "" ""
    Write-ColorOutput "选项:" "Blue"
    Write-ColorOutput "  -Shards N      分片数量 (默认: 4)" "Yellow"
    Write-ColorOutput "  -NodesPerShard N   每分片节点数 (默认: 4)" "Yellow"
    Write-ColorOutput "" ""
    Write-ColorOutput "示例:" "Blue"
    Write-ColorOutput "  .\run-local-system.ps1 start" "Green"
    Write-ColorOutput "  .\run-local-system.ps1 start -Shards 2 -NodesPerShard 4" "Green"
    Write-ColorOutput "  .\run-local-system.ps1 status" "Green"
    Write-ColorOutput "  .\run-local-system.ps1 logs supervisor" "Green"
    Write-ColorOutput "  .\run-local-system.ps1 stop" "Green"
    Write-ColorOutput "" ""
    Write-ColorOutput "💡 系统特性:" "Blue"
    Write-ColorOutput "  - 集成EvolveGCN四步分片算法" "Cyan"
    Write-ColorOutput "  - 支持动态分片重配置" "Cyan"
    Write-ColorOutput "  - 实时日志监控" "Cyan"
    Write-ColorOutput "  - 多进程并发运行" "Cyan"
}

# 主逻辑
switch ($Action) {
    "start"  { Start-System }
    "stop"   { Stop-System }
    "status" { Show-Status }
    "logs"   { Show-Logs $args[0] }
    "clean"  { Clear-Environment }
    "help"   { Show-Help }
    default  { Show-Help }
}
