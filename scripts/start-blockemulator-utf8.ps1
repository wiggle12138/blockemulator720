# UTF-8编码支持的BlockEmulator启动脚本
# 解决GBK编码问题，支持中文字符处理

param(
    [ValidateSet("Start", "Stop", "Status", "Restart", "Test")]
    [string]$Action = "Start",
    [int]$Nodes = 4,
    [int]$Shards = 2,
    [switch]$EnableUTF8,
    [switch]$Verbose,
    [switch]$TestMode
)

# 强制设置UTF-8编码
$OutputEncoding = [console]::InputEncoding = [console]::OutputEncoding = New-Object System.Text.UTF8Encoding

# 设置进程环境变量支持UTF-8
$env:PYTHONIOENCODING = "utf-8"
$env:LANG = "en_US.UTF-8"
$env:LC_ALL = "en_US.UTF-8"

Write-Host "===============================================" -ForegroundColor Green
Write-Host "  BlockEmulator UTF-8启动器 v2.0" -ForegroundColor Green  
Write-Host "===============================================" -ForegroundColor Green

if ($EnableUTF8) {
    Write-Host "[UTF-8] 启用UTF-8编码支持" -ForegroundColor Yellow
    # 设置Windows控制台代码页为UTF-8 (65001)
    chcp 65001 > $null
}

# 全局变量
$script:ProcessIds = @()
$script:PidFile = "blockemulator_utf8_pids.txt"

function Write-LogMessage {
    param([string]$Message, [string]$Level = "INFO")
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $color = switch ($Level) {
        "SUCCESS" { "Green" }
        "WARNING" { "Yellow" }
        "ERROR" { "Red" }
        "DEBUG" { "Cyan" }
        default { "White" }
    }
    
    Write-Host "[$timestamp] [$Level] $Message" -ForegroundColor $color
}

function Test-UTF8Encoding {
    Write-LogMessage "测试UTF-8编码支持..." "INFO"
    
    # 创建包含中文的测试文件
    $testContent = @"
{
    "测试": "UTF-8编码支持",
    "中文字符": ["成功", "失败", "警告"],
    "特殊字符": "🚀⚡🔧",
    "timestamp": "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}
"@
    
    try {
        [System.IO.File]::WriteAllText("utf8_encoding_test.json", $testContent, [System.Text.Encoding]::UTF8)
        Write-LogMessage "UTF-8测试文件创建成功" "SUCCESS"
        
        # 验证文件内容
        $readContent = [System.IO.File]::ReadAllText("utf8_encoding_test.json", [System.Text.Encoding]::UTF8)
        if ($readContent.Contains("测试") -and $readContent.Contains("🚀")) {
            Write-LogMessage "UTF-8编码验证通过" "SUCCESS"
            return $true
        } else {
            Write-LogMessage "UTF-8编码验证失败" "ERROR"
            return $false
        }
    } catch {
        Write-LogMessage "UTF-8编码测试失败: $_" "ERROR"
        return $false
    }
}

function Get-UTF8Executable {
    # 优先使用UTF-8编译版本
    $utf8Executables = @(
        "blockEmulator_Windows_UTF8.exe",
        "blockEmulator_UTF8.exe",
        "blockEmulator.exe",
        "blockEmulator_Windows_Precompile.exe"
    )
    
    foreach ($exe in $utf8Executables) {
        if (Test-Path $exe) {
            Write-LogMessage "找到可执行文件: $exe" "INFO"
            return $exe
        }
    }
    
    Write-LogMessage "未找到BlockEmulator可执行文件" "ERROR"
    return $null
}

function Start-BlockEmulatorSystem {
    Write-LogMessage "启动BlockEmulator系统 (UTF-8编码支持)" "INFO"
    Write-LogMessage "配置: $Shards个分片，每分片$Nodes个节点" "INFO"
    
    $executable = Get-UTF8Executable
    if (-not $executable) {
        Write-LogMessage "请先编译BlockEmulator或确保可执行文件存在" "ERROR"
        return $false
    }
    
    # 清理旧的PID文件
    if (Test-Path $script:PidFile) {
        Remove-Item $script:PidFile -Force
    }
    
    # 启动各分片节点
    for ($shard = 0; $shard -lt $Shards; $shard++) {
        Write-LogMessage "启动分片${shard}节点..." "INFO"
        
        for ($node = 0; $node -lt $Nodes; $node++) {
            $processArgs = @(
                "-n", $node,
                "-N", $Nodes,
                "-s", $shard,
                "-S", $Shards
            )
            
            try {
                # 使用UTF-8环境启动进程
                $process = Start-Process -FilePath $executable -ArgumentList $processArgs -PassThru -WindowStyle Minimized
                $script:ProcessIds += $process.Id
                
                # 记录PID到文件
                "$($process.Id)" | Add-Content $script:PidFile
                
                Write-LogMessage "  节点 S${shard}-N${node} 启动中... (PID: $($process.Id))" "SUCCESS"
                
                # 给进程时间初始化
                Start-Sleep -Milliseconds 500
                
            } catch {
                Write-LogMessage "启动节点 S${shard}-N${node} 失败: $_" "ERROR"
            }
        }
    }
    
    # 启动Supervisor
    Write-LogMessage "启动Supervisor..." "INFO"
    try {
        $supervisorArgs = @("-c", "-N", $Nodes, "-S", $Shards)
        $supervisorProcess = Start-Process -FilePath $executable -ArgumentList $supervisorArgs -PassThru -WindowStyle Normal
        $script:ProcessIds += $supervisorProcess.Id
        
        "$($supervisorProcess.Id)" | Add-Content $script:PidFile
        Write-LogMessage "  Supervisor 启动中... (PID: $($supervisorProcess.Id))" "SUCCESS"
        
    } catch {
        Write-LogMessage "启动Supervisor失败: $_" "ERROR"
    }
    
    Write-LogMessage "系统启动完成!" "SUCCESS"
    Write-LogMessage "总共启动了 $($script:ProcessIds.Count) 个进程" "INFO"
    Write-LogMessage "进程ID已保存到: $script:PidFile" "INFO"
    
    return $true
}

function Stop-BlockEmulatorSystem {
    Write-LogMessage "停止BlockEmulator系统..." "WARNING"
    
    $stoppedCount = 0
    
    # 从PID文件读取进程ID
    if (Test-Path $script:PidFile) {
        $savedPids = Get-Content $script:PidFile | Where-Object { $_ -match '^\d+$' }
        
        foreach ($pid in $savedPids) {
            try {
                $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
                if ($process) {
                    Stop-Process -Id $pid -Force
                    Write-LogMessage "  停止进程 PID: $pid" "INFO"
                    $stoppedCount++
                }
            } catch {
                Write-LogMessage "停止进程 $pid 时出错: $_" "WARNING"
            }
        }
        
        # 删除PID文件
        Remove-Item $script:PidFile -Force -ErrorAction SilentlyContinue
    } else {
        Write-LogMessage "未找到PID文件，尝试根据进程名停止..." "WARNING"
        
        # 根据进程名停止
        $processes = Get-Process | Where-Object { $_.ProcessName -match "blockEmulator" }
        foreach ($process in $processes) {
            try {
                Stop-Process -Id $process.Id -Force
                Write-LogMessage "  停止进程: $($process.ProcessName) (PID: $($process.Id))" "INFO"
                $stoppedCount++
            } catch {
                Write-LogMessage "停止进程失败: $_" "ERROR"
            }
        }
    }
    
    Write-LogMessage "系统停止完成，共停止 $stoppedCount 个进程" "SUCCESS"
}

function Get-SystemStatus {
    Write-LogMessage "检查系统状态..." "INFO"
    
    $activeProcesses = @()
    
    if (Test-Path $script:PidFile) {
        $savedPids = Get-Content $script:PidFile | Where-Object { $_ -match '^\d+$' }
        
        foreach ($pid in $savedPids) {
            $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
            if ($process) {
                $activeProcesses += $process
            }
        }
    }
    
    if ($activeProcesses.Count -gt 0) {
        Write-LogMessage "发现 $($activeProcesses.Count) 个活动进程:" "SUCCESS"
        foreach ($proc in $activeProcesses) {
            $cpuUsage = try { $proc.CPU } catch { "N/A" }
            $memoryMB = [math]::Round($proc.WorkingSet64 / 1MB, 2)
            Write-LogMessage "  PID: $($proc.Id), CPU: ${cpuUsage}%, 内存: ${memoryMB}MB" "INFO"
        }
    } else {
        Write-LogMessage "未发现活动的BlockEmulator进程" "WARNING"
    }
}

function Test-PythonIntegration {
    Write-LogMessage "测试Python集成..." "INFO"
    
    # 检查Python环境
    try {
        $pythonVersion = python --version 2>&1
        Write-LogMessage "Python版本: $pythonVersion" "INFO"
    } catch {
        Write-LogMessage "Python环境未找到" "ERROR"
        return $false
    }
    
    # 测试关键Python文件
    $pythonFiles = @(
        "evolvegcn_go_interface.py",
        "integrated_four_step_pipeline.py"
    )
    
    foreach ($file in $pythonFiles) {
        if (Test-Path $file) {
            Write-LogMessage "检查Python文件: $file" "INFO"
            
            # 简单的语法检查
            try {
                python -m py_compile $file
                Write-LogMessage "  语法检查通过: $file" "SUCCESS"
            } catch {
                Write-LogMessage "  语法检查失败: $file" "ERROR"
            }
        } else {
            Write-LogMessage "缺少Python文件: $file" "WARNING"
        }
    }
    
    return $true
}

# 主执行逻辑
switch ($Action) {
    "Start" {
        if ($TestMode) {
            Write-LogMessage "测试模式启动" "DEBUG"
            if (-not (Test-UTF8Encoding)) {
                exit 1
            }
        }
        
        Start-BlockEmulatorSystem
    }
    
    "Stop" {
        Stop-BlockEmulatorSystem
    }
    
    "Status" {
        Get-SystemStatus
    }
    
    "Restart" {
        Stop-BlockEmulatorSystem
        Start-Sleep -Seconds 2
        Start-BlockEmulatorSystem
    }
    
    "Test" {
        Write-LogMessage "运行综合测试..." "INFO"
        
        $utf8Test = Test-UTF8Encoding
        $pythonTest = Test-PythonIntegration
        
        if ($utf8Test -and $pythonTest) {
            Write-LogMessage "所有测试通过" "SUCCESS"
        } else {
            Write-LogMessage "部分测试失败" "ERROR"
        }
    }
}

Write-LogMessage "操作完成" "INFO"

# 如果是交互模式，显示管理提示
if (-not $TestMode) {
    Write-Host "`n管理命令:" -ForegroundColor Cyan
    Write-Host "  查看状态: .\start-blockemulator-utf8.ps1 -Action Status" -ForegroundColor Yellow
    Write-Host "  停止系统: .\start-blockemulator-utf8.ps1 -Action Stop" -ForegroundColor Yellow
    Write-Host "  测试集成: .\start-blockemulator-utf8.ps1 -Action Test" -ForegroundColor Yellow
}
