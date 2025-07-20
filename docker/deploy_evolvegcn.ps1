# PowerShell 版本的Docker部署与管理脚本 v2.1 - EvolveGCN集成版
# 包含了构建、启动、停止、状态、日志、执行和清理等完整功能
# 新增: Python虚拟环境配置和EvolveGCN集成

# --- 函数定义 ---

# 函数：打印带颜色的消息
function Print-Info { param($message) Write-Host "[INFO] $message" -ForegroundColor Blue }
function Print-Success { param($message) Write-Host "[SUCCESS] $message" -ForegroundColor Green }
function Print-Warning { param($message) Write-Host "[WARNING] $message" -ForegroundColor Yellow }
function Print-Error { param($message) Write-Host "[ERROR] $message" -ForegroundColor Red }

# 函数：显示帮助信息
function Show-Help {
    Write-Host "Docker部署脚本 (PowerShell版本 v2.1 - EvolveGCN集成版)"
    Write-Host ""
    Write-Host "用法: .\deploy_evolvegcn.ps1 [命令] [参数]"
    Write-Host ""
    Write-Host "命令:"
    Write-Host "  setup          配置Python虚拟环境和EvolveGCN集成"
    Write-Host "  build          构建共享Docker镜像 (block-emulator:latest)"
    Write-Host "  start          在后台启动所有节点容器 (自动创建输出目录)"
    Write-Host "  stop           停止并移除所有节点容器"
    Write-Host "  restart        重启所有节点容器"
    Write-Host "  status         查看服务状态、资源使用和健康检查"
    Write-Host "  logs           查看所有节点的实时日志"
    Write-Host "  logs [节点名]  查看指定节点的实时日志 (例如: logs shard0-node0)"
    Write-Host "  exec [节点名]  进入指定容器的shell环境 (例如: exec supervisor)"
    Write-Host "  test-python    测试Python虚拟环境和EvolveGCN集成"
    Write-Host "  cleanup        清理所有Docker资源 (停止容器、删除网络和卷)"
    Write-Host "  help           显示此帮助信息"
    Write-Host ""
    Write-Host "EvolveGCN集成特性:"
    Write-Host "  - 自动检测和配置Python虚拟环境"
    Write-Host "  - 集成四步EvolveGCN分片算法"
    Write-Host "  - 替换CLPA占位算法为真实EvolveGCN"
    Write-Host "  - 支持反馈循环和自适应分片"
}

# 函数：检查Docker环境
function Check-Docker {
    Print-Info "检查Docker环境..."
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Print-Error "Docker未安装，请先安装Docker Desktop"
        exit 1
    }
    docker info > $null
    if ($LASTEXITCODE -ne 0) {
        Print-Error "Docker服务未启动，请启动Docker Desktop"
        exit 1
    }
    Print-Success "Docker检查通过"
}

# 函数：配置Python虚拟环境
function Setup-PythonEnvironment {
    Print-Info "配置Python虚拟环境和EvolveGCN集成..."
    
    # 检查Python虚拟环境配置脚本（从主目录查找）
    if (-not (Test-Path "..\config_python_venv.py")) {
        Print-Error "未找到Python环境配置脚本: config_python_venv.py"
        return $false
    }
    
    # 运行Python环境配置（切换到主目录运行）
    Print-Info "正在配置Python虚拟环境..."
    try {
        $originalDir = Get-Location
        Set-Location ".."
        $result = python config_python_venv.py
        Set-Location $originalDir
        if ($LASTEXITCODE -eq 0) {
            Print-Success "Python虚拟环境配置完成"
        } else {
            Print-Warning "Python虚拟环境配置可能有警告，继续执行..."
        }
    } catch {
        Print-Error "Python虚拟环境配置失败: $_"
        Set-Location $originalDir
        return $false
    }
    
    # 检查关键文件（从主目录查找）
    $requiredFiles = @(
        "..\evolvegcn_go_interface.py",
        "..\integrated_four_step_pipeline.py",
        "..\python_config.json"
    )
    
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path $file)) {
            Print-Error "缺少关键文件: $file"
            return $false
        } else {
            Print-Info "发现关键文件: $(Split-Path $file -Leaf)"
        }
    }
    
    Print-Success "EvolveGCN集成环境检查完成"
    return $true
}

# 函数：测试Python集成
function Test-PythonIntegration {
    Print-Info "测试Python虚拟环境和EvolveGCN集成..."
    
    # 读取Python配置（从主目录查找）
    if (Test-Path "..\python_config.json") {
        $config = Get-Content "..\python_config.json" | ConvertFrom-Json
        $pythonPath = $config.python_path
        Print-Info "使用Python路径: $pythonPath"
    } else {
        Print-Warning "未找到python_config.json，使用默认Python"
        $pythonPath = "python"
    }
    
    # 测试Python环境
    Print-Info "测试Python基础环境..."
    try {
        & $pythonPath --version
        if ($LASTEXITCODE -ne 0) {
            Print-Error "Python环境不可用"
            return $false
        }
    } catch {
        Print-Error "无法执行Python: $_"
        return $false
    }
    
    # 测试EvolveGCN接口
    Print-Info "测试EvolveGCN Go接口..."
    try {
        & $pythonPath evolvegcn_go_interface.py --help
        if ($LASTEXITCODE -eq 0) {
            Print-Success "EvolveGCN接口测试通过"
        } else {
            Print-Warning "EvolveGCN接口可能需要调试"
        }
    } catch {
        Print-Error "EvolveGCN接口测试失败: $_"
        return $false
    }
    
    # 创建测试输入
    Print-Info "创建EvolveGCN测试输入..."
    $testInput = @{
        node_features = @(
            @{
                node_id = "test_node_1"
                features = @(0.1, 0.2, 0.3, 0.4, 0.5)
                metadata = @{
                    shard_id = 0
                    timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
                }
            },
            @{
                node_id = "test_node_2"
                features = @(0.6, 0.7, 0.8, 0.9, 1.0)
                metadata = @{
                    shard_id = 1
                    timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
                }
            }
        )
        transaction_graph = @{
            edges = @(
                @("test_node_1", "test_node_2", 1.0)
            )
            metadata = @{
                timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
                graph_type = "test"
            }
        }
        config = @{
            enable_feedback = $true
            max_iterations = 1
        }
    }
    
    # 保存测试输入
    $testInput | ConvertTo-Json -Depth 10 | Out-File -FilePath "test_evolvegcn_input.json" -Encoding UTF8
    
    # 运行测试
    Print-Info "运行EvolveGCN集成测试..."
    try {
        & $pythonPath evolvegcn_go_interface.py --input "test_evolvegcn_input.json" --output "test_evolvegcn_output.json"
        if ($LASTEXITCODE -eq 0 -and (Test-Path "test_evolvegcn_output.json")) {
            Print-Success "EvolveGCN集成测试通过"
            
            # 显示测试结果
            $result = Get-Content "test_evolvegcn_output.json" | ConvertFrom-Json
            if ($result.success) {
                Print-Info "测试结果:"
                Print-Info "  - 成功状态: $($result.success)"
                Print-Info "  - 分片映射: $($result.partition_map | ConvertTo-Json -Compress)"
                Print-Info "  - 跨分片边数: $($result.cross_shard_edges)"
            } else {
                Print-Warning "测试完成但有错误: $($result.error)"
            }
            
            # 清理测试文件
            Remove-Item "test_evolvegcn_input.json" -ErrorAction SilentlyContinue
            Remove-Item "test_evolvegcn_output.json" -ErrorAction SilentlyContinue
            
            return $true
        } else {
            Print-Error "EvolveGCN集成测试失败"
            return $false
        }
    } catch {
        Print-Error "运行EvolveGCN测试失败: $_"
        return $false
    }
}

# 函数：检查必要文件
function Check-Files {
    Print-Info "检查必要文件..."
    $missingFiles = 0

    # 检查可执行文件
    if (-not (Test-Path ".\Files\blockEmulator_linux_Precompile")) {
        Print-Error "未找到可执行文件: .\Files\blockEmulator_linux_Precompile"
        $missingFiles++
    }

    # 检查配置文件
    if (-not (Test-Path ".\Files\paramsConfig.json")) {
        Print-Error "缺少配置文件: .\Files\paramsConfig.json"
        $missingFiles++
    } else {
        Print-Info "找到配置文件: paramsConfig.json"
    }

    if (-not (Test-Path ".\Files\ipTable.json")) {
        Print-Error "缺少配置文件: .\Files\ipTable.json"
        $missingFiles++
    } else {
        Print-Info "找到配置文件: ipTable.json"
    }

    # 检查EvolveGCN相关文件（从主目录查找）
    $evolvegcnFiles = @(
        "..\evolvegcn_go_interface.py",
        "..\integrated_four_step_pipeline.py",
        "..\python_config.json"
    )
    
    foreach ($file in $evolvegcnFiles) {
        if (-not (Test-Path $file)) {
            Print-Warning "EvolveGCN文件缺失: $file"
            $missingFiles++
        } else {
            Print-Info "找到EvolveGCN文件: $(Split-Path $file -Leaf)"
        }
    }

    if ($missingFiles -gt 0) {
        Print-Error "缺少必要文件，无法继续"
        exit 1
    }
    
    Print-Success "文件检查通过"
}

# 函数：构建镜像
function Build-Images {
    Print-Info "开始构建Docker镜像..."
    Print-Info "这可能需要几分钟，请耐心等待..."
    
    # 构建镜像并标记为 block-emulator:latest
    docker build -t block-emulator:latest .
    if ($LASTEXITCODE -ne 0) { Print-Error "镜像构建失败"; exit 1 }
    
    Print-Success "Docker镜像构建完成: block-emulator:latest"
    Print-Info "所有节点将共享此镜像，并支持EvolveGCN集成"
}

# 函数：启动服务
function Start-Services {
    Print-Info "检查镜像是否存在..."
    docker images | findstr "block-emulator" > $null
    if ($LASTEXITCODE -ne 0) {
        Print-Error "镜像 block-emulator:latest 不存在，请先运行 '.\deploy_evolvegcn.ps1 build'"
        exit 1
    }
    
    # 确保Python环境已配置（从主目录查找）
    if (-not (Test-Path "..\python_config.json")) {
        Print-Warning "未找到Python配置，正在自动配置..."
        if (-not (Setup-PythonEnvironment)) {
            Print-Error "Python环境配置失败，启动中止"
            exit 1
        }
    }
    
    Print-Info "在后台启动所有节点 (支持EvolveGCN分片算法)..."
    docker-compose up -d
    if ($LASTEXITCODE -ne 0) { Print-Error "启动服务失败"; exit 1 }
    Print-Success "所有节点已启动，EvolveGCN分片算法已集成。使用 '.\deploy_evolvegcn.ps1 status' 查看状态。"
}

# 函数：停止服务
function Stop-Services {
    Print-Info "停止并移除所有节点容器..."
    docker-compose down
    if ($LASTEXITCODE -ne 0) { Print-Warning "停止服务时遇到问题，可能部分资源未清理" }
    Print-Success "所有节点已停止"
}

# 函数：重启服务
function Restart-Services {
    Print-Info "重启所有节点..."
    docker-compose restart
    if ($LASTEXITCODE -ne 0) { Print-Error "重启服务失败"; exit 1 }
    Print-Success "所有节点已重启"
}

# 函数：获取服务状态
function Get-Status {
    Print-Info "容器状态概览:"
    docker-compose ps
    Write-Host ""
    
    # 显示EvolveGCN配置状态
    if (Test-Path "python_config.json") {
        Print-Info "EvolveGCN配置状态:"
        $config = Get-Content "python_config.json" | ConvertFrom-Json
        Write-Host "  - EvolveGCN启用: $($config.enable_evolve_gcn)" -ForegroundColor Cyan
        Write-Host "  - 反馈机制启用: $($config.enable_feedback)" -ForegroundColor Cyan
        Write-Host "  - Python路径: $($config.python_path)" -ForegroundColor Cyan
    } else {
        Print-Warning "EvolveGCN配置未找到"
    }
    
    Write-Host ""
    Print-Info "容器资源实时使用情况 (按Ctrl+C退出):"
    docker stats
}

# 函数：查看日志
function View-Logs {
    param($serviceName)
    if ([string]::IsNullOrEmpty($serviceName)) {
        Print-Info "查看所有节点的实时日志 (按Ctrl+C退出)..."
        docker-compose logs -f --timestamps
    } else {
        Print-Info "查看节点 $serviceName 的实时日志 (按Ctrl+C退出)..."
        docker-compose logs -f --timestamps $serviceName
    }
}

# 函数：进入容器
function Enter-Container {
    param($serviceName)
    if ([string]::IsNullOrEmpty($serviceName)) {
        Print-Error "请指定要进入的容器名称。用法: .\deploy_evolvegcn.ps1 exec [节点名]"
        exit 1
    }
    Print-Info "进入容器 $serviceName 的shell环境..."
    docker-compose exec $serviceName /bin/sh
}

# 函数：清理所有资源
function Cleanup-Resources {
    Print-Warning "警告：此操作将停止并删除所有容器、网络和数据卷！"
    $confirmation = Read-Host "是否继续? (y/n)"
    if ($confirmation -ne 'y') {
        Print-Info "操作已取消"
        exit 0
    }
    Print-Info "开始清理所有Docker资源..."
    docker-compose down -v
    
    # 清理EvolveGCN临时文件
    Print-Info "清理EvolveGCN临时文件..."
    $tempFiles = @(
        "evolvegcn_input.json",
        "evolvegcn_output.json",
        "test_evolvegcn_input.json",
        "test_evolvegcn_output.json"
    )
    
    foreach ($file in $tempFiles) {
        if (Test-Path $file) {
            Remove-Item $file -Force
            Print-Info "已删除: $file"
        }
    }
    
    Print-Success "清理完成"
}

# --- 主逻辑 ---
function Main {
    param(
        [string]$command,
        [string]$parameter
    )

    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host " 区块链节点部署系统 - EvolveGCN集成版 v2.1" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    echo ""

    # 对于 build 和 start 命令，需要检查文件
    if ($command -in @("build", "start")) {
        Check-Files
    }
    
    # 检查Docker环境（除help外所有命令都需要）
    if ($command -notin @("help", "setup", "test-python")) {
        Check-Docker
    }

    switch ($command) {
        "setup"       { Setup-PythonEnvironment }
        "build"       { Build-Images }
        "start"       { Start-Services }
        "stop"        { Stop-Services }
        "restart"     { Restart-Services }
        "status"      { Get-Status }
        "logs"        { View-Logs -serviceName $parameter }
        "exec"        { Enter-Container -serviceName $parameter }
        "test-python" { Test-PythonIntegration }
        "cleanup"     { Cleanup-Resources }
        "help"        { Show-Help }
        default {
            Print-Warning "未知命令或未提供命令。"
            Show-Help
        }
    }
}

# --- 脚本入口 ---
# 执行主函数，并传递脚本的第一个和第二个参数
Main $args[0] $args[1]
