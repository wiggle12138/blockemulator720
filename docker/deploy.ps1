# PowerShell 版本的Docker部署与管理脚本 v2.0
# 包含了构建、启动、停止、状态、日志、执行和清理等完整功能

# --- 函数定义 ---

# 函数：打印带颜色的消息
function Print-Info { param($message) Write-Host "[INFO] $message" -ForegroundColor Blue }
function Print-Success { param($message) Write-Host "[SUCCESS] $message" -ForegroundColor Green }
function Print-Warning { param($message) Write-Host "[WARNING] $message" -ForegroundColor Yellow }
function Print-Error { param($message) Write-Host "[ERROR] $message" -ForegroundColor Red }

# 函数：显示帮助信息
function Show-Help {
    Write-Host "Docker部署脚本 (PowerShell版本 v2.0 - 优化版)"
    Write-Host ""
    Write-Host "用法: .\deploy.ps1 [命令] [参数]"
    Write-Host ""
    Write-Host "命令:"
    Write-Host "  build          构建共享Docker镜像 (block-emulator:latest)"
    Write-Host "  start          在后台启动所有节点容器 (自动创建输出目录)"
    Write-Host "  stop           停止并移除所有节点容器"
    Write-Host "  restart        重启所有节点容器"
    Write-Host "  status         查看服务状态、资源使用和健康检查"
    Write-Host "  logs           查看所有节点的实时日志"
    Write-Host "  logs [节点名]  查看指定节点的实时日志 (例如: logs shard0-node0)"
    Write-Host "  exec [节点名]  进入指定容器的shell环境 (例如: exec supervisor)"
    Write-Host "  cleanup        清理所有Docker资源 (停止容器、删除网络和卷)"
    Write-Host "  help           显示此帮助信息"
    Write-Host ""
    Write-Host "优化特性:"
    Write-Host "  - 所有节点共享同一个镜像，减少存储空间"
    Write-Host "  - 每个节点有独立的输出目录，避免文件冲突"
    Write-Host "  - 自动创建必要的目录结构"
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

    if ($missingFiles -gt 0) {
        Print-Error "缺少必要文件，无法继续"
        exit 1
    }
}

# 函数：构建镜像
function Build-Images {
    Print-Info "开始构建Docker镜像..."
    Print-Info "这可能需要几分钟，请耐心等待..."
    
    # 构建镜像并标记为 block-emulator:latest
    docker build -t block-emulator:latest .
    if ($LASTEXITCODE -ne 0) { Print-Error "镜像构建失败"; exit 1 }
    
    Print-Success "Docker镜像构建完成: block-emulator:latest"
    Print-Info "所有节点将共享此镜像，提高部署效率"
}

# 函数：启动服务
function Start-Services {
    Print-Info "检查镜像是否存在..."
    docker images | findstr "block-emulator" > $null
    if ($LASTEXITCODE -ne 0) {
        Print-Error "镜像 block-emulator:latest 不存在，请先运行 '.\deploy.ps1 build'"
        exit 1
    }
    
    Print-Info "在后台启动所有节点..."
    docker-compose up -d
    if ($LASTEXITCODE -ne 0) { Print-Error "启动服务失败"; exit 1 }
    Print-Success "所有节点已启动。使用 '.\deploy.ps1 status' 查看状态。"
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
        Print-Error "请指定要进入的容器名称。用法: .\deploy.ps1 exec [节点名]"
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
    Print-Success "清理完成"
}


# --- 主逻辑 ---
function Main {
    param(
        [string]$command,
        [string]$parameter
    )

    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host " 区块链节点部署系统 - PowerShell版本 v2.0 (优化版)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    echo ""

    # 对于 build 和 start 命令，需要检查文件
    if ($command -in @("build", "start")) {
        Check-Files
    }
    
    # 检查Docker环境（除help外所有命令都需要）
    if ($command -ne "help") {
        Check-Docker
    }

    switch ($command) {
        "build"   { Build-Images }
        "start"   { Start-Services }
        "stop"    { Stop-Services }
        "restart" { Restart-Services }
        "status"  { Get-Status }
        "logs"    { View-Logs -serviceName $parameter }
        "exec"    { Enter-Container -serviceName $parameter }
        "cleanup" { Cleanup-Resources }
        "help"    { Show-Help }
        default {
            Print-Warning "未知命令或未提供命令。"
            Show-Help
        }
    }
}

# --- 脚本入口 ---
# 执行主函数，并传递脚本的第一个和第二个参数
Main $args[0] $args[1]