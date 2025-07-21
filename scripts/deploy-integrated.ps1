#!/usr/bin/env pwsh
# 轻量化BlockEmulator + EvolveGCN集成部署脚本
# 目标：将6GB镜像减少到1.5GB以下，实现快速部署

param(
    [Parameter(Position=0)]
    [string]$Action = "help",
    
    [Parameter(Position=1)]
    [string]$Target = "",
    
    [switch]$Force,
    [switch]$NoBuild,
    [switch]$QuickTest
)

# 颜色输出函数
function Print-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Print-Success($msg) { Write-Host "[SUCCESS] $msg" -ForegroundColor Green }
function Print-Warning($msg) { Write-Host "[WARNING] $msg" -ForegroundColor Yellow }
function Print-Error($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red }

Write-Host "🚀 轻量化BlockEmulator + EvolveGCN集成系统" -ForegroundColor Magenta
Write-Host "   目标: 6GB → 1.5GB 镜像优化" -ForegroundColor Magenta
Write-Host "=" * 50 -ForegroundColor Magenta

function Show-Help {
    Write-Host "轻量化集成部署脚本 v1.0" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "命令列表:"
    Write-Host "  build          构建轻量化集成镜像 (约1.5GB)"
    Write-Host "  start          启动集成系统"
    Write-Host "  stop           停止系统"
    Write-Host "  restart        重启系统"
    Write-Host "  status         查看系统状态"
    Write-Host "  logs [node]    查看日志"
    Write-Host "  test           运行快速集成测试"
    Write-Host "  size           显示镜像大小对比"
    Write-Host "  cleanup        清理资源"
    Write-Host "  help           显示帮助"
    Write-Host ""
    Write-Host "优化特性:"
    Write-Host "  ✅ 多阶段构建减少镜像体积"
    Write-Host "  ✅ CPU版本PyTorch (小3GB)"
    Write-Host "  ✅ 选择性文件复制"
    Write-Host "  ✅ 统一镜像多实例部署"
    Write-Host "  ✅ 资源限制和健康检查"
}

function Test-Prerequisites {
    Print-Info "检查系统环境..."
    
    # 检查Docker
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Print-Error "Docker未安装"
        return $false
    }
    
    # 检查Docker运行状态
    try {
        docker info | Out-Null
    } catch {
        Print-Error "Docker未启动"
        return $false
    }
    
    # 检查必要文件
    $requiredFiles = @(
        "docker/Files/blockEmulator_linux_Precompile",
        "docker/Files/paramsConfig.json",
        "docker/Files/ipTable.json",
        "Dockerfile.integrated",
        "docker-compose.integrated.yml"
    )
    
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path $file)) {
            Print-Error "缺少必要文件: $file"
            return $false
        }
    }
    
    Print-Success "环境检查通过"
    return $true
}

function Build-IntegratedImage {
    Print-Info "构建轻量化集成镜像..."
    
    if (-not (Test-Prerequisites)) {
        return $false
    }
    
    # 显示构建信息
    Print-Info "镜像优化策略:"
    Print-Info "  - 多阶段构建"
    Print-Info "  - CPU版本PyTorch"
    Print-Info "  - 选择性文件复制"
    Print-Info "  - 压缩层优化"
    
    # 构建镜像
    Print-Info "开始构建 blockemulator-integrated:latest..."
    docker build -f Dockerfile.integrated -t blockemulator-integrated:latest .
    
    if ($LASTEXITCODE -eq 0) {
        Print-Success "镜像构建成功"
        Show-ImageSize
        return $true
    } else {
        Print-Error "镜像构建失败"
        return $false
    }
}

function Show-ImageSize {
    Print-Info "镜像大小对比:"
    
    $images = @(
        "blockemulator-integrated:latest",
        "block-emulator:latest"
    )
    
    foreach ($image in $images) {
        $size = docker images $image --format "{{.Size}}" 2>$null
        if ($size) {
            Write-Host "  $image : $size" -ForegroundColor Green
        } else {
            Write-Host "  $image : 未构建" -ForegroundColor Gray
        }
    }
}

function Start-IntegratedSystem {
    Print-Info "启动轻量化集成系统..."
    
    # 创建必要目录
    $dirs = @("outputs", "data_exchange", "outputs/supervisor")
    foreach ($dir in $dirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Print-Info "创建目录: $dir"
        }
    }
    
    # 为每个节点创建输出目录
    for ($s = 0; $s -lt 4; $s++) {
        for ($n = 0; $n -lt 4; $n++) {
            $nodeDir = "outputs/shard$s-node$n"
            if (-not (Test-Path $nodeDir)) {
                New-Item -ItemType Directory -Path $nodeDir -Force | Out-Null
            }
        }
    }
    
    # 启动服务
    Print-Info "使用docker-compose启动集成系统..."
    docker-compose -f docker-compose.integrated.yml up -d
    
    if ($LASTEXITCODE -eq 0) {
        Print-Success "系统启动成功"
        Start-Sleep 3
        Get-SystemStatus
        return $true
    } else {
        Print-Error "系统启动失败"
        return $false
    }
}

function Stop-IntegratedSystem {
    Print-Info "停止集成系统..."
    docker-compose -f docker-compose.integrated.yml down
    
    if ($LASTEXITCODE -eq 0) {
        Print-Success "系统已停止"
    } else {
        Print-Error "停止失败"
    }
}

function Get-SystemStatus {
    Print-Info "系统状态:"
    
    # 容器状态
    $containers = docker-compose -f docker-compose.integrated.yml ps
    Write-Host $containers
    
    # 资源使用
    Print-Info "资源使用情况:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" $(docker-compose -f docker-compose.integrated.yml ps -q)
}

function Show-Logs {
    param([string]$NodeName = "")
    
    if ($NodeName) {
        Print-Info "查看节点 $NodeName 的日志:"
        docker-compose -f docker-compose.integrated.yml logs -f $NodeName
    } else {
        Print-Info "查看所有节点日志:"
        docker-compose -f docker-compose.integrated.yml logs -f
    }
}

function Run-IntegrationTest {
    Print-Info "运行集成测试..."
    
    # 检查系统是否运行
    $supervisorStatus = docker-compose -f docker-compose.integrated.yml ps supervisor
    if ($supervisorStatus -notmatch "Up") {
        Print-Warning "系统未运行，先启动系统"
        Start-IntegratedSystem
        Start-Sleep 10
    }
    
    # 运行测试
    Print-Info "执行EvolveGCN集成测试..."
    docker exec supervisor python3 /app/evolvegcn_go_interface.py --test
    
    if ($LASTEXITCODE -eq 0) {
        Print-Success "集成测试通过"
    } else {
        Print-Error "集成测试失败"
    }
}

function Cleanup-Resources {
    Print-Info "清理系统资源..."
    
    # 停止并删除容器
    docker-compose -f docker-compose.integrated.yml down -v
    
    # 删除镜像 (可选)
    if ($Force) {
        Print-Warning "强制删除镜像..."
        docker rmi blockemulator-integrated:latest 2>$null
    }
    
    # 清理输出目录
    if (Test-Path "outputs") {
        Remove-Item -Recurse -Force outputs 2>$null
        Print-Info "清理输出目录"
    }
    
    Print-Success "清理完成"
}

# 主逻辑
switch ($Action.ToLower()) {
    "build" { Build-IntegratedImage }
    "start" { Start-IntegratedSystem }
    "stop" { Stop-IntegratedSystem }
    "restart" { 
        Stop-IntegratedSystem
        Start-Sleep 2
        Start-IntegratedSystem 
    }
    "status" { Get-SystemStatus }
    "logs" { Show-Logs -NodeName $Target }
    "test" { Run-IntegrationTest }
    "size" { Show-ImageSize }
    "cleanup" { Cleanup-Resources }
    "help" { Show-Help }
    default {
        Print-Error "未知命令: $Action"
        Show-Help
        exit 1
    }
}

Write-Host ""
Write-Host "🎯 轻量化集成完成!" -ForegroundColor Green
