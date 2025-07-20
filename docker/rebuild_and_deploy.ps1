#!/usr/bin/env pwsh
# 重新构建和部署EvolveGCN集成系统
# 自动处理：Go代码编译 → Linux可执行文件生成 → Docker镜像构建 → 容器部署

param(
    [string]$Action = "full"  # full, compile, build, deploy
)

# 颜色输出函数
function Print-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Print-Success($msg) { Write-Host "[SUCCESS] $msg" -ForegroundColor Green }
function Print-Warning($msg) { Write-Host "[WARNING] $msg" -ForegroundColor Yellow }
function Print-Error($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red }

Write-Host "===========================================" -ForegroundColor Magenta
Write-Host " EvolveGCN集成系统 - 自动构建部署工具 v1.0" -ForegroundColor Magenta
Write-Host "===========================================" -ForegroundColor Magenta
Write-Host ""

function Test-Prerequisites {
    Print-Info "检查前置条件..."
    
    # 检查Docker
    try {
        docker --version | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Docker不可用" }
        Print-Success "✅ Docker环境正常"
    } catch {
        Print-Error "❌ Docker环境检查失败: $_"
        return $false
    }
    
    # 检查Go环境
    try {
        go version | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Go不可用" }
        Print-Success "✅ Go环境正常"
    } catch {
        Print-Error "❌ Go环境检查失败: $_"
        return $false
    }
    
    # 检查关键文件
    $requiredFiles = @(
        "..\main.go",
        "..\evolvegcn_go_interface.py",
        "..\integrated_four_step_pipeline.py",
        "..\python_config.json"
    )
    
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path $file)) {
            Print-Error "❌ 缺少关键文件: $file"
            return $false
        }
    }
    Print-Success "✅ 关键文件检查通过"
    
    return $true
}

function Compile-GoCode {
    Print-Info "编译Go代码为Linux可执行文件..."
    
    # 返回主目录进行编译
    $currentDir = Get-Location
    Set-Location ".."
    
    try {
        # 直接交叉编译，更可靠
        Print-Info "执行交叉编译..."
        $env:GOOS = "linux"
        $env:GOARCH = "amd64"
        go build -o blockEmulator_linux_Precompile main.go
        if ($LASTEXITCODE -ne 0) { throw "Go交叉编译失败" }
        
        # 检查生成的文件
        if (-not (Test-Path "blockEmulator_linux_Precompile")) {
            throw "Linux可执行文件未生成"
        }
        
        # 移动到Docker Files目录
        if (-not (Test-Path "docker\Files")) {
            New-Item -ItemType Directory -Path "docker\Files" -Force | Out-Null
        }
        
        Copy-Item "blockEmulator_linux_Precompile" "docker\Files\" -Force
        Print-Success "✅ Linux可执行文件已生成并复制到docker/Files/"
        
    } catch {
        Print-Error "❌ 编译失败: $_"
        return $false
    } finally {
        Set-Location $currentDir
    }
    
    return $true
}

function Build-DockerImage {
    Print-Info "构建Docker镜像（包含Python环境）..."
    
    try {
        # 确保EvolveGCN文件存在于Docker构建上下文中
        Print-Info "复制EvolveGCN文件到Docker构建上下文..."
        Copy-Item "..\evolvegcn_go_interface.py" "." -Force
        Copy-Item "..\integrated_four_step_pipeline.py" "." -Force  
        Copy-Item "..\python_config.json" "." -Force
        
        # 强制重新构建，不使用缓存
        docker build --no-cache -t block-emulator:latest .
        if ($LASTEXITCODE -ne 0) { throw "Docker镜像构建失败" }
        
        Print-Success "✅ Docker镜像构建完成"
    } catch {
        Print-Error "❌ Docker镜像构建失败: $_"
        return $false
    }
    
    return $true
}

function Deploy-Containers {
    Print-Info "部署容器..."
    
    try {
        # 停止现有容器
        Print-Info "停止现有容器..."
        & ".\deploy_evolvegcn.ps1" stop
        
        # 启动新容器
        Print-Info "启动新容器..."
        & ".\deploy_evolvegcn.ps1" start
        if ($LASTEXITCODE -ne 0) { throw "容器启动失败" }
        
        Print-Success "✅ 容器部署完成"
    } catch {
        Print-Error "❌ 容器部署失败: $_"
        return $false
    }
    
    return $true
}

function Show-Usage {
    Write-Host "用法: .\rebuild_and_deploy.ps1 [Action]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Actions:" -ForegroundColor Cyan
    Write-Host "  full     - 完整流程：编译 → 构建 → 部署 (默认)" -ForegroundColor White
    Write-Host "  compile  - 仅编译Go代码" -ForegroundColor White
    Write-Host "  build    - 仅构建Docker镜像" -ForegroundColor White
    Write-Host "  deploy   - 仅部署容器" -ForegroundColor White
    Write-Host ""
    Write-Host "示例:" -ForegroundColor Yellow
    Write-Host "  .\rebuild_and_deploy.ps1              # 完整流程"
    Write-Host "  .\rebuild_and_deploy.ps1 compile      # 仅编译"
    Write-Host "  .\rebuild_and_deploy.ps1 build        # 仅构建镜像"
}

# 主逻辑
if ($Action -eq "help" -or $Action -eq "-h" -or $Action -eq "--help") {
    Show-Usage
    exit 0
}

# 检查前置条件
if (-not (Test-Prerequisites)) {
    Print-Error "前置条件检查失败，退出"
    exit 1
}

$success = $true

switch ($Action) {
    "compile" {
        $success = Compile-GoCode
    }
    "build" {
        $success = Build-DockerImage
    }
    "deploy" {
        $success = Deploy-Containers
    }
    "full" {
        Print-Info "执行完整构建和部署流程..."
        Write-Host ""
        
        Print-Info "=== 第1步：编译Go代码 ==="
        if (-not (Compile-GoCode)) { $success = $false }
        
        if ($success) {
            Write-Host ""
            Print-Info "=== 第2步：构建Docker镜像 ==="
            if (-not (Build-DockerImage)) { $success = $false }
        }
        
        if ($success) {
            Write-Host ""
            Print-Info "=== 第3步：部署容器 ==="
            if (-not (Deploy-Containers)) { $success = $false }
        }
    }
    default {
        Print-Error "未知操作: $Action"
        Show-Usage
        exit 1
    }
}

Write-Host ""
if ($success) {
    Print-Success "🎉 操作完成！"
    if ($Action -eq "full" -or $Action -eq "deploy") {
        Write-Host ""
        Print-Info "查看部署状态："
        Write-Host "  .\deploy_evolvegcn.ps1 status" -ForegroundColor Cyan
        Write-Host ""
        Print-Info "查看supervisor日志："
        Write-Host "  .\deploy_evolvegcn.ps1 logs supervisor" -ForegroundColor Cyan
    }
} else {
    Print-Error "❌ 操作失败！"
    exit 1
}
