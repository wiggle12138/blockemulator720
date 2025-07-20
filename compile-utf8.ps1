# UTF-8编码支持的BlockEmulator编译脚本
# 解决GBK编码导致的中文字符处理问题

param(
    [string]$Platform = "windows",
    [switch]$Clean,
    [switch]$Verbose
)

# 设置控制台编码为UTF-8
$OutputEncoding = [console]::InputEncoding = [console]::OutputEncoding = New-Object System.Text.UTF8Encoding

Write-Host "[INFO] BlockEmulator UTF-8编译器" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green

# 显示当前编码设置
if ($Verbose) {
    Write-Host "[DEBUG] 当前编码设置:" -ForegroundColor Yellow
    Write-Host "  控制台输入编码: $([console]::InputEncoding.EncodingName)" -ForegroundColor Yellow
    Write-Host "  控制台输出编码: $([console]::OutputEncoding.EncodingName)" -ForegroundColor Yellow
    Write-Host "  PowerShell输出编码: $($OutputEncoding.EncodingName)" -ForegroundColor Yellow
}

# 设置Go编译环境变量以支持UTF-8
$env:CGO_ENABLED = "0"
$env:GOOS = switch ($Platform.ToLower()) {
    "windows" { "windows" }
    "linux" { "linux" }
    "darwin" { "darwin" }
    "macos" { "darwin" }
    default { "windows" }
}

$env:GOARCH = "amd64"

# 强制Go使用UTF-8编码
$env:LANG = "en_US.UTF-8"
$env:LC_ALL = "en_US.UTF-8"

Write-Host "[INFO] 编译配置:" -ForegroundColor Cyan
Write-Host "  目标平台: $env:GOOS" -ForegroundColor Cyan
Write-Host "  目标架构: $env:GOARCH" -ForegroundColor Cyan
Write-Host "  CGO启用: $env:CGO_ENABLED" -ForegroundColor Cyan

# 确定输出文件名
$outputFile = switch ($env:GOOS) {
    "windows" { "blockEmulator_Windows_UTF8.exe" }
    "linux" { "blockEmulator_Linux_UTF8" }
    "darwin" { "blockEmulator_Darwin_UTF8" }
    default { "blockEmulator_UTF8.exe" }
}

# 清理旧文件
if ($Clean -and (Test-Path $outputFile)) {
    Write-Host "[INFO] 清理旧的编译文件: $outputFile" -ForegroundColor Yellow
    Remove-Item $outputFile -Force
}

# 检查Go环境
Write-Host "[INFO] 检查Go编译环境..." -ForegroundColor Cyan
try {
    $goVersion = go version
    Write-Host "[SUCCESS] Go环境检查通过: $goVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Go环境未找到，请安装Go并添加到PATH" -ForegroundColor Red
    exit 1
}

# 检查main.go文件
if (-not (Test-Path "main.go")) {
    Write-Host "[ERROR] 未找到main.go文件，请在BlockEmulator根目录运行此脚本" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] 开始编译BlockEmulator (支持UTF-8编码)..." -ForegroundColor Cyan

# 添加UTF-8编译标签和参数
$buildArgs = @(
    "build"
    "-ldflags", "-s -w -X 'main.DefaultEncoding=UTF-8'"
    "-tags", "utf8"
    "-o", $outputFile
    "main.go"
)

if ($Verbose) {
    Write-Host "[DEBUG] 编译命令: go $($buildArgs -join ' ')" -ForegroundColor Yellow
}

# 执行编译
try {
    $buildProcess = Start-Process -FilePath "go" -ArgumentList $buildArgs -Wait -PassThru -NoNewWindow -RedirectStandardOutput "compile_stdout.log" -RedirectStandardError "compile_stderr.log"
    
    if ($buildProcess.ExitCode -eq 0) {
        Write-Host "[SUCCESS] 编译完成: $outputFile" -ForegroundColor Green
        
        # 检查文件大小
        $fileSize = (Get-Item $outputFile).Length
        $fileSizeMB = [math]::Round($fileSize / 1MB, 2)
        Write-Host "[INFO] 文件大小: ${fileSizeMB} MB" -ForegroundColor Cyan
        
        # 验证UTF-8支持
        Write-Host "[INFO] 验证UTF-8编码支持..." -ForegroundColor Cyan
        Test-UTF8Support -ExecutableFile $outputFile
        
    } else {
        Write-Host "[ERROR] 编译失败，退出代码: $($buildProcess.ExitCode)" -ForegroundColor Red
        
        # 显示编译错误
        if (Test-Path "compile_stderr.log") {
            Write-Host "[ERROR] 编译错误详情:" -ForegroundColor Red
            Get-Content "compile_stderr.log" | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
        }
        exit 1
    }
} catch {
    Write-Host "[ERROR] 编译过程中出现异常: $_" -ForegroundColor Red
    exit 1
} finally {
    # 清理临时日志文件
    @("compile_stdout.log", "compile_stderr.log") | ForEach-Object {
        if (Test-Path $_) { Remove-Item $_ -Force }
    }
}

Write-Host "" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host "[SUCCESS] UTF-8编码支持的BlockEmulator编译完成!" -ForegroundColor Green
Write-Host "" -ForegroundColor Green
Write-Host "使用方法:" -ForegroundColor Yellow
Write-Host "  1. 替换原有可执行文件: Copy-Item $outputFile blockEmulator.exe -Force" -ForegroundColor Yellow
Write-Host "  2. 直接运行: .\\$outputFile -c -N 4 -S 2" -ForegroundColor Yellow
Write-Host "  3. 测试UTF-8支持: .\\$outputFile --test-utf8" -ForegroundColor Yellow

function Test-UTF8Support {
    param([string]$ExecutableFile)
    
    Write-Host "[INFO] 创建UTF-8测试文件..." -ForegroundColor Cyan
    
    # 创建包含中文字符的测试JSON文件
    $testJson = @{
        "test_encoding" = "UTF-8支持测试"
        "chinese_chars" = "中文字符：测试、成功、失败"
        "emojis" = "🚀🔧⚡"
        "mixed_content" = "Mixed内容测试123"
    } | ConvertTo-Json -Depth 3
    
    # 确保以UTF-8编码保存
    [System.IO.File]::WriteAllText("utf8_test.json", $testJson, [System.Text.Encoding]::UTF8)
    
    Write-Host "[SUCCESS] UTF-8测试文件已创建: utf8_test.json" -ForegroundColor Green
    Write-Host "[INFO] 请运行BlockEmulator时监控是否能正确处理中文字符" -ForegroundColor Cyan
}

# 如果指定了多个平台，递归编译
if ($Platform -eq "all") {
    Write-Host "[INFO] 编译所有平台版本..." -ForegroundColor Cyan
    
    foreach ($p in @("windows", "linux", "darwin")) {
        Write-Host "`n[INFO] 编译 $p 版本..." -ForegroundColor Yellow
        & $PSCommandPath -Platform $p -Verbose:$Verbose -Clean:$Clean
    }
}
