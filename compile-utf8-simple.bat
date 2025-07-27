chcp 65001 >nul
@echo off
REM BlockEmulator UTF-8 Windows编译脚本
REM 参照zPrecompileScripts\windows_precomplieScript.bat
echo [INFO] BlockEmulator UTF-8编译器 (Windows)
echo ======================================

REM 设置Go编译环境
set GOOS=windows
set GOARCH=amd64
set CGO_ENABLED=0

REM 设置UTF-8编码环境变量
set LANG=en_US.UTF-8
set LC_ALL=en_US.UTF-8

echo [INFO] 编译配置:
echo   目标平台: %GOOS%
echo   目标架构: %GOARCH%
echo   CGO启用: %CGO_ENABLED%

REM 检查Go环境
echo [INFO] 检查Go编译环境...
go version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Go未安装或不在PATH中
    pause
    exit /b 1
)

REM 显示Go版本
for /f "tokens=*" %%i in ('go version') do echo [SUCCESS] Go环境检查通过: %%i

REM 清理旧文件（如果存在）
if exist "blockEmulator_Windows_UTF8.exe" (
    echo [INFO] 清理旧的编译文件...
    del "blockEmulator_Windows_UTF8.exe"
)

REM 开始编译（使用简单的go build命令，类似原始脚本）
echo [INFO] 开始编译BlockEmulator (支持UTF-8编码)...
go build -o blockEmulator_Windows_UTF8.exe main.go

REM 检查编译结果
if errorlevel 1 (
    echo [ERROR] 编译失败
    pause
    exit /b 1
)

if not exist "blockEmulator_Windows_UTF8.exe" (
    echo [ERROR] 编译成功但未找到输出文件
    pause
    exit /b 1
)

echo [SUCCESS] 编译完成: blockEmulator_Windows_UTF8.exe
echo [INFO] 文件大小: 
for %%F in (blockEmulator_Windows_UTF8.exe) do echo   %%~zF bytes

REM 创建UTF-8版本的软链接
echo [INFO] 创建UTF-8版本别名...
if exist "blockEmulator.exe" del "blockEmulator.exe"
copy "blockEmulator_Windows_UTF8.exe" "blockEmulator.exe" >nul

echo [SUCCESS] UTF-8编译完成！
echo [INFO] 现在可以使用以下命令测试UTF-8支持:
echo   blockEmulator_Windows_UTF8.exe -help
echo   blockEmulator.exe -help
pause
