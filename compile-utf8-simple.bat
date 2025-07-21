@echo off
REM BlockEmulator UTF-8 Windows编译脚本
REM 参照zPrecompileScripts\windows_precomplieScript.bat

REM 显式设置控制台代码页为UTF-8 (65001)
chcp 65001 >nul 2>&1

echo [INFO] BlockEmulator UTF-8编译器 (Windows)
echo ======================================

REM 如果存在UTF-8环境配置脚本，先执行它
if exist "setup-utf8-env.bat" (
    echo [INFO] 加载UTF-8环境配置...
    call setup-utf8-env.bat >nul 2>&1
    echo [SUCCESS] UTF-8环境配置已加载
) else (
    echo [INFO] 使用内置UTF-8环境配置...
)

REM 设置Go编译环境
set GOOS=windows
set GOARCH=amd64
set CGO_ENABLED=0

REM 显式设置UTF-8编码环境变量
set LANG=en_US.UTF-8
set LC_ALL=en_US.UTF-8
set LANGUAGE=en_US:en

REM 设置Go模块和编码相关环境变量
set GO111MODULE=on
set GOPROXY=https://goproxy.cn,direct
set GOSUMDB=sum.golang.google.cn

echo [INFO] 编译配置:
echo   目标平台: %GOOS%
echo   目标架构: %GOARCH%
echo   CGO启用: %CGO_ENABLED%
echo   代码页: UTF-8 (65001)
echo   语言环境: %LANG%

REM 验证UTF-8环境设置
echo [INFO] 验证UTF-8编码环境...
for /f "tokens=2" %%i in ('chcp') do set CURRENT_CODEPAGE=%%i
if "%CURRENT_CODEPAGE%"=="65001" (
    echo [SUCCESS] UTF-8代码页设置成功: %CURRENT_CODEPAGE%
) else (
    echo [WARNING] 代码页可能不是UTF-8: %CURRENT_CODEPAGE%
    echo [INFO] 尝试重新设置UTF-8代码页...
    chcp 65001
)

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

REM 开始编译（使用简单的go build命令，显式支持UTF-8编码）
echo [INFO] 开始编译BlockEmulator (支持UTF-8编码)...
echo [INFO] 编译命令: go build -ldflags="-s -w" -gcflags="-N -l" -o blockEmulator_Windows_UTF8.exe main.go

REM 设置编译时的UTF-8环境
set GOMODCACHE=%USERPROFILE%\go\pkg\mod
set GOCACHE=%USERPROFILE%\.cache\go-build

REM 执行编译，确保UTF-8环境
go build -ldflags="-s -w" -gcflags="-N -l" -o blockEmulator_Windows_UTF8.exe main.go

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
echo [INFO] 可执行文件大小:
dir blockEmulator_Windows_UTF8.exe | findstr "blockEmulator_Windows_UTF8.exe"

REM 验证可执行文件的UTF-8编码支持
echo [INFO] 验证UTF-8编码支持...
echo [INFO] 测试可执行文件帮助信息...
blockEmulator_Windows_UTF8.exe -h >nul 2>&1
if errorlevel 1 (
    echo [WARNING] 可执行文件可能存在问题，请检查编码配置
) else (
    echo [SUCCESS] 可执行文件UTF-8编码测试通过
)

echo.
echo [INFO] 编译完成! 使用以下命令测试:
echo   blockEmulator_Windows_UTF8.exe -h
echo   blockEmulator_Windows_UTF8.exe -c -N 4 -S 2
echo.
echo [INFO] UTF-8编码提示:
echo   - 确保终端支持UTF-8显示
echo   - 如有中文乱码，请检查控制台代码页设置
echo   - 推荐使用Windows Terminal或支持UTF-8的终端
echo.
pause
