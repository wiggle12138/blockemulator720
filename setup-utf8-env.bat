@echo off
REM UTF-8编码环境配置脚本
REM 用于确保BlockEmulator编译和运行环境的UTF-8编码一致性

echo [INFO] UTF-8编码环境配置工具
echo ====================================

REM 第一步：设置控制台代码页为UTF-8
echo [INFO] 设置控制台UTF-8代码页...
chcp 65001 >nul 2>&1

REM 验证代码页设置
for /f "tokens=2" %%i in ('chcp') do set CURRENT_CODEPAGE=%%i
if "%CURRENT_CODEPAGE%"=="65001" (
    echo [SUCCESS] 控制台代码页已设置为UTF-8: %CURRENT_CODEPAGE%
) else (
    echo [ERROR] 控制台代码页设置失败: %CURRENT_CODEPAGE%
    echo [INFO] 尝试强制设置UTF-8代码页...
    chcp 65001
    for /f "tokens=2" %%i in ('chcp') do set CURRENT_CODEPAGE=%%i
    if not "%CURRENT_CODEPAGE%"=="65001" (
        echo [WARNING] 无法设置UTF-8代码页，可能影响中文显示
    )
)

REM 第二步：设置系统级UTF-8环境变量
echo [INFO] 配置系统UTF-8环境变量...
set LANG=en_US.UTF-8
set LC_ALL=en_US.UTF-8
set LC_CTYPE=en_US.UTF-8
set LC_COLLATE=en_US.UTF-8
set LC_TIME=en_US.UTF-8
set LC_NUMERIC=en_US.UTF-8
set LC_MESSAGES=en_US.UTF-8
set LANGUAGE=en_US:en

REM 第三步：设置Go语言UTF-8编码相关环境变量
echo [INFO] 配置Go语言UTF-8编码环境...
set GO111MODULE=on
set GOPROXY=https://goproxy.cn,direct
set GOSUMDB=sum.golang.google.cn
set GOOS=windows
set GOARCH=amd64
set CGO_ENABLED=0

REM 设置Go编译缓存目录（确保路径支持UTF-8）
set GOMODCACHE=%USERPROFILE%\go\pkg\mod
set GOCACHE=%USERPROFILE%\.cache\go-build
set GOTMPDIR=%TEMP%\go-build-tmp

REM 创建临时目录（如果不存在）
if not exist "%GOTMPDIR%" mkdir "%GOTMPDIR%"

REM 第四步：验证UTF-8编码设置
echo [INFO] 验证UTF-8编码配置...
echo   控制台代码页: %CURRENT_CODEPAGE%
echo   系统语言环境: %LANG%
echo   字符集设置: %LC_ALL%
echo   Go模块代理: %GOPROXY%
echo   Go编译目标: %GOOS%/%GOARCH%

REM 第五步：测试UTF-8编码支持
echo [INFO] 测试UTF-8编码支持...
echo   测试字符: 中文UTF-8编码测试 ✓
echo   特殊字符: αβγδεζηθικλμνξοπρστυφχψω
echo   Emoji支持: 🎉📊🔧💻🌟

REM 验证Go环境
echo [INFO] 验证Go编译环境...
go version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Go未安装或不在PATH中
    echo [INFO] 请先安装Go语言环境
    pause
    exit /b 1
) else (
    for /f "tokens=*" %%i in ('go version') do echo [SUCCESS] Go环境: %%i
)

echo [SUCCESS] UTF-8编码环境配置完成！
echo [INFO] 环境变量已设置，可以开始编译BlockEmulator
echo.
