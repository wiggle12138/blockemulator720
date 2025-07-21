@echo off
REM BlockEmulator UTF-8 版本启动脚本
REM 使用新的UTF-8编译版本：blockEmulator_Windows_UTF8.exe

REM 设置UTF-8环境
chcp 65001 >nul 2>&1
set LANG=en_US.UTF-8
set LC_ALL=en_US.UTF-8

echo [INFO] BlockEmulator 动态分片系统启动 (UTF-8版本)
echo ================================================

REM 检查可执行文件
set EXE_NAME=blockEmulator_Windows_UTF8.exe
if not exist "%EXE_NAME%" (
    echo [ERROR] 找不到UTF-8版本的可执行文件: %EXE_NAME%
    echo [INFO] 请先运行编译脚本: .\compile-utf8-simple.bat
    pause
    exit /b 1
)

echo [INFO] 使用可执行文件: %EXE_NAME%
echo [INFO] 启动配置: 2分片，每分片4节点 + 1个supervisor
echo.

REM 启动分片0的节点
echo [INFO] 启动分片0节点...
start "分片0-节点0" cmd /k "%EXE_NAME% -n 0 -N 4 -s 0 -S 2"
timeout /t 1 /nobreak >nul

start "分片0-节点1" cmd /k "%EXE_NAME% -n 1 -N 4 -s 0 -S 2"
timeout /t 1 /nobreak >nul

start "分片0-节点2" cmd /k "%EXE_NAME% -n 2 -N 4 -s 0 -S 2"
timeout /t 1 /nobreak >nul

start "分片0-节点3" cmd /k "%EXE_NAME% -n 3 -N 4 -s 0 -S 2"
timeout /t 2 /nobreak >nul

REM 启动分片1的节点
echo [INFO] 启动分片1节点...
start "分片1-节点0" cmd /k "%EXE_NAME% -n 0 -N 4 -s 1 -S 2"
timeout /t 1 /nobreak >nul

start "分片1-节点1" cmd /k "%EXE_NAME% -n 1 -N 4 -s 1 -S 2"
timeout /t 1 /nobreak >nul

start "分片1-节点2" cmd /k "%EXE_NAME% -n 2 -N 4 -s 1 -S 2"
timeout /t 1 /nobreak >nul

start "分片1-节点3" cmd /k "%EXE_NAME% -n 3 -N 4 -s 1 -S 2"
timeout /t 3 /nobreak >nul

REM 启动supervisor（最重要）
echo [INFO] 启动Supervisor (EvolveGCN控制器)...
start "Supervisor-EvolveGCN" cmd /k "%EXE_NAME% -c -N 4 -S 2"

echo.
echo [SUCCESS] 系统启动完成！
echo [INFO] 启动了以下组件:
echo   - 分片0: 4个节点
echo   - 分片1: 4个节点  
echo   - Supervisor: EvolveGCN动态分片控制器
echo.
echo [INFO] 监控提示:
echo   - 查看Supervisor窗口了解EvolveGCN分片算法运行状态
echo   - 各节点窗口显示共识和交易处理状态
echo   - 系统支持UTF-8编码，中文日志正常显示
echo.

pause
