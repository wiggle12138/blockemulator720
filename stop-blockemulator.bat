@echo off
REM BlockEmulator 系统停止脚本

REM 设置UTF-8环境
chcp 65001 >nul 2>&1

echo [INFO] 停止BlockEmulator系统...
echo ================================

REM 停止所有BlockEmulator进程
taskkill /im "blockEmulator_Windows_UTF8.exe" /f >nul 2>&1
taskkill /im "blockEmulator.exe" /f >nul 2>&1
taskkill /im "blockEmulator_Windows_Precompile.exe" /f >nul 2>&1

REM 等待进程完全终止
timeout /t 2 /nobreak >nul

REM 检查是否还有残留进程
tasklist | findstr "blockEmulator" >nul 2>&1
if errorlevel 1 (
    echo [SUCCESS] 所有BlockEmulator进程已停止
) else (
    echo [WARNING] 仍有BlockEmulator进程在运行
    echo [INFO] 请手动检查任务管理器
)

REM 停止相关的Python进程（如果有）
taskkill /im "python.exe" /fi "WINDOWTITLE eq *evolvegcn*" /f >nul 2>&1

echo [INFO] 清理完成
pause
