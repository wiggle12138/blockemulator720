@echo off
echo Starting Python Integration Modules Only...

REM 检查Python环境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found. Please install Python and add it to PATH.
    pause
    exit /b 1
)

REM 创建数据交换目录
if not exist "data_exchange" mkdir data_exchange

REM 启动Python集成模块
echo Starting Python integration modules...
python integrated_test.py --mode %1 --max_iterations %2 --epochs_per_iteration %3

pause
