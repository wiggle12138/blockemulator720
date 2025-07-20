@echo off
echo Starting BlockEmulator with Python Integration...

REM 检查Python环境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found. Please install Python and add it to PATH.
    pause
    exit /b 1
)

REM 安装必要的Python依赖
echo Installing Python dependencies...
pip install torch numpy scikit-learn matplotlib pandas tqdm

REM 创建数据交换目录
if not exist "data_exchange" mkdir data_exchange

REM 启动 BlockEmulator 主程序（带Python集成）
echo Starting BlockEmulator with Python integration...
go run main.go -c -N 4 -S 2 -p

echo BlockEmulator with Python Integration started!
pause
