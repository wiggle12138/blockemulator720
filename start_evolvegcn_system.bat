@echo off
setlocal EnableDelayedExpansion

echo =======================================================
echo      EvolveGCN BlockEmulator 集成启动脚本
echo =======================================================
echo.

:: 检查Python环境
echo [1/5] 检查Python环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python未安装或未添加到PATH
    echo 💡 请安装Python 3.8+并添加到系统PATH
    pause
    exit /b 1
)
echo ✅ Python环境正常

:: 检查Python依赖
echo.
echo [2/5] 检查Python依赖...
python -c "import torch, numpy, pandas" >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Python依赖缺失，正在安装...
    pip install torch numpy pandas scikit-learn matplotlib seaborn
    if !errorlevel! neq 0 (
        echo ❌ 依赖安装失败
        pause
        exit /b 1
    )
)
echo ✅ Python依赖完整

:: 测试EvolveGCN接口
echo.
echo [3/5] 测试EvolveGCN接口...
python test_evolvegcn_integration.py
if %errorlevel% neq 0 (
    echo ❌ EvolveGCN接口测试失败
    echo 💡 请检查Python脚本是否正确配置
    pause
    exit /b 1
)
echo ✅ EvolveGCN接口测试通过

:: 检查BlockEmulator可执行文件
echo.
echo [4/5] 检查BlockEmulator可执行文件...
if exist "blockEmulator_Windows_Precompile.exe" (
    echo ✅ 找到预编译版本
    copy /Y "blockEmulator_Windows_Precompile.exe" "blockEmulator.exe" >nul
) else if exist "blockEmulator.exe" (
    echo ✅ 找到可执行文件
) else (
    echo ⚠️  未找到可执行文件，尝试编译...
    go build -o blockEmulator.exe main.go
    if !errorlevel! neq 0 (
        echo ❌ 编译失败，请检查Go环境
        pause
        exit /b 1
    )
    echo ✅ 编译成功
)

:: 启动系统
echo.
echo [5/5] 启动EvolveGCN BlockEmulator系统...
echo.
echo 🚀 系统启动中，请监控以下关键日志：
echo    - EvolveGCN Epoch X: Triggering comprehensive node feature collection...
echo    - EvolveGCN: Starting four-step partition pipeline...
echo    - EvolveGCN: Python pipeline completed, processed X nodes with X cross-shard edges
echo.
echo 💡 按 Ctrl+C 停止系统
echo =======================================================
echo.

:: 启动主程序
blockEmulator.exe

echo.
echo =======================================================
echo 系统已停止
echo =======================================================
pause
