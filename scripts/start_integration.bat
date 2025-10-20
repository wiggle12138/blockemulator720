@echo off
echo BlockEmulator EvolveGCN & Feedback 集成系统启动器
echo =============================================

REM 检查Python环境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: Python未安装或不在PATH中
    pause
    exit /b 1
)

REM 创建必要的目录
if not exist "data_exchange" mkdir data_exchange
if not exist "logs" mkdir logs

echo.
echo 请选择运行模式:
echo 1. 运行完整集成测试
echo 2. 运行单次模式
echo 3. 运行连续模式
echo 4. 运行系统测试
echo 5. 退出
echo.

set /p choice=请输入选择 (1-5): 

if "%choice%"=="1" (
    echo 运行完整集成测试...
    python test_integration.py
) else if "%choice%"=="2" (
    echo 运行单次模式...
    python integration_complete.py --mode single --config integration_config.json
) else if "%choice%"=="3" (
    echo 运行连续模式...
    python integration_complete.py --mode continuous --config integration_config.json
) else if "%choice%"=="4" (
    echo 运行系统测试...
    python test_integration.py
) else if "%choice%"=="5" (
    echo 退出...
    exit /b 0
) else (
    echo 无效选择，请重新运行
    pause
    exit /b 1
)

echo.
echo 操作完成！
pause
