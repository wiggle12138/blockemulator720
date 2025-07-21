@echo off
REM BlockEmulator UTF-8 系统管理脚本

REM 设置UTF-8环境
chcp 65001 >nul 2>&1

:MENU
cls
echo.
echo ================================================
echo    BlockEmulator 动态分片系统管理 (UTF-8版本)
echo ================================================
echo.
echo 请选择操作:
echo.
echo 1. 编译系统 (UTF-8版本)
echo 2. 启动系统 (2分片 x 4节点 + supervisor)
echo 3. 检查系统状态
echo 4. 停止系统
echo 5. 测试UTF-8编码
echo 6. 运行数据接口测试
echo 7. 清理系统日志
echo 0. 退出
echo.
set /p choice=请输入选项 (0-7): 

if "%choice%"=="1" goto COMPILE
if "%choice%"=="2" goto START
if "%choice%"=="3" goto STATUS
if "%choice%"=="4" goto STOP
if "%choice%"=="5" goto TEST_UTF8
if "%choice%"=="6" goto TEST_DATA
if "%choice%"=="7" goto CLEANUP
if "%choice%"=="0" goto EXIT
echo 无效选项，请重试...
timeout /t 2 /nobreak >nul
goto MENU

:COMPILE
cls
echo [编译] 开始编译BlockEmulator UTF-8版本...
call compile-utf8-simple.bat
pause
goto MENU

:START
cls
echo [启动] 启动BlockEmulator系统...
call start-blockemulator-utf8.bat
pause
goto MENU

:STATUS
cls
call status-blockemulator.bat
goto MENU

:STOP
cls
echo [停止] 停止BlockEmulator系统...
call stop-blockemulator.bat
goto MENU

:TEST_UTF8
cls
echo [测试] UTF-8编码支持测试...
call test-utf8-encoding.bat
goto MENU

:TEST_DATA
cls
echo [测试] 数据接口对齐测试...
python test_data_interface_alignment.py
pause
goto MENU

:CLEANUP
cls
echo [清理] 清理系统日志和临时文件...
if exist "expTest\" (
    echo 清理实验数据...
    rmdir /s /q "expTest" 2>nul
)
if exist "outputs\" (
    echo 清理输出文件...
    del /q "outputs\*" 2>nul
)
if exist "data_exchange\" (
    echo 清理数据交换文件...
    del /q "data_exchange\*" 2>nul
)
echo [SUCCESS] 清理完成
pause
goto MENU

:EXIT
echo 退出系统管理...
exit /b 0
