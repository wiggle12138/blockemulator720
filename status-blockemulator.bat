@echo off
REM BlockEmulator 系统状态检查脚本

REM 设置UTF-8环境
chcp 65001 >nul 2>&1

echo [INFO] BlockEmulator 系统状态检查
echo ===================================

REM 检查可执行文件
echo [检查] 可执行文件状态:
if exist "blockEmulator_Windows_UTF8.exe" (
    echo ✅ blockEmulator_Windows_UTF8.exe - 存在
) else (
    echo ❌ blockEmulator_Windows_UTF8.exe - 不存在（需要编译）
)

if exist "blockEmulator.exe" (
    echo ✅ blockEmulator.exe - 存在（备用版本）
) else (
    echo ⚠️ blockEmulator.exe - 不存在
)

echo.

REM 检查运行中的进程
echo [检查] 运行中的进程:
set PROCESS_COUNT=0
for /f %%i in ('tasklist /fi "imagename eq blockEmulator_Windows_UTF8.exe" ^| find /c "blockEmulator_Windows_UTF8.exe"') do set PROCESS_COUNT=%%i

if %PROCESS_COUNT% gtr 0 (
    echo ✅ 发现 %PROCESS_COUNT% 个 blockEmulator_Windows_UTF8.exe 进程
    echo [详细信息]:
    tasklist /fi "imagename eq blockEmulator_Windows_UTF8.exe"
) else (
    echo ⚠️ 没有发现运行中的 blockEmulator_Windows_UTF8.exe 进程
)

REM 检查其他版本进程
for /f %%i in ('tasklist /fi "imagename eq blockEmulator.exe" ^| find /c "blockEmulator.exe"') do (
    if %%i gtr 0 (
        echo ✅ 发现 %%i 个 blockEmulator.exe 进程（旧版本）
    )
)

echo.

REM 检查配置文件
echo [检查] 配置文件:
if exist "paramsConfig.json" (
    echo ✅ paramsConfig.json - 存在
) else (
    echo ❌ paramsConfig.json - 不存在
)

if exist "ipTable.json" (
    echo ✅ ipTable.json - 存在
) else (
    echo ❌ ipTable.json - 不存在
)

echo.

REM 检查数据文件
echo [检查] 数据文件:
if exist "selectedTxs_300K.csv" (
    echo ✅ selectedTxs_300K.csv - 存在
) else (
    echo ❌ selectedTxs_300K.csv - 不存在
)

echo.

REM 检查Python环境
echo [检查] Python环境:
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python - 不可用
) else (
    for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo ✅ Python - %%i
)

REM 检查虚拟环境
if exist ".venv\Scripts\python.exe" (
    echo ✅ 虚拟环境 (.venv) - 存在
) else (
    echo ⚠️ 虚拟环境 (.venv) - 不存在
)

echo.

REM 网络连接检查
echo [检查] 网络端口使用情况:
netstat -an | findstr ":32216\|:32217\|:32218" >nul 2>&1
if errorlevel 1 (
    echo ⚠️ 没有发现BlockEmulator相关端口活动
) else (
    echo ✅ 发现BlockEmulator相关端口活动:
    netstat -an | findstr ":32216\|:32217\|:32218"
)

echo.
echo [总结] 系统状态检查完成
echo ===================================

if %PROCESS_COUNT% gtr 0 (
    echo 🟢 系统状态: 运行中
    echo 📊 活跃进程: %PROCESS_COUNT% 个
    echo 🔧 建议: 系统正常运行，可通过各节点窗口监控状态
) else (
    echo 🟡 系统状态: 未运行
    echo 🚀 建议: 使用 start-blockemulator-utf8.bat 启动系统
)

echo.
pause
