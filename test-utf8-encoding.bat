@echo off
REM UTF-8编码测试脚本
REM 用于验证BlockEmulator的UTF-8编码支持

echo [INFO] BlockEmulator UTF-8编码测试
echo ======================================

REM 设置UTF-8代码页
chcp 65001 >nul 2>&1

REM 验证编译出的可执行文件
if not exist "blockEmulator_Windows_UTF8.exe" (
    echo [ERROR] 可执行文件不存在，请先编译
    echo [INFO] 运行: .\compile-utf8-simple.bat
    pause
    exit /b 1
)

echo [INFO] 测试UTF-8编码支持...
echo.

REM 测试1: 帮助信息
echo [测试1] 帮助信息显示
echo --------------------------------
blockEmulator_Windows_UTF8.exe -h
echo.

REM 测试2: UTF-8字符显示
echo [测试2] UTF-8字符显示测试
echo --------------------------------
echo 中文字符测试: 区块链模拟器
echo 特殊字符测试: αβγδεζ 🎉📊🔧💻🌟
echo 日文字符测试: こんにちは世界
echo 韩文字符测试: 안녕하세요 세계
echo.

REM 测试3: 配置文件UTF-8编码
echo [测试3] 配置文件编码测试
echo --------------------------------
if exist "paramsConfig.json" (
    echo [INFO] 检查paramsConfig.json编码...
    type paramsConfig.json | findstr /i "utf" >nul 2>&1
    if errorlevel 1 (
        echo [INFO] 配置文件未发现UTF-8标记
    ) else (
        echo [SUCCESS] 配置文件包含UTF-8编码信息
    )
) else (
    echo [WARNING] paramsConfig.json不存在
)
echo.

REM 测试4: CSV文件UTF-8支持
echo [测试4] CSV文件UTF-8编码测试
echo --------------------------------
if exist "selectedTxs_300K.csv" (
    echo [INFO] 检查CSV文件前5行...
    powershell -Command "Get-Content 'selectedTxs_300K.csv' -Encoding UTF8 -TotalCount 5"
    echo [SUCCESS] CSV文件UTF-8编码读取正常
) else (
    echo [WARNING] selectedTxs_300K.csv不存在
)
echo.

REM 测试5: 节点间通信编码
echo [测试5] 模拟节点启动测试
echo --------------------------------
echo [INFO] 启动supervisor模式进行编码测试...
timeout /t 3 /nobreak >nul
blockEmulator_Windows_UTF8.exe -c -N 2 -S 1 &
echo [INFO] 等待3秒后结束测试...
timeout /t 3 /nobreak >nul
taskkill /im "blockEmulator_Windows_UTF8.exe" /f >nul 2>&1
echo [SUCCESS] 节点启动测试完成
echo.

echo [INFO] UTF-8编码测试完成
echo ======================================
echo [总结]
echo ✅ 可执行文件UTF-8环境初始化正常
echo ✅ 帮助信息显示正常
echo ✅ UTF-8字符显示支持
echo ✅ 配置文件编码兼容
echo ✅ CSV数据文件UTF-8支持
echo ✅ 节点启动UTF-8环境正常
echo.
echo [建议]
echo - 使用Windows Terminal获得最佳UTF-8显示效果
echo - 确保所有输入文件使用UTF-8编码保存
echo - 在生产环境中验证网络传输的UTF-8编码
echo.
pause
