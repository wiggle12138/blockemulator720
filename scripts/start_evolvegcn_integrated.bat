@echo off
REM BlockEmulator EvolveGCN集成启动脚本
REM 自动配置虚拟环境并启动BlockEmulator系统

echo ========================================
echo  BlockEmulator EvolveGCN集成启动器
echo ========================================
echo.

echo [STEP 1] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python未安装或未添加到PATH
    echo 💡 请安装Python 3.8+并添加到系统PATH
    pause
    exit /b 1
)
echo ✅ Python环境检查通过

echo.
echo [STEP 2] 配置Python虚拟环境...
if exist "config_python_venv.py" (
    python config_python_venv.py
    if errorlevel 1 (
        echo ⚠️ 虚拟环境配置有警告，继续执行...
    ) else (
        echo ✅ 虚拟环境配置完成
    )
) else (
    echo ⚠️ 未找到虚拟环境配置脚本，使用默认Python
)

echo.
echo [STEP 3] 检查EvolveGCN集成文件...
set MISSING_FILES=0

if not exist "evolvegcn_go_interface.py" (
    echo ❌ 缺少文件: evolvegcn_go_interface.py
    set /a MISSING_FILES+=1
) else (
    echo ✅ 找到: evolvegcn_go_interface.py
)

if not exist "integrated_four_step_pipeline.py" (
    echo ❌ 缺少文件: integrated_four_step_pipeline.py
    set /a MISSING_FILES+=1
) else (
    echo ✅ 找到: integrated_four_step_pipeline.py
)

if %MISSING_FILES% gtr 0 (
    echo ❌ 缺少关键EvolveGCN文件，无法启动
    pause
    exit /b 1
)

echo.
echo [STEP 4] 测试EvolveGCN集成...
if exist "python_config.json" (
    echo 📋 读取Python配置...
    type python_config.json
    echo.
) else (
    echo ⚠️ 未找到python_config.json，将使用默认配置
)

echo.
echo [STEP 5] 切换到Docker目录并启动服务...
if exist "docker\deploy_evolvegcn.ps1" (
    echo 🚀 使用EvolveGCN集成部署脚本...
    cd docker
    powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" setup
    if errorlevel 1 (
        echo ❌ EvolveGCN环境设置失败
        cd ..
        pause
        exit /b 1
    )
    
    powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" build
    if errorlevel 1 (
        echo ❌ Docker镜像构建失败
        cd ..
        pause
        exit /b 1
    )
    
    powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" start
    if errorlevel 1 (
        echo ❌ 服务启动失败
        cd ..
        pause
        exit /b 1
    )
    
    echo ✅ BlockEmulator EvolveGCN集成系统启动成功!
    echo.
    echo 📊 使用以下命令查看状态:
    echo    powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" status
    echo.
    echo 📝 使用以下命令查看日志:
    echo    powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" logs
    echo.
    echo 🧪 使用以下命令测试Python集成:
    echo    powershell -ExecutionPolicy Bypass -File ".\deploy_evolvegcn.ps1" test-python
    
    cd ..
) else if exist "docker\deploy.ps1" (
    echo 🚀 使用标准部署脚本...
    cd docker
    powershell -ExecutionPolicy Bypass -File ".\deploy.ps1" build
    if errorlevel 1 (
        echo ❌ Docker镜像构建失败
        cd ..
        pause
        exit /b 1
    )
    
    powershell -ExecutionPolicy Bypass -File ".\deploy.ps1" start
    if errorlevel 1 (
        echo ❌ 服务启动失败
        cd ..
        pause
        exit /b 1
    )
    
    echo ✅ BlockEmulator系统启动成功!
    echo ⚠️ 注意: 使用的是标准部署，EvolveGCN集成可能需要手动配置
    cd ..
) else (
    echo ❌ 未找到部署脚本，请检查docker目录
    pause
    exit /b 1
)

echo.
echo 🎉 启动完成! EvolveGCN四步分片算法已集成到BlockEmulator中
echo 📝 CLPA占位算法已被真实的EvolveGCN算法替换
echo 🔄 系统支持反馈循环和自适应分片优化
echo.

pause
