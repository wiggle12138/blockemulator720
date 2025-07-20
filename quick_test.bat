@echo off
echo ==========================================
echo 动态分片系统快速检测工具
echo ==========================================

echo.
echo 🔍 检查Python环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python环境未找到，请安装Python
    pause
    exit /b 1
)
echo ✅ Python环境正常

echo.
echo 🔧 检查依赖库...
python -c "import torch; print('✅ PyTorch:', torch.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo ⚠️  PyTorch未安装，尝试安装...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

python -c "import numpy; print('✅ NumPy:', numpy.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo ⚠️  NumPy未安装，尝试安装...
    pip install numpy
)

echo.
echo 🚀 开始快速系统检测...
python quick_sharding_test.py

if %errorlevel% equ 0 (
    echo.
    echo =========================================
    echo ✅ 快速检测完成！系统基本可用
    echo =========================================
    echo.
    echo 📋 下一步操作建议：
    echo.
    echo 1. 完整功能测试：
    echo    python test_iterative_sharding_feedback.py
    echo.
    echo 2. 系统集成测试：
    echo    python simplified_integration_fixed.py --mode single --generate_sample
    echo.
    echo 3. 性能基准测试：
    echo    python run_enhanced_pipeline.py
    echo.
    echo 4. 查看测试结果：
    echo    type data_exchange\quick_test_results.json
    echo.
) else (
    echo.
    echo =========================================
    echo ❌ 快速检测发现问题
    echo =========================================
    echo.
    echo 请检查以下可能的问题：
    echo 1. Python依赖库是否完整安装
    echo 2. 系统目录结构是否正确
    echo 3. 查看详细错误信息进行调试
    echo.
)

echo 查看完整测试指南: type optimized_testing_guide.md
echo.
pause
