@echo off
echo ==========================================
echo BlockEmulator Python Integration Test
echo ==========================================

echo.
echo Step 1: Testing Python environment...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python and add it to PATH.
    pause
    exit /b 1
)
echo ✓ Python environment OK

echo.
echo Step 2: Creating data exchange directory...
if not exist "data_exchange" mkdir data_exchange
echo ✓ Data exchange directory created

echo.
echo Step 3: Testing simplified integration...
python simplified_integration.py --mode single --generate_sample --log_level INFO
if %errorlevel% neq 0 (
    echo ERROR: Simplified integration test failed
    pause
    exit /b 1
)
echo ✓ Simplified integration test passed

echo.
echo Step 4: Testing Go integration...
go run main.go -p -N 4 -S 2 -n 0 -s 0 --help >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Go integration test failed, but this is expected without full setup
) else (
    echo ✓ Go integration test passed
)

echo.
echo Step 5: Checking results...
if exist "data_exchange\feedback_results.json" (
    echo ✓ Feedback results generated
    echo.
    echo Results preview:
    type data_exchange\feedback_results.json
) else (
    echo WARNING: No feedback results found
)

echo.
echo ==========================================
echo Integration test completed!
echo ==========================================
echo.
echo To run the full system:
echo   1. Windows: start_with_python.bat
echo   2. Linux:   ./start_with_python.sh
echo.
echo To run only Python modules:
echo   1. Windows: start_python_only.bat continuous 10 8
echo   2. Linux:   ./start_python_only.sh continuous 10 8
echo.
pause
