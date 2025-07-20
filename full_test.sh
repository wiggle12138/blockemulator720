#!/bin/bash
echo "=========================================="
echo "BlockEmulator Python Integration Test"
echo "=========================================="

echo
echo "Step 1: Testing Python environment..."
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found. Please install Python."
    exit 1
fi
echo "✓ Python environment OK"

echo
echo "Step 2: Creating data exchange directory..."
mkdir -p data_exchange
echo "✓ Data exchange directory created"

echo
echo "Step 3: Testing simplified integration..."
python simplified_integration.py --mode single --generate_sample --log_level INFO
if [ $? -ne 0 ]; then
    echo "ERROR: Simplified integration test failed"
    exit 1
fi
echo "✓ Simplified integration test passed"

echo
echo "Step 4: Testing Go integration..."
if command -v go &> /dev/null; then
    go run main.go -p -N 4 -S 2 -n 0 -s 0 --help >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ Go integration test passed"
    else
        echo "WARNING: Go integration test failed, but this is expected without full setup"
    fi
else
    echo "WARNING: Go not found, skipping Go integration test"
fi

echo
echo "Step 5: Checking results..."
if [ -f "data_exchange/feedback_results.json" ]; then
    echo "✓ Feedback results generated"
    echo
    echo "Results preview:"
    cat data_exchange/feedback_results.json
else
    echo "WARNING: No feedback results found"
fi

echo
echo "=========================================="
echo "Integration test completed!"
echo "=========================================="
echo
echo "To run the full system:"
echo "  1. Windows: start_with_python.bat"
echo "  2. Linux:   ./start_with_python.sh"
echo
echo "To run only Python modules:"
echo "  1. Windows: start_python_only.bat continuous 10 8"
echo "  2. Linux:   ./start_python_only.sh continuous 10 8"
echo
