#!/bin/bash
echo "======================================================================"
echo "          CREDIT CARD DEFAULT PREDICTOR - LAUNCHER"
echo "                AI Application Lab - Chandigarh University"
echo "======================================================================"
echo ""

# Find Python 3
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v py &>/dev/null; then
    PYTHON_CMD="py -3.10"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python was not found. Please install Python 3.10+."
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version)"

# Check dependencies
$PYTHON_CMD -c "import streamlit, xgboost, fpdf, pptx" &>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required dependencies..."
    $PYTHON_CMD -m pip install -r requirements.txt
fi

# Check models
if [ ! -f "models/xgboost.joblib" ]; then
    echo "Training ML models..."
    $PYTHON_CMD model.py
fi

echo ""
echo "Launching Streamlit application at http://localhost:8501..."
$PYTHON_CMD -m streamlit run app.py --browser.gatherUsageStats false

echo ""
read -p "Press Enter to exit..."
