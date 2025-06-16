#!/bin/bash
# Quick Chrome installation script for MiniWob++

set -e

echo "Installing Chrome/Chromium for MiniWob++..."
echo "=========================================="

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux system"
    
    # Check if we have apt (Ubuntu/Debian)
    if command -v apt &> /dev/null; then
        echo "Installing Chromium and dependencies..."
        sudo apt update
        sudo apt install -y chromium-browser chromium-chromedriver xvfb
        echo "✓ Chromium installation completed"
        
    # Check if we have yum (CentOS/RHEL)
    elif command -v yum &> /dev/null; then
        echo "Installing Chromium and dependencies..."
        sudo yum install -y chromium chromium-headless xorg-x11-server-Xvfb
        echo "✓ Chromium installation completed"
        
    # Check if we have dnf (Fedora)
    elif command -v dnf &> /dev/null; then
        echo "Installing Chromium and dependencies..."
        sudo dnf install -y chromium chromium-headless xorg-x11-server-Xvfb
        echo "✓ Chromium installation completed"
        
    else
        echo "❌ Unsupported Linux distribution"
        echo "Please install Chromium manually:"
        echo "  - Chromium browser"
        echo "  - ChromeDriver"
        echo "  - Xvfb (for headless environments)"
        exit 1
    fi
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS system"
    
    if command -v brew &> /dev/null; then
        echo "Installing Chrome and ChromeDriver..."
        brew install --cask google-chrome
        brew install chromedriver
        echo "✓ Chrome installation completed"
    else
        echo "❌ Homebrew not found"
        echo "Please install Homebrew first: https://brew.sh/"
        echo "Or install Chrome manually: https://www.google.com/chrome/"
        exit 1
    fi
    
else
    echo "❌ Unsupported operating system: $OSTYPE"
    echo "Please install Chrome manually:"
    echo "  Windows: https://www.google.com/chrome/"
    echo "  Linux: sudo apt install chromium-browser"
    echo "  macOS: brew install --cask google-chrome"
    exit 1
fi

# Test installation
echo ""
echo "Testing installation..."
echo "======================"

# Test Chrome binary
if command -v chromium-browser &> /dev/null; then
    CHROME_VERSION=$(chromium-browser --version 2>/dev/null || echo "Unknown")
    echo "✓ Chrome/Chromium: $CHROME_VERSION"
elif command -v google-chrome &> /dev/null; then
    CHROME_VERSION=$(google-chrome --version 2>/dev/null || echo "Unknown")
    echo "✓ Chrome: $CHROME_VERSION"
else
    echo "❌ Chrome/Chromium not found in PATH"
fi

# Test ChromeDriver
if command -v chromedriver &> /dev/null; then
    DRIVER_VERSION=$(chromedriver --version 2>/dev/null | head -n1 || echo "Unknown")
    echo "✓ ChromeDriver: $DRIVER_VERSION"
else
    echo "❌ ChromeDriver not found in PATH"
fi

# Test Xvfb (Linux only)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v Xvfb &> /dev/null; then
        echo "✓ Xvfb: Available"
    else
        echo "❌ Xvfb not found"
    fi
fi

echo ""
echo "Installation completed!"
echo "======================"
echo ""
echo "You can now run MiniWob++ benchmarks:"
echo "  python -m webexp.benchmark.run_miniwob -c configs/example_miniwob_config.yaml"
echo ""
echo "For headless environments, use:"
echo "  DISPLAY=:99 python -m webexp.benchmark.run_miniwob -c configs/example_miniwob_config.yaml"
echo ""
echo "If you encounter issues, see CHROME_SETUP_GUIDE.md for troubleshooting."