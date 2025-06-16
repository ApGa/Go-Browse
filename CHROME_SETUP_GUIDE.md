# Chrome Setup Guide for MiniWob++

This guide helps you resolve Chrome/Chromium installation and configuration issues when running Go-Browse with MiniWob++.

## Quick Fix

If you're getting Chrome-related errors, try this automated setup:

```bash
python setup_chrome.py
```

## Manual Setup Instructions

### Ubuntu/Debian

#### Option 1: Install Chromium (Recommended)
```bash
sudo apt update
sudo apt install -y chromium-browser chromium-chromedriver
```

#### Option 2: Install Google Chrome
```bash
# Add Google Chrome repository
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list

# Install Chrome
sudo apt update
sudo apt install -y google-chrome-stable

# Install ChromeDriver
sudo apt install -y chromium-chromedriver
```

### CentOS/RHEL/Fedora

#### CentOS/RHEL
```bash
sudo yum install -y chromium chromium-headless
```

#### Fedora
```bash
sudo dnf install -y chromium chromium-headless
```

### macOS

#### Using Homebrew
```bash
brew install --cask google-chrome
brew install chromedriver
```

### Windows

1. Download Chrome from: https://www.google.com/chrome/
2. Download ChromeDriver from: https://chromedriver.chromium.org/
3. Add ChromeDriver to your PATH

## Headless Environment Setup

If you're running in a headless environment (no GUI), you need a virtual display:

### Install Xvfb
```bash
sudo apt install -y xvfb
```

### Start Virtual Display
```bash
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 &
```

### Run with Virtual Display
```bash
DISPLAY=:99 python -m webexp.benchmark.run_miniwob -c configs/example_miniwob_config.yaml
```

## Common Error Solutions

### Error: "Unable to obtain driver for chrome"

**Solution 1: Install ChromeDriver**
```bash
# Ubuntu/Debian
sudo apt install chromium-chromedriver

# Or download manually
wget https://chromedriver.storage.googleapis.com/LATEST_RELEASE
LATEST=$(cat LATEST_RELEASE)
wget https://chromedriver.storage.googleapis.com/${LATEST}/chromedriver_linux64.zip
unzip chromedriver_linux64.zip
sudo mv chromedriver /usr/local/bin/
```

**Solution 2: Use Selenium Manager (Automatic)**
```bash
pip install --upgrade selenium
```

### Error: "Chrome binary not found"

**Solution: Specify Chrome path**
```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.binary_location = "/usr/bin/chromium-browser"  # or your Chrome path
driver = webdriver.Chrome(options=options)
```

### Error: "DevToolsActivePort file doesn't exist"

**Solution: Add Chrome arguments**
```python
options = Options()
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-gpu')
```

### Error: Network connectivity issues

**Solution: Offline ChromeDriver setup**
```bash
# Download ChromeDriver manually
wget https://chromedriver.storage.googleapis.com/114.0.5735.90/chromedriver_linux64.zip
unzip chromedriver_linux64.zip
sudo mv chromedriver /usr/local/bin/
chmod +x /usr/local/bin/chromedriver
```

## Environment Variables

Set these environment variables for better compatibility:

```bash
export DISPLAY=:99                    # For headless environments
export CHROME_BIN=/usr/bin/chromium   # Chrome binary path
export CHROMEDRIVER_PATH=/usr/local/bin/chromedriver  # ChromeDriver path
```

## Docker Setup

If you're using Docker, add this to your Dockerfile:

```dockerfile
# Install Chrome and dependencies
RUN apt-get update && apt-get install -y \
    chromium-browser \
    chromium-chromedriver \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV DISPLAY=:99
ENV CHROME_BIN=/usr/bin/chromium-browser
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Start virtual display
RUN Xvfb :99 -screen 0 1024x768x24 &
```

## Testing Your Setup

### Test Chrome Installation
```bash
# Test Chrome binary
chromium-browser --version
# or
google-chrome --version

# Test ChromeDriver
chromedriver --version
```

### Test Selenium Integration
```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

try:
    driver = webdriver.Chrome(options=options)
    driver.get('data:text/html,<html><body><h1>Test</h1></body></html>')
    print("✓ Selenium Chrome test successful!")
    driver.quit()
except Exception as e:
    print(f"✗ Selenium Chrome test failed: {e}")
```

### Test MiniWob++ Environment
```python
import gymnasium
import miniwob

# Register environments
gymnasium.register_envs(miniwob)

# Test environment creation
try:
    env = gymnasium.make('miniwob/click-test-v1', render_mode=None)
    obs, info = env.reset()
    print("✓ MiniWob++ environment test successful!")
    env.close()
except Exception as e:
    print(f"✗ MiniWob++ environment test failed: {e}")
```

## Alternative Solutions

### Use BrowserGym with Different Backends

If Chrome continues to cause issues, you can try different backends:

```python
# Use Firefox instead
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

options = Options()
options.add_argument('--headless')
driver = webdriver.Firefox(options=options)
```

### Use Playwright Instead of Selenium

```bash
pip install playwright
playwright install chromium
```

## Performance Optimization

For better performance in headless environments:

```python
options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-gpu')
options.add_argument('--disable-extensions')
options.add_argument('--disable-plugins')
options.add_argument('--disable-images')
options.add_argument('--disable-javascript')  # Only if not needed
options.add_argument('--memory-pressure-off')
options.add_argument('--max_old_space_size=4096')
```

## Getting Help

If you're still having issues:

1. **Check the logs**: Look at the error messages carefully
2. **Verify installation**: Make sure Chrome and ChromeDriver are properly installed
3. **Test step by step**: Use the test scripts above to isolate the issue
4. **Check permissions**: Make sure you have proper permissions to run Chrome
5. **Try different versions**: Sometimes version mismatches cause issues

### Common Version Compatibility

| Chrome Version | ChromeDriver Version |
|----------------|---------------------|
| 114.x.x        | 114.0.5735.90      |
| 115.x.x        | 115.0.5790.102     |
| 116.x.x        | 116.0.5845.96      |

### Useful Commands

```bash
# Check Chrome version
chromium-browser --version
google-chrome --version

# Check ChromeDriver version
chromedriver --version

# Find Chrome binary location
which chromium-browser
which google-chrome

# Check if display is available
echo $DISPLAY

# Test virtual display
xdpyinfo -display :99
```

## Automated Setup Script

The `setup_chrome.py` script automates most of these steps:

```bash
python setup_chrome.py
```

This script will:
1. Check for existing Chrome installation
2. Install Chrome/Chromium if needed
3. Install ChromeDriver
4. Set up virtual display for headless environments
5. Test the complete setup

After running the setup script, you should be able to run MiniWob++ benchmarks without issues.