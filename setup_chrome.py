#!/usr/bin/env python3
"""
Chrome/Chromium setup script for MiniWob++ environments.
This script helps install and configure Chrome/Chromium for use with Selenium.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.CalledProcessError as e:
        return e.stdout.strip(), e.stderr.strip(), e.returncode

def check_chrome_installed():
    """Check if Chrome or Chromium is installed."""
    chrome_paths = [
        '/usr/bin/google-chrome',
        '/usr/bin/google-chrome-stable',
        '/usr/bin/chromium',
        '/usr/bin/chromium-browser',
        '/opt/google/chrome/chrome',
        '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
        'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
    ]
    
    for path in chrome_paths:
        if os.path.exists(path):
            print(f"✓ Found Chrome/Chromium at: {path}")
            return path
    
    # Try which/where command
    for cmd in ['google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser']:
        stdout, stderr, code = run_command(f"which {cmd}", check=False)
        if code == 0 and stdout:
            print(f"✓ Found Chrome/Chromium at: {stdout}")
            return stdout
    
    return None

def install_chrome_linux():
    """Install Chrome on Linux systems."""
    print("Installing Chrome on Linux...")
    
    # Try different installation methods
    install_commands = [
        # Ubuntu/Debian - try chromium first (easier)
        "sudo apt update && sudo apt install -y chromium-browser",
        # Ubuntu/Debian - Google Chrome
        """
        wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add - &&
        echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list &&
        sudo apt update && sudo apt install -y google-chrome-stable
        """,
        # CentOS/RHEL/Fedora
        "sudo yum install -y chromium",
        "sudo dnf install -y chromium",
    ]
    
    for cmd in install_commands:
        print(f"Trying: {cmd.split('&&')[0].strip()}")
        stdout, stderr, code = run_command(cmd, check=False)
        if code == 0:
            print("✓ Installation successful!")
            return True
        else:
            print(f"✗ Failed: {stderr}")
    
    return False

def install_chromedriver():
    """Install ChromeDriver."""
    print("Installing ChromeDriver...")
    
    # Try different installation methods
    install_commands = [
        # Ubuntu/Debian
        "sudo apt install -y chromium-chromedriver",
        # Using pip
        "pip install chromedriver-autoinstaller",
        # Manual download (as fallback)
    ]
    
    for cmd in install_commands:
        print(f"Trying: {cmd}")
        stdout, stderr, code = run_command(cmd, check=False)
        if code == 0:
            print("✓ ChromeDriver installation successful!")
            return True
        else:
            print(f"✗ Failed: {stderr}")
    
    return False

def setup_headless_display():
    """Set up virtual display for headless environments."""
    print("Setting up virtual display...")
    
    # Install xvfb
    stdout, stderr, code = run_command("sudo apt install -y xvfb", check=False)
    if code != 0:
        print("⚠ Could not install xvfb, trying without sudo...")
        stdout, stderr, code = run_command("apt install -y xvfb", check=False)
    
    if code == 0:
        print("✓ xvfb installed successfully!")
        
        # Set display environment variable
        os.environ['DISPLAY'] = ':99'
        
        # Start virtual display
        stdout, stderr, code = run_command("Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &", check=False)
        if code == 0:
            print("✓ Virtual display started!")
            return True
    
    print("⚠ Could not set up virtual display")
    return False

def create_chrome_wrapper():
    """Create a Chrome wrapper script for better compatibility."""
    wrapper_content = '''#!/bin/bash
# Chrome wrapper script for MiniWob++

# Find Chrome binary
CHROME_BIN=""
for path in /usr/bin/google-chrome /usr/bin/google-chrome-stable /usr/bin/chromium /usr/bin/chromium-browser; do
    if [ -x "$path" ]; then
        CHROME_BIN="$path"
        break
    fi
done

if [ -z "$CHROME_BIN" ]; then
    echo "Error: Chrome/Chromium not found"
    exit 1
fi

# Run Chrome with appropriate flags
exec "$CHROME_BIN" \\
    --no-sandbox \\
    --disable-dev-shm-usage \\
    --disable-gpu \\
    --disable-extensions \\
    --disable-plugins \\
    --disable-images \\
    --disable-javascript \\
    --disable-default-apps \\
    --disable-background-timer-throttling \\
    --disable-backgrounding-occluded-windows \\
    --disable-renderer-backgrounding \\
    --disable-features=TranslateUI \\
    --disable-ipc-flooding-protection \\
    --remote-debugging-port=9222 \\
    "$@"
'''
    
    wrapper_path = Path.home() / '.local' / 'bin' / 'chrome-wrapper'
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    
    os.chmod(wrapper_path, 0o755)
    print(f"✓ Created Chrome wrapper at: {wrapper_path}")
    return str(wrapper_path)

def test_selenium_chrome():
    """Test if Selenium can start Chrome."""
    print("Testing Selenium Chrome integration...")
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        
        # Try to start Chrome
        driver = webdriver.Chrome(options=options)
        driver.get('data:text/html,<html><body><h1>Test</h1></body></html>')
        title = driver.title
        driver.quit()
        
        print("✓ Selenium Chrome test successful!")
        return True
        
    except Exception as e:
        print(f"✗ Selenium Chrome test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("Chrome/Chromium Setup for MiniWob++")
    print("=" * 40)
    
    system = platform.system().lower()
    print(f"Detected system: {system}")
    
    # Check if Chrome is already installed
    chrome_path = check_chrome_installed()
    if not chrome_path:
        print("Chrome/Chromium not found. Installing...")
        
        if system == 'linux':
            if not install_chrome_linux():
                print("✗ Failed to install Chrome/Chromium")
                print("Please install manually:")
                print("  Ubuntu/Debian: sudo apt install chromium-browser")
                print("  CentOS/RHEL: sudo yum install chromium")
                return False
        else:
            print(f"✗ Automatic installation not supported for {system}")
            print("Please install Chrome manually:")
            print("  macOS: brew install --cask google-chrome")
            print("  Windows: Download from https://www.google.com/chrome/")
            return False
    
    # Install ChromeDriver
    print("\nSetting up ChromeDriver...")
    install_chromedriver()
    
    # Set up virtual display for headless environments
    if system == 'linux' and not os.environ.get('DISPLAY'):
        setup_headless_display()
    
    # Create Chrome wrapper
    wrapper_path = create_chrome_wrapper()
    
    # Test Selenium integration
    print("\nTesting setup...")
    if test_selenium_chrome():
        print("\n✓ Setup completed successfully!")
        print("\nYou can now run MiniWob++ benchmarks:")
        print("  python -m webexp.benchmark.run_miniwob -c configs/example_miniwob_config.yaml")
        return True
    else:
        print("\n✗ Setup incomplete. Please check the errors above.")
        print("\nTroubleshooting tips:")
        print("1. Make sure Chrome/Chromium is installed")
        print("2. Install ChromeDriver: sudo apt install chromium-chromedriver")
        print("3. For headless environments, install xvfb: sudo apt install xvfb")
        print("4. Try running with DISPLAY=:99 if using virtual display")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)