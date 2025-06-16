#!/usr/bin/env python3
"""
Test script to verify Chrome setup for MiniWob++.
Run this after installing Chrome to make sure everything works.
"""

import os
import sys
import subprocess
import platform

def test_chrome_binary():
    """Test if Chrome binary is available."""
    print("Testing Chrome binary...")
    
    chrome_commands = ['chromium-browser', 'google-chrome', 'chromium']
    
    for cmd in chrome_commands:
        try:
            result = subprocess.run([cmd, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✓ Found Chrome: {result.stdout.strip()}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    print("✗ Chrome binary not found")
    return False

def test_chromedriver():
    """Test if ChromeDriver is available."""
    print("Testing ChromeDriver...")
    
    try:
        result = subprocess.run(['chromedriver', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✓ Found ChromeDriver: {result.stdout.strip()}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("✗ ChromeDriver not found")
    return False

def test_xvfb():
    """Test if Xvfb is available (Linux only)."""
    if platform.system().lower() != 'linux':
        print("Skipping Xvfb test (not Linux)")
        return True
    
    print("Testing Xvfb...")
    
    try:
        result = subprocess.run(['Xvfb', '-help'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 or 'Xvfb' in result.stderr:
            print("✓ Xvfb is available")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("✗ Xvfb not found")
    return False

def test_selenium_chrome():
    """Test Selenium with Chrome."""
    print("Testing Selenium Chrome integration...")
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        
        # Set display for headless environments
        if platform.system().lower() == 'linux' and not os.environ.get('DISPLAY'):
            os.environ['DISPLAY'] = ':99'
        
        driver = webdriver.Chrome(options=options)
        driver.get('data:text/html,<html><body><h1>Test</h1></body></html>')
        title = driver.title
        driver.quit()
        
        print("✓ Selenium Chrome test successful")
        return True
        
    except Exception as e:
        print(f"✗ Selenium Chrome test failed: {e}")
        return False

def test_miniwob_environment():
    """Test MiniWob++ environment creation."""
    print("Testing MiniWob++ environment...")
    
    try:
        import gymnasium
        import miniwob
        
        # Set display for headless environments
        if platform.system().lower() == 'linux' and not os.environ.get('DISPLAY'):
            os.environ['DISPLAY'] = ':99'
        
        gymnasium.register_envs(miniwob)
        env = gymnasium.make('miniwob/click-test-v1', render_mode=None)
        obs, info = env.reset()
        env.close()
        
        print("✓ MiniWob++ environment test successful")
        return True
        
    except Exception as e:
        print(f"✗ MiniWob++ environment test failed: {e}")
        return False

def setup_virtual_display():
    """Set up virtual display for headless environments."""
    if platform.system().lower() != 'linux':
        return True
    
    if os.environ.get('DISPLAY'):
        print(f"Display already set: {os.environ['DISPLAY']}")
        return True
    
    print("Setting up virtual display...")
    
    try:
        # Start Xvfb in background
        subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1024x768x24'], 
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Set display environment variable
        os.environ['DISPLAY'] = ':99'
        
        # Wait a moment for Xvfb to start
        import time
        time.sleep(2)
        
        print("✓ Virtual display started on :99")
        return True
        
    except Exception as e:
        print(f"✗ Failed to start virtual display: {e}")
        return False

def main():
    """Run all tests."""
    print("Chrome Setup Test for MiniWob++")
    print("=" * 40)
    
    # Setup virtual display first if needed
    setup_virtual_display()
    
    tests = [
        ("Chrome Binary", test_chrome_binary),
        ("ChromeDriver", test_chromedriver),
        ("Xvfb", test_xvfb),
        ("Selenium Chrome", test_selenium_chrome),
        ("MiniWob++ Environment", test_miniwob_environment),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 20)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your Chrome setup is ready for MiniWob++")
        print("\nYou can now run:")
        print("  python -m webexp.benchmark.run_miniwob -c configs/example_miniwob_config.yaml")
        return 0
    else:
        print(f"\n⚠ {len(results) - passed} test(s) failed")
        print("\nTroubleshooting:")
        print("1. Run: ./install_chrome.sh")
        print("2. Or: python setup_chrome.py")
        print("3. See: CHROME_SETUP_GUIDE.md")
        return 1

if __name__ == "__main__":
    exit(main())