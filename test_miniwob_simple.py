#!/usr/bin/env python3
"""
Simple test script to verify MiniWob++ basic functionality.
"""

import gymnasium
import miniwob

def test_miniwob_environment():
    """Test basic MiniWob++ environment functionality."""
    print("Testing MiniWob++ environment setup...")
    
    # Register environments
    gymnasium.register_envs(miniwob)
    
    # Test creating an environment
    env = gymnasium.make('miniwob/click-test-2-v1', render_mode=None)
    
    try:
        obs, info = env.reset()
        print(f"Environment reset successful!")
        print(f"Observation keys: {obs.keys()}")
        print(f"Utterance: {obs.get('utterance', 'N/A')}")
        print(f"Fields: {obs.get('fields', 'N/A')}")
        print(f"Number of DOM elements: {len(obs.get('dom_elements', []))}")
        
        # Test a simple action
        if obs.get('dom_elements'):
            first_element = obs['dom_elements'][0]
            print(f"First DOM element: {first_element}")
            
            # Try to create a click action
            from miniwob.action import ActionTypes
            action = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=first_element['ref'])
            print(f"Created action: {action}")
        
        return True
        
    except Exception as e:
        print(f"Error testing environment: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        env.close()

def main():
    """Run basic test."""
    print("Starting basic MiniWob++ test...\n")
    
    result = test_miniwob_environment()
    
    if result:
        print("\nBasic MiniWob++ test passed!")
        return 0
    else:
        print("\nBasic MiniWob++ test failed!")
        return 1

if __name__ == "__main__":
    exit(main())