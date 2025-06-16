#!/usr/bin/env python3
"""
Simple test script to verify MiniWob++ integration works.
"""

import gymnasium
import miniwob
import os
import sys

# Add the project root to the path
sys.path.insert(0, '/workspace/Go-Browse')

# Import directly to avoid webexp __init__.py issues
from webexp.agents.miniwob_solver_agent import MiniWobSolverAgent

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
        
        return True
        
    except Exception as e:
        print(f"Error testing environment: {e}")
        return False
    finally:
        env.close()

def test_agent_creation():
    """Test creating the MiniWob++ agent."""
    print("\nTesting MiniWob++ agent creation...")
    
    try:
        # Create agent with dummy config
        agent = MiniWobSolverAgent(
            model_id="gpt-3.5-turbo",  # This won't be used in this test
            temperature=0.0,
            char_limit=10000
        )
        
        print("Agent created successfully!")
        print(f"Agent config: {agent.get_config()}")
        
        return True
        
    except Exception as e:
        print(f"Error creating agent: {e}")
        return False

def test_observation_preprocessing():
    """Test observation preprocessing."""
    print("\nTesting observation preprocessing...")
    
    try:
        # Create agent
        agent = MiniWobSolverAgent(
            model_id="gpt-3.5-turbo",
            temperature=0.0,
            char_limit=10000
        )
        
        # Create environment and get observation
        gymnasium.register_envs(miniwob)
        env = gymnasium.make('miniwob/click-test-2-v1', render_mode=None)
        
        obs, info = env.reset()
        
        # Test preprocessing
        processed_obs = agent.obs_preprocessor(obs)
        
        print("Observation preprocessing successful!")
        print(f"Processed observation keys: {processed_obs.keys()}")
        print(f"Utterance: {processed_obs.get('utterance', 'N/A')}")
        print(f"Number of DOM elements: {len(processed_obs.get('dom_elements', []))}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"Error in observation preprocessing: {e}")
        return False

def test_action_processing():
    """Test action processing."""
    print("\nTesting action processing...")
    
    try:
        # Create agent
        agent = MiniWobSolverAgent(
            model_id="gpt-3.5-turbo",
            temperature=0.0,
            char_limit=10000
        )
        
        # Test various action formats
        test_actions = [
            '{"thought": "I need to click the button", "action": "click(\\"button_1\\")"}',
            '{"thought": "I should type some text", "action": "type(\\"hello world\\")"}',
            '{"thought": "Press enter key", "action": "key(\\"Enter\\")"}',
            '{"thought": "Task is done", "action": "done()"}',
        ]
        
        for action_str in test_actions:
            processed_action = agent.action_processor(action_str)
            print(f"Action: {action_str}")
            print(f"Processed: {processed_action}")
            print()
        
        return True
        
    except Exception as e:
        print(f"Error in action processing: {e}")
        return False

def main():
    """Run all tests."""
    print("Starting MiniWob++ integration tests...\n")
    
    tests = [
        test_miniwob_environment,
        test_agent_creation,
        test_observation_preprocessing,
        test_action_processing,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print(f"\nTest Results:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("All tests passed! MiniWob++ integration is working.")
        return 0
    else:
        print("Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())