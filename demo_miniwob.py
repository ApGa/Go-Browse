#!/usr/bin/env python3
"""
Demo script showing how to run Go-Browse on a single MiniWob++ task.
This is useful for testing and development.
"""

import gymnasium
import miniwob
import os
import sys
from webexp.agents.miniwob_solver_agent import MiniWobSolverAgent

def demo_single_task(env_name="click-test-2", max_steps=10):
    """
    Run a single MiniWob++ task with the Go-Browse agent.
    
    Args:
        env_name (str): Name of the MiniWob++ environment
        max_steps (int): Maximum number of steps to run
    """
    
    print(f"Demo: Running Go-Browse on MiniWob++ task '{env_name}'")
    print("=" * 60)
    
    # Register MiniWob++ environments
    gymnasium.register_envs(miniwob)
    
    # Create environment
    env = gymnasium.make(f'miniwob/{env_name}-v1', render_mode=None)
    
    # Create agent (using dummy model for demo - won't actually call API)
    agent = MiniWobSolverAgent(
        model_id="gpt-3.5-turbo",
        temperature=0.0,
        char_limit=10000
    )
    
    try:
        # Reset environment and agent
        obs, info = env.reset()
        agent.reset()
        
        print(f"Task: {obs.get('utterance', 'N/A')}")
        print(f"Fields: {obs.get('fields', 'N/A')}")
        print(f"DOM Elements: {len(obs.get('dom_elements', []))}")
        print()
        
        # Process observation
        processed_obs = agent.obs_preprocessor(obs)
        print("Processed observation keys:", list(processed_obs.keys()))
        
        # Show DOM elements
        if processed_obs.get('dom_elements'):
            print("\nDOM Elements:")
            for i, elem in enumerate(processed_obs['dom_elements'][:5]):  # Show first 5
                print(f"  {i+1}. {elem.get('tag', 'N/A')} - '{elem.get('text', 'N/A')}' (ref: {elem.get('ref', 'N/A')})")
            if len(processed_obs['dom_elements']) > 5:
                print(f"  ... and {len(processed_obs['dom_elements']) - 5} more elements")
        
        print("\nThis demo shows the observation processing.")
        print("To run with an actual LLM, set up your API credentials and run:")
        print(f"python -m webexp.benchmark.run_miniwob -c configs/example_miniwob_config.yaml")
        
        # Demonstrate action processing
        print("\nDemo: Action Processing")
        print("-" * 30)
        
        sample_actions = [
            '{"thought": "I need to click on button ONE", "action": "click(\\"2\\")"}',
            '{"thought": "Let me type some text", "action": "type(\\"hello\\")"}',
            '{"thought": "Task completed", "action": "done()"}',
        ]
        
        for action in sample_actions:
            processed = agent.action_processor(action)
            print(f"Input:  {action}")
            print(f"Output: {processed}")
            print()
        
        return True
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        env.close()

def list_available_environments():
    """List some popular MiniWob++ environments."""
    
    environments = [
        # Simple tasks
        ("click-test", "Click on a specific button"),
        ("click-test-2", "Click on a button with specific text"),
        ("click-button", "Click on a button"),
        ("enter-text", "Enter text into a field"),
        ("enter-text-2", "Enter text with specific formatting"),
        
        # Form tasks
        ("login-user", "Fill out a login form"),
        ("form-sequence", "Fill out a multi-step form"),
        ("book-flight", "Book a flight with specific criteria"),
        
        # Navigation tasks
        ("click-tab", "Click on a specific tab"),
        ("click-tab-2", "Navigate between tabs"),
        ("navigate-tree", "Navigate a tree structure"),
        
        # Complex tasks
        ("email-inbox", "Manage email inbox"),
        ("search-engine", "Use a search engine"),
        ("use-autocomplete", "Use autocomplete functionality"),
    ]
    
    print("Available MiniWob++ Environments:")
    print("=" * 50)
    for env_name, description in environments:
        print(f"  {env_name:<20} - {description}")
    print()
    print("For a complete list, see: https://miniwob.farama.org/environments/list/")

def main():
    """Main demo function."""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            list_available_environments()
            return
        else:
            env_name = sys.argv[1]
    else:
        env_name = "click-test-2"
    
    print("Go-Browse MiniWob++ Demo")
    print("=" * 40)
    print()
    
    # Check if environment exists
    try:
        gymnasium.register_envs(miniwob)
        env = gymnasium.make(f'miniwob/{env_name}-v1', render_mode=None)
        env.close()
    except Exception as e:
        print(f"Error: Environment '{env_name}' not found or invalid.")
        print("Use --list to see available environments.")
        return 1
    
    # Run demo
    success = demo_single_task(env_name)
    
    if success:
        print("\nDemo completed successfully!")
        print("\nNext steps:")
        print("1. Set up your LLM API credentials (OpenAI API key, etc.)")
        print("2. Configure configs/example_miniwob_config.yaml")
        print("3. Run: python -m webexp.benchmark.run_miniwob -c configs/example_miniwob_config.yaml")
        return 0
    else:
        print("\nDemo failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    exit(main())