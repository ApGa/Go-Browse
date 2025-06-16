#!/usr/bin/env python3
"""
Demo script showing all available agent types for MiniWob++:
1. OpenAI API (original)
2. Azure AI (GPT-4o)
3. Amazon Bedrock (Claude)
"""

import gymnasium
import miniwob
import os
import sys

def demo_agent_comparison(env_name="click-test-2"):
    """
    Compare all three agent types on a single MiniWob++ task.
    
    Args:
        env_name (str): Name of the MiniWob++ environment
    """
    
    print(f"Demo: Comparing Go-Browse agents on MiniWob++ task '{env_name}'")
    print("=" * 70)
    
    # Register MiniWob++ environments
    gymnasium.register_envs(miniwob)
    
    # Create environment
    env = gymnasium.make(f'miniwob/{env_name}-v1', render_mode=None)
    
    try:
        # Reset environment
        obs, info = env.reset()
        
        print(f"Task: {obs.get('utterance', 'N/A')}")
        print(f"Fields: {obs.get('fields', 'N/A')}")
        print(f"DOM Elements: {len(obs.get('dom_elements', []))}")
        print()
        
        # Show DOM elements
        if obs.get('dom_elements'):
            print("DOM Elements:")
            for i, elem in enumerate(obs['dom_elements'][:5]):  # Show first 5
                print(f"  {i+1}. {elem.get('tag', 'N/A')} - '{elem.get('text', 'N/A')}' (ref: {elem.get('ref', 'N/A')})")
            if len(obs['dom_elements']) > 5:
                print(f"  ... and {len(obs['dom_elements']) - 5} more elements")
        print()
        
        # Test all agent types
        agents_to_test = []
        
        # 1. Original OpenAI agent
        try:
            from webexp.agents.miniwob_solver_agent import MiniWobSolverAgent
            agents_to_test.append(("OpenAI API", MiniWobSolverAgent, {
                "model_id": "gpt-3.5-turbo",
                "temperature": 0.0,
                "char_limit": 10000
            }))
        except Exception as e:
            print(f"⚠ OpenAI agent not available: {e}")
        
        # 2. Azure AI agent
        try:
            os.environ["api_key_azure"] = "dummy_key"
            os.environ["api_base_azure_ai"] = "https://dummy.openai.azure.com/"
            
            from webexp.agents.cloud_miniwob_solver_agent import AzureGPTMiniWobAgent
            agents_to_test.append(("Azure AI", AzureGPTMiniWobAgent, {
                "model_name": "gpt-4o",
                "temperature": 0.0,
                "char_limit": 10000
            }))
        except Exception as e:
            print(f"⚠ Azure AI agent not available: {e}")
        
        # 3. Bedrock agent
        try:
            os.environ["AWS_KEY"] = "dummy_key"
            os.environ["AWS_SECRET_KEY"] = "dummy_secret"
            os.environ["AWS_REGION"] = "us-east-1"
            
            from webexp.agents.cloud_miniwob_solver_agent import BedrockClaudeMiniWobAgent
            agents_to_test.append(("Amazon Bedrock", BedrockClaudeMiniWobAgent, {
                "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "temperature": 0.0,
                "max_tokens": 10000
            }))
        except Exception as e:
            print(f"⚠ Bedrock agent not available: {e}")
        
        # Test each agent
        for agent_name, agent_class, agent_kwargs in agents_to_test:
            print(f"Testing {agent_name} Agent:")
            print("-" * 40)
            
            try:
                agent = agent_class(**agent_kwargs)
                
                # Test observation preprocessing
                processed_obs = agent.obs_preprocessor(obs)
                print(f"✓ Observation preprocessing: {list(processed_obs.keys())}")
                
                # Test action processing
                sample_action = '{"thought": "I need to click on button ONE", "action": "click(\\"4\\")"}'
                processed_action = agent.action_processor(sample_action)
                print(f"✓ Action processing: {processed_action}")
                
                print(f"✓ {agent_name} agent working correctly!")
                
            except Exception as e:
                print(f"✗ Error with {agent_name} agent: {e}")
            
            print()
        
        return True
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        env.close()

def show_configuration_examples():
    """Show configuration examples for all agent types."""
    
    print("\n" + "="*70)
    print("CONFIGURATION EXAMPLES")
    print("="*70)
    
    print("\n1. OPENAI API (Original):")
    print("   Config file: configs/example_miniwob_config.yaml")
    print("   Environment variables:")
    print("     export OPENAI_API_KEY='your-openai-api-key'")
    print("   Agent name: MiniWobSolverAgent")
    
    print("\n2. AZURE AI:")
    print("   Config file: configs/azure_gpt_miniwob.yaml")
    print("   Environment variables:")
    print("     export api_key_azure='your-azure-api-key'")
    print("     export api_base_azure_ai='https://your-resource.openai.azure.com/'")
    print("   Agent name: AzureGPTMiniWobAgent")
    
    print("\n3. AMAZON BEDROCK:")
    print("   Config file: configs/bedrock_claude_miniwob.yaml")
    print("   Environment variables:")
    print("     export AWS_KEY='your-aws-access-key'")
    print("     export AWS_SECRET_KEY='your-aws-secret-key'")
    print("     export AWS_REGION='us-east-1'")
    print("   Agent name: BedrockClaudeMiniWobAgent")

def show_model_comparison():
    """Show comparison of available models."""
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    models = [
        ("OpenAI API", [
            ("gpt-4o", "Latest GPT-4 model, excellent performance"),
            ("gpt-4", "Previous generation, still very capable"),
            ("gpt-3.5-turbo", "Faster and cheaper, good for testing"),
        ]),
        ("Azure AI", [
            ("gpt-4o", "Same as OpenAI but through Azure"),
            ("gpt-4", "Previous generation through Azure"),
            ("gpt-35-turbo", "Azure naming for GPT-3.5-turbo"),
        ]),
        ("Amazon Bedrock", [
            ("claude-3-5-sonnet", "Latest Claude, excellent reasoning"),
            ("claude-3-opus", "Most capable Claude model"),
            ("claude-3-haiku", "Fastest Claude model"),
        ]),
    ]
    
    for provider, model_list in models:
        print(f"\n{provider}:")
        for model, description in model_list:
            print(f"  • {model:<20} - {description}")

def main():
    """Main demo function."""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("Usage: python demo_cloud_agents.py [environment_name]")
            print("Example: python demo_cloud_agents.py click-test-2")
            return
        else:
            env_name = sys.argv[1]
    else:
        env_name = "click-test-2"
    
    print("Go-Browse Cloud Agents Demo")
    print("=" * 40)
    print()
    
    # Check if environment exists
    try:
        gymnasium.register_envs(miniwob)
        env = gymnasium.make(f'miniwob/{env_name}-v1', render_mode=None)
        env.close()
    except Exception as e:
        print(f"Error: Environment '{env_name}' not found or invalid.")
        print("Use 'python demo_miniwob.py --list' to see available environments.")
        return 1
    
    # Run demo
    success = demo_agent_comparison(env_name)
    
    if success:
        show_configuration_examples()
        show_model_comparison()
        
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\n1. Choose your preferred cloud provider")
        print("2. Set up credentials (see configuration examples above)")
        print("3. Install required dependencies:")
        print("   - For Azure AI: pip install langchain-openai")
        print("   - For Bedrock: pip install boto3 (already installed)")
        print("4. Run benchmark:")
        print("   - OpenAI: python -m webexp.benchmark.run_miniwob -c configs/example_miniwob_config.yaml")
        print("   - Azure AI: python -m webexp.benchmark.run_miniwob -c configs/azure_gpt_miniwob.yaml")
        print("   - Bedrock: python -m webexp.benchmark.run_miniwob -c configs/bedrock_claude_miniwob.yaml")
        
        print("\n✓ Demo completed successfully!")
        return 0
    else:
        print("\nDemo failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    exit(main())