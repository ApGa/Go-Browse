# Go-Browse MiniWob++ Adaptation

This document describes the adaptation of Go-Browse from WebArena to the MiniWob++ benchmark.

## Overview

The MiniWob++ adaptation allows Go-Browse to work with the MiniWob++ benchmark, which provides over 100 web interaction tasks in a simpler, more accessible format compared to WebArena. This adaptation maintains the core Go-Browse methodology while making it easier to set up and test.

## Key Differences from WebArena

### Advantages of MiniWob++
- **Easier Setup**: No complex server infrastructure required
- **Faster Execution**: Lightweight environments with quick reset times
- **Diverse Tasks**: 100+ different web interaction scenarios
- **Standardized Interface**: Consistent observation and action spaces
- **Better for Development**: Ideal for rapid prototyping and testing

### Environment Differences
- **Observation Format**: MiniWob++ provides DOM elements, utterances, and fields instead of accessibility trees
- **Action Space**: Simplified action space with click, type, key, and scroll actions
- **Task Structure**: Self-contained HTML pages vs. full web applications
- **Evaluation**: Immediate reward signals vs. complex task completion criteria

## Installation

1. Install the base Go-Browse requirements:
```bash
pip install -r requirements.txt
pip install -e .
```

2. Install MiniWob++ and BrowserGym:
```bash
pip install miniwob browsergym omegaconf
```

3. Install Chrome/Chromium and ChromeDriver (if not already installed):
```bash
# On Ubuntu/Debian
sudo apt install chromium-driver

# Or download ChromeDriver manually and add to PATH
```

4. Install cloud provider dependencies (optional):
```bash
# For Azure AI support
pip install langchain-openai

# For Amazon Bedrock support
pip install boto3
```

## Usage

### Running the MiniWob++ Benchmark

Go-Browse supports three different model providers:

#### Option 1: OpenAI API (Original)
1. **Configure your experiment**:
```bash
cp configs/example_miniwob_config.yaml configs/my_miniwob_config.yaml
```

2. **Set up environment variables**:
```bash
export OPENAI_API_KEY='your-openai-api-key'
```

3. **Update configuration**:
```yaml
agent_factory_args:
  name: MiniWobSolverAgent
  model_id: "gpt-4o"  # or gpt-3.5-turbo
  temperature: 0.0
  char_limit: 40000
```

4. **Run the benchmark**:
```bash
python -m webexp.benchmark.run_miniwob -c configs/my_miniwob_config.yaml
```

#### Option 2: Azure AI (GPT-4o)
1. **Configure your experiment**:
```bash
cp configs/azure_gpt_miniwob.yaml configs/my_azure_config.yaml
```

2. **Set up environment variables**:
```bash
export api_key_azure='your-azure-api-key'
export api_base_azure_ai='https://your-resource.openai.azure.com/'
```

3. **Update configuration**:
```yaml
agent_factory_args:
  name: AzureGPTMiniWobAgent
  model_name: "gpt-4o"  # Your Azure deployment name
  api_version: "2024-08-01-preview"
  temperature: 0.0
  char_limit: 40000
```

4. **Run the benchmark**:
```bash
python -m webexp.benchmark.run_miniwob -c configs/my_azure_config.yaml
```

#### Option 3: Amazon Bedrock (Claude)
1. **Configure your experiment**:
```bash
cp configs/bedrock_claude_miniwob.yaml configs/my_bedrock_config.yaml
```

2. **Set up environment variables**:
```bash
export AWS_KEY='your-aws-access-key'
export AWS_SECRET_KEY='your-aws-secret-key'
export AWS_REGION='us-east-1'
```

3. **Update configuration**:
```yaml
agent_factory_args:
  name: BedrockClaudeMiniWobAgent
  model_id: "anthropic.claude-3-5-sonnet-20240620-v1:0"
  temperature: 0.0
  max_tokens: 10000
  char_limit: 40000
```

4. **Run the benchmark**:
```bash
python -m webexp.benchmark.run_miniwob -c configs/my_bedrock_config.yaml
```

### Configuration Options

#### Agent Configuration
- `model_id`: Hugging Face model ID or OpenAI model name
- `base_url`: Custom API endpoint (optional)
- `base_url_2`: Backup API endpoint for longer contexts (optional)
- `api_key`: API key (optional if using environment variables)
- `temperature`: Sampling temperature (0.0 for deterministic)
- `char_limit`: Maximum characters in prompt

#### Benchmark Configuration
- `exp_dir`: Directory to save results
- `environments`: List of MiniWob++ environment names
- `num_episodes`: Episodes per environment
- `max_steps`: Maximum steps per episode
- `timeout`: Timeout per episode in seconds

### Available Environments

The MiniWob++ benchmark includes over 100 environments. Here are some recommended starting points:

#### Simple Tasks (Good for Testing)
- `click-test`, `click-test-2`: Basic clicking
- `click-button`: Button clicking
- `enter-text`, `enter-text-2`: Text input

#### Form Tasks
- `login-user`: Login forms
- `form-sequence`, `form-sequence-2`: Multi-step forms
- `book-flight`: Flight booking

#### Navigation Tasks
- `click-tab`, `click-tab-2`: Tab navigation
- `navigate-tree`: Tree navigation
- `click-menu`: Menu interaction

#### Complex Tasks
- `email-inbox`: Email management
- `search-engine`: Search interactions
- `use-autocomplete`: Autocomplete usage

## Architecture

### New Components

1. **MiniWobSolverAgent** (`webexp/agents/miniwob_solver_agent.py`)
   - Specialized agent for MiniWob++ environments using OpenAI API
   - Handles MiniWob++ observation format
   - Converts actions to MiniWob++ action objects

2. **AzureGPTMiniWobAgent** (`webexp/agents/cloud_miniwob_solver_agent.py`)
   - Azure AI version using GPT-4o through Azure OpenAI Service
   - Same functionality as MiniWobSolverAgent but with Azure backend
   - Uses LangChain for Azure integration

3. **BedrockClaudeMiniWobAgent** (`webexp/agents/cloud_miniwob_solver_agent.py`)
   - Amazon Bedrock version using Claude models
   - Direct boto3 integration with Bedrock runtime
   - Supports Claude 3.5 Sonnet, Opus, and Haiku

4. **MiniWobSolverPromptBuilder** (`webexp/agents/prompt_builders/miniwob_solver_prompt_builder.py`)
   - Builds prompts specific to MiniWob++ tasks
   - Formats DOM elements and task instructions
   - Provides MiniWob++-specific action examples
   - Shared across all agent types

5. **MiniWobBenchmark** (`webexp/benchmark/run_miniwob.py`)
   - Benchmark runner for MiniWob++ environments
   - Handles episode execution and result collection
   - Converts between agent actions and MiniWob++ actions
   - Works with all agent types

6. **MiniWobAgentStepData** (`webexp/agents/trajectory_data.py`)
   - Data structure for MiniWob++ step information
   - Stores utterances, DOM elements, and actions

### Action Format

The agent outputs actions in a structured format:

```json
{
  "thought": "Reasoning about the action",
  "action": "action_type(parameters)"
}
```

Supported action types:
- `click("element_ref")`: Click on an element
- `type("text")`: Type text
- `key("key_name")`: Press a key (Enter, Tab, etc.)
- `scroll("direction")`: Scroll (up, down, left, right)
- `done()`: Indicate task completion

### Observation Format

MiniWob++ observations include:
- `utterance`: Task instruction
- `fields`: Task-specific fields
- `dom_elements`: List of DOM elements with properties
- `screenshot`: PIL Image (optional)

## Results and Evaluation

### Output Format

Results are saved in JSON format with the following structure:

```json
{
  "config": { ... },
  "results": [
    {
      "env_name": "click-test",
      "episodes": [ ... ],
      "summary": {
        "total_episodes": 5,
        "successful_episodes": 4,
        "success_rate": 0.8,
        "average_reward": 0.75,
        "average_duration": 12.3
      }
    }
  ],
  "overall_summary": {
    "total_environments": 8,
    "total_episodes": 40,
    "successful_episodes": 32,
    "average_success_rate": 0.8
  }
}
```

### Metrics

- **Success Rate**: Percentage of episodes completed successfully
- **Average Reward**: Mean reward across episodes
- **Average Duration**: Mean episode duration in seconds
- **Step Efficiency**: Average steps to completion

## Testing

### Quick Test

Run a basic integration test:
```bash
python test_miniwob_simple.py
```

### Full Integration Test

Run comprehensive tests:
```bash
python test_miniwob_integration.py
```

### Example Test Run

Test with a small set of environments:
```bash
python -m webexp.benchmark.run_miniwob -c configs/example_miniwob_config.yaml
```

## Troubleshooting

### Common Issues

1. **ChromeDriver not found**
   - Install ChromeDriver and add to PATH
   - Or install via package manager: `sudo apt install chromium-driver`

2. **Model API errors**
   - Check API key and endpoint configuration
   - Verify model ID is correct
   - Check rate limits and quotas

3. **Environment setup errors**
   - Ensure all dependencies are installed
   - Check Python version compatibility (3.8+)

4. **Action parsing errors**
   - Check prompt format and examples
   - Verify action syntax in agent output

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Extending the Adaptation

### Adding New Environments

1. Add environment names to the configuration:
```yaml
environments:
  - your-new-environment
```

2. Test with a small number of episodes first

### Customizing Prompts

Modify `MiniWobSolverPromptBuilder` to:
- Add task-specific instructions
- Include additional examples
- Customize action descriptions

### Adding New Action Types

1. Update action parsing in `MiniWobSolverAgent`
2. Add action conversion in `MiniWobBenchmark`
3. Update prompt examples

## Performance Comparison

| Aspect | WebArena | MiniWob++ |
|--------|----------|-----------|
| Setup Complexity | High | Low |
| Environment Diversity | 4 domains | 100+ tasks |
| Task Complexity | High | Variable |
| Execution Speed | Slow | Fast |
| Development Ease | Difficult | Easy |
| Evaluation Clarity | Complex | Clear |

## Future Improvements

1. **Multi-modal Support**: Add screenshot analysis
2. **Curriculum Learning**: Progressive difficulty ordering
3. **Few-shot Learning**: Task-specific examples
4. **Error Recovery**: Better handling of failed actions
5. **Batch Processing**: Parallel environment execution

## Contributing

To contribute to the MiniWob++ adaptation:

1. Test your changes with the integration tests
2. Add new environments to the recommended lists
3. Improve prompt engineering for better performance
4. Add support for new action types or observation formats

## References

- [MiniWob++ Paper](https://arxiv.org/abs/1802.08802)
- [MiniWob++ Documentation](https://miniwob.farama.org/)
- [Go-Browse Paper](https://arxiv.org/abs/2506.03533)
- [BrowserGym Documentation](https://github.com/ServiceNow/BrowserGym)