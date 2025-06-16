from .base_agent import AgentFactory, BaseAgent
from .trajectory_data import MiniWobAgentStepData
import ast
import logging
import os
import re
import time
import json
import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def extract_action_and_thought(raw_string):
    """Extract thought and action from potentially malformed JSON string.
    
    Args:
        raw_string (str): Raw string containing thought and action
        
    Returns:
        tuple: (action, thought) or (None, None) if extraction fails
    """
    # Initialize defaults
    thought = None
    action = None
    
    try:
        # Look for thought pattern using non-greedy match
        thought_match = re.search(r'"thought"\s*:\s*"(.*?)"(?=\s*[,}])', raw_string, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1)
            # Clean up escaped quotes
            thought = thought.replace('\\"', '"')
            
        # Look for action pattern using non-greedy match    
        action_match = re.search(r'"action"\s*:\s*"(.*?)"(?=\s*[,}])', raw_string, re.DOTALL)
        if action_match:
            action = action_match.group(1)
            # Clean up escaped quotes
            action = action.replace('\\"', '"')
            
    except Exception as e:
        print(f"Error parsing string: {e}")
        return None, None
        
    return action, thought


@AgentFactory.register
class AzureGPTMiniWobAgent(BaseAgent):
    """
    Agent using Azure AI for GPT-4o on MiniWob++ environments.
    """

    def __init__(
            self,
            model_name: str = "gpt-4o",
            api_version: str = "2024-08-01-preview",
            temperature: float = 0.0,
            char_limit: int = -1,
            demo_mode: str = 'off',
    ):
        """
        Initialize the Azure GPT MiniWob++ agent.

        Args:
            model_name (str): The Azure model deployment name.
            api_version (str): Azure API version.
            temperature (float): The temperature to use for sampling.
            char_limit (int): Character limit for prompt truncation.
            demo_mode (str): Whether to run in demo mode.
        """
        
        super().__init__(model_name=model_name, temperature=temperature, char_limit=char_limit, demo_mode=demo_mode)
        
        self.model_name = model_name
        self.api_version = api_version
        self.temperature = temperature
        self.char_limit = char_limit
        self.demo_mode = demo_mode
        
        # Azure AI credentials
        self.api_key_azure = os.getenv("api_key_azure")
        self.api_base_azure = os.getenv("api_base_azure_ai")
        
        if not self.api_key_azure or not self.api_base_azure:
            raise ValueError("Azure AI credentials not found. Please set api_key_azure and api_base_azure_ai environment variables.")
        
        # Import here to avoid dependency issues if not using Azure
        try:
            from langchain_openai import AzureChatOpenAI
            self.llm_azure = AzureChatOpenAI(
                temperature=self.temperature,
                api_key=self.api_key_azure,
                api_version=self.api_version,
                azure_endpoint=self.api_base_azure,
                model_name=self.model_name,
            )
        except ImportError:
            raise ImportError("langchain_openai is required for Azure AI support. Install with: pip install langchain-openai")

        # Import here to avoid circular imports
        from .prompt_builders.miniwob_solver_prompt_builder import MiniWobSolverPromptBuilder
        self.prompt_builder = MiniWobSolverPromptBuilder()
        self.history: list[MiniWobAgentStepData] = []

    def reset(self):
        self.history.clear()

    def obs_preprocessor(self, obs: dict) -> dict:
        """
        Preprocess MiniWob++ observation to extract relevant information.
        """
        
        # Extract DOM elements information
        dom_info = []
        if 'dom_elements' in obs:
            for element in obs['dom_elements']:
                element_info = {
                    'ref': element.get('ref', ''),
                    'tag': element.get('tag', ''),
                    'text': element.get('text', ''),
                    'value': element.get('value', ''),
                    'id': element.get('id', ''),
                    'classes': element.get('classes', []),
                    'bbox': element.get('bbox', {}),
                    'focused': element.get('focused', False),
                    'tampered': element.get('tampered', False)
                }
                dom_info.append(element_info)
        
        return {
            "utterance": obs.get("utterance", ""),
            "fields": obs.get("fields", []),
            "dom_elements": dom_info,
            "screenshot": obs.get("screenshot"),
        }
    
    def action_processor(self, action: str) -> object:
        """
        Process the action string into a MiniWob++ action object.
        """
        parsed_action, thought = extract_action_and_thought(action)
        action_str = parsed_action if parsed_action else action
        
        try:
            return self._parse_action_string(action_str)
        except Exception as e:
            logger.error(f"Error processing action '{action_str}': {str(e)}")
            return {'type': 'none'}
    
    def _parse_action_string(self, action_str: str):
        """Parse action string into MiniWob++ action."""
        action_str = action_str.strip()
        
        # Handle click actions
        if action_str.startswith('click(') and action_str.endswith(')'):
            ref = action_str[6:-1].strip().strip('"\'')
            return {'type': 'click', 'ref': ref}
        
        # Handle type actions
        elif action_str.startswith('type(') and action_str.endswith(')'):
            text = action_str[5:-1].strip().strip('"\'')
            return {'type': 'type', 'text': text}
        
        # Handle key actions
        elif action_str.startswith('key(') and action_str.endswith(')'):
            key = action_str[4:-1].strip().strip('"\'')
            return {'type': 'key', 'key': key}
        
        # Handle scroll actions
        elif action_str.startswith('scroll(') and action_str.endswith(')'):
            direction = action_str[7:-1].strip().strip('"\'')
            if direction.lower() in ['up', 'down', 'left', 'right']:
                return {'type': 'scroll', 'direction': direction}
        
        # Handle done action
        elif action_str.lower() in ['done()', 'done', 'finish()', 'finish']:
            return {'type': 'done'}
        
        # Default to no action if parsing fails
        logger.warning(f"Could not parse action: {action_str}")
        return {'type': 'none'}
    
    def get_action(self, obs: dict, oracle_action: tuple[str, str] = None, **kwargs) -> tuple[str, dict]:
        """
        Get the action for the given observation using Azure AI.
        """

        current_step = MiniWobAgentStepData(
            action=None,
            thought=None,
            utterance=obs.get("utterance", ""),
            dom_elements=obs.get("dom_elements", []),
            misc={}
        )

        if oracle_action is None:
            # Use Azure AI to get response
            response = self.make_azure_llm_call(obs, current_step)
            
            raw_action = response
            action, thought = extract_action_and_thought(raw_action)
            current_step.misc["model_usage"] = {"provider": "azure_ai", "model": self.model_name}
        
        else:
            action, thought = oracle_action
            raw_action = f'{{"thought": "{thought}", "action": "{action}"}}'
            
        print(f"Raw Action:\n {raw_action}")

        current_step.action = action
        current_step.thought = thought
        current_step.misc.update({"thought": thought, "parsed_action": action})
        
        self.history.append(current_step)

        return raw_action, current_step.misc
        
    def make_azure_llm_call(self, obs: dict, current_step: MiniWobAgentStepData) -> str:
        """
        Make a call to Azure AI.
        """
        max_attempts = 3
        attempt = 0
        current_char_limit = self.char_limit
        
        while attempt < max_attempts:
            try:
                # Build messages with current character limit
                messages_dict = self.prompt_builder.build_messages(
                    utterance=obs["utterance"],
                    current_step=current_step,
                    history=self.history,
                    char_limit=current_char_limit if (attempt == 0) or (current_char_limit < 0) else current_char_limit * 2
                )
                
                # Convert to LangChain format
                from langchain_core.messages import SystemMessage, HumanMessage
                
                messages = []
                for msg in messages_dict['prompt']:
                    if msg['role'] == 'system':
                        messages.append(SystemMessage(content=msg['content']))
                    elif msg['role'] == 'user':
                        messages.append(HumanMessage(content=msg['content']))
                
                print(f"Azure AI attempt {attempt+1}: Using char_limit={current_char_limit}")
                
                # Make the actual API call
                response = self.llm_azure.invoke(messages)
                return response.content
                
            except Exception as e:
                attempt += 1
                if attempt >= max_attempts:
                    logger.error(f"Failed after {max_attempts} attempts: {str(e)}")
                    raise
                    
                if attempt > 1:
                    current_char_limit = int(current_char_limit * 0.95)
                logger.warning(f"Retrying with {current_char_limit} character limit after error: {str(e)}")
                
                wait_time = 1.5 * (2 ** (attempt-1)) + (0.1 * attempt)
                logger.info(f"Waiting {wait_time:.2f} seconds before retry")
                time.sleep(wait_time)


@AgentFactory.register
class BedrockClaudeMiniWobAgent(BaseAgent):
    """
    Agent using Amazon Bedrock for Claude on MiniWob++ environments.
    """

    def __init__(
            self,
            model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0",
            temperature: float = 0.0,
            max_tokens: int = 10000,
            char_limit: int = -1,
            demo_mode: str = 'off',
    ):
        """
        Initialize the Bedrock Claude MiniWob++ agent.

        Args:
            model_id (str): The Bedrock model ID.
            temperature (float): The temperature to use for sampling.
            max_tokens (int): Maximum tokens to generate.
            char_limit (int): Character limit for prompt truncation.
            demo_mode (str): Whether to run in demo mode.
        """
        
        super().__init__(model_id=model_id, temperature=temperature, char_limit=char_limit, demo_mode=demo_mode)
        
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.char_limit = char_limit
        self.demo_mode = demo_mode
        
        # AWS credentials
        self.aws_key = os.getenv("AWS_KEY")
        self.aws_secret_key = os.getenv("AWS_SECRET_KEY")
        self.aws_region = os.getenv("AWS_REGION")
        
        if not all([self.aws_key, self.aws_secret_key, self.aws_region]):
            raise ValueError("AWS credentials not found. Please set AWS_KEY, AWS_SECRET_KEY, and AWS_REGION environment variables.")
        
        # Initialize Bedrock client
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.aws_region,
            aws_access_key_id=self.aws_key,
            aws_secret_access_key=self.aws_secret_key,
        )

        # Import here to avoid circular imports
        from .prompt_builders.miniwob_solver_prompt_builder import MiniWobSolverPromptBuilder
        self.prompt_builder = MiniWobSolverPromptBuilder()
        self.history: list[MiniWobAgentStepData] = []

    def reset(self):
        self.history.clear()

    def obs_preprocessor(self, obs: dict) -> dict:
        """
        Preprocess MiniWob++ observation to extract relevant information.
        """
        
        # Extract DOM elements information
        dom_info = []
        if 'dom_elements' in obs:
            for element in obs['dom_elements']:
                element_info = {
                    'ref': element.get('ref', ''),
                    'tag': element.get('tag', ''),
                    'text': element.get('text', ''),
                    'value': element.get('value', ''),
                    'id': element.get('id', ''),
                    'classes': element.get('classes', []),
                    'bbox': element.get('bbox', {}),
                    'focused': element.get('focused', False),
                    'tampered': element.get('tampered', False)
                }
                dom_info.append(element_info)
        
        return {
            "utterance": obs.get("utterance", ""),
            "fields": obs.get("fields", []),
            "dom_elements": dom_info,
            "screenshot": obs.get("screenshot"),
        }
    
    def action_processor(self, action: str) -> object:
        """
        Process the action string into a MiniWob++ action object.
        """
        parsed_action, thought = extract_action_and_thought(action)
        action_str = parsed_action if parsed_action else action
        
        try:
            return self._parse_action_string(action_str)
        except Exception as e:
            logger.error(f"Error processing action '{action_str}': {str(e)}")
            return {'type': 'none'}
    
    def _parse_action_string(self, action_str: str):
        """Parse action string into MiniWob++ action."""
        action_str = action_str.strip()
        
        # Handle click actions
        if action_str.startswith('click(') and action_str.endswith(')'):
            ref = action_str[6:-1].strip().strip('"\'')
            return {'type': 'click', 'ref': ref}
        
        # Handle type actions
        elif action_str.startswith('type(') and action_str.endswith(')'):
            text = action_str[5:-1].strip().strip('"\'')
            return {'type': 'type', 'text': text}
        
        # Handle key actions
        elif action_str.startswith('key(') and action_str.endswith(')'):
            key = action_str[4:-1].strip().strip('"\'')
            return {'type': 'key', 'key': key}
        
        # Handle scroll actions
        elif action_str.startswith('scroll(') and action_str.endswith(')'):
            direction = action_str[7:-1].strip().strip('"\'')
            if direction.lower() in ['up', 'down', 'left', 'right']:
                return {'type': 'scroll', 'direction': direction}
        
        # Handle done action
        elif action_str.lower() in ['done()', 'done', 'finish()', 'finish']:
            return {'type': 'done'}
        
        # Default to no action if parsing fails
        logger.warning(f"Could not parse action: {action_str}")
        return {'type': 'none'}
    
    def get_action(self, obs: dict, oracle_action: tuple[str, str] = None, **kwargs) -> tuple[str, dict]:
        """
        Get the action for the given observation using Bedrock Claude.
        """

        current_step = MiniWobAgentStepData(
            action=None,
            thought=None,
            utterance=obs.get("utterance", ""),
            dom_elements=obs.get("dom_elements", []),
            misc={}
        )

        if oracle_action is None:
            # Use Bedrock to get response
            response = self.make_bedrock_llm_call(obs, current_step)
            
            raw_action = response
            action, thought = extract_action_and_thought(raw_action)
            current_step.misc["model_usage"] = {"provider": "bedrock", "model": self.model_id}
        
        else:
            action, thought = oracle_action
            raw_action = f'{{"thought": "{thought}", "action": "{action}"}}'
            
        print(f"Raw Action:\n {raw_action}")

        current_step.action = action
        current_step.thought = thought
        current_step.misc.update({"thought": thought, "parsed_action": action})
        
        self.history.append(current_step)

        return raw_action, current_step.misc
        
    def make_bedrock_llm_call(self, obs: dict, current_step: MiniWobAgentStepData) -> str:
        """
        Make a call to Bedrock Claude.
        """
        max_attempts = 3
        attempt = 0
        current_char_limit = self.char_limit
        
        while attempt < max_attempts:
            try:
                # Build messages with current character limit
                messages_dict = self.prompt_builder.build_messages(
                    utterance=obs["utterance"],
                    current_step=current_step,
                    history=self.history,
                    char_limit=current_char_limit if (attempt == 0) or (current_char_limit < 0) else current_char_limit * 2
                )
                
                # Convert to Bedrock format
                system_prompt = ""
                user_message = ""
                
                for msg in messages_dict['prompt']:
                    if msg['role'] == 'system':
                        system_prompt = msg['content']
                    elif msg['role'] == 'user':
                        user_message = msg['content']
                
                # Add JSON format instruction to system prompt
                system_prompt += "\n\nAll your responses should be in JSON format with 'thought' and 'action' fields."
                
                messages = [{"role": "user", "content": user_message}]
                
                body = json.dumps({
                    "messages": messages,
                    "system": system_prompt,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": 1,
                    "anthropic_version": "bedrock-2023-05-31"
                })
                
                print(f"Bedrock attempt {attempt+1}: Using char_limit={current_char_limit}")
                
                # Make the actual API call
                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model_id,
                    body=body,
                )
                
                response_body = json.loads(response.get('body').read())
                result = response_body.get('content', [])
                
                if result and len(result) > 0:
                    return result[0].get('text', '')
                else:
                    raise Exception("Empty response from Bedrock")
                
            except Exception as e:
                attempt += 1
                if attempt >= max_attempts:
                    logger.error(f"Failed after {max_attempts} attempts: {str(e)}")
                    raise
                    
                if attempt > 1:
                    current_char_limit = int(current_char_limit * 0.95)
                logger.warning(f"Retrying with {current_char_limit} character limit after error: {str(e)}")
                
                wait_time = 1.5 * (2 ** (attempt-1)) + (0.1 * attempt)
                logger.info(f"Waiting {wait_time:.2f} seconds before retry")
                time.sleep(wait_time)