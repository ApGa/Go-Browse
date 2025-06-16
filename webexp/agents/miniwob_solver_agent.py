from .base_agent import AgentFactory, BaseAgent
from .trajectory_data import MiniWobAgentStepData
from openai import OpenAI
import ast
import logging
import os
import re
import time
import json

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
class MiniWobSolverAgent(BaseAgent):
    """
    Agent specifically designed for MiniWob++ environments.
    """

    def __init__(
            self,
            model_id: str,
            model_id_2: str | None = None,
            base_url: str | None = None,
            base_url_2: str | None = None,
            api_key: str | None = None,
            temperature: float = 1.0,
            char_limit: int = -1,
            demo_mode: str = 'off',
    ):
        """
        Initialize the MiniWob++ agent.

        Args:
            model_id (str): The name of the model to use.
            temperature (float): The temperature to use for sampling.
            char_limit (int): Character limit for prompt truncation.
            demo_mode (str): Whether to run in demo mode.
        """
        
        super().__init__(model_id=model_id, temperature=temperature, char_limit=char_limit, demo_mode=demo_mode)
        
        self.model_id = model_id
        self.model_id_2 = model_id_2 or model_id
        self.temperature = temperature
        self.char_limit = char_limit
        self.demo_mode = demo_mode
        
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        base_url_2 = base_url_2 or os.getenv("OPENAI_BASE_URL")
        api_key = api_key or os.getenv("OPENAI_API_KEY", "Unspecified!")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.client_long = OpenAI(base_url=base_url_2, api_key=api_key)

        # Import here to avoid circular imports
        from .prompt_builders.miniwob_solver_prompt_builder import MiniWobSolverPromptBuilder
        self.prompt_builder = MiniWobSolverPromptBuilder()
        self.history: list[MiniWobAgentStepData] = []

    def reset(self):
        self.history.clear()

    def obs_preprocessor(self, obs: dict) -> dict:
        """
        Preprocess MiniWob++ observation to extract relevant information.
        
        MiniWob++ observations contain:
        - utterance: The task instruction
        - fields: Task-specific fields
        - dom_elements: List of DOM elements with properties
        - screenshot: PIL Image (optional)
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
        
        Args:
            action (str): The action string from the agent.
            
        Returns:
            object: MiniWob++ action object.
        """
        parsed_action, thought = extract_action_and_thought(action)
        action_str = parsed_action if parsed_action else action
        
        try:
            # Parse the action string to extract action type and parameters
            return self._parse_action_string(action_str)
        except Exception as e:
            logger.error(f"Error processing action '{action_str}': {str(e)}")
            # Return a no-op action if parsing fails
            return ActionTypes.NONE
    
    def _parse_action_string(self, action_str: str):
        """
        Parse action string into MiniWob++ action.
        
        Expected formats:
        - click(ref)
        - type(text)
        - key(key_name)
        - scroll(direction)
        - done()
        """
        import gymnasium
        import miniwob
        from miniwob.action import ActionTypes
        
        action_str = action_str.strip()
        
        # Handle click actions
        if action_str.startswith('click(') and action_str.endswith(')'):
            ref = action_str[6:-1].strip().strip('"\'')
            # For MiniWob++, we need to create the action using the environment's action creation method
            # This will be handled by the environment wrapper
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
        Get the action for the given observation.

        Args:
            obs (dict): The observation from the environment.
            oracle_action tuple[str, str]: Tuple of (action, thought) to use if available.

        Returns:
            tuple: (raw_action_string, action_info_dict)
        """

        current_step = MiniWobAgentStepData(
            action=None,
            thought=None,
            utterance=obs.get("utterance", ""),
            dom_elements=obs.get("dom_elements", []),
            misc={}
        )

        if oracle_action is None:
            # Use adaptive retry mechanism with character limit reduction
            response = self.make_llm_call_with_adaptive_retry(obs, current_step)
            
            raw_action = response.choices[0].message.content
            action, thought = extract_action_and_thought(raw_action)
            current_step.misc["model_usage"] = response.usage.to_dict()
        
        else:
            action, thought = oracle_action
            raw_action = f'{{"thought": "{thought}", "action": "{action}"}}'
            
        print(f"Raw Action:\n {raw_action}")

        current_step.action = action
        current_step.thought = thought
        current_step.misc.update({"thought": thought, "parsed_action": action})
        
        self.history.append(current_step)

        return raw_action, current_step.misc
        
    def make_llm_call_with_adaptive_retry(self, obs: dict, current_step: MiniWobAgentStepData) -> dict:
        """
        Make a call to the LLM with adaptive retry that reduces character limit on failures.
        
        Args:
            obs (dict): The observation from the environment.
            current_step (MiniWobAgentStepData): The current step data.
            
        Returns:
            dict: The response from the LLM.
        """
        max_attempts = 5
        attempt = 0
        current_char_limit = self.char_limit
        
        while attempt < max_attempts:
            try:
                # Build messages with current character limit
                messages = self.prompt_builder.build_messages(
                    utterance=obs["utterance"],
                    current_step=current_step,
                    history=self.history,
                    char_limit=current_char_limit if (attempt == 0) or (current_char_limit < 0) else current_char_limit * 2
                )['prompt']
                
                print(f"Attempt {attempt+1}: Using char_limit={current_char_limit}")
                
                if attempt == 0:
                    # Make the actual API call
                    return self.client.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                        temperature=self.temperature
                    )
                else:
                    return self.client_long.chat.completions.create(
                        model=self.model_id_2,
                        messages=messages,
                        temperature=self.temperature
                    )
                
            except Exception as e:
                attempt += 1
                if attempt >= max_attempts:
                    logger.error(f"Failed after {max_attempts} attempts: {str(e)}")
                    raise
                    
                if attempt > 1:
                    current_char_limit = int(current_char_limit * 0.95)
                logger.warning(f"Retrying with {current_char_limit} character limit after error: {str(e)}")
                
                if attempt > 1:  # Skip delay for first retry
                    wait_time = 1.5 * (2 ** (attempt-1)) + (0.1 * attempt)
                    logger.info(f"Waiting {wait_time:.2f} seconds before retry")
                    time.sleep(wait_time)
                else:
                    logger.info("Retrying immediately")