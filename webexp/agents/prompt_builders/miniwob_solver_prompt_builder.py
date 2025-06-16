from . import BasePromptBuilder, flatten_messages
from ..trajectory_data import MiniWobAgentStepData
from dataclasses import dataclass
from textwrap import dedent
import json

class MiniWobSolverPromptBuilder(BasePromptBuilder):

    def __init__(self):
        pass

    def build_messages(self, obs: dict):
        messages = []
        if "message" in obs:
            messages.append({"text": obs["message"]})
        return messages

    def format_thought_and_action(self, thought: str, action: str) -> str:
        d = {}
        if thought:
            d['thought'] = thought
        if action:
            d['action'] = action
        return json.dumps(d)
    
    def trim_dom_elements(self, dom_elements: list, num_elements_to_remove: int) -> list:
        """Trim DOM elements by removing less important ones."""
        if num_elements_to_remove <= 0 or len(dom_elements) <= num_elements_to_remove:
            return dom_elements[:max(1, len(dom_elements) - num_elements_to_remove)]
        return dom_elements[:-num_elements_to_remove]
    
    def format_dom_elements(self, dom_elements: list) -> str:
        """Format DOM elements for the prompt."""
        if not dom_elements:
            return "No DOM elements available."
        
        formatted_elements = []
        for i, element in enumerate(dom_elements):
            element_str = f"Element {i+1}:"
            element_str += f"\n  - ref: {element.get('ref', 'N/A')}"
            element_str += f"\n  - tag: {element.get('tag', 'N/A')}"
            element_str += f"\n  - text: {element.get('text', 'N/A')}"
            element_str += f"\n  - value: {element.get('value', 'N/A')}"
            if element.get('id'):
                element_str += f"\n  - id: {element.get('id')}"
            if element.get('classes'):
                element_str += f"\n  - classes: {element.get('classes')}"
            if element.get('focused'):
                element_str += f"\n  - focused: {element.get('focused')}"
            formatted_elements.append(element_str)
        
        return "\n\n".join(formatted_elements)
    
    def build_messages(self, utterance: str, current_step: MiniWobAgentStepData, history: list[MiniWobAgentStepData], char_limit: int=-1) -> dict:
        past_thoughts = [step.thought for step in history]
        past_actions = [step.action for step in history]
        
        dom_elements = current_step.dom_elements or []
        completion_thought = current_step.thought
        completion_action = current_step.action

        add_completion = completion_thought or completion_action
        
        messages = self._build_messages(
            utterance,
            past_thoughts,
            past_actions,
            dom_elements,
            completion_thought,
            completion_action
        )
        
        curr_char_count = self.count_message_chars(messages['prompt'] + (messages['completion'] if add_completion else []))
        if char_limit > 0 and curr_char_count > char_limit:
            # First try trimming history
            past_thoughts, past_actions = self.trim_past_thoughts_and_actions(past_thoughts, past_actions, max_allowed=5)
            messages = self._build_messages(
                utterance,
                past_thoughts,
                past_actions,
                dom_elements,
                completion_thought,
                completion_action
            )
            
            curr_char_count = self.count_message_chars(messages['prompt'] + (messages['completion'] if add_completion else []))
            remaining_overflow = curr_char_count - char_limit
            if remaining_overflow > 0:
                # Then try trimming DOM elements
                num_elements_to_remove = min(len(dom_elements) // 2, remaining_overflow // 100)
                dom_elements = self.trim_dom_elements(dom_elements, num_elements_to_remove)
                messages = self._build_messages(    
                    utterance,
                    past_thoughts,
                    past_actions,
                    dom_elements,
                    completion_thought,
                    completion_action
                )

        return {k : flatten_messages(v) for k, v in messages.items() if v}
        
    def count_message_chars(self, messages: list[dict]) -> int:
        return sum([len(m['text']) for message in messages for m in message['content']])
    
    def trim_past_thoughts_and_actions(self, past_thoughts: list[str | None], past_actions: list[str], max_allowed: int=3) -> tuple[list[str | None], list[str]]:
        if len(past_thoughts) > max_allowed:
            past_thoughts = past_thoughts[-max_allowed:]
            past_actions = past_actions[-max_allowed:]
        return past_thoughts, past_actions

    def _build_messages(
        self,
        utterance: str,
        thoughts: list[str | None],
        actions: list[str | None],
        dom_elements: list,
        completion_thought: str | None = None,
        completion_action: str | None = None
    ):
        system_messages = {"role": "system", "content": [self.system_message()]}
        user_messages = {
            "role": "user",
            "content": [
                self.task_message(utterance),
                self.dom_elements_message(dom_elements),
                self.action_space_message(),
                self.action_history_messages(thoughts, actions),
                self.next_action_request_message()
            ]
        }
        
        output = { "prompt": [system_messages, user_messages] }
        
        if completion_thought or completion_action:
            assistant_messages = {
                "role": "assistant",
                "content": [self.completion_message(completion_thought, completion_action)]
            }
            output["completion"] = [assistant_messages]
        
        return output

    def system_message(self):
        return  {
                "type": "text",
                "text": dedent("""\
                    # Instructions
                    You are a web automation assistant designed to complete tasks on web pages. 
                    You will be given a task description and the current state of a web page represented as DOM elements.
                    Your goal is to complete the task by interacting with the web page elements.
                    
                    You must respond with a JSON object containing your reasoning in a "thought" field and your action in an "action" field.
                    """
                )
        }

    def task_message(self, utterance: str):
        return  {
                "type": "text",
                "text": (
                    "# Task\n"
                    f"{utterance}"
                )
        }
        
    def action_space_message(self):
        return  {
                "type": "text",
                "text": dedent("""\
                    # Available Actions
                    
                    You can perform the following actions:
                    
                    1. **click(ref)** - Click on an element
                       - ref: The reference ID of the element to click
                       - Example: click("button_1")
                    
                    2. **type(text)** - Type text into the currently focused input field
                       - text: The text to type
                       - Example: type("Hello World")
                    
                    3. **key(key_name)** - Press a specific key
                       - key_name: The name of the key (e.g., "Enter", "Tab", "Escape")
                       - Example: key("Enter")
                    
                    4. **scroll(direction)** - Scroll the page
                       - direction: "up", "down", "left", or "right"
                       - Example: scroll("down")
                    
                    5. **done()** - Indicate that the task is complete
                       - Example: done()
                    
                    ## Action Examples with Chain-of-Thought:
                    
                    {"thought": "I need to click on the submit button to complete the form submission.", "action": "click('submit_btn')"}
                    
                    {"thought": "I should type the required text into the input field.", "action": "type('example text')"}
                    
                    {"thought": "I need to press Enter to confirm the input.", "action": "key('Enter')"}
                    
                    {"thought": "The task has been completed successfully.", "action": "done()"}
                    """
                )
        }

    def dom_elements_message(self, dom_elements: list):
        return  {
                "type": "text",
                "text": (
                    "# Current Page Elements\n"
                    f"{self.format_dom_elements(dom_elements)}"
                )
        }
        
    def action_history_messages(self, thoughts: list[str | None], actions: list[str]):
        if not thoughts and not actions:
            return {
                "type": "text",
                "text": "# Action History\nNo previous actions."
            }
        
        newline = "\n"
        history_items = []
        for i, (thought, action) in enumerate(zip(thoughts, actions), 1):
            history_items.append(f"Step {i}: {self.format_thought_and_action(thought, action)}")
        
        return  {
                "type": "text",
                "text": (
                    "# Action History\n"
                    f"{newline.join(history_items)}"
                )
        }
        
    def next_action_request_message(self):
        return  {
                "type": "text",
                "text": (
                    "# Next Action\n\n"
                    "Analyze the current page state and task requirements. Think step by step about what action to take next. "
                    "Provide your response as a JSON object with 'thought' and 'action' fields. "
                    "The 'thought' should contain your reasoning, and the 'action' should contain the specific action to perform."
                )
        }
        
    def completion_message(self, completion_thought: str, completion_action: str):
        return  {
                "type": "text",
                "text": f"{self.format_thought_and_action(completion_thought, completion_action)}"
        }