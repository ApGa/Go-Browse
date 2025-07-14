from dataclasses import dataclass
import numpy as np
from PIL import Image
import json
import os

def _extract_text_obs(observation):
    """
    Recursively extract all textual observations from a nested structure.
    """
    if isinstance(observation, str):
        return observation
    elif isinstance(observation, dict):
        return {k: _extract_text_obs(v) for k, v in observation.items() if isinstance(v, (str, dict, list, tuple))}
    elif isinstance(observation, list):
        return [_extract_text_obs(item) for item in observation if isinstance(item, (str, dict, list, tuple))]
    elif isinstance(observation, tuple):
        return tuple(_extract_text_obs(item) for item in observation if isinstance(item, (str, dict, list, tuple)))
    return None

@dataclass
class TrajectoryStep:
    action: str | None
    parsed_action: str | None
    thought: str | None
    observation: dict
    misc: dict | None = None

    def __post_init__(self):
        self._last_saved_dir = None
    
    def save(self, save_dir: str, keep_image_in_memory: bool=False, save_image: bool=True):
        # Extract all textual observations, including nested collections
        text_obs = _extract_text_obs(self.observation)
        step_info = {
            "action": self.action,
            "parsed_action": self.parsed_action,
            "thought": self.thought,
            "observation": text_obs,
            "misc": self.misc
        }
        
        with open(os.path.join(save_dir, "step_info.json"), "w") as f:
            json.dump(step_info, f, indent=4)
        
        if 'screenshot' in self.observation:
            if save_image:
                # Save screenshot
                screenshot = self.observation["screenshot"]
                img = Image.fromarray(screenshot)
                img.save(os.path.join(save_dir, "screenshot.png"))

            if not keep_image_in_memory:
                # Remove the screenshot from memory to save space
                del self.observation["screenshot"]
                self.observation = {k: v for k, v in self.observation.items() if k != "screenshot"}
            
        self._last_saved_dir = save_dir

    @staticmethod
    def load(load_dir: str, load_image: bool=True):
        with open(os.path.join(load_dir, "step_info.json"), "r") as f:
            step_info = json.load(f)
        
        if load_image:
            screenshot = np.asarray(Image.open(os.path.join(load_dir, "screenshot.png")))
            step_info["observation"]["screenshot"] = screenshot
        
        return TrajectoryStep(step_info["action"], step_info["parsed_action"], step_info["thought"], step_info["observation"], step_info["misc"])
    
    @property
    def last_saved_dir(self) -> str | None:
        return self._last_saved_dir


@dataclass
class Trajectory:
    steps: list[TrajectoryStep]
    final_state: TrajectoryStep | None
    goal: str
    reward: float
    success: bool
    response: str
    agent_info: dict
    misc: dict
    
    def add_step(self, action: str, parsed_action: str | None, thought: str | None, observation: dict, misc: dict = None):
        self.steps.append(TrajectoryStep(action, parsed_action, thought, observation, misc))
    
    def extract_response(self, env):
        chat_messages = env.chat.messages
        if chat_messages and chat_messages[-1]["role"] == "assistant":
            self.response = chat_messages[-1]["message"]
        elif chat_messages and chat_messages[-1]["role"] == "infeasible":
            self.response = "User goal/request is infeasible."
            
        return self.response
    
    def save(self, save_dir: str):
        traj_info = {
            "goal": self.goal,
            "reward": self.reward,
            "success": self.success,
            "response": self.response,
            "agent_info": self.agent_info,
            "misc": self.misc
        }
        
        with open(os.path.join(save_dir, "traj_info.json"), "w") as f:
            json.dump(traj_info, f, indent=4)
        
        for i, step in enumerate(self.steps):
            
            step_save_dir = os.path.join(save_dir, f"step_{i}")
            os.makedirs(step_save_dir, exist_ok=True)
            
            step.save(step_save_dir)

        final_state_save_dir = os.path.join(save_dir, "final_state")
        os.makedirs(final_state_save_dir, exist_ok=True)
        if self.final_state is not None:
            self.final_state.save(final_state_save_dir)

    @staticmethod
    def load(load_dir: str, load_steps: bool=True, load_images: bool=True):
        with open(os.path.join(load_dir, "traj_info.json"), "r") as f:
            traj_info = json.load(f)
        
        steps = []
        if load_steps:
            i = 0
            while os.path.exists(os.path.join(load_dir, f"step_{i}")):
                step_load_dir = os.path.join(load_dir, f"step_{i}")
                steps.append(TrajectoryStep.load(step_load_dir, load_image=load_images))
                i += 1

        final_state_load_dir = os.path.join(load_dir, "final_state")
        final_state = TrajectoryStep.load(final_state_load_dir, load_image=load_images)
        
        return Trajectory(steps, final_state, traj_info["goal"], traj_info["reward"], traj_info["success"], traj_info["response"], traj_info["agent_info"], traj_info["misc"])
        
    def __len__(self):
        return len(self.steps)
    
    @staticmethod    
    def from_goal(goal: str, agent_info: dict = None, misc: dict = None):
        if agent_info is None:
            agent_info = {}
        if misc is None:
            misc = {}
        return Trajectory([], None, goal, 0.0, False, "N/A", agent_info=agent_info, misc=misc)
