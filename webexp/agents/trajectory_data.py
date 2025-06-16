from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy

@dataclass
class StepData:
    misc : dict | None = None

    def process_for_dataset(self) -> StepData:
        return deepcopy(self)
    
    def process_for_prompt(self) -> StepData:
        return deepcopy(self)

@dataclass
class TrajectoryData:
    steps: list[StepData]
    goal: str
    reward: float
    misc: dict | None = None

    def process_for_dataset(self) -> TrajectoryData:
        return deepcopy(self)
    
    def process_for_prompt(self) -> TrajectoryData:
        return deepcopy(self)
    

@dataclass
class BrowserGymAgentStepData(StepData):
    action: str | None = None
    thought: str | None = None
    axtree: str | None = None
    last_action_error: str | None = None
    
    def process_for_dataset(self) -> BrowserGymAgentStepData:
        if self.misc is None:
            self.misc = {}
        
        if self.action is None or self.thought is None or self.axtree is None:
            self.misc["skip"] = True
        
        
        return BrowserGymAgentStepData(
            action=self.action,
            thought=self.thought,
            axtree=self.axtree,
            last_action_error=self.last_action_error,
            misc=self.misc
        )


@dataclass
class MiniWobAgentStepData(StepData):
    action: str | None = None
    thought: str | None = None
    utterance: str | None = None
    dom_elements: list | None = None
    
    def process_for_dataset(self) -> MiniWobAgentStepData:
        if self.misc is None:
            self.misc = {}
        
        if self.action is None or self.thought is None or self.utterance is None:
            self.misc["skip"] = True
        
        return MiniWobAgentStepData(
            action=self.action,
            thought=self.thought,
            utterance=self.utterance,
            dom_elements=self.dom_elements,
            misc=self.misc
        )


@dataclass
class BrowserGymAgentTrajectoryData(TrajectoryData):
    steps: list[BrowserGymAgentStepData]
    goal: str
    reward: float
    misc: dict | None = None

    def process_for_dataset(self) -> BrowserGymAgentTrajectoryData:
        processed_steps = [step.process_for_dataset() for step in self.steps]
        processed_steps = [step for step in processed_steps if not step.misc.get("skip", False)]

        if not processed_steps:
            self.misc["skip"] = True
            return BrowserGymAgentTrajectoryData(
                steps=processed_steps,
                goal=self.goal,
                reward=self.reward,
                misc=self.misc
            )

        reward = self.reward
        
        if processed_steps[-1].action and "report_infeasible" in processed_steps[-1].action and reward > 0:
            self.misc["skip"] = True
            return BrowserGymAgentTrajectoryData(
                steps=processed_steps,
                goal=self.goal,
                reward=0.0,
                misc=self.misc
            )

        # If there are a trailing set of "noop" actions, we should remove them from the end of the trajectory
        while len(processed_steps) > 1 and processed_steps[-1].action and "noop" in processed_steps[-1].action:
            processed_steps.pop()
            
        # TODO: Consider removing repeated action. Though, this might not always be accurate.

        return BrowserGymAgentTrajectoryData(
            steps=processed_steps,
            goal=self.goal,
            reward=self.reward,
            misc=self.misc
        )
    
    def process_for_prompt(self) -> BrowserGymAgentTrajectoryData:
        processed_steps = [step.process_for_prompt() for step in self.steps]
        return BrowserGymAgentTrajectoryData(
            steps=processed_steps,
            goal=self.goal,
            reward=self.reward,
            misc=self.misc
        )
