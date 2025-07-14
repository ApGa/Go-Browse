from .task import Task
from .trace import Trace
from .trajectory import TrajectoryStep, Trajectory
from dataclasses import dataclass
import json
import logging
import os
import shutil
import glob

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def delete_folder_contents(folder: str):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


@dataclass    
class Node: 
    url: str
    tasks: dict[str, Task]
    exploration_tasks: dict[str, Task]
    children: list[str]
    description: str
    prefixes: list[Trace]
    visited: bool
    exp_dir: str
    misc: dict = None
   
    
    def __post_init__(self):
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
            node_info = {
                "url": self.url,
                "description": self.description,
                "children": self.children,
                #"prefix_source": self.prefix_source,
                "visited": self.visited,
                "misc": self.misc
            }
            with open(os.path.join(self.exp_dir, "node_info.json"), "w") as f:
                json.dump(node_info, f, indent=4)
                

            # Save prefixes
            prefix_save_dir = os.path.join(self.exp_dir, "prefixes")
            os.makedirs(prefix_save_dir)
            for i, trace in enumerate(self.prefixes):
                trace_save_dir = os.path.join(prefix_save_dir, f"prefix_{i}")
                os.makedirs(trace_save_dir)
                trace.save(trace_save_dir)
                
    @staticmethod
    def node_from_url(url: str, exp_dir: str):
        return Node(
            url=url,
            tasks={},
            exploration_tasks={},
            children=[],
            description="",
            prefixes=[],
            visited=False,
            exp_dir=exp_dir,
        )
        
            
    @staticmethod
    def load(load_dir: str, load_steps: bool=True, load_prefix: bool=True, load_images: bool=True):
        
        logger.info(f"Loading node from {load_dir}")
        
        with open(os.path.join(load_dir, "node_info.json"), "r") as f:
            node_info = json.load(f)
        
        visited = node_info["visited"]
        
        tasks = {}
        exploration_tasks = {}
        
        if visited:
            if os.path.exists(os.path.join(load_dir, "tasks")):
                for i in range(len(os.listdir(os.path.join(load_dir, "tasks")))):
                    task_load_dir = os.path.join(load_dir, "tasks", f"task_{i}")
                    task = Task.load(task_load_dir, load_steps=load_steps, load_images=load_images)
                    tasks[task.goal] = task
            
            if os.path.exists(os.path.join(load_dir, "exploration_tasks")):
                for i in range(len(os.listdir(os.path.join(load_dir, "exploration_tasks")))):
                    task_load_dir = os.path.join(load_dir, "exploration_tasks", f"task_{i}")
                    task = Task.load(task_load_dir, load_steps=load_steps, load_images=load_images)
                    exploration_tasks[task.goal] = task
            
        prefixes = []
        if load_prefix:

            if os.path.exists(os.path.join(load_dir, "prefixes")):
                # Use glob to find all prefix directories
                prefix_dirs = glob.glob(os.path.join(load_dir, "prefixes", "prefix_*"))
                
                # Sort numerically by extracting the number from each path
                prefix_dirs.sort(key=lambda x: int(x.split("_")[-1]))
                
                for prefix_dir in prefix_dirs:
                    try:
                        prefixes.append(Trace.load(prefix_dir, load_steps=load_steps, load_images=False))
                    except Exception as e:
                        logger.warning(f"Failed to load prefix from {prefix_dir}: {e}")
   

        return Node(
            node_info["url"],
            tasks,
            exploration_tasks,
            node_info["children"],
            node_info["description"],
            prefixes,
            node_info["visited"],
            load_dir,
            misc=node_info.get("misc", None) 
        )
        
    def update_save(self, save_prefix=False, save_info=True):
        if save_info:
            node_info = {
                "url": self.url,
                "description": self.description,
                "children": self.children,
                "visited": self.visited,
                "misc": self.misc
            }
            with open(os.path.join(self.exp_dir, "node_info.json"), "w") as f:
                json.dump(node_info, f, indent=4)
        
        if save_prefix:

            prefix_save_dir = os.path.join(self.exp_dir, "prefixes")
            #delete existing prefixes
            delete_folder_contents(prefix_save_dir)
            for i, trace in enumerate(self.prefixes):
                trace_save_dir = os.path.join(prefix_save_dir, f"prefix_{i}")
                os.makedirs(trace_save_dir, exist_ok=True)
                trace.save(trace_save_dir)


    def add_task(self, goal: str, task_misc: dict = None) -> Task:
        task_dir = os.path.join(self.exp_dir, "tasks", f"task_{len(self.tasks)}")
        processed_goal, _ = Task.process_raw_goal(goal)
        if processed_goal not in self.tasks:
            self.tasks[processed_goal] = Task.from_goal(goal, task_dir, misc=task_misc)
            logger.info(f"Added task for goal: {processed_goal}")
        else:
            logger.warning(f"Task for goal: {processed_goal} already exists. Not adding again.")

        return self.tasks[processed_goal]
    

    def add_tasks(self, goals: list[str], task_misc: dict = None) -> list[Task]:
        return [self.add_task(goal, task_misc) for goal in goals]
        

    def add_exploration_task(self, goal: str, task_misc: dict = None) -> Task:
        task_exp_dir = os.path.join(self.exp_dir, "exploration_tasks", f"task_{len(self.exploration_tasks)}")
        processed_goal, _ = Task.process_raw_goal(goal)
        if processed_goal not in self.exploration_tasks:
            self.exploration_tasks[processed_goal] = Task.from_goal(goal, task_exp_dir, misc=task_misc)
            logger.info(f"Added exploration task for goal: {processed_goal}")
        else:
            logger.warning(f"Exploration task for goal: {processed_goal} already exists. Not adding again.")

        return self.exploration_tasks[processed_goal]
    
    def add_exploration_tasks(self, goals: list[str], task_misc: dict = None) -> list[Task]:
        return [self.add_exploration_task(goal, task_misc) for goal in goals]
 
    def add_trajectory(self, traj: Trajectory, task_misc: dict = None):
        task = self.add_task(traj.goal, task_misc)
        task.add_trajectory(traj)

        
    def add_trajectories(self, trajs: list[Trajectory], task_misc: dict = None):
        for traj in trajs:
            self.add_trajectory(traj, task_misc)

    
    def add_exploration_traj(self, traj: Trajectory):
        task = self.add_exploration_task(traj.goal)
        task.add_trajectory(traj)


    def get_feasible_tasks(self) -> list[Task]:
        
        feasible_tasks = []

        for task in self.tasks.values():
            if task.is_feasible():
                feasible_tasks.append(task)

        return feasible_tasks


    def add_prefix(self, prefix: Trace):
        prefix_save_dir = os.path.join(self.exp_dir, "prefixes")
        os.makedirs(prefix_save_dir, exist_ok=True)
        
        # Find the highest existing prefix number
        existing_prefixes = glob.glob(os.path.join(prefix_save_dir, "prefix_*"))
        if existing_prefixes:
            # Extract numbers and find the highest one
            highest_num = max([int(p.split("_")[-1]) for p in existing_prefixes])
            next_num = highest_num + 1
        else:
            next_num = 0
        
        # Use the next available number
        trace_save_dir = os.path.join(prefix_save_dir, f"prefix_{next_num}")
        os.makedirs(trace_save_dir)
        prefix.save(trace_save_dir)
        self.prefixes.append(prefix)
