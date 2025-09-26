"""Script to run a leisaac replay with leisaac in the simulation."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac replay for leisaac in the simulation.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to load recorded demos.")
parser.add_argument("--replay_mode", type=str, default="action", choices=["action", "state"], help="Replay mode, action: replay the action, state: replay the state.")
parser.add_argument("--select_episodes", type=int, nargs="+", default=[], help="A list of episode indices to replayed. Keep empty to replay all episodes.")
parser.add_argument("--task_type", type=str, default=None, help="Specify task type. If your dataset is recorded with keyboard, you should set it to 'keyboard', otherwise not to set it and keep default value None.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import os
import time
import torch
import gymnasium as gym
import contextlib

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.datasets import HDF5DatasetFileHandler, EpisodeData

import leisaac  # noqa: F401
from leisaac.utils.env_utils import get_task_type, dynamic_reset_gripper_effort_limit_sim
from collections import deque

class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def get_next_action(episode_data: EpisodeData, return_state: bool = False, task_type: str = None):
    if return_state:
        next_state = episode_data.get_next_state()
        if next_state is None:
            return None
        if task_type == "bi-so101leader":
            left_joint_pos = next_state['articulation']['left_arm']['joint_position']
            right_joint_pos = next_state['articulation']['right_arm']['joint_position']
            return torch.cat([left_joint_pos, right_joint_pos], dim=0)
        else:
            return next_state['articulation']['robot']['joint_position']
        print("joint_pos")
        print(next_state['articulation']['robot']['joint_position'])
    else:
        return episode_data.get_next_action()


def main():
    """Replay episodes loaded from a file."""

    # Load dataset
    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    episode_indices_to_replay = args_cli.select_episodes
    if len(episode_indices_to_replay) == 0:
        episode_indices_to_replay = list(range(episode_count))

    num_envs = args_cli.num_envs

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    task_type = get_task_type(args_cli.task, args_cli.task_type)
    env_cfg.use_teleop_device(task_type)

    # Disable all recorders and terminations
    env_cfg.recorders = {}
    env_cfg.terminations = {}

    # create environment from loaded config
    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Get idle action (idle actions are applied to envs without next action)
    # Get idle action (idle actions are applied to envs without next action)
    action_dim = env.action_space.shape[-1]
    if hasattr(env_cfg, "idle_action") and env_cfg.idle_action is not None:
        base_idle = env_cfg.idle_action
        if not torch.is_tensor(base_idle):
            base_idle = torch.as_tensor(base_idle, dtype=torch.float32)
        base_idle = base_idle.to(env.device).view(-1)  # (action_dim,)
        assert base_idle.numel() == action_dim, f"idle_action dim {base_idle.numel()} != action_dim {action_dim}"
        idle_action = base_idle.unsqueeze(0).repeat(num_envs, 1)     # (num_envs, action_dim)
    else:
        idle_action = torch.zeros((num_envs, action_dim), device=env.device, dtype=torch.float32)

    # reset before starting
    env.reset()

    rate_limiter = RateLimiter(args_cli.step_hz)

    episode_names = list(dataset_file_handler.get_episode_names())
    episode_count = len(episode_names)
    print(f"[replay] Dataset has {episode_count} episode(s): {episode_names}")

    # requested indices (0-based; if empty -> all)
    req = list(args_cli.select_episodes or [])
    # support negative indexing
    req = [i if i >= 0 else episode_count + i for i in req]
    # filter valid
    req = [i for i in req if 0 <= i < episode_count]
    if not req:
        req = list(range(episode_count))
    print(f"[replay] Will replay indices (0-based): {req}")

    # queue & counters
    queue = deque(req)
    replayed_episode_count = 0

    # per-env state
    env_episode_data_map: dict[int, EpisodeData | None] = {i: None for i in range(num_envs)}

    # helpers for safe action assignment
    def _to_tensor(x, device, dtype):
        if x is None:
            return None
        t = x if torch.is_tensor(x) else torch.as_tensor(x)
        return t.to(device=device, dtype=dtype).view(-1)

    def _adapt_action_dim(x: torch.Tensor, target_dim: int, last_action: torch.Tensor | None = None) -> torch.Tensor | None:
        """Truncate or pad with zeros (or reuse trailing dims from last_action) to match target_dim."""
        if x is None:
            return None
        src = x.numel()
        if src == target_dim:
            return x
        if src > target_dim:
            return x[:target_dim]
        # pad
        pad = torch.zeros(target_dim - src, device=x.device, dtype=x.dtype)
        if last_action is not None and last_action.numel() == target_dim:
            pad = last_action[src:].clone()
        return torch.cat([x, pad], dim=0)

    # keep last action per env for smoother padding
    last_actions: dict[int, torch.Tensor | None] = {i: None for i in range(num_envs)}

    def load_episode_into_env(env_id: int) -> bool:
        """Pop next index from queue, load it into env_id, reset env to its initial state."""
        nonlocal replayed_episode_count
        if not queue:
            return False
        idx = queue.pop(0) if isinstance(queue, list) else queue.popleft()
        name = episode_names[idx]
        replayed_episode_count += 1
        print(f"{replayed_episode_count:4}: Loading #{idx} ('{name}') into env_{env_id}")
        ep = dataset_file_handler.load_episode(name, env.device)
        env_episode_data_map[env_id] = ep
        init_state = ep.get_initial_state()
        env.reset_to(
            init_state,
            torch.tensor([env_id], device=env.device),
            seed=int(ep.seed) if getattr(ep, "seed", None) is not None else None,
            is_relative=True,
        )
        last_actions[env_id] = None
        return True

    def get_env_joint_names(env, task_type):
        # try a few likely places
        names = None
        try:
            robot = env.scene["robot"]
        except Exception:
            robot = None
        candidates = []
        if robot is not None:
            for attr in ("joint_names",):
                if hasattr(robot, attr):
                    candidates.append(getattr(robot, attr))
            if hasattr(robot, "data") and hasattr(robot.data, "joint_names"):
                candidates.append(robot.data.joint_names)
            if hasattr(robot, "root_physx_view") and hasattr(robot.root_physx_view, "get_dof_names"):
                try:
                    candidates.append(robot.root_physx_view.get_dof_names())
                except Exception:
                    pass
        for c in candidates:
            if c:
                names = list(c)
                break
        if names is None:
            raise RuntimeError("Could not query env joint names/order.")
        return names

    def remap_action_by_names(action_vec, recorded_names, env_names):
        """Return action ordered for env_names, given action in recorded_names order."""
        idx_map = {n: i for i, n in enumerate(recorded_names)}
        out = torch.zeros(len(env_names), device=action_vec.device, dtype=action_vec.dtype)
        for j, name in enumerate(env_names):
            if name in idx_map:
                out[j] = action_vec[idx_map[name]]
            else:
                # missing joint in dataset: keep zero (or carry last action if you prefer)
                out[j] = 0.0
        return out

    # preload first batch
    for env_id in range(num_envs):
        if not load_episode_into_env(env_id):
            break

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            # stop when no active episodes and nothing left in queue
            if all(v is None for v in env_episode_data_map.values()) and not queue:
                break

            # start with idle actions so envs without next action won't move
            actions = idle_action.clone()  # (num_envs, action_dim)
            #print("actions")
            print(actions)
            #print("stop")

            for env_id in range(num_envs):
                ep = env_episode_data_map[env_id]
                if ep is None:
                    # try to load a new one if available
                    load_episode_into_env(env_id)
                    ep = env_episode_data_map[env_id]
                    recorded_action_names = None
                    if hasattr(ep, "metadata") and isinstance(ep.metadata, dict):
                        print("metadata")
                        recorded_action_names = ep.metadata.get("action_joint_names", None)

                    # If you have direct access to the HDF5 group, it might be:
                    # recorded_action_names = ep.h5_group["actions"].attrs.get("joint_names", None)

                    if recorded_action_names:

                        recorded_action_names = [n if isinstance(n, str) else n.decode("utf-8")
                                                for n in recorded_action_names]
                        print(recorded_action_names)
                    env_joint_names = get_env_joint_names(env, task_type)
                    print("env_joint_names")
                    if ep is None:
                        continue

                # 1) try action or state based on mode
                env_next_action = get_next_action(
                    ep,
                    return_state=(args_cli.replay_mode == "state"),
                    task_type=task_type,
                )
                print("envnext")
                print(env_next_action)

                # 2) if user asked for action but dataset lacks it, fallback to state (joint positions)
                if env_next_action is None and args_cli.replay_mode == "action":
                    maybe_state = get_next_action(ep, return_state=True, task_type=task_type)
                    env_next_action = maybe_state
                    print("state:")
                    print(maybe_state)
                
                # 3) if still None, the episode likely ended -> load next episode
                if env_next_action is None:
                    env_episode_data_map[env_id] = None
                    if load_episode_into_env(env_id):
                        ep = env_episode_data_map[env_id]
                        env_next_action = get_next_action(
                            ep,
                            return_state=(args_cli.replay_mode == "state"),
                            task_type=task_type,
                        )
                        if env_next_action is None and args_cli.replay_mode == "action":
                            env_next_action = get_next_action(ep, return_state=True, task_type=task_type)
                        print("2")
                        print(env_next_action)
                    else:
                        continue  # no more episodes → idle this tick

                

                # 4) still None? keep idle for this env
                if env_next_action is None:
                    continue

                # 5) coerce to tensor and adapt dims to env.action_space
                env_next_action = _to_tensor(env_next_action, device=env.device, dtype=idle_action.dtype)
                env_next_action = _adapt_action_dim(env_next_action, actions.shape[-1], last_actions[env_id])

                print("adapt")
                print(env_next_action)

                # 6) if somehow adaptation failed, skip; otherwise assign
                if env_next_action is None:
                    continue

                actions[env_id] = env_next_action
                last_actions[env_id] = env_next_action  # cache

            if args_cli.replay_mode == "action":
                dynamic_reset_gripper_effort_limit_sim(env, task_type)

            # step even if all idle — keeps sim/render alive
            env.step(actions)
            rate_limiter.sleep(env)

    # Close environment after replay is complete
    plural = "s" if replayed_episode_count != 1 else ""
    print(f"Finished replaying {replayed_episode_count} episode{plural}.")
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
