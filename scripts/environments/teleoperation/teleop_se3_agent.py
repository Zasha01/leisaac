# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a leisaac teleoperation with leisaac manipulation environments."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

from isaaclab.app import AppLauncher



# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac teleoperation for leisaac environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", choices=['keyboard', 'so101leader', 'bi-so101leader'], help="Device for interacting with environment")
parser.add_argument("--port", type=str, default='/dev/ttyACM0', help="Port for the teleop device:so101leader, default is /dev/ttyACM0")
parser.add_argument("--left_arm_port", type=str, default='/dev/ttyACM0', help="Port for the left teleop device:bi-so101leader, default is /dev/ttyACM0")
parser.add_argument("--right_arm_port", type=str, default='/dev/ttyACM1', help="Port for the right teleop device:bi-so101leader, default is /dev/ttyACM1")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

# recorder_parameter
parser.add_argument("--record", action="store_true", help="whether to enable record function")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos.")
parser.add_argument("--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite.")

parser.add_argument("--recalibrate", action="store_true", help="recalibrate SO101-Leader or Bi-SO101Leader")
parser.add_argument("--quality", action="store_true", help="whether to enable quality render mode.")

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

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.managers import TerminationTermCfg

from leisaac.devices import Se3Keyboard, SO101Leader, BiSO101Leader
from leisaac.enhance.managers import StreamingRecorderManager
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim

import random
from pxr import UsdGeom, Gf

import omni.usd
from pxr import UsdPhysics, PhysxSchema

from pxr import UsdGeom, Gf
import re

def spawn_red_cube_over_chess():
    stage = omni.usd.get_context().get_stage()
    envs_root = stage.GetPrimAtPath("/World/envs")
    if not envs_root or not envs_root.IsValid():
        print("[cube] /World/envs not ready.")
        return

    for env_prim in envs_root.GetChildren():
        env_ns = env_prim.GetPath().pathString
        chess_root = stage.GetPrimAtPath(f"{env_ns}/Scene/chess/chess/chess_001")
        if not chess_root or not chess_root.IsValid():
            continue

        board_prim = stage.GetPrimAtPath(f"{chess_root.GetPath()}/Board")
        if not board_prim or not board_prim.IsValid():
            print(f"[cube] No Board under {chess_root.GetPath()}")
            continue

        # Create cube prim above the board
        cube_path = f"{chess_root.GetPath()}/RedCube"
        cube = UsdGeom.Cube.Define(stage, cube_path)

        # Highlight chess field
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3f(0.024, 0.024, -0.00832))   
        xform.AddScaleOp().Set(Gf.Vec3f(0.022, 0.022, 0.0002))    

        # Make it red
        cube.CreateDisplayColorAttr([(1.0, 0.0, 0.0)])

        print(f"[cube] Spawned static red cube at {cube_path}")


import re
from pxr import UsdGeom, Gf, Usd
import omni.usd

from leisaac.chess import init_random_position_from_db, compute_and_highlight_best_move
# --- Board extents and step (meters) ---
X_MIN, X_MAX = -0.164,  0.164
Y_MIN, Y_MAX = -0.164,  0.164
Z_PLANE       = -0.00832
STEP_X = (X_MAX - X_MIN) / 7.0   # 0.328 / 7
STEP_Y = (Y_MAX - Y_MIN) / 7.0

# If you ever need to flip orientation, toggle these:
FLIP_FILES = True   # True makes 'a' on +X
FLIP_RANKS = True   # True makes rank 1 on +Y

def _square_center(square: str) -> Gf.Vec3f:
    m = re.fullmatch(r'([a-hA-H])([1-8])', square.strip())
    if not m:
        raise ValueError(f"Invalid square '{square}'. Use like 'e5'.")
    file_idx = ord(m.group(1).lower()) - ord('a')   # 0..7
    rank_idx = int(m.group(2)) - 1                  # 0..7

    if FLIP_FILES:
        file_idx = 7 - file_idx
    if FLIP_RANKS:
        rank_idx = 7 - rank_idx

    x = X_MIN + file_idx * STEP_X
    y = Y_MIN + rank_idx * STEP_Y
    return Gf.Vec3f(x, y, Z_PLANE)

def highlight_square(square: str):
    """Spawn or move a static red overlay cube to the given square in all envs."""
    stage = omni.usd.get_context().get_stage()
    envs_root = stage.GetPrimAtPath("/World/envs")
    if not envs_root or not envs_root.IsValid():
        print("[highlight] /World/envs not ready.")
        return

    pos = _square_center(square)

    for env_prim in envs_root.GetChildren():
        env_ns = env_prim.GetPath().pathString
        chess_root = stage.GetPrimAtPath(f"{env_ns}/Scene/chess/chess/chess_001")
        if not chess_root or not chess_root.IsValid():
            continue

        cube_path = f"{chess_root.GetPath()}/Highlight_{square.lower()}"
        cube = UsdGeom.Cube.Define(stage, cube_path)

        # Reset xform ops so repeated calls overwrite cleanly
        xform = UsdGeom.Xformable(cube)
        if xform.GetOrderedXformOps():
            xform.ClearXformOpOrder()

        # Use your calibrated size and an exact position on the board plane
        xform.AddTranslateOp().Set(pos)
        xform.AddScaleOp().Set(Gf.Vec3f(0.022, 0.022, 0.0002))  # your size

        cube.CreateDisplayColorAttr([(1.0, 0.0, 0.0)])

    print(f"[highlight] {square} → ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.5f})")


def _xy_to_square_in_chess_local(x: float, y: float) -> str:
    # snap to nearest 0..7 index in each axis
    fx = round((x - X_MIN) / STEP_X)
    fy = round((y - Y_MIN) / STEP_Y)
    fx = max(0, min(7, int(fx)))
    fy = max(0, min(7, int(fy)))
    if FLIP_FILES:
        fx = 7 - fx
    if FLIP_RANKS:
        fy = 7 - fy
    return chr(ord('a') + fx) + str(fy + 1)

def print_chess_positions():
    """Map each piece to a chess square by projecting into '.../chess_001' local frame."""
    stage = omni.usd.get_context().get_stage()
    envs_root = stage.GetPrimAtPath("/World/envs")
    if not envs_root or not envs_root.IsValid():
        print("[chess] /World/envs not ready.")
        return

    # caches for robust bounds & transforms
    time = Usd.TimeCode.Default()
    bbox_cache = UsdGeom.BBoxCache(time, includedPurposes=[UsdGeom.Tokens.default_], useExtentsHint=True)
    xform_cache = UsdGeom.XformCache(time)

    for env_prim in envs_root.GetChildren():
        env_ns = env_prim.GetPath().pathString
        chess_root = stage.GetPrimAtPath(f"{env_ns}/Scene/chess/chess/chess_001")
        if not chess_root or not chess_root.IsValid():
            continue

        # chess_001 local->world, then invert to get world->chess_001
        M_world_from_chess = xform_cache.GetLocalToWorldTransform(chess_root)
        M_chess_from_world = Gf.Matrix4d(M_world_from_chess).GetInverse()

        print(f"\n[chess] Positions in {env_ns}:")
        for prim in chess_root.GetChildren():
            name = prim.GetName().lower()
            if "board" in name or name.startswith("highlight_"):
                continue

            # world-space AABB center
            aabb = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            c_world = (aabb.GetMin() + aabb.GetMax()) * 0.5  # Gf.Vec3d

            # project into chess_001 local frame
            c_local = M_chess_from_world.Transform(c_world)

            sq = _xy_to_square_in_chess_local(c_local[0], c_local[1])
            print(f"  {prim.GetName():<12} -> {sq}    (local x={c_local[0]: .3f}, y={c_local[1]: .3f})")



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


def main():
    """Running lerobot teleoperation with leisaac manipulation environment."""

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(args_cli.teleop_device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    task_name = args_cli.task

    if args_cli.quality:
        env_cfg.sim.render.antialiasing_mode = 'FXAA'
        env_cfg.sim.render.rendering_mode = 'quality'

    # precheck task and teleop device
    if "BiArm" in task_name:
        assert args_cli.teleop_device == "bi-so101leader", "only support bi-so101leader for bi-arm task"

    # modify configuration
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    if args_cli.record:
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        if not hasattr(env_cfg.terminations, "success"):
            setattr(env_cfg.terminations, "success", None)
        env_cfg.terminations.success = TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    else:
        env_cfg.recorders = None

    # create environment
    env: ManagerBasedRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped
    #highlight_square("c5")
    #init_random_position_from_db()
    #print_chess_positions()
    # replace the original recorder manager with the streaming recorder manager
    if args_cli.record:
        del env.recorder_manager
        env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
        env.recorder_manager.flush_steps = 100
        env.recorder_manager.compression = 'lzf'

    # create controller
    if args_cli.teleop_device == "keyboard":
        teleop_interface = Se3Keyboard(env, sensitivity=0.25 * args_cli.sensitivity)
    elif args_cli.teleop_device == "so101leader":
        teleop_interface = SO101Leader(env, port=args_cli.port, recalibrate=args_cli.recalibrate)
    elif args_cli.teleop_device == "bi-so101leader":
        teleop_interface = BiSO101Leader(env, left_port=args_cli.left_arm_port, right_port=args_cli.right_arm_port, recalibrate=args_cli.recalibrate)
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'so101leader', 'bi-so101leader'."
        )

    # add teleoperation key for env reset
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # add teleoperation key for task success
    should_reset_task_success = False

    def reset_task_success():
        nonlocal should_reset_task_success
        should_reset_task_success = True
        reset_recording_instance()

    teleop_interface.add_callback("R", reset_recording_instance)
    teleop_interface.add_callback("N", reset_task_success)
    print(teleop_interface)

    rate_limiter = RateLimiter(args_cli.step_hz)

    # reset environment
    env.reset()
    teleop_interface.reset()

    compute_and_highlight_best_move()

    current_recorded_demo_count = 0

    start_record_state = False

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            dynamic_reset_gripper_effort_limit_sim(env, args_cli.teleop_device)
            actions = teleop_interface.advance()
            if should_reset_task_success:
                print("Task Success!!!")
                should_reset_task_success = False
                if args_cli.record:
                    env.termination_manager.set_term_cfg("success", TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)))
                    env.termination_manager.compute()
            if should_reset_recording_instance:
                env.reset()
                compute_and_highlight_best_move()
                should_reset_recording_instance = False
                if start_record_state:
                    if args_cli.record:
                        print("Stop Recording!!!")
                    start_record_state = False
                if args_cli.record:
                    env.termination_manager.set_term_cfg("success", TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)))
                    env.termination_manager.compute()
                # print out the current demo count if it has changed
                if args_cli.record and env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                    current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                    print(f"Recorded {current_recorded_demo_count} successful demonstrations.")
                if args_cli.record and args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                    print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                    break

            elif actions is None:
                env.render()
            # apply actions
            else:
                if not start_record_state:
                    if args_cli.record:
                        print("Start Recording!!!")
                    start_record_state = True
                env.step(actions)
            if rate_limiter:
                rate_limiter.sleep(env)

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
