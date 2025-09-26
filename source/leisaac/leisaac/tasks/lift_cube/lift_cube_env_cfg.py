import torch

from typing import Dict, List

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from leisaac.assets.scenes.simple import TABLE_WITH_CUBE_CFG, TABLE_WITH_CUBE_USD_PATH
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.domain_randomization import randomize_object_uniform, randomize_camera_uniform, domain_randomization
from leisaac.enhance.envs.manager_based_rl_digital_twin_env_cfg import ManagerBasedRLDigitalTwinEnvCfg
from leisaac.utils.env_utils import delete_attribute

from . import mdp
from ..template import SingleArmTaskSceneCfg, SingleArmTaskEnvCfg, SingleArmTerminationsCfg, SingleArmObservationsCfg


# --- Chess stability tweaks for Isaac Sim 4.5 ---
import omni.usd
from pxr import UsdPhysics, PhysxSchema, UsdGeom, Sdf



def stabilize_chess_set():
    stage = omni.usd.get_context().get_stage()
    envs_root = stage.GetPrimAtPath("/World/envs")
    for env in envs_root.GetChildren():
        print(f"Env: {env.GetPath()}")
        CHESS_ROOT = stage.GetPrimAtPath(f"{env.GetPath()}/Scene/chess")

    BOARD_PATH = f"{chess_root}/Board"

    # 1) Make the board static-ish (no gravity; high damping)
    board = stage.GetPrimAtPath(BOARD_PATH)
    if board:
        # Ensure it’s a rigid body (so we can set PhysX props), but keep it effectively static
        UsdPhysics.RigidBodyAPI.Apply(board).CreateRigidBodyEnabledAttr(True)
        pxa = PhysxSchema.PhysxRigidBodyAPI.Apply(board)
        pxa.CreateDisableGravityAttr(True)
        # Extra damping helps kill any residual vibrations from contacts
        pxa.CreateLinearDampingAttr(2.0)
        pxa.CreateAngularDampingAttr(2.0)

    # 2) Tune every child under CHESS_ROOT except the Board
    root = stage.GetPrimAtPath(CHESS_ROOT)
    if not root:
        print(f"[chess] Root not found: {CHESS_ROOT}")
        return

    for prim in root.GetChildren():
        name = prim.GetName()
        if prim.GetPath().pathString == BOARD_PATH or "board" in name.lower():
            continue

        # Ensure piece is a rigid body
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)

        # Set mass (per piece type if you like)
        mapi = UsdPhysics.MassAPI.Apply(prim)
        if any(k in name for k in ["King", "Queen"]):
            mapi.CreateMassAttr(0.40)   # kg
        elif any(k in name for k in ["Rook"]):
            mapi.CreateMassAttr(0.30)
        elif any(k in name for k in ["Bishop", "Knight"]):
            mapi.CreateMassAttr(0.25)
        else:  # Pawn or other
            mapi.CreateMassAttr(0.18)

        # PhysX stability niceties for small objects
        pxa = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        pxa.CreateEnableCCDAttr(True)                 # continuous collision for small fast pieces
        pxa.CreateLinearDampingAttr(0.05)
        pxa.CreateAngularDampingAttr(0.05)

        # If the piece uses very thin visual meshes, set sensible contact/rest offsets
        # (restOffset < contactOffset). Helps avoid initial interpenetration jitter.
        coll_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        coll_api.CreateContactOffsetAttr(0.003)       # 3 mm
        coll_api.CreateRestOffsetAttr(0.001)          # 1 mm

    print("[chess] Board set static; piece masses and physics tuned.")


@configclass
class LiftCubeSceneCfg(SingleArmTaskSceneCfg):
    """Scene configuration for the lift task with chess board baked into scene.usd."""

    scene: AssetBaseCfg = TABLE_WITH_CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    # chess board already lives inside the scene.usd → just point to the prim
    chess: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene/chess",   # NOTE: if yours is "{ENV}/Scene/chess/chess", change to that
        spawn=None,                               # important: we are NOT spawning a new USD
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.6, -0.75, 0.38), rot=(0.77337, 0.55078, -0.2374, -0.20537), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=40.6,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )

    def __post_init__(self):
        super().__post_init__()
        delete_attribute(self, "wrist")
        delete_attribute(self, "cube")  # we don’t use the cube anymore



@configclass
class ObservationsCfg(SingleArmObservationsCfg):
    def __post_init__(self):
        super().__post_init__()
        delete_attribute(self.policy, "wrist")
        # drop cube-related terms if any
        if hasattr(self, "subtask_terms"):
            delete_attribute(self, "subtask_terms")


@configclass
class TerminationsCfg(SingleArmTerminationsCfg):
    def __post_init__(self):
        super().__post_init__()
        if hasattr(self, "success"):
            delete_attribute(self, "success")



@configclass
class LiftCubeEnvCfg(SingleArmTaskEnvCfg):
    scene: LiftCubeSceneCfg = LiftCubeSceneCfg(env_spacing=8.0)
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (-0.4, -0.6, 0.5)
        self.viewer.lookat = (0.9, 0.0, -0.3)

        self.scene.robot.init_state.pos = (0.35, -0.64, 0.05)

        # parse subassets from the baked scene (includes chess now)
        parse_usd_and_create_subassets(TABLE_WITH_CUBE_USD_PATH, self)
        #stabilize_chess_set()





@configclass
class LiftCubeDigitalTwinEnvCfg(LiftCubeEnvCfg, ManagerBasedRLDigitalTwinEnvCfg):
    """Configuration for the lift cube digital twin environment."""

    rgb_overlay_mode: str = "background"

    rgb_overlay_paths: Dict[str, str] = {
        "front": "greenscreen/background-lift-cube.png"
    }

    render_objects: List[SceneEntityCfg] = [
        SceneEntityCfg("cube"),
        SceneEntityCfg("robot"),
    ]
