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
