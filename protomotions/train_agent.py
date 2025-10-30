import os

os.environ["WANDB_DISABLE_SENTRY"] = "true"  # Must be first environment variable
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_DISABLE_CODE"] = "true"

import sys
import platform
from pathlib import Path
import logging
import hydra
from hydra.utils import instantiate

has_robot_arg = False
simulator = None
for arg in sys.argv:
    # This hack ensures that isaacgym is imported before any torch modules.
    # The reason it is here (and not in the main func) is due to pytorch lightning multi-gpu behavior.
    if "robot" in arg:
        has_robot_arg = True
    if "simulator" in arg:
        if not has_robot_arg:
            raise ValueError("+robot argument should be provided before +simulator")
        if "isaacgym" in arg.split("=")[-1]:
            import isaacgym  # noqa: F401

            simulator = "isaacgym"
        elif "isaaclab" in arg.split("=")[-1]:
            from isaaclab.app import AppLauncher

            simulator = "isaaclab"

import wandb  # noqa: E402
from lightning.pytorch.loggers import WandbLogger  # noqa: E402
import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402
from utils.config_utils import *  # noqa: E402, F403
from utils.common import seeding  # noqa: E402, F403

from protomotions.agents.ppo.agent import PPO  # noqa: E402

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="base")
def main(config: OmegaConf):
    import torch
    from omegaconf import OmegaConf
    import os
    import platform
    from pathlib import Path

    log.info("==== Train Agent Entry ====")

    # ---- Auto-enable 4-GPU DDP if available ----
    if "fabric" not in config:
        config.fabric = OmegaConf.create({})

    if torch.cuda.device_count() >= 4:
        config.fabric.accelerator = "gpu"
        config.fabric.strategy = "ddp"
        config.fabric.devices = 4
        if "precision" not in config.fabric or config.fabric.precision is None:
            config.fabric.precision = "bf16-mixed"
        log.info(f"Launching with {torch.cuda.device_count()} GPUs (DDP with 4 devices).")
    else:
        config.fabric.accelerator = "gpu"
        config.fabric.strategy = "auto"
        config.fabric.devices = 1
        if "precision" not in config.fabric or config.fabric.precision is None:
            config.fabric.precision = "bf16-mixed"
        log.info("Launching with single GPU (auto).")

    # Fabric init
    fabric: Fabric = instantiate(config.fabric)
    fabric.launch()

    if platform.system() == "Darwin":
        log.info("Found OSX device")
        config.fabric.accelerator = "mps"
        config.fabric.strategy = "auto"

    # resolve=False is important otherwise overrides at inference time won't work properly
    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    os.chdir(hydra.utils.get_original_cwd())

    torch.set_float32_matmul_precision("high")

    save_dir = Path(config.save_dir)
    pre_existing_checkpoint = save_dir / "last.ckpt"
    checkpoint_config_path = save_dir / "config.yaml"
    if pre_existing_checkpoint.exists():
        log.info(f"Found latest checkpoint at {pre_existing_checkpoint}")
        if checkpoint_config_path.exists():
            log.info(f"Loading config from {checkpoint_config_path}")
            config = OmegaConf.load(checkpoint_config_path)
        config.checkpoint = pre_existing_checkpoint

    # launch App / Env
    if simulator == "isaaclab":
        app_launcher_flags = {"headless": config.headless}
        if fabric.world_size > 1:
            app_launcher_flags["distributed"] = True
            os.environ["LOCAL_RANK"] = str(fabric.local_rank)
            os.environ["RANK"] = str(fabric.global_rank)
        app_launcher = AppLauncher(app_launcher_flags)
        simulation_app = app_launcher.app
        env = instantiate(config.env, device=fabric.device, simulation_app=simulation_app)
    else:
        env = instantiate(config.env, device=fabric.device)

    agent: PPO = instantiate(config.agent, env=env, fabric=fabric)
    agent.setup()
    agent.fabric.strategy.barrier()
    agent.load(config.checkpoint)

    # Save config with wandb id if first run
    if fabric.global_rank == 0 and not checkpoint_config_path.exists():
        if "wandb" in config:
            for logger in fabric.loggers:
                if isinstance(logger, WandbLogger):
                    logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))
            wandb_id = wandb.run.id
            log.info(f"wandb_id found {wandb_id}")
            unresolved_conf["wandb"]["wandb_id"] = wandb_id
        log.info(f"Saving config file to {save_dir}")
        with open(checkpoint_config_path, "w") as file:
            OmegaConf.save(unresolved_conf, file)

    agent.fabric.strategy.barrier()
    agent.fit()



if __name__ == "__main__":
    main()
