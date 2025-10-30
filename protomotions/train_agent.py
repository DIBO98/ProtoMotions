import os
import sys
import platform
from pathlib import Path
import logging
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

# wandb는 선택적 사용
try:
    import wandb
    from lightning.pytorch.loggers import WandbLogger
except Exception:
    wandb = None
    WandbLogger = tuple()  # isinstance 체크 회피용

from lightning.fabric import Fabric

# --- 기존 코드와 동일한 시뮬레이터 자동 감지 로직 ---
has_robot_arg = False
simulator = None
for arg in sys.argv:
    if "robot" in arg:
        has_robot_arg = True
    if "simulator" in arg:
        if not has_robot_arg:
            raise ValueError("+robot argument should be provided before +simulator")
        sim_val = arg.split("=", 1)[-1]
        if "isaacgym" in sim_val:
            import isaacgym  # noqa: F401
            simulator = "isaacgym"
        elif "isaaclab" in sim_val:
            # AppLauncher는 isaaclab 경로 하위에 있음
            from isaaclab.app import AppLauncher
            simulator = "isaaclab"

# 로거
log = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="base")
def main(config):
    log.info("==== Train Agent Entry ====")

    # ----- Fabric(DDP) 설정 자동화 -----
    # 사용자가 config에 fabric을 안 넣었을 수도 있으므로 보정
    if "fabric" not in config or config.fabric is None:
        config.fabric = OmegaConf.create({})

    # 기본값 보정
    cfg_fabric = config.fabric
    # (우선순위: 커맨드라인/기존 설정 > 자동 추천)
    if "accelerator" not in cfg_fabric or cfg_fabric.accelerator is None:
        cfg_fabric.accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # 4장 이상이면 자동으로 DDP 4장 권장, 아니면 1장
    if torch.cuda.is_available() and torch.cuda.device_count() >= 4:
        cfg_fabric.strategy = cfg_fabric.get("strategy", "ddp")
        cfg_fabric.devices = cfg_fabric.get("devices", 4)
        log.info(f"Launching with {torch.cuda.device_count()} GPUs (DDP with {cfg_fabric.devices} devices).")
    else:
        cfg_fabric.strategy = cfg_fabric.get("strategy", "auto")
        cfg_fabric.devices = cfg_fabric.get("devices", 1)
        log.info("Launching with single GPU (auto).")

    if "precision" not in cfg_fabric or cfg_fabric.precision is None:
        # A100이면 bf16-mixed 권장
        cfg_fabric.precision = "bf16-mixed" if torch.cuda.is_available() else "32-true"

    # macOS 보정
    if platform.system() == "Darwin":
        log.info("Found OSX device")
        cfg_fabric.accelerator = "mps"
        cfg_fabric.strategy = "auto"

    # ----- Fabric 생성/런치 전에 GPU 인덱스 환경변수 고정 준비 -----
    # LOCAL_RANK는 fabric.launch() 이후에도 접근 가능하지만,
    # Isaac(Omni) 쪽은 초기화 시점에 GPU를 읽으므로 사전 설정
    if torch.cuda.is_available():
        local_rank_pre = int(os.environ.get("LOCAL_RANK", 0))
        os.environ["OMNI_KIT_GPU_INDEX"] = str(local_rank_pre)

    # Fabric init & launch
    fabric: Fabric = instantiate(cfg_fabric)
    fabric.launch()

    # launch 이후 최종 디바이스 고정 (PyTorch 쪽)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    # 해석 안 된 상태 보존(추후 저장용). inference override 위해 resolve=False 유지
    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    os.chdir(hydra.utils.get_original_cwd())

    # matmul 정밀도
    torch.set_float32_matmul_precision("high")

    # 체크포인트 경로 세팅
    save_dir = Path(config.save_dir)
    pre_existing_checkpoint = save_dir / "last.ckpt"
    checkpoint_config_path = save_dir / "config.yaml"
    if pre_existing_checkpoint.exists():
        log.info(f"Found latest checkpoint at {pre_existing_checkpoint}")
        if checkpoint_config_path.exists():
            log.info(f"Loading config from {checkpoint_config_path}")
            config = OmegaConf.load(checkpoint_config_path)
        config.checkpoint = pre_existing_checkpoint

    # ----- IsaacLab / IsaacGym 환경 생성 -----
    if simulator == "isaaclab":
        # rank별 동일 GPU 사용 보장
        os.environ["OMNI_KIT_GPU_INDEX"] = str(fabric.local_rank)

        app_launcher_flags = {"headless": config.headless}
        if fabric.world_size > 1:
            app_launcher_flags["distributed"] = True
            os.environ["LOCAL_RANK"] = str(fabric.local_rank)
            os.environ["RANK"] = str(fabric.global_rank)

        # 일부 빌드에서 RTX 디바이스 ordinal 키가 먹는 경우가 있어 주석으로 남김
        # app_launcher_flags["/rtx/device/ordinal"] = fabric.local_rank

        app_launcher = AppLauncher(app_launcher_flags)
        simulation_app = app_launcher.app
        env = instantiate(config.env, device=fabric.device, simulation_app=simulation_app)
    else:
        # isaacgym 또는 기타
        env = instantiate(config.env, device=fabric.device)

    # ----- 에이전트 생성/로드 -----
    from protomotions.agents.ppo.agent import PPO  # 지연 import로 시작시간 감소
    agent: PPO = instantiate(config.agent, env=env, fabric=fabric)
    agent.setup()
    agent.fabric.strategy.barrier()
    agent.load(config.checkpoint)

    # ----- 초기 1회 config 저장(wandb id 포함) -----
    if fabric.global_rank == 0 and not checkpoint_config_path.exists():
        if wandb is not None and "wandb" in config:
            for logger in fabric.loggers:
                if isinstance(logger, WandbLogger):
                    logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))
            if wandb.run is not None:
                wandb_id = wandb.run.id
                log.info(f"wandb_id found {wandb_id}")
                # unresolved_conf가 dict이므로 안전하게 접근
                if "wandb" in unresolved_conf and isinstance(unresolved_conf["wandb"], dict):
                    unresolved_conf["wandb"]["wandb_id"] = wandb_id

        log.info(f"Saving config file to {save_dir}")
        with open(checkpoint_config_path, "w") as file:
            OmegaConf.save(unresolved_conf, file)

    agent.fabric.strategy.barrier()

    # ----- 학습 시작 -----
    agent.fit()


if __name__ == "__main__":
    main()

