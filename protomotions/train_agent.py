import os
import sys
import platform
from pathlib import Path
import logging
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from omegaconf import OmegaConf
import math

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

def _safe_eval(expr: str):
    """OmegaConf ${eval:...}용 안전한 eval(수학 연산만 허용)."""
    return eval(expr, {"__builtins__": None, "math": math}, {})

def _safe_register_resolvers():
    # 멀티랭크/중복 호출에도 안전하도록 교체 등록
    OmegaConf.register_new_resolver("len", lambda x: len(x), replace=True)
    OmegaConf.register_new_resolver("eval", _safe_eval, replace=True)

_safe_register_resolvers()


@hydra.main(config_path="config", config_name="base")
def main(config: OmegaConf):
    # --- 기본 로그 ---
    log.info("==== Train Agent Entry ====")

    # --- Fabric 설정 자동 보정 (GPU 개수 기반) ---
    if "fabric" not in config:
        config.fabric = OmegaConf.create({})

    cuda_ok = torch.cuda.is_available()
    ngpu = torch.cuda.device_count() if cuda_ok else 0

    # precision 기본값
    if "precision" not in config.fabric or config.fabric.get("precision") is None:
        config.fabric.precision = "bf16-mixed" if cuda_ok else "32-true"

    if ngpu >= 4:
        # 4장 사용할 수 있으면 DDP로
        config.fabric.accelerator = "gpu"
        config.fabric.strategy = "ddp"
        config.fabric.devices = 4
        log.info(f"Launching with {ngpu} GPUs (use 4 via DDP).")
    elif ngpu >= 1:
        config.fabric.accelerator = "gpu"
        config.fabric.strategy = "auto"
        config.fabric.devices = 1
        log.info("Launching with single GPU.")
    else:
        config.fabric.accelerator = "cpu"
        config.fabric.strategy = "auto"
        config.fabric.devices = 1
        log.info("Launching on CPU.")

    # --- Fabric 인스턴스화 (Hydra instantiate or direct) ---
    def _make_fabric(cfg_node):
        if isinstance(cfg_node, dict) and "_target_" in cfg_node:
            return instantiate(cfg_node)
        # 직접 생성 경로
        return Fabric(
            accelerator=cfg_node.get("accelerator", "gpu" if cuda_ok else "cpu"),
            devices=cfg_node.get("devices", 1),
            strategy=cfg_node.get("strategy", "auto"),
            precision=cfg_node.get("precision", "bf16-mixed" if cuda_ok else "32-true"),
        )

    # Hydra가 작업 디렉토리 바꾸기 전에 원본 CWD 확보
    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    os.chdir(hydra.utils.get_original_cwd())

    # 행렬 정밀도 최적화
    torch.set_float32_matmul_precision("high")

    # --- Fabric 런치 ---
    fabric: Fabric = _make_fabric(config.fabric)
    fabric.launch()

    # 랭크별 CUDA/Omni 장치 고정
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if cuda_ok:
        torch.cuda.set_device(local_rank)

    # Isaac Sim(Omni) 쪽은 PyTorch와 장치 열거가 다를 수 있음 → 랭크별 GPU 고정
    # (Omni 경고 회피: CUDA_VISIBLE_DEVICES에 의존하지 않고 OMNI_KIT_GPU_INDEX로 지정)
    os.environ["OMNI_KIT_GPU_INDEX"] = str(local_rank)

    # --- 기존 macOS 처리(참고) ---
    if platform.system() == "Darwin":
        config.fabric.accelerator = "mps"
        config.fabric.strategy = "auto"
        log.info("Found macOS (MPS)")

    # --- 체크포인트 경로/저장 디렉토리 준비 ---
    save_dir = Path(config.save_dir)
    pre_existing_checkpoint = save_dir / "last.ckpt"
    checkpoint_config_path = save_dir / "config.yaml"
    if pre_existing_checkpoint.exists():
        log.info(f"Found latest checkpoint at {pre_existing_checkpoint}")
        if checkpoint_config_path.exists():
            log.info(f"Loading config from {checkpoint_config_path}")
            config = OmegaConf.load(checkpoint_config_path)
        config.checkpoint = pre_existing_checkpoint

    # --- 시뮬레이터별 App/Env 생성 ---
    if simulator == "isaaclab":
        # 분산 시뮬 실행 시 필수 플래그 전달
        app_launcher_flags = {"headless": config.headless}
        if fabric.world_size > 1:
            app_launcher_flags["distributed"] = True
            os.environ["LOCAL_RANK"] = str(fabric.local_rank)
            os.environ["RANK"] = str(fabric.global_rank)

        # App 실행
        app_launcher = AppLauncher(app_launcher_flags)
        simulation_app = app_launcher.app

        # Env 생성 (Hydra instantiate 사용)
        env = instantiate(config.env, device=fabric.device, simulation_app=simulation_app)
    else:
        # isaacgym 또는 기타 시뮬레이터
        env = instantiate(config.env, device=fabric.device)

    # --- 에이전트 생성/셋업 ---
    from protomotions.agents.ppo.agent import PPO  # 지연 임포트(환경 먼저 준비)
    agent: PPO = instantiate(config.agent, env=env, fabric=fabric)
    agent.setup()
    agent.fabric.strategy.barrier()

    # --- 체크포인트 로드 ---
    agent.load(config.get("checkpoint", None))

    # --- (옵션) wandb 설정 및 최초 실행 시 config 저장 ---
    if fabric.global_rank == 0 and not checkpoint_config_path.exists():
        if wandb is not None and "wandb" in config:
            for logger in getattr(fabric, "loggers", []):
                if isinstance(logger, WandbLogger):
                    logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))
            if wandb.run is not None:
                wandb_id = wandb.run.id
                log.info(f"wandb_id found {wandb_id}")
                if isinstance(unresolved_conf, dict) and "wandb" in unresolved_conf:
                    unresolved_conf["wandb"]["wandb_id"] = wandb_id
        log.info(f"Saving config file to {save_dir}")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_config_path, "w") as file:
            OmegaConf.save(unresolved_conf, file)

    agent.fabric.strategy.barrier()

    # --- 학습 시작 ---
    agent.fit()



if __name__ == "__main__":
    main()

