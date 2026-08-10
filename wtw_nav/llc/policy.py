"""Walk These Ways (WTW) 하위 제어기(LLC)를 MJX/JAX에서 구동하기 위한 래퍼.

배포된 체크포인트 `pretrain-v0`의 torch JIT 가중치를 **순수 JAX MLP로 재구현**한다.
torch 콜백 방식은 `jit`/`vmap`/`scan` 안에 넣을 수 없어 병렬 학습이 불가능하므로,
가중치만 추출해 JAX 배열로 들고 있는다.

정책은 모듈 **2개**로 구성된다 (`go1_gym_deploy/scripts/deploy_policy.py:59-66`)::

    latent = adaptation_module(obs_history)          # 2100 -> 2
    action = body(concat([obs_history, latent]))     # 2102 -> 12

`body`만 로드하면 입력 차원(2102)부터 맞지 않는다.

이 파일의 모든 상수는 체크포인트에 동봉된 `parameters.pkl`(이 가중치를 만든 config
그 자체)에서 읽은 값이다. 리포지토리의 `scripts/train.py`를 근거로 삼지 말 것 —
자세한 배경은 `docs/01_llc.md` §0.1·§14.
"""

from __future__ import annotations

import os
import platform

# mujoco를 import 하기 전에 GL 백엔드를 정해야 한다 (Colab/headless에서 렌더링이 되도록).
# policy.py가 이 패키지의 최초 진입점인 경우가 많으므로 여기서도 설정한다.
if platform.system() == "Linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

# --------------------------------------------------------------------------------------
# 체크포인트 실측 상수 (parameters.pkl)
# --------------------------------------------------------------------------------------

#: WTW/Isaac의 DOF 순서 — **FL, FR, RL, RR** (각 다리 안에서는 hip, thigh, calf).
#:
#: 주의 — **여기가 이 이식에서 가장 함정이었던 지점이다.**
#: `go1.urdf` 파일의 joint 등장 순서는 `FR, FL, RR, RL`이지만 Isaac Gym이 돌려주는
#: DOF 순서는 그것이 아니다. 근거:
#:   - `legged_robot.py:878-902`가 `foot_indices[0]`을 **FL**로, [1]을 FR, [2]를 RL,
#:     [3]을 RR로 라벨링한다 (WTW 소스 자신의 라벨).
#:   - 이 순서로 놓아야 `scripts/play.py:102`의 gait 프리셋 이름이 앞뒤가 맞는다:
#:     offset=0.5 → 앞다리쌍 = bounding, bound=0.5 → 좌우쌍 = pacing.
#:   - 폐루프 전수 조사에서 이 순서만 보행이 성립했다 (`FL,FR,RL,RR` 12/12 성공,
#:     다른 순서 36/36 실패). `FR,FL,RR,RL`로 두면 로봇이 액션 포화(±10)와 함께
#:     주저앉는다.
#:
#: 관절 부호·영점·프레임 규약 자체는 URDF와 MJCF가 완전히 동일하다(순기구학 대조 오차 0).
#: 틀린 것은 **순서뿐**이었다.
JOINT_NAMES_WTW = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)

#: Cfg.init_state.default_joint_angles 를 위 DOF 순서로 편 것.
DEFAULT_DOF_POS_WTW = jnp.array([
     0.1, 0.8, -1.5,   # FL
    -0.1, 0.8, -1.5,   # FR
     0.1, 1.0, -1.5,   # RL
    -0.1, 1.0, -1.5,   # RR
])

#: WTW 관절 순서에서 hip(abduction) 관절의 위치. `_compute_torques`의 `[0, 3, 6, 9]`.
HIP_INDICES = (0, 3, 6, 9)

#: Cfg.init_state.pos — 리셋 시 몸통 높이. 이 자세에서 발은 지면 위 1.5~2.5 cm에 뜬다.
#: 주의 — `mj_resetData`는 모델 기본값(z=0.445, 관절 0 = 다리를 편 상태로 발이 지면을 관통)을
#: 주므로 **반드시 `wtw_init_qpos`로 덮어써야 한다.** 그러지 않으면 t=0에 접촉이 잡힌 채
#: PD가 웅크린 자세로 당기면서 로봇이 튕겨 나가고 물리가 발산한다.
INIT_BASE_POS = (0.0, 0.0, 0.34)

#: Cfg.obs_scales 로부터 조립된 15D 명령 스케일 (`legged_robot.py:1196-1203`).
COMMANDS_SCALE = jnp.array([
    2.0,   # 0  vx            <- lin_vel
    2.0,   # 1  vy            <- lin_vel
    0.25,  # 2  yaw_v         <- ang_vel
    2.0,   # 3  height        <- body_height_cmd
    1.0,   # 4  step_freq     <- gait_freq_cmd
    1.0,   # 5  gait_phase    <- gait_phase_cmd
    1.0,   # 6  gait_offset   <- gait_phase_cmd
    1.0,   # 7  gait_bound    <- gait_phase_cmd
    1.0,   # 8  duty          <- gait_phase_cmd
    0.15,  # 9  footswing     <- footswing_height_cmd
    0.3,   # 10 pitch         <- body_pitch_cmd
    0.3,   # 11 roll          <- body_roll_cmd
    1.0,   # 12 stance_width  <- stance_width_cmd
    1.0,   # 13 stance_length <- stance_length_cmd
    1.0,   # 14 aux           <- aux_reward_cmd
])

OBS_SCALE_DOF_POS = 1.0    # Cfg.obs_scales.dof_pos
OBS_SCALE_DOF_VEL = 0.05   # Cfg.obs_scales.dof_vel

ACTION_SCALE = 0.25        # Cfg.control.action_scale
HIP_SCALE_REDUCTION = 0.5  # Cfg.control.hip_scale_reduction
CLIP_ACTIONS = 10.0        # Cfg.normalization.clip_actions
CLIP_OBSERVATIONS = 100.0  # Cfg.normalization.clip_observations

NUM_OBS = 70               # Cfg.env.num_observations
HISTORY_LEN = 30           # Cfg.env.num_observation_history
OBS_HISTORY_DIM = NUM_OBS * HISTORY_LEN      # 2100 (adaptation module 입력)
LATENT_DIM = 2             # Cfg.env.num_privileged_obs
BODY_INPUT_DIM = OBS_HISTORY_DIM + LATENT_DIM  # 2102

#: 물리 파라미터. Cfg.control (stiffness 20 / damping 0.5 / decimation 4) + sim dt 0.005.
KP = 20.0
KD = 0.5
SIM_TIMESTEP = 0.005
DECIMATION = 4
POLICY_DT = SIM_TIMESTEP * DECIMATION  # 0.02 s -> 50 Hz

#: 명령 인덱스 (가독성용).
CMD_STEP_FREQ, CMD_PHASE, CMD_OFFSET, CMD_BOUND, CMD_DUTY = 4, 5, 6, 7, 8


# --------------------------------------------------------------------------------------
# 정책 로드 (torch JIT -> JAX MLP)
# --------------------------------------------------------------------------------------

def _extract_mlp(module) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    """`nn.Sequential` JIT 모듈의 state_dict에서 (weight, bias) 층 목록을 순서대로 뽑는다."""
    sd = module.state_dict()
    layer_ids = sorted({int(k.split(".")[0]) for k in sd})
    layers = []
    for i in layer_ids:
        w = jnp.asarray(sd[f"{i}.weight"].detach().cpu().numpy())
        b = jnp.asarray(sd[f"{i}.bias"].detach().cpu().numpy())
        layers.append((w, b))
    return layers


def _mlp_forward(layers, x):
    """AC_Args.activation = 'elu'. 마지막 층에는 활성화가 없다."""
    for w, b in layers[:-1]:
        x = jax.nn.elu(x @ w.T + b)
    w, b = layers[-1]
    return x @ w.T + b


def load_policy(ckpt_dir: str):
    """`body_latest.jit` + `adaptation_module_latest.jit`를 로드해 JAX 함수로 반환한다.

    Args:
        ckpt_dir: `.../025417.456545/checkpoints` 디렉토리 경로.

    Returns:
        ``policy_fn(obs_history: (2100,)) -> action: (12,)``  (WTW 관절 순서)
    """
    import torch  # 로드 시점에만 필요 — 학습 루프에는 들어가지 않는다.

    body = torch.jit.load(f"{ckpt_dir}/body_latest.jit", map_location="cpu")
    adaptation = torch.jit.load(f"{ckpt_dir}/adaptation_module_latest.jit", map_location="cpu")

    body_layers = _extract_mlp(body)
    adapt_layers = _extract_mlp(adaptation)

    # 로드된 가중치가 예상 구조와 맞는지 즉시 검증 — 여기서 틀리면 이후 전부가 무의미하다.
    assert adapt_layers[0][0].shape[1] == OBS_HISTORY_DIM, (
        f"adaptation module 입력 {adapt_layers[0][0].shape[1]} != {OBS_HISTORY_DIM}")
    assert adapt_layers[-1][0].shape[0] == LATENT_DIM, (
        f"latent 차원 {adapt_layers[-1][0].shape[0]} != {LATENT_DIM}")
    assert body_layers[0][0].shape[1] == BODY_INPUT_DIM, (
        f"body 입력 {body_layers[0][0].shape[1]} != {BODY_INPUT_DIM} "
        "(adaptation module을 빼먹으면 여기서 걸린다)")
    assert body_layers[-1][0].shape[0] == 12

    def policy_fn(obs_history: jnp.ndarray) -> jnp.ndarray:
        latent = _mlp_forward(adapt_layers, obs_history)
        return _mlp_forward(body_layers, jnp.concatenate([obs_history, latent], axis=-1))

    return policy_fn


# --------------------------------------------------------------------------------------
# MuJoCo / MJX 환경
# --------------------------------------------------------------------------------------

class JointIndex(dict):
    """WTW 관절 순서 ↔ MuJoCo 인덱스 매핑. 하드코딩하지 않고 모델에서 유도한다."""


def _build_joint_index(mj_model) -> JointIndex:
    qpos_adr, qvel_adr, act_adr = [], [], []
    for name in JOINT_NAMES_WTW:
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"모델에 관절 '{name}'이 없습니다.")
        qpos_adr.append(mj_model.jnt_qposadr[jid])
        qvel_adr.append(mj_model.jnt_dofadr[jid])

        aid = next((a for a in range(mj_model.nu) if mj_model.actuator_trnid[a, 0] == jid), None)
        if aid is None:
            raise ValueError(f"관절 '{name}'을 구동하는 액추에이터가 없습니다.")
        act_adr.append(aid)

    return JointIndex(
        qpos_adr=jnp.asarray(qpos_adr),   # x_wtw = data.qpos[qpos_adr]
        qvel_adr=jnp.asarray(qvel_adr),   # v_wtw = data.qvel[qvel_adr]
        act_adr=jnp.asarray(act_adr),     # ctrl = zeros(nu).at[act_adr].set(target_wtw)
        n_act=mj_model.nu,
    )


def _apply_collision_filter(mj_model, mode: str, verbose: bool = False) -> None:
    """geom 충돌 쌍을 줄인다. **MJX 처리량의 지배 요인**이다.

    menagerie `go1.xml`은 43개 geom을 전부 `contype=1/conaff=1`로 두므로 서로가
    서로와 충돌한다 -> MJX가 매 스텝 **약 900쌍**을 계산한다. 그런데 평지에서 실제로
    생기는 접촉은 **4개**(발 4개 vs 바닥)뿐이고, 발 외 39개는 전부 `condim=1`
    (무마찰) — 즉 자기충돌 방지 전용이다.

    실측(L4, 512 env): 솔버를 4/8로 낮춰도 L0가 11%밖에 안 빨라졌다. 솔버가 아니라
    **충돌 검사가 병목**이라는 뜻이다.

    비트마스크 규칙은 `(contype_a & conaff_b) | (contype_b & conaff_a) != 0`이면 충돌.

    Args:
        mode:
            ``"full"``  — 원본 유지. WTW는 `self_collisions = 0`(=켜짐)으로 학습했으므로
                          이것이 원본 충실 설정이다.
            ``"world"`` — 자기충돌만 끈다(로봇 geom끼리 필터). 다리 vs 지형은 그대로
                          남으므로 P2의 장애물 접촉이 보존된다. **권장 기본값.**
            ``"feet"``  — 발·몸통만 세계와 충돌. 가장 빠르지만 다리로 장애물을 짚거나
                          걸리는 것을 모사하지 못하므로 **평지 전용**이다.
    ★★ **미로가 느릴 때 여기를 더 깎지 말 것 (2026-08-10 결론).** P4에서 처리량이
    평지 3,096 -> 78 HLC steps/s로 떨어졌을 때 이 함수를 지렛대로 삼으려 했고,
    `feet`이 실제로 15.4배(1,205)를 냈다. **그럼에도 채택하지 않았다** — `feet`은
    다리 캡슐이 벽을 통과하고, 그러면 "통로를 실제보다 넓게" 학습한다. 미로만
    푸는 것이 목적이 아니므로 **모든 충돌을 살려 둔다.**

    답은 geom 축소가 아니라 **`HLCConfig.max_geom_pairs`(MJX 근사 broadphase)** 였다.
    `world`(전 geom 유지) + k=32로 **708 steps/s (9.1배)**, ETA 21시간 -> 141분.
    게다가 그쪽은 맵 크기에 O(1)이다. 측정 경위는 `docs/01_llc.md` §8.5.

    주의 — 폐기 기록 — `"maze"` 모드(바닥/벽 비트마스크 분리, 벽은 몸통·정강이·발만)를
    만들었다가 **지웠다.** 78 -> 109로 1.4배뿐이었다. 쌍은 403 -> 247(1.6배)인데
    `feet`(143쌍)은 11배라 초선형이었고, 차이는 정강이 캡슐 8개 = `CAPSULE-BOX`
    104쌍뿐이다. 그 조합이 비용을 지배한다고 볼 수밖에 없으나 **11배의 기전은
    확인하지 못했다.** 다시 만들지 말 것 — 이득이 작고, broadphase가 상위 호환이다.

    비트: 바닥 `(1,1)` / 벽 `(4,4)` / 로봇 코어 `(2,5)` / 로봇 나머지 `(2,1)`
      · 바닥 vs 로봇 전부  : (1&1)|(2&1) = 1   -> 충돌 (평지 성능 그대로)
      · 벽   vs 코어       : (4&5)|(2&4) = 4   -> 충돌
      · 벽   vs 나머지     : (4&1)|(2&4) = 0   -> **필터**
      · 로봇 vs 로봇       : (2&1)|(2&1) = 0   -> 필터
      · 바닥 vs 벽         : (1&4)|(4&1) = 0   -> 필터 (덤)

    주의 — **지형 사다리에 쓰지 말 것.** 사다리는 바닥 평면을 지우고 전부 BOX로 만들기
    때문에(`modules._spec`) 전 지형이 "벽"으로 분류되고, 그러면 다리가 턱·요철을
    통과한다 — `feet`을 평지 전용으로 못박은 것과 같은 이유다. 호출부에서 막는다.
    """
    if mode == "full":
        return
    if mode == "maze":
        raise ValueError(
            "collision='maze'는 2026-08-10에 폐기됐습니다 (78 -> 109 steps/s, "
            "1.4배뿐). 미로가 느리면 geom을 깎지 말고 "
            "`HLCConfig.max_geom_pairs`(MJX 근사 broadphase)를 쓰십시오 — "
            "전 geom을 유지한 채 9.1배이고 맵 크기에 O(1)입니다.")
    if mode not in ("world", "feet"):
        raise ValueError(f"collision must be full|world|feet, got {mode!r}")

    body = mj_model.geom_bodyid
    is_world = body == 0
    active = (mj_model.geom_contype != 0) | (mj_model.geom_conaffinity != 0)

    # world: contype=1/conaff=1, robot: contype=2/conaff=1
    #   robot-robot: (2&1)=0, (2&1)=0            -> 필터됨
    #   robot-world: (world 1 & robot conaff 1)=1 -> 충돌 유지
    mj_model.geom_contype[active & is_world] = 1
    mj_model.geom_conaffinity[active & is_world] = 1
    mj_model.geom_contype[active & ~is_world] = 2
    mj_model.geom_conaffinity[active & ~is_world] = 1

    if mode == "feet":
        # 발(condim>=3) 과 몸통(body 1) 외에는 충돌에서 완전히 제외
        keep = is_world | (mj_model.geom_condim >= 3) | (body == 1)
        mj_model.geom_contype[~keep] = 0
        mj_model.geom_conaffinity[~keep] = 0

    if verbose:
        n = int(((mj_model.geom_contype != 0) |
                 (mj_model.geom_conaffinity != 0)).sum())
        print(f"  [collision={mode}] 충돌 참여 geom {int(active.sum())} -> {n}, "
              f"자기충돌 {'유지' if mode == 'full' else '해제'}")


def create_env(xml_path: str, kp: float = KP, kd: float = KD,
               timestep: float = SIM_TIMESTEP, match_wtw_joints: bool = True,
               mjx_friendly: bool = True, iterations: int = 8,
               ls_iterations: int = 16, collision: str = "world",
               verbose: bool = False):
    """MuJoCo/MJX 모델을 WTW 학습 조건에 맞춰 설정한다.

    WTW는 `control_type = "actuator_net"` — 학습된 액추에이터 네트워크를 씁니다.
    주의 — 여기서는 이를 이상적인 PD(Kp=20, Kd=0.5)로 **근사**합니다. 이는 제거할 수 없는
    sim-to-sim 격차이며, 추종 오차가 남는다면 1순위 용의자입니다.

    Args:
        match_wtw_joints: True면 menagerie 모델의 관절 damping/frictionloss를 0으로 두어
            속도 항이 액추에이터 Kd만 남게 합니다(= WTW의 PD 법칙과 동일한 형태).
            False면 menagerie 기본값(damping 2, frictionloss 0.2)을 유지합니다.
        mjx_friendly: menagerie의 `go1.xml`은 정확도 우선으로 작성되어 MJX에서 컴파일·실행이
            **매우** 느립니다(elliptic cone, impratio 100, 발 condim 6, **iterations 100 /
            ls_iterations 50**). MJX는 솔버 반복을 컴파일 그래프에 그대로 펼치므로 이대로면
            XLA 컴파일만 수십 분이 걸립니다. True면 pyramidal cone, impratio 1,
            발 condim 3, Newton `iterations`/`ls_iterations`로 바꿉니다.
            주의 — 접촉 해의 정확도가 낮아지는 **근사**입니다.
        iterations, ls_iterations: 기본 **8/16**. 주의 — 이 값을 낮추지 마십시오.
            MJX는 접촉 해가 덜 수렴하면 **정책이 보행에 실패**합니다. 실측 A/B:

                4/8    vx=0.8 -> 0.218, min_z 0.058, 낙상
                8/16   vx=0.8 -> 0.777, min_z 0.155, 정상
                16/32  vx=0.8 -> 0.777, min_z 0.155, 정상  (8/16과 동일 = 수렴)

            MuJoCo C 엔진은 같은 모델을 iterations=4에서도 잘 굴리므로(vx 0.865),
            정적 유지 테스트나 C 엔진 결과로 이 값을 정하면 안 됩니다.
            integrator는 **Euler를 유지**합니다 — `implicitfast`+`iterations=1`은
            이 모델·게인 조합에서 발산하는 것을 실측으로 확인했습니다.
        collision: 충돌 쌍 필터. 기본 `"world"`(자기충돌 해제). MJX 처리량을 가장
            크게 좌우하는 값이므로 `_apply_collision_filter`의 설명을 읽으십시오.

    Returns:
        (mj_model, mj_data, mjx_model)
    """
    # `xml_path`는 경로 또는 **이미 컴파일된 MjModel**. 후자는 `terrain.modules`가
    # `MjSpec`으로 조립한 지형을 그대로 넘기기 위한 것이다 (include는 meshdir이
    # 깨져서 못 쓴다 — `terrain/modules.py` 상단 참조).
    mj_model = (xml_path if isinstance(xml_path, mujoco.MjModel)
                else mujoco.MjModel.from_xml_path(xml_path))
    mj_model.opt.timestep = timestep

    # position 액추에이터를 tau = kp*(ctrl - q) - kd*qd 로 만든다.
    mj_model.actuator_gainprm[:, 0] = kp
    mj_model.actuator_biasprm[:, 1] = -kp
    mj_model.actuator_biasprm[:, 2] = -kd

    if mjx_friendly:
        before = (int(mj_model.opt.cone), float(mj_model.opt.impratio),
                  int(mj_model.opt.iterations), int(mj_model.opt.ls_iterations))
        mj_model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        mj_model.opt.impratio = 1.0
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        mj_model.opt.iterations = iterations
        mj_model.opt.ls_iterations = ls_iterations
        # 발 접촉의 condim 6(비틀림·구름 마찰)은 MJX에서 특히 비싸다 -> 3으로.
        mj_model.geom_condim[mj_model.geom_condim > 3] = 3
        # 주의 — integrator는 Euler(menagerie 기본값)를 유지한다. mujoco_playground의
        #    implicitfast + iterations=1 조합은 이 모델·게인에서 발산한다(z가 100 m 이상으로
        #    튐). 실측으로 확인함 — 바꾸지 말 것.
        if verbose:
            print(f"  [mjx_friendly] cone/impratio/iters/ls: {before} -> "
                  f"(PYRAMIDAL, 1.0, {iterations}, {ls_iterations}), "
                  f"geom_condim>3 -> 3, integrator=Euler 유지")

    if match_wtw_joints:
        # 12개 구동 관절의 dof만 정확히 지목한다 (free joint dof를 건드리지 않도록,
        # 그리고 "free joint는 항상 앞 6개"라는 가정에 기대지 않도록).
        for name in JOINT_NAMES_WTW:
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"모델에 관절 '{name}'이 없습니다.")
            dof = mj_model.jnt_dofadr[jid]
            mj_model.dof_damping[dof] = 0.0
            mj_model.dof_frictionloss[dof] = 0.0

    _apply_collision_filter(mj_model, collision, verbose)

    mj_data = mujoco.MjData(mj_model)
    mjx_model = mjx.put_model(mj_model)
    return mj_model, mj_data, mjx_model


def wtw_init_qpos(mj_model, jidx) -> np.ndarray:
    """WTW의 리셋 자세 qpos (`Cfg.init_state`).

    `mj_resetData`가 주는 모델 기본값은 **관절이 전부 0 = 다리를 쭉 편 자세**이고
    base z=0.445라 발이 지면을 4 mm 관통한 채(ncon=4) 시작한다. 거기서 PD가 웅크린
    default 자세로 당기면 로봇이 튕겨 나가고 물리가 발산한다.

    이 함수가 주는 자세에서는 발이 지면 위 1.5~2.5 cm에 떠 있고 접촉이 없다(ncon=0).
    """
    qpos = np.array(mj_model.qpos0, dtype=np.float64)

    free = np.flatnonzero(mj_model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free):
        adr = mj_model.jnt_qposadr[free[0]]
        qpos[adr:adr + 3] = INIT_BASE_POS
        qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)

    qpos[np.asarray(jidx["qpos_adr"])] = np.asarray(DEFAULT_DOF_POS_WTW)
    return qpos


def reset_data(mj_model, mj_data, mjx_model, jidx):
    """WTW 리셋 자세로 초기화된 MJX data를 만든다.

    `mj_forward`를 먼저 돌려 파생량(접촉 등)을 일관되게 만든 뒤 `put_data` 한다.
    """
    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[:] = wtw_init_qpos(mj_model, jidx)
    mj_data.qvel[:] = 0.0
    mj_data.ctrl[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)
    return mjx.put_data(mj_model, mj_data)


# --------------------------------------------------------------------------------------
# 관측 (70D)
# --------------------------------------------------------------------------------------

def _projected_gravity(quat: jnp.ndarray) -> jnp.ndarray:
    """중력 벡터 [0,0,-1]을 몸통 좌표계로 회전. quat은 MuJoCo 규약 (w, x, y, z)."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    # R^T @ [0, 0, -1] = -(R의 3번째 열을 몸통 좌표로 옮긴 것)
    return jnp.array([
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
    ])


def clock_inputs(gait_indices: jnp.ndarray, commands: jnp.ndarray) -> jnp.ndarray:
    """`_step_contact_targets`의 clock_inputs 4개. 발 순서는 DOF와 같은 **FL, FR, RL, RR**.

    duty(`commands[8]`)로 위상을 워핑한 뒤 `sin(2*pi*.)`를 취한다.
    참고: 이 체크포인트는 duty가 0.5로 고정 학습되어 있고, duty=0.5에서 워핑은 항등이다.

    프리셋 검산 (`play.py:102`와 일치):
      phase=0.5  -> FL·RR 대 FR·RL  = 대각쌍 = trot
      offset=0.5 -> FL·FR 대 RL·RR  = 앞뒤쌍 = bound
      bound=0.5  -> FL·RL 대 FR·RR  = 좌우쌍 = pace
    """
    phases, offsets = commands[CMD_PHASE], commands[CMD_OFFSET]
    bounds, durations = commands[CMD_BOUND], commands[CMD_DUTY]

    feet = jnp.array([
        gait_indices + phases + offsets + bounds,  # FL
        gait_indices + offsets,                    # FR
        gait_indices + bounds,                     # RL
        gait_indices + phases,                     # RR
    ])

    x = jnp.remainder(feet, 1.0)
    stance = x * (0.5 / durations)
    swing = 0.5 + (x - durations) * (0.5 / (1.0 - durations))
    warped = jnp.where(x < durations, stance, swing)
    return jnp.sin(2.0 * jnp.pi * warped)


def get_obs(data, gait_indices, actions, last_actions, commands, jidx) -> jnp.ndarray:
    """WTW의 70D 관측 벡터 (`legged_robot.py:302-338`, `observe_command=True`).

    구성: proj_gravity(3) + commands*scale(15) + dof_pos_err(12) + dof_vel(12)
          + actions(12) + last_actions(12) + clock_inputs(4) = 70

    Args:
        actions: 방금 실행된 행동 a_t (WTW 관절 순서, 스케일 전).
        last_actions: 그 직전 행동 a_{t-1}.
    """
    dof_pos = data.qpos[jidx["qpos_adr"]]
    dof_vel = data.qvel[jidx["qvel_adr"]]

    obs = jnp.concatenate([
        _projected_gravity(data.qpos[3:7]),
        commands * COMMANDS_SCALE,
        (dof_pos - DEFAULT_DOF_POS_WTW) * OBS_SCALE_DOF_POS,
        dof_vel * OBS_SCALE_DOF_VEL,
        actions,
        last_actions,
        clock_inputs(gait_indices, commands),
    ])
    return jnp.clip(obs, -CLIP_OBSERVATIONS, CLIP_OBSERVATIONS)


# --------------------------------------------------------------------------------------
# 스텝 함수
# --------------------------------------------------------------------------------------

def make_step_fn(mjx_model, policy_fn, jidx, decimation: int = DECIMATION):
    """LLC 1스텝(50 Hz)을 수행하는 JAX 함수를 만든다.

    반환 함수 시그니처::

        (data, obs_history, gait_indices, last_actions, last_last_actions, commands)
            -> (data, obs_history, gait_indices, last_actions, last_last_actions)

    `commands`는 15D 원시 명령(스케일 적용 전)이며 이 스텝 동안 고정이다.
    """
    hip_idx = jnp.asarray(HIP_INDICES)
    act_adr, n_act = jidx["act_adr"], jidx["n_act"]

    def policy_step_fn(data, obs_history, gait_indices, last_actions, last_last_actions,
                       commands):
        action = jnp.clip(policy_fn(obs_history), -CLIP_ACTIONS, CLIP_ACTIONS)

        # 행동 -> 관절 목표각 (`_compute_torques:919-925`)
        scaled = action * ACTION_SCALE
        scaled = scaled.at[hip_idx].multiply(HIP_SCALE_REDUCTION)
        target_wtw = scaled + DEFAULT_DOF_POS_WTW

        ctrl = jnp.zeros(n_act).at[act_adr].set(target_wtw)
        data = data.replace(ctrl=ctrl)

        def sim_step(d, _):
            return mjx.step(mjx_model, d), None

        data, _ = jax.lax.scan(sim_step, data, jnp.arange(decimation))

        # post_physics_step 순서: 보행 위상 갱신 -> 관측 계산 -> 행동 버퍼 시프트
        gait_indices = jnp.remainder(
            gait_indices + POLICY_DT * commands[CMD_STEP_FREQ], 1.0)

        obs = get_obs(data, gait_indices, action, last_actions, commands, jidx)
        obs_history = jnp.concatenate([obs_history[NUM_OBS:], obs])

        return data, obs_history, gait_indices, action, last_actions

    return policy_step_fn


def init_llc_state():
    """LLC의 초기 캐리 상태. `reset_idx`가 전부 0으로 두는 것과 같다.

    dtype을 명시해 둔다 — weak-typed 스칼라를 넘기면 두 번째 호출에서 불필요한
    재컴파일이 일어난다.
    """
    return (
        jnp.zeros(OBS_HISTORY_DIM, jnp.float32),  # obs_history
        jnp.zeros((), jnp.float32),               # gait_indices
        jnp.zeros(12, jnp.float32),               # last_actions
        jnp.zeros(12, jnp.float32),               # last_last_actions
    )


def make_rollout_fn(mjx_model, policy_fn, jidx, decimation: int = DECIMATION):
    """T 스텝 롤아웃 전체를 `lax.scan` **하나**로 도는 함수를 만든다.

    Python for 루프로 `step_fn`을 반복 호출하면 스텝마다 dispatch가 일어나고,
    중간에 `np.asarray(...)`로 값을 꺼내면 매번 host 동기화까지 걸린다. 단일 env MJX는
    스텝당 비용이 커서 이 조합이 수십 분으로 불어난다. 전체를 하나의 scan에 넣으면
    컴파일 1회 + 디스패치 1회로 끝난다.

    반환 함수 시그니처::

        (data, obs_history, gait, last_a, last_last_a, cmds[T, 15])
            -> (carry_final, qpos[T, nq], qvel[T, nv])

    주의 — T가 바뀌면 재컴파일된다. 롤아웃 길이는 몇 종류로 고정해서 쓸 것.
    """
    step = make_step_fn(mjx_model, policy_fn, jidx, decimation)

    def rollout_fn(data, obs_history, gait, last_a, last_last_a, cmds):
        def body(carry, cmd):
            d, h, g, a, b = step(*carry, cmd)
            return (d, h, g, a, b), (d.qpos, d.qvel)

        carry, (qpos, qvel) = jax.lax.scan(
            body, (data, obs_history, gait, last_a, last_last_a), cmds)
        return carry, qpos, qvel

    return rollout_fn


def make_closed_rollout_fn(mjx_model, policy_fn, jidx, cmd_fn,
                           decimation: int = DECIMATION,
                           hlc_decimation: int = 5):
    """★ **되먹임이 있는** 롤아웃. `cmd_fn(cmd, data) -> cmd`로 명령을 매 순간 고친다.

    `make_rollout_fn`은 `cmds[T,15]`를 미리 다 정해 놓고 돈다 — 순수 열린 루프다.
    그것으로 잰 지형 능력은 배포와 무관하다는 것이 실측으로 드러났다(표류가 8초에
    3.3 m). 이 함수는 그 되먹임 경로를 만든다.

    주의 — **`hlc_decimation` 스텝마다 한 번만 갱신한다.** LLC 50 Hz마다 보정하면
    실제 HLC(10 Hz)보다 5배 민첩한 제어기를 재게 되고, 그러면 측정이 다시 배포와
    어긋난다 — 방향만 반대인 같은 실수다. 갱신 사이에는 값을 물고 있는다.

    시그니처는 `make_rollout_fn`과 같다 (`cmds`는 **공칭** 명령이 된다).
    """
    step = make_step_fn(mjx_model, policy_fn, jidx, decimation)

    def rollout_fn(data, obs_history, gait, last_a, last_last_a, cmds):
        def body(carry, x):
            d, h, g, a, b, held = carry
            i, nominal = x
            # 갱신 시점에만 되먹임을 다시 계산하고, 아니면 직전 값을 유지한다.
            held = jnp.where(i % hlc_decimation == 0, cmd_fn(nominal, d), held)
            d, h, g, a, b = step(d, h, g, a, b, held)
            return (d, h, g, a, b, held), (d.qpos, d.qvel)

        init = (data, obs_history, gait, last_a, last_last_a, cmds[0])
        carry, (qpos, qvel) = jax.lax.scan(
            body, init, (jnp.arange(cmds.shape[0]), cmds))
        return carry[:5], qpos, qvel

    return rollout_fn


# --------------------------------------------------------------------------------------
# 명령 헬퍼
# --------------------------------------------------------------------------------------

#: WTW 보행 프리셋 (phase, offset, bound). `scripts/play.py:102`.
#: DOF 순서를 FL,FR,RL,RR로 잡으면 이 이름들이 `clock_inputs`의 실제 발 조합과 일치한다.
#: 주의 — 인덱스 이름과 gait 이름이 어긋난다. `commands[6]`의 필드명은 `gait_offset`이지만
#: 그것을 0.5로 두면 **bound**가 되고, `commands[7]`의 필드명은 `gait_bound`인데
#: 0.5로 두면 **pace**가 된다. 검산 (DOF 순서 FL,FR,RL,RR):
#:   offset=0.5 -> FL·FR 대 RL·RR = 앞뒤쌍 = bound
#:   bound=0.5  -> FL·RL 대 FR·RR = 좌우쌍 = pace
#: 아래 값이 `play.py:102`와 일치하는 정답이다.
GAITS = {
    "pronk": (0.0, 0.0, 0.0),
    "trot":  (0.5, 0.0, 0.0),
    "bound": (0.0, 0.5, 0.0),
    "pace":  (0.0, 0.0, 0.5),
}


def make_commands(vx=0.0, vy=0.0, yaw=0.0, height=0.0, step_freq=3.0, gait="trot",
                  footswing=0.08, pitch=0.0, stance_width=0.25,
                  stance_length=0.45) -> jnp.ndarray:
    """15D 명령 벡터를 조립한다. 고정 차원은 학습된 상수값으로 채운다.

    주의 — `duty`(8)와 `roll`(11)은 이 체크포인트에서 상수로 학습되어 조작할 수 없으므로
    인자로 노출하지 않는다. 근거는 `docs/01_llc.md` §0.1·§14 — 커리큘럼
    상한 `limit_*`까지 폭이 0이라 **이 가중치는 그 축을 본 적이 없다.**

    ★ `stance_length`(13)는 그것들과 **다르다.** 학습 범위가 `[0.35, 0.45]` 전 구간이고
    `num_bins=1`이라 균등 샘플됐다 — **조작 가능하다.** HLC 설계에서 고정한 것은
    "지형 돌파에 불필요해서"라는 **추측**이었지, 측정 결과가 아니다(2026-07-29 발견).
    앞뒤 다리 간격은 틈 step-over에서 발이 닿는 거리와 직접 관련된 기하량이므로
    최소한 실측은 해봐야 한다. `terrain.measure.axis_screen()` 참조.

    주의 — **HLC는 `gait="trot"` 만 쓸 것.** `pronk`는 몸통을 띄우지 못하면서
    (정점 상승 +0.001 m) MJX에서 전복을 유발한다 — `docs/03_results.md` §2.
    `pace`/`bound`는 검증하지 않았다. 다른 gait는 진단·비교 목적으로만 쓴다.
    """
    phase, offset, bound = GAITS[gait] if isinstance(gait, str) else gait
    return jnp.array([
        vx, vy, yaw,
        height, step_freq,
        phase, offset, bound,
        0.5,              # duty  — 학습 시 상수 고정
        footswing,
        pitch,
        0.0,              # roll  — 학습 시 상수 고정
        stance_width,
        stance_length,
        0.0,              # aux
    ])
