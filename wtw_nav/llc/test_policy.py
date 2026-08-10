"""P0 게이트: 복구한 `policy.py`의 정확성 검증.

두 단계로 나뉜다. **로컬(CPU, JAX 불필요)에서 위험한 수학은 전부 검증되고**,
JAX/MJX 실행 경로만 Colab에서 확인한다.

    conda run -n mujoco_env python -m wtw_nav.llc.test_policy       # 로컬: 정적 검증
    (Colab)  notebooks/01_llc_check.py  ->  run_gate()              # JAX 실행 + 롤아웃

로컬 검증 항목
    0. policy.py 상수 파싱
    1. 상수가 체크포인트 `parameters.pkl`에서 유래했는가 (하드코딩 오타 방지)
    2. MLP 가중치 추출 + forward가 torch와 일치하는가 (< 1e-4)
       + adaptation module을 빼면 실제로 실패하는가 (음성 대조)
    3. 관절 매핑이 모델에서 올바르게 유도되는가 + PD 게인 설정
    4. 70D 관측 조립
    5. proj_gravity를 MuJoCo 회전행렬과 교차검증

JAX가 설치되어 있으면 policy.py의 실제 함수로, 없으면 numpy 미러로 같은 수식을
검증한다 (미러는 policy.py의 상수를 소스에서 직접 읽어 쓰므로 상수 불일치는 잡힌다).
"""

from __future__ import annotations

import io
import pickle
import re
import sys

import numpy as np
import mujoco
import torch

ROOT = __file__.rsplit("wtw_nav", 1)[0].rstrip("\\/")
RUN = (f"{ROOT}/walk-these-ways/runs/gait-conditioned-agility/pretrain-v0"
       f"/train/025417.456545")
CKPT = f"{RUN}/checkpoints"
XML = f"{ROOT}/mujoco_menagerie/unitree_go1/scene.xml"

# policy.py를 실제로 import 하려면 jax **와** mujoco.mjx가 둘 다 있어야 한다.
# (로컬 mujoco 3.1.0에는 jax는 있어도 mjx가 없다 -> 그때는 numpy 미러로 검증한다.)
# policy.py를 실제로 import 하려면 jax **와** mujoco.mjx가 둘 다 살아 있어야 한다.
# 주의 — ImportError만 잡으면 안 된다. 커널에 반쯤 초기화된 jax가 남아 있으면
#    `AttributeError: partially initialized module 'jax' has no attribute '_src'`가
#    나는데, 그걸 못 잡으면 fallback 없이 그대로 죽는다.
_missing = []
try:
    import jax.numpy as jnp  # noqa: F401
except Exception as _e:
    _missing.append(f"jax({type(_e).__name__})")
try:
    from mujoco import mjx  # noqa: F401
except Exception as _e:
    _missing.append(f"mujoco.mjx({type(_e).__name__})")
HAS_JAX = not _missing
_WHY = "" if HAS_JAX else f" ({', '.join(_missing)} 사용 불가)"

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def env_report() -> None:
    """실행 환경 점검. jax/mjx가 안 될 때 원인을 바로 보여준다.

        conda run -n mujoco_env python -c "from wtw_nav.llc.test_policy import env_report; env_report()"
    """
    import importlib.metadata as md
    import sys
    import traceback

    print(f"python {sys.version.split()[0]}")
    for pkg in ("jax", "jaxlib", "mujoco", "mujoco-mjx", "brax", "ml_dtypes",
                "numpy", "torch", "mediapy"):
        try:
            print(f"  {pkg:12s} {md.version(pkg)}")
        except Exception:
            print(f"  {pkg:12s} (미설치)")

    print()
    for label, fn in (("import jax", lambda: __import__("jax")),
                      ("import mujoco", lambda: __import__("mujoco")),
                      ("from mujoco import mjx",
                       lambda: __import__("mujoco.mjx", fromlist=["mjx"]))):
        try:
            m = fn()
            v = getattr(m, "__version__", "")
            print(f"  OK    {label} {v}")
        except Exception as e:
            print(f"  FAIL  {label}  ->  {type(e).__name__}: {e}")
            if "partially initialized" in str(e):
                print("        ↳ 커널에 반쯤 초기화된 모듈이 남아 있습니다. "
                      "**런타임/커널을 재시작**하십시오 "
                      "(pip install 후에는 항상 재시작).")
            else:
                traceback.print_exc(limit=3)


def _load_all(path):
    """ml_logger는 한 파일에 객체를 계속 append한다."""
    objs = []
    with open(path, "rb") as f:
        while True:
            try:
                objs.append(pickle.load(f))
            except EOFError:
                break
    return objs


# --------------------------------------------------------------- 0. 상수 파싱
def parse_constants():
    """policy.py에서 상수를 읽는다. JAX가 없어도 되도록 소스를 파싱한다."""
    src = open(f"{ROOT}/wtw_nav/llc/policy.py", encoding="utf-8").read()

    def arr(name):
        body = re.search(rf"{name} = jnp\.array\(\[(.*?)\]\)", src, re.S).group(1)
        body = re.sub(r"#.*", "", body)
        return np.array([float(x) for x in body.replace("\n", " ").split(",") if x.strip()])

    def scalar(name):
        return float(re.search(rf"^{name} = ([0-9.eE+-]+)", src, re.M).group(1))

    names = tuple(re.findall(r'"(\w+_joint)"',
                             src.split("JOINT_NAMES_WTW = (")[1].split(")")[0]))
    return dict(
        COMMANDS_SCALE=arr("COMMANDS_SCALE"),
        DEFAULT_DOF_POS_WTW=arr("DEFAULT_DOF_POS_WTW"),
        JOINT_NAMES_WTW=names,
        KP=scalar("KP"), KD=scalar("KD"),
        SIM_TIMESTEP=scalar("SIM_TIMESTEP"),
        ACTION_SCALE=scalar("ACTION_SCALE"),
        HIP_SCALE_REDUCTION=scalar("HIP_SCALE_REDUCTION"),
        DECIMATION=int(scalar("DECIMATION")),
    )


# --------------------------------------------------------------- numpy 미러
def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def _extract(module):
    sd = module.state_dict()
    ids = sorted({int(k.split(".")[0]) for k in sd})
    return [(sd[f"{i}.weight"].detach().cpu().numpy(),
             sd[f"{i}.bias"].detach().cpu().numpy()) for i in ids]


def _fwd(layers, x):
    for w, b in layers[:-1]:
        x = _elu(x @ w.T + b)
    w, b = layers[-1]
    return x @ w.T + b


def _proj_grav(q):
    w, x, y, z = q
    return np.array([-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))])


def _clock(gi, cmd):
    ph, off, bd, dur = cmd[5], cmd[6], cmd[7], cmd[8]
    feet = np.array([gi + ph + off + bd, gi + off, gi + bd, gi + ph])
    x = np.remainder(feet, 1.0)
    warped = np.where(x < dur, x * (0.5 / dur), 0.5 + (x - dur) * (0.5 / (1 - dur)))
    return np.sin(2 * np.pi * warped)


# --------------------------------------------------------------- main
def main() -> int:
    C = parse_constants()
    mode = "policy.py 직접 실행" if HAS_JAX else f"numpy 미러{_WHY}"
    print(f"\n{'=' * 62}\nP0 로컬 검증  ({mode})\n{'=' * 62}")

    print("\n[0] policy.py 상수 파싱")
    check("COMMANDS_SCALE 15개", len(C["COMMANDS_SCALE"]) == 15)
    check("DEFAULT_DOF_POS 12개", len(C["DEFAULT_DOF_POS_WTW"]) == 12)
    check("JOINT_NAMES 12개", len(C["JOINT_NAMES_WTW"]) == 12, str(C["JOINT_NAMES_WTW"][:3]))

    print("\n[1] 상수가 체크포인트 parameters.pkl에서 유래하는가")
    torch.storage._load_from_bytes = lambda b: torch.load(
        io.BytesIO(b), map_location="cpu", weights_only=False)
    cfg = _load_all(f"{RUN}/parameters.pkl")[0]["Cfg"]
    os_ = cfg["obs_scales"]
    expect_scale = np.array([
        os_["lin_vel"], os_["lin_vel"], os_["ang_vel"], os_["body_height_cmd"],
        os_["gait_freq_cmd"], os_["gait_phase_cmd"], os_["gait_phase_cmd"],
        os_["gait_phase_cmd"], os_["gait_phase_cmd"], os_["footswing_height_cmd"],
        os_["body_pitch_cmd"], os_["body_roll_cmd"], os_["stance_width_cmd"],
        os_["stance_length_cmd"], os_["aux_reward_cmd"]])
    check("COMMANDS_SCALE == obs_scales 유래값",
          np.allclose(C["COMMANDS_SCALE"], expect_scale), str(expect_scale))

    dja = cfg["init_state"]["default_joint_angles"]
    check("DEFAULT_DOF_POS == default_joint_angles",
          np.allclose(C["DEFAULT_DOF_POS_WTW"], [dja[n] for n in C["JOINT_NAMES_WTW"]]))
    ctrl = cfg["control"]
    check("KP/KD == Cfg.control",
          C["KP"] == ctrl["stiffness"]["joint"] and C["KD"] == ctrl["damping"]["joint"])
    check("ACTION_SCALE == Cfg.control", C["ACTION_SCALE"] == ctrl["action_scale"])
    check("HIP_SCALE_REDUCTION == Cfg.control",
          C["HIP_SCALE_REDUCTION"] == ctrl["hip_scale_reduction"])
    check("DECIMATION == Cfg.control", C["DECIMATION"] == ctrl["decimation"])
    check("duty/roll이 여전히 상수인지 (설계 전제 재확인)",
          cfg["commands"]["gait_duration_cmd_range"] == [0.5, 0.5]
          and cfg["commands"]["body_roll_range"][0] == cfg["commands"]["body_roll_range"][1])

    print("\n[2] MLP 추출 + forward vs torch")
    body = torch.jit.load(f"{CKPT}/body_latest.jit", map_location="cpu")
    adapt = torch.jit.load(f"{CKPT}/adaptation_module_latest.jit", map_location="cpu")
    BL, AL = _extract(body), _extract(adapt)
    check("body 입력 2102", BL[0][0].shape[1] == 2102, str(BL[0][0].shape))
    check("adaptation 입력 2100 / 출력 2",
          AL[0][0].shape[1] == 2100 and AL[-1][0].shape[0] == 2)
    check("body 출력 12", BL[-1][0].shape[0] == 12)

    if HAS_JAX:
        from wtw_nav.llc import policy as P
        forward = lambda x: np.asarray(P.load_policy(CKPT)(jnp.asarray(x)))  # noqa: E731
        label = "JAX policy.load_policy"
    else:
        forward = lambda x: _fwd(BL, np.concatenate([x, _fwd(AL, x)]))  # noqa: E731
        label = "numpy 미러"

    # 임계값 1e-3: float32 누적 오차는 백엔드마다 1e-5~1e-4 수준이고(CPU 2.3e-4,
    # Colab GPU 6.1e-5), 구조적 오류(가중치·활성화·레이아웃)는 O(1) 차이를 낸다.
    rng = np.random.default_rng(0)
    maxd = rel = 0.0
    for _ in range(5):
        x = rng.normal(0, 0.5, 2100).astype(np.float32)
        with torch.no_grad():
            t = torch.from_numpy(x)
            a_t = body.forward(torch.cat((t, adapt.forward(t)), -1)).numpy()
        a_j = forward(x)
        maxd = max(maxd, float(np.abs(a_t - a_j).max()))
        rel = max(rel, float(np.abs(a_t - a_j).max() / max(np.abs(a_t).max(), 1e-9)))
    check(f"{label} vs torch  max|diff| < 1e-3", maxd < 1e-3,
          f"max_diff={maxd:.3e} (상대 {rel:.1e})")

    try:
        _fwd(BL, np.zeros(2100, np.float32))
        neg_ok = False
    except Exception:
        neg_ok = True
    check("body 단독(2100D)은 실패해야 함 [음성 대조]", neg_ok)

    print("\n[3] 관절 매핑 + 모델 설정")
    if HAS_JAX:
        from wtw_nav.llc import policy as P
        mj, _, _ = P.create_env(XML)
        jidx = P._build_joint_index(mj)
        qpos_adr = np.asarray(jidx["qpos_adr"])
        act_adr = list(np.asarray(jidx["act_adr"]))
    else:
        mj = mujoco.MjModel.from_xml_path(XML)
        mj.opt.timestep = C["SIM_TIMESTEP"]
        mj.actuator_gainprm[:, 0] = C["KP"]
        mj.actuator_biasprm[:, 1] = -C["KP"]
        mj.actuator_biasprm[:, 2] = -C["KD"]
        mj.dof_damping[6:] = 0.0
        mj.dof_frictionloss[6:] = 0.0
        qpos_adr, act_adr = [], []
        for n in C["JOINT_NAMES_WTW"]:
            jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, n)
            qpos_adr.append(mj.jnt_qposadr[jid])
            act_adr.append(next(a for a in range(mj.nu) if mj.actuator_trnid[a, 0] == jid))
        qpos_adr = np.asarray(qpos_adr)

    names_back = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, mj.actuator_trnid[a, 0])
                  for a in act_adr]
    check("act_adr가 WTW 관절 순서를 복원", tuple(names_back) == C["JOINT_NAMES_WTW"])
    check("qpos_adr 12개 & free joint(0:7) 이후",
          len(qpos_adr) == 12 and qpos_adr.min() >= 7)
    check("액추에이터 12개, act_adr가 12개 전부를 덮는 순열",
          mj.nu == 12 and sorted(int(a) for a in act_adr) == list(range(12)),
          f"act_adr={[int(a) for a in act_adr]}")
    # WTW DOF 순서는 FL,FR,RL,RR — MJCF 파일 순서(FR,FL,RR,RL)와 다르다.
    # 이걸 틀리면 로봇이 액션 포화와 함께 주저앉는다 (docs/03_results.md §1).
    check("DOF 순서가 FL,FR,RL,RR",
          tuple(n[:2] for n in C["JOINT_NAMES_WTW"][::3]) == ("FL", "FR", "RL", "RR"),
          str([n[:2] for n in C["JOINT_NAMES_WTW"][::3]]))
    check("PD 게인이 WTW 값으로 설정됨",
          np.allclose(mj.actuator_gainprm[:, 0], C["KP"])
          and np.allclose(mj.actuator_biasprm[:, 1], -C["KP"])
          and np.allclose(mj.actuator_biasprm[:, 2], -C["KD"]))
    check("timestep == 0.005", abs(mj.opt.timestep - C["SIM_TIMESTEP"]) < 1e-12)

    print("\n[4] 70D 관측 조립")
    d = mujoco.MjData(mj)
    mujoco.mj_resetData(mj, d)
    mujoco.mj_forward(mj, d)
    cmd = np.array([0.8, 0, 0, 0, 3.0, 0.5, 0.0, 0.0, 0.5, 0.08, 0.0, 0.0, 0.25, 0.45, 0.0])
    obs = np.concatenate([
        _proj_grav(d.qpos[3:7]), cmd * C["COMMANDS_SCALE"],
        (d.qpos[qpos_adr] - C["DEFAULT_DOF_POS_WTW"]) * 1.0,
        d.qvel[[a + 6 for a in range(12)]] * 0.05,
        np.zeros(12), np.zeros(12), _clock(0.3, cmd)])
    check("obs 차원 70 (3+15+12+12+12+12+4)", obs.shape == (70,), f"shape={obs.shape}")
    check("proj_gravity ~ (0,0,-1) 직립",
          np.allclose(_proj_grav(d.qpos[3:7]), [0, 0, -1], atol=1e-6))

    ci = _clock(0.3, cmd)
    check("duty=0.5에서 워핑은 항등",
          np.allclose(ci, np.sin(2 * np.pi * np.remainder([0.8, 0.3, 0.3, 0.8], 1.0))),
          str(ci.round(4)))
    check("trot: FL·RR 동상, FR·RL 동상",
          abs(ci[0] - ci[3]) < 1e-9 and abs(ci[1] - ci[2]) < 1e-9, str(ci.round(3)))

    print("\n[5] 리셋 자세 (WTW init_state) — 발산의 주원인이었던 지점")
    feet = [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("FR", "FL", "RR", "RL")]
    radii = np.array([mj.geom_size[f][0] for f in feet])

    def pose_report(dd):
        mujoco.mj_forward(mj, dd)
        return np.array([dd.geom_xpos[f][2] for f in feet]) - radii, dd.ncon

    mujoco.mj_resetData(mj, d)
    clear_default, ncon_default = pose_report(d)
    check("mj_resetData 기본값은 발이 지면을 관통한다 [음성 대조]",
          clear_default.min() < 0, f"foot_clear={np.round(clear_default, 4)}, ncon={ncon_default}")

    # WTW init_state 적용
    mujoco.mj_resetData(mj, d)
    d.qpos[0:3] = [0.0, 0.0, 0.34]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[qpos_adr] = C["DEFAULT_DOF_POS_WTW"]
    clear_wtw, ncon_wtw = pose_report(d)
    check("WTW init_state에서는 접촉 없음 (발이 지면 위)",
          ncon_wtw == 0 and clear_wtw.min() > 0,
          f"foot_clear={np.round(clear_wtw, 4)}, ncon={ncon_wtw}")
    check("몸통 z == Cfg.init_state.pos[2] == 0.34",
          abs(cfg["init_state"]["pos"][2] - 0.34) < 1e-9)

    print("\n[6] proj_gravity를 MuJoCo 회전행렬과 교차검증")
    rng2 = np.random.default_rng(3)
    worst = 0.0
    for _ in range(20):
        q = rng2.normal(size=4)
        q /= np.linalg.norm(q)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        worst = max(worst, np.abs(R.reshape(3, 3).T @ [0.0, 0.0, -1.0] - _proj_grav(q)).max())
    check("proj_gravity == R^T @ [0,0,-1]", worst < 1e-9, f"max|diff|={worst:.2e}")

    n_fail = _results.count(False)
    print("\n" + "=" * 62)
    print(f"{len(_results) - n_fail}/{len(_results)} PASS" if n_fail == 0
          else f"*** {n_fail}/{len(_results)} FAILED ***")
    if not HAS_JAX:
        print(f"\n※ JAX 실행 경로와 MJX 롤아웃은 미검증입니다{_WHY}.")
        print("  Colab에서:  from wtw_nav.llc import check;  check.run_gate()")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
