"""LLC 검증 — P0 / P0.5 게이트 (JAX 필요, Colab에서 실행).

로컬(CPU)에서는 `python -m wtw_nav.llc.test_policy`로 정적 검증만 돌린다.
JAX 실행 경로와 MJX 롤아웃은 이 모듈에서 확인한다.

Colab 사용법::

    from google.colab import drive; drive.mount('/content/drive')
    %cd "/content/drive/Othercomputers/BPC/D:/02_projects/2026_wtw"
    import sys; sys.path.append(".")
    from wtw_nav.dev import reload_wtw   # %autoreload 금지 (Colab 3.12에 imp 없음)

    from wtw_nav.llc import check, policy as P
    env = check.build()                 # 정책 + MJX 모델 (JIT 컴파일 1회, 재사용)
    check.run_gate(env)                 # P0  게이트 — 통과해야 다음 단계로

    # 영상 — 노트북이면 인라인 표시가 자동으로 켜진다
    check.preview(env, P.make_commands(vx=0.8, gait="trot"))   # 하나만 빠르게
    check.sweep_video(env=env)                                 # P0.5 전체
    check.sweep_video("pronk_slow", env=env)                   # 하나만

영상이 안 보일 때
    - `sweep_video`는 노트북이 아니면 인라인 표시를 생략하고 mp4만 남긴다
      (`checkpoints/llc_check/`). 강제하려면 `show=True`.
    - mp4 저장에는 ffmpeg이 필요하다. 없으면 저장만 실패하고 **표시는 계속**되며,
      동영상 표시까지 막히면 정지 프레임으로 대체된다.
    - 화면이 검으면 GL 백엔드 문제다. Colab은 `MUJOCO_GL=egl`이어야 하며
      **mujoco를 import 하기 전에** 정해져야 한다 (이 패키지가 자동 설정한다).

P0 게이트 (통과 기준)
    1. torch ↔ JAX 출력 차 < 1e-4
    2. trot `(0.5, 0, 0)`으로 vx=0.8 명령 시 실측 0.6~0.9 m/s 직진, 5 s 무낙상

P0.5 게이트 (판정)
    height 하한 / stance_width 하한·상한 / footswing 상한 / **pronk 4足 동시 체공**
    / vx–yaw 마름모 경계 추종. 특히 pronk 체공 여부가 duty 고정 상태에서 점프가
    가능한지를 결정한다 — `docs/02_hlc.md` §9 참조.
"""

from __future__ import annotations

import os
import platform

def default_gl() -> str:
    """이 런타임에서 쓸 GL 백엔드.

    ★ **EGL은 GPU가 있어야 한다.** Colab을 CPU 런타임으로 두면 EGL 초기화가
    실패하고, 그때 뜨는 `GL_HELP`는 "MUJOCO_GL=egl로 설정하라"고 안내한다 —
    **이미 그렇게 되어 있는데** 말이다. 원인을 엉뚱한 곳에서 찾게 되므로
    여기서 미리 갈라 놓는다 (2026-08-09).

    ⚠️ CPU 런타임에서 osmesa를 쓰려면 시스템 라이브러리가 필요하다:
        !apt-get -qq install -y libosmesa6-dev
    없으면 물리는 돌지만 **영상만 실패**한다.
    """
    import glob

    return "egl" if glob.glob("/dev/nvidia*") else "osmesa"


if platform.system() == "Linux":
    os.environ.setdefault("MUJOCO_GL", default_gl())

import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from wtw_nav.llc import policy as P

CKPT = ("walk-these-ways/runs/gait-conditioned-agility/pretrain-v0"
        "/train/025417.456545/checkpoints")
XML = "mujoco_menagerie/unitree_go1/scene.xml"


# --------------------------------------------------------------------------------------
def build(ckpt=CKPT, xml=XML, verbose=True, iterations=None, ls_iterations=None,
          collision=None):
    """정책 + MJX 모델 + 롤아웃 함수를 만든다.

    solver 설정의 **단일 출처는 `policy.create_env`** 다. 여기서 기본값을 중복 정의하지
    말 것 — 실제로 그렇게 해뒀다가 `create_env`만 고치고 이쪽을 놓쳐서, 기본값이 8/16인데
    실행은 4/8로 도는 일이 있었다. `None`이면 `create_env`의 기본값을 그대로 쓴다.

    A/B 목적으로만 명시적으로 넘기고, 평상시엔 건드리지 말 것.
    """
    t0 = time.time()
    policy_fn = P.load_policy(ckpt)
    kw = {}
    if iterations is not None:
        kw["iterations"] = iterations
    if ls_iterations is not None:
        kw["ls_iterations"] = ls_iterations
    if collision is not None:
        kw["collision"] = collision
    mj_model, mj_data, mjx_model = P.create_env(xml, verbose=verbose, **kw)
    iterations = int(mj_model.opt.iterations)
    ls_iterations = int(mj_model.opt.ls_iterations)
    jidx = P._build_joint_index(mj_model)
    rollout_fn = jax.jit(P.make_rollout_fn(mjx_model, policy_fn, jidx))
    if verbose:
        print(f"  build: {time.time() - t0:.1f}s  "
              f"(nq={mj_model.nq}, nv={mj_model.nv}, ngeom={mj_model.ngeom}, "
              f"backend={jax.default_backend()}, iters={iterations}/{ls_iterations})")
        if iterations < 8 or ls_iterations < 16:
            print(f"  ⚠️ solver {iterations}/{ls_iterations} 는 **검증된 수렴 설정(8/16) 미만**입니다.\n"
                  "     이 구간에서는 접촉 해가 수렴하지 않아 보행이 실패하고, 결과가\n"
                  "     하드웨어(CPU/GPU)에 따라 달라집니다. 기본값이 8/16인데 4/8이 보인다면\n"
                  "     **수정된 policy.py가 반영되지 않은 것**입니다 (Drive 동기화 / 모듈 리로드 확인).")
        ok, msg = gl_status()
        if not ok:
            print("  ⚠️ 렌더링 불가 (수치 지표는 정상 동작). 자세한 내용은 아래.\n" + msg)
    return dict(policy_fn=policy_fn, mj_model=mj_model, mj_data=mj_data,
                mjx_model=mjx_model, jidx=jidx, rollout_fn=rollout_fn)


def rollout(env, commands, seconds=5.0, settle=0.5, verbose=False):
    """LLC를 단독 구동한다. 전체가 `lax.scan` 하나로 돈다 (컴파일 1회, 디스패치 1회).

    Args:
        env: `build()`의 반환값.
        commands: (15,) 또는 (T, 15). 전자는 전 구간 동일 명령으로 확장된다.
        settle: 낙하·정렬용 초기 구간 (속도 측정과 낙상 판정에서 제외).

    Returns:
        dict(qpos[T,nq], qvel[T,nv], vel_b[T,3](몸통 좌표계 선속도),
             mean_vx/mean_vy(몸통 기준), mean_wz, fell, diverged_at)
    """
    mj_model, mj_data = env["mj_model"], env["mj_data"]
    n_steps = int(seconds / P.POLICY_DT)
    n_settle = int(settle / P.POLICY_DT)

    cmds = jnp.asarray(commands, jnp.float32)
    if cmds.ndim == 1:
        cmds = jnp.broadcast_to(cmds, (n_steps, 15))
    n_steps = cmds.shape[0]

    data = P.reset_data(mj_model, mj_data, env["mjx_model"], env["jidx"])
    obs_history, gait, last_a, last_last_a = P.init_llc_state()

    t0 = time.time()
    _, qpos, qvel = env["rollout_fn"](
        data, obs_history, gait, last_a, last_last_a, cmds)
    qpos = np.asarray(jax.block_until_ready(qpos))   # 여기서 한 번만 동기화
    qvel = np.asarray(qvel)
    if verbose:
        dt = time.time() - t0
        print(f"       rollout {n_steps} steps ({seconds:.1f}s sim): "
              f"{dt:.1f}s wall  ({n_steps / dt:.0f} steps/s, 컴파일 포함)")

    vel_b = body_velocity(qpos, qvel)

    finite = np.all(np.isfinite(qpos), axis=1) & np.all(np.isfinite(qvel), axis=1)
    if not finite.all():
        first = int(np.argmin(finite))
        print(f"       ⚠️ 스텝 {first}/{n_steps} ({first * P.POLICY_DT:.2f}s)에서 발산")
        if first > 0:
            j = max(0, first - 1)
            print(f"          직전: z={qpos[j, 2]:.4f}  |qvel|max={np.abs(qvel[j]).max():.1f}  "
                  f"관절범위=[{qpos[j, 7:].min():+.2f}, {qpos[j, 7:].max():+.2f}]")
        print("          점검: ① 초기 자세(P.reset_data 사용 여부) ② solver iterations")
        print("               ③ PD 게인 ④ integrator (Euler 유지해야 함)")
        return dict(qpos=qpos, qvel=qvel, vel_b=vel_b, mean_vx=float("nan"),
                    mean_vy=float("nan"), mean_wz=float("nan"), fell=True,
                    diverged_at=first)

    return dict(qpos=qpos, qvel=qvel, vel_b=vel_b,
                mean_vx=float(np.mean(vel_b[n_settle:, 0])),
                mean_vy=float(np.mean(vel_b[n_settle:, 1])),
                mean_wz=float(np.mean(qvel[n_settle:, 5])),
                fell=bool(np.any(qpos[n_settle:, 2] < 0.15)),
                diverged_at=None)


GL_HELP = """MuJoCo Renderer 생성 실패: {err}
  현재 MUJOCO_GL={gl!r}, 이미 로드된 백엔드={loaded}

  ⚠️ MUJOCO_GL은 **mujoco를 import 하는 순간** 읽힙니다. 노트북에서 이미 mujoco가
     import된 뒤에 값을 바꿔도 소용이 없습니다 (glfw가 잡혀 있으면 headless에서 실패).

  해결: 런타임을 재시작하고, **첫 셀에서 다른 어떤 import보다 먼저** 설정하십시오.

      import os
      os.environ["MUJOCO_GL"] = "{want}"
      # ↑ 이 두 줄이 mujoco / mujoco.mjx / wtw_nav 어떤 import보다도 먼저 와야 합니다
      from wtw_nav.llc import check

  ⚠️ **GPU 런타임이 아니면 egl은 되지 않습니다** (EGL은 GPU 드라이버를 씁니다).
     CPU 런타임에서는 소프트웨어 렌더러를 씁니다 — 라이브러리를 먼저 깔아야 합니다:

      !apt-get -qq install -y libosmesa6-dev
      os.environ["MUJOCO_GL"] = "osmesa"

     이 런타임의 자동 판정: {want!r} (/dev/nvidia* {nv})

  렌더링 없이 수치 지표만 보려면: check.sweep_video(..., video=False)
"""


def _gl_ctx() -> dict:
    """`GL_HELP.format()`에 항상 넣어야 하는 진단 값."""
    import glob

    nv = glob.glob("/dev/nvidia*")
    return {"want": default_gl(), "nv": "있음" if nv else "없음"}


def gl_status():
    """(사용 가능 여부, 설명). 렌더링을 시도하기 전에 값싸게 확인한다."""
    loaded = [m for m in ("glfw", "OpenGL.EGL", "mujoco.egl", "mujoco.glfw", "mujoco.osmesa")
              if m in sys.modules]
    try:
        m = mujoco.MjModel.from_xml_string("<mujoco/>")
        r = mujoco.Renderer(m, height=16, width=16)
        r.close()
        return True, f"MUJOCO_GL={os.environ.get('MUJOCO_GL')!r}"
    except Exception as e:
        return False, GL_HELP.format(err=f"{type(e).__name__}: {e}",
                                     gl=os.environ.get("MUJOCO_GL"),
                                     loaded=loaded or "(없음)", **_gl_ctx())


def in_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def body_velocity(qpos, qvel):
    """월드 선속도를 몸통 좌표계로 회전 (T, 3).

    WTW의 `vx`/`vy` 명령은 **몸통 기준**이다. 월드 x 속도로 재면 로봇이 방향을 튼
    만큼 과소평가된다 — 추종 판정에는 반드시 이 값을 써야 한다.
    """
    w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    # R^T의 행 = R의 열
    r0 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1)
    r1 = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], -1)
    r2 = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)
    v = qvel[:, 0:3]
    return np.stack([(r0 * v).sum(-1), (r1 * v).sum(-1), (r2 * v).sum(-1)], -1)


def render(mj_model, qpos, fps=25, width=640, height=480, track=True,
           elevation=-20.0, azimuth=120.0, every=None, distance=2.0,
           geomgroups=None):
    """로그된 qpos에서 프레임을 만든다.

    MJX 롤아웃 도중에 렌더링하면 스텝마다 host 동기화가 걸린다. 시뮬레이션이 끝난 뒤
    일반 MuJoCo로 재생하는 편이 훨씬 빠르고, 물리 결과는 동일하다.

    Args:
        every: 몇 프레임마다 그릴지. `None`이면 **qpos가 LLC 주기(50 Hz)로
            기록됐다고 가정**하고 `fps`에서 역산한다. ⚠️ HLC 롤아웃
            (`envs.scripted`)은 10 Hz로 기록하므로 반드시 `every=1`을 넘겨야
            한다 — 안 그러면 5프레임 중 4개를 버려 2 fps 영상이 나온다.
        distance: 추적 카메라 거리 (m). 사다리 코스는 3~4가 보기 좋다.
        geomgroups: 그릴 geom group들. `None`이면 로봇 시각(0~2) + 지형
            (`modules.TERRAIN_GROUP`). ⚠️ 아래 주석 참조 — 이걸 안 켜면 지형이
            통째로 안 보인다.

    실패하면 조용히 넘어가지 않고 원인을 알려준다 (Colab에서는 `MUJOCO_GL=egl` 필요).
    """
    if every is None:
        every = max(1, int(round(1.0 / (fps * P.POLICY_DT))))
    try:
        renderer = mujoco.Renderer(mj_model, height=height, width=width)
    except (TypeError, ValueError):
        # ⚠️ 인자가 잘못된 것을 GL 문제로 포장하지 않는다 (2026-08-06). `height`가
        #    실수로 명령값(-0.22)이 되어 죽었는데 "런타임을 재시작하십시오"라고
        #    안내하는 바람에 원인에서 멀어졌다. 원본 예외를 그대로 올린다.
        raise
    except Exception as e:
        raise RuntimeError(GL_HELP.format(
            err=f"{type(e).__name__}: {e}", gl=os.environ.get("MUJOCO_GL"),
            loaded=[m for m in ("glfw", "mujoco.egl", "mujoco.glfw") if m in sys.modules]
            or "(없음)", **_gl_ctx())) from e

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(mj_model, camera)
    camera.distance = distance
    camera.elevation = elevation
    camera.azimuth = azimuth

    # ★ **지형을 보이게 한다** (2026-08-06). 렌더러 기본 옵션은 geom group 4·5를
    #   그리지 않는데, 우리 지형은 전부 `modules.TERRAIN_GROUP`(=5)이다 — 라이다가
    #   로봇 자기 다리를 맞는 것을 막으려고 옮겼던 그 그룹이다. 그대로 두면 물리는
    #   정상인데 **화면에서는 로봇이 허공을 걷는다.** 실제로 경사 영상에서 그렇게
    #   나왔고, 하마터면 "지형이 안 만들어졌다"로 오진할 뻔했다.
    #   로봇 충돌 geom(group 3)은 끈 채로 둔다 — 켜면 시각 메시와 겹쳐 지저분하다.
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    if geomgroups is None:
        from wtw_nav.terrain.modules import TERRAIN_GROUP
        geomgroups = (0, 1, 2, TERRAIN_GROUP)
    opt.geomgroup[:] = 0
    for g in geomgroups:
        opt.geomgroup[g] = 1

    d = mujoco.MjData(mj_model)
    frames = []
    try:
        for q in qpos[::every]:
            d.qpos[:] = q
            mujoco.mj_forward(mj_model, d)
            if track:
                camera.lookat[:] = d.qpos[:3]
            renderer.update_scene(d, camera=camera, scene_option=opt)
            frames.append(renderer.render())
    finally:
        renderer.close()

    if frames and float(np.mean(frames[len(frames) // 2])) < 1.0:
        print("       ⚠️ 프레임이 거의 검습니다 — GL 백엔드(EGL) 문제일 수 있습니다.")
    return frames


def preview(env=None, cmd=None, seconds=4.0, fps=25, save=None,
            return_data=False, **render_kw):
    """명령 하나를 굴려 바로 보여준다. 노트북에서 가장 간단한 확인 경로.

        from wtw_nav.llc import check, policy as P
        env = check.build()
        check.preview(env, P.make_commands(vx=0.8, gait="trot"))

    기본적으로 **아무것도 반환하지 않는다** — 노트북이 qpos/프레임 배열을 통째로
    찍어버리는 것을 막기 위해서다. 데이터가 필요하면 `return_data=True`.
    """
    env = env or build()
    cmd = P.make_commands(vx=0.8, gait="trot") if cmd is None else cmd
    out = rollout(env, cmd, seconds=seconds, verbose=True)
    frames = render(env["mj_model"], out["qpos"], fps=fps, **render_kw)
    print(f"       {len(frames)} 프레임  vx={out['mean_vx']:+.3f} vy={out['mean_vy']:+.3f} "
          f"wz={out['mean_wz']:+.3f}  min_z={np.nanmin(out['qpos'][:, 2]):.3f}  "
          f"낙상={out['fell']}")
    _show(frames, fps, save, title="preview", show=True)
    if return_data:
        return out, frames
    return None


def _show(frames, fps, save=None, title="", show=None):
    """노트북이면 인라인 표시, 경로가 주어지면 mp4로도 저장.

    저장(ffmpeg)과 표시는 **서로 독립**이다. mp4 인코딩이 실패해도 표시는 계속하고,
    동영상 표시가 안 되면 정지 프레임 몇 장으로라도 보여준다 — 렌더링은 됐는데
    인코더가 없어서 아무것도 못 보는 상황을 막는다.
    """
    if not frames:
        print("       ⚠️ 프레임이 없습니다.")
        return
    if show is None:
        show = in_notebook()
    try:
        import mediapy
    except ImportError:
        print("       ⚠️ mediapy 없음 → `pip install mediapy`. 표시를 건너뜁니다.")
        return

    if save:
        try:
            mediapy.write_video(save, frames, fps=fps)
            print(f"       저장: {save}")
        except Exception as e:                       # 대개 ffmpeg 부재
            print(f"       ⚠️ mp4 저장 실패({type(e).__name__}: {e}). "
                  "표시는 계속합니다. 필요하면 `apt install ffmpeg`.")

    if not show:
        print(f"       (인라인 표시 꺼짐 — 프레임 {len(frames)}개 준비됨)")
        return
    try:
        mediapy.show_video(frames, fps=fps, title=title)
    except Exception as e:
        print(f"       ⚠️ 동영상 표시 실패({type(e).__name__}) → 정지 프레임으로 대체")
        k = max(1, len(frames) // 6)
        try:
            mediapy.show_images(frames[::k][:6], columns=6,
                                titles=[f"{i*k}" for i in range(len(frames[::k][:6]))])
        except Exception as e2:
            print(f"       ⚠️ 정지 프레임 표시도 실패({type(e2).__name__}: {e2})")


def smoke_test(env=None, steps=10):
    """롤아웃이 실제로 도는지 최소 비용으로 확인한다. 컴파일 시간을 분리해 보여준다."""
    env = env or build()
    print(f"[smoke] {steps} 스텝 롤아웃 — 컴파일 시간 측정")
    out = rollout(env, P.make_commands(vx=0.5, gait="trot"),
                  seconds=steps * P.POLICY_DT, settle=0.0, verbose=True)
    print(f"        z: {out['qpos'][0, 2]:.3f} -> {out['qpos'][-1, 2]:.3f}, "
          f"x: {out['qpos'][-1, 0]:+.4f}")
    if not np.all(np.isfinite(out["qpos"])):
        print("        ⚠️ qpos에 NaN/Inf — 물리가 발산했습니다.")
    return out


# --------------------------------------------------------------------------------------
def run_gate(env=None, verbose=True):
    """P0 게이트. 통과하면 True."""
    import torch

    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))

    print("=" * 62)
    print("P0 게이트 (JAX / MJX)")
    print("=" * 62)

    env = env or build()

    print("\n[1] torch vs JAX 출력 일치")
    body = torch.jit.load(f"{CKPT}/body_latest.jit", map_location="cpu")
    adapt = torch.jit.load(f"{CKPT}/adaptation_module_latest.jit", map_location="cpu")
    # 임계값 1e-3 — float32 누적 오차는 백엔드마다 1e-5~1e-4(CPU 2.3e-4, GPU 6.1e-5),
    # 구조적 오류는 O(1). 자세한 근거는 test_policy.py 참조.
    rng = np.random.default_rng(0)
    maxd = 0.0
    for _ in range(5):
        x = rng.normal(0, 0.5, P.OBS_HISTORY_DIM).astype(np.float32)
        with torch.no_grad():
            t = torch.from_numpy(x)
            a_t = body.forward(torch.cat((t, adapt.forward(t)), -1)).numpy()
        maxd = max(maxd, np.abs(a_t - np.asarray(env["policy_fn"](jnp.asarray(x)))).max())
    check("torch vs JAX  max|diff| < 1e-3", maxd < 1e-3, f"max_diff={maxd:.3e}")

    print("\n[2] 리셋 자세 (WTW init_state)")
    mj_model, mj_data, jidx = env["mj_model"], env["mj_data"], env["jidx"]
    P.reset_data(mj_model, mj_data, env["mjx_model"], jidx)
    clear = foot_clearance(mj_data.qpos[None, :], mj_model)[0]
    check("몸통 z == 0.34", abs(mj_data.qpos[2] - 0.34) < 1e-6, f"z={mj_data.qpos[2]:.4f}")
    check("리셋 시 접촉 없음 (발이 지면을 관통하지 않음)", mj_data.ncon == 0,
          f"ncon={mj_data.ncon}, foot_clear={np.round(clear, 4)}")

    print("\n[3] 스모크 테스트 (10 스텝) — 컴파일 시간 확인")
    smoke = smoke_test(env, steps=10)
    check("qpos 유한 (물리 발산 없음)", smoke["diverged_at"] is None)

    print("\n[4] trot vx=0.8 직진 롤아웃 (5 s)")
    out = rollout(env, P.make_commands(vx=0.8, gait="trot"), seconds=5.0, verbose=True)
    check("발산 없음", out["diverged_at"] is None)
    check("낙상하지 않음 (몸통 z >= 0.15 유지)", not out["fell"],
          f"min_z={np.nanmin(out['qpos'][:, 2]):.3f}")
    check("실측 전진속도 0.6~0.9 m/s", 0.6 <= out["mean_vx"] <= 0.9,
          f"mean_vx={out['mean_vx']:.3f} m/s")

    # 정지 명령에서 이 정책은 **약 +0.09 m/s의 지속적 전진 편향**을 보인다.
    # 초기 착지 과도현상이 아니라 선형 편향이며(1~5s 0.0948, 5~10s 0.0930 m/s),
    # MJX와 MuJoCo C 엔진이 일치하므로 이식 오류가 아니라 sim-to-sim 격차의
    # 결과다(actuator_net→PD 근사, 질량 차이 등). 명령 범위(0.5~1.9 m/s) 대비
    # 작고 HLC가 폐루프로 보정하므로 P0 기준으로는 허용한다.
    # 기준값 0.15 m/s는 실측 0.09에 여유를 둔 값이며, 이보다 커지면 회귀 신호다.
    print("\n[5] 정지 명령 (편향 실측 ≈ +0.09 m/s)")
    out0 = rollout(env, P.make_commands(vx=0.0, gait="trot"), seconds=5.0)
    check("정지 편향 |vx| < 0.15 m/s", abs(out0["mean_vx"]) < 0.15 and not out0["fell"],
          f"vx={out0['mean_vx']:+.3f} m/s, x_end={out0['qpos'][-1, 0]:+.3f} m")

    # 관절 순서(FL,FR,RL,RR)와 solver iterations가 틀리면 여기가 무너진다 — 회귀 검사.
    # 속도는 반드시 몸통 좌표계로 잰다 (명령이 몸통 기준이므로).
    print("\n[6] 명령 추종 (관절 순서 · solver 회귀 검사)")
    for label, kw, key, lo, hi in (
            ("후진 vx=-0.5", dict(vx=-0.5), "mean_vx", -0.65, -0.25),
            ("게걸음 vy=+0.4", dict(vy=0.4), "mean_vy", 0.25, 0.65),
            ("회전 yaw=+0.8", dict(yaw=0.8), "mean_wz", 0.5, 1.4)):
        o = rollout(env, P.make_commands(**kw), seconds=4.0)
        v = o[key]
        check(f"{label} 추종", lo <= v <= hi and not o["fell"], f"측정={v:+.3f}")

    n_fail = results.count(False)
    print("\n" + "=" * 62)
    if n_fail == 0:
        print(f"{len(results)}/{len(results)} PASS  →  P0 통과. P0.5(sweep_video)로 진행.")
    else:
        print(f"*** {n_fail}/{len(results)} FAILED — 다음 단계로 넘어가지 말 것 ***")
        print("점검 순서: ① adaptation module 로드 여부 ② 관절 순서 ③ PD 게인/timestep")
        print("④ WTW는 actuator_net을 쓰지만 여기서는 PD 근사 — 추종 오차의 1순위 용의자")
    return n_fail == 0


# --------------------------------------------------------------------------------------
SWEEPS = {
    # name          : (설명, t -> 15D 명령)
    "trot_forward":   ("trot, vx 0 -> 0.8", lambda t: P.make_commands(
        vx=float(np.clip(t / 2.0, 0, 1) * 0.8), gait="trot")),
    "height_low":     ("몸통 높이 하한 (-0.22)", lambda t: P.make_commands(
        vx=0.4, height=-0.22, gait="trot")),
    "stance_narrow":  ("stance_width 하한 (0.12)", lambda t: P.make_commands(
        vx=0.4, stance_width=0.12, gait="trot")),
    "stance_wide":    ("stance_width 상한 (0.42)", lambda t: P.make_commands(
        vx=0.4, stance_width=0.42, gait="trot")),
    "footswing_high": ("footswing 상한 (0.32)", lambda t: P.make_commands(
        vx=0.4, footswing=0.32, gait="trot")),
    "pitch_sweep":    ("pitch ±0.35", lambda t: P.make_commands(
        vx=0.4, pitch=0.35 * float(np.sin(2 * np.pi * t / 4.0)), gait="trot")),
    # ★ P0.5 핵심 판정: duty 고정 상태에서 점프(4足 동시 체공)가 나오는가
    "pronk_slow":     ("pronk, step_freq 하한 2.0", lambda t: P.make_commands(
        vx=0.3, step_freq=2.0, footswing=0.32, gait="pronk")),
    "pronk_fast":     ("pronk, step_freq 상한 3.9", lambda t: P.make_commands(
        vx=0.3, step_freq=3.9, footswing=0.32, gait="pronk")),
    # vx-yaw 마름모 경계 (trot: VX_MAX=1.9, YAW_MAX=2.4의 L1 경계 근처 3점)
    "diamond_a":      ("vx=1.7, yaw=0.2", lambda t: P.make_commands(vx=1.7, yaw=0.2)),
    "diamond_b":      ("vx=1.0, yaw=1.1", lambda t: P.make_commands(vx=1.0, yaw=1.1)),
    "diamond_c":      ("vx=0.3, yaw=2.0", lambda t: P.make_commands(vx=0.3, yaw=2.0)),
}


def jump_metrics(qpos, mj_model, settle=25, min_steps=2):
    """점프의 '품질'을 잰다 — 체공률만으로는 판단할 수 없다.

    ⚠️ 체공률(4발이 동시에 뜬 비율)은 **점프의 증거가 아니다.** 다리를 몸쪽으로
    접기만 해도 발은 뜬다. 실제로 이 정책의 pronk가 그렇다: 체공률 20~30%인데
    **정점 상승이 +0.001 m**로 몸통이 전혀 뜨지 않는다.

    틈을 건널 수 있는지 판단하려면 다음이 필요하다:
      apex_rise : 체공 구간에서 몸통이 실제로 상승한 높이 (진짜 점프면 > 0)
      air_dx    : 체공 중 수평 이동 거리  ← **건널 수 있는 틈 폭의 상한**
      air_ms    : 연속 체공 지속시간

    Returns:
        dict(flight_frac, air_ms, air_dx, apex_rise, z_amp, front_contact, rear_contact)
    """
    cl = foot_clearance(qpos, mj_model)
    if cl.size == 0:
        return {}
    q = qpos[settle:]
    cl = cl[settle:]
    z = q[:, 2]
    # 직립도 (proj_gravity z). 넘어진 상태를 체공으로 세지 않기 위해 필요하다.
    w, x, y, zq = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
    gz = -(1 - 2 * (x * x + y * y))
    air = (cl.min(axis=1) > 0.03) & (gz < -0.8) & (z > 0.20)

    runs, s = [], None
    for i, f in enumerate(air):
        if f and s is None:
            s = i
        elif not f and s is not None:
            runs.append((s, i)); s = None
    if s is not None:
        runs.append((s, len(air)))
    runs = [r for r in runs if r[1] - r[0] >= min_steps]

    out = dict(flight_frac=float(air.mean()),
               z_amp=float(z.max() - z.min()),
               front_contact=float((cl[:, :2] < 0.005).mean()),
               rear_contact=float((cl[:, 2:] < 0.005).mean()),
               air_ms=0.0, air_dx=0.0, apex_rise=0.0, n_flights=len(runs))
    if runs:
        out["air_ms"] = float(np.mean([(b - a) * P.POLICY_DT for a, b in runs]) * 1000)
        out["air_dx"] = float(np.max([q[b - 1, 0] - q[a, 0] for a, b in runs]))
        out["apex_rise"] = float(np.max([z[a:b].max() - z[a] for a, b in runs]))
    return out


def foot_clearance(qpos, mj_model):
    """스텝별 발 4개의 지면 클리어런스 (m).

    발 geom은 반경 0.023 m의 구이므로 중심 높이에서 반경을 뺀 값이 실제 클리어런스다.
    """
    feet = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("FR", "FL", "RR", "RL")]
    feet = [f for f in feet if f >= 0]
    if not feet:
        return np.empty((0, 0))
    radii = np.array([mj_model.geom_size[f][0] for f in feet])

    d = mujoco.MjData(mj_model)
    out = []
    for q in qpos:
        d.qpos[:] = q
        mujoco.mj_kinematics(mj_model, d)
        out.append(np.array([d.geom_xpos[f][2] for f in feet]) - radii)
    return np.array(out)


def version_check():
    """Colab이 최신 코드를 보고 있는지 확인한다 (Drive 동기화 / 모듈 리로드 점검).

        from wtw_nav.llc import check; check.version_check()
    """
    import inspect
    import datetime as _dt

    ok = True
    for mod in (P, sys.modules[__name__]):
        path = inspect.getfile(mod)
        mtime = _dt.datetime.fromtimestamp(os.path.getmtime(path))
        print(f"  {mod.__name__:22s} {mtime:%Y-%m-%d %H:%M}  {path}")

    # ⚠️ 시그니처 기본값만 보면 안 된다 — build()가 자기 기본값으로 덮어쓰던 버그를
    #    이 방식으로는 못 잡았다. **실제로 모델을 만들어** 최종 반영값을 확인한다.
    m, _, _ = P.create_env(XML, verbose=False)
    it, ls = int(m.opt.iterations), int(m.opt.ls_iterations)
    good = (it >= 8 and ls >= 16)
    ok &= good
    print(f"\n  실제 생성된 모델의 solver = {it}/{ls}  "
          f"{'OK (수렴 설정)' if good else '❌ 8/16 이어야 합니다'}")

    sig = inspect.signature(build)
    bd = (sig.parameters["iterations"].default, sig.parameters["ls_iterations"].default)
    good = bd == (None, None)
    ok &= good
    print(f"  build()의 solver 기본값 = {bd}  "
          f"{'OK (create_env에 위임)' if good else '❌ 기본값을 중복 정의하고 있습니다'}")

    legs = tuple(n[:2] for n in P.JOINT_NAMES_WTW[::3])
    good = legs == ("FL", "FR", "RL", "RR")
    ok &= good
    print(f"  DOF 순서 = {legs}  {'OK' if good else '❌ FL,FR,RL,RR 이어야 합니다'}")

    if not ok:
        print("\n  → Drive 동기화를 기다린 뒤 **런타임 재시작**하거나,\n"
              "     `from wtw_nav.dev import reload_wtw; reload_wtw()` 후 다시 import 하십시오.\n"
              "     (Colab은 Python 3.12라 %autoreload 를 쓸 수 없습니다 — imp 모듈이 없습니다)")
    return ok


def solver_ab(iters=(8, 16), ls=None, seconds=4.0, cmds=None):
    """MJX 솔버 반복 횟수 A/B (몸통 좌표계 속도로 판정).

    MJX는 접촉 해가 덜 수렴하면 정책이 보행에 실패한다. **MuJoCo C 엔진에서는
    iterations=4로도 잘 걷기 때문에**, C 엔진이나 정적 유지 테스트로 이 값을 정하면
    안 된다. 2026-07-28 실측: 4/8 낙상, 8/16 정상, 16/32는 8/16과 동일(수렴).

        from wtw_nav.llc import check
        check.solver_ab()
    """
    if cmds is None:
        cmds = [("vx=+0.8", P.make_commands(vx=0.8), "mean_vx"),
                ("vy=+0.4", P.make_commands(vy=0.4), "mean_vy"),
                ("vx=-0.5", P.make_commands(vx=-0.5), "mean_vx")]
    print("MuJoCo C 엔진 참조: vx 0.865 / vy 0.490 / -0.432,  min_z 0.14~0.18")
    print(f"{'iters/ls':>10s} {'명령':10s} {'min_z':>6s} {'추종':>7s} "
          f"{'vx_b':>7s} {'vy_b':>7s} {'낙상':>5s}")
    print("=" * 64)
    out = {}
    for it in iters:
        lsi = (2 * it) if ls is None else ls
        env = build(iterations=it, ls_iterations=lsi, verbose=False)
        for label, c, key in cmds:
            r = rollout(env, c, seconds=seconds)
            mz = float(np.nanmin(r["qpos"][:, 2]))
            out[(it, label)] = (mz, r["mean_vx"], r["mean_vy"], r["fell"])
            print(f"{f'{it}/{lsi}':>10s} {label:10s} {mz:6.3f} {r[key]:7.3f} "
                  f"{r['mean_vx']:7.3f} {r['mean_vy']:7.3f} {str(r['fell']):>5s}")
    return out


def collision_ab(modes=("full", "world", "feet"), seconds=4.0, cmds=None):
    """충돌 필터 A/B — **속도가 아니라 추종 성능**으로 판정한다.

        from wtw_nav.llc import check
        check.collision_ab()

    menagerie 원본은 43개 geom이 서로 충돌해 매 스텝 843쌍을 검사하지만, 평지에서
    실제 접촉은 4개뿐이다. 자기충돌을 끄면(L4 실측) 물리 처리량이 **7.1배** 오른다.
    다만 WTW는 `self_collisions = 0`(자기충돌 켬)으로 학습했으므로 원본과의 편차다.
    `full` 대비 추종·min_z가 유지되어야 채택할 수 있다.

    속도 비교는 `python -m wtw_nav.bench collision` 쪽이다.
    """
    if cmds is None:
        cmds = [("vx=+0.8", P.make_commands(vx=0.8), "mean_vx"),
                ("vy=+0.4", P.make_commands(vy=0.4), "mean_vy"),
                ("yaw=+0.8", P.make_commands(yaw=0.8), "mean_wz")]
    print("판정: `full` 대비 추종 오차 0.05 이내, min_z 유지, 낙상 없음")
    print(f"{'mode':>7s} {'쌍':>5s} {'명령':10s} {'min_z':>6s} {'추종':>7s} "
          f"{'vx_b':>7s} {'vy_b':>7s} {'낙상':>5s}")
    print("=" * 64)
    out = {}
    for mode in modes:
        env = build(collision=mode, verbose=False)
        npair = _candidate_pairs(env["mj_model"])
        for label, c, key in cmds:
            r = rollout(env, c, seconds=seconds)
            mz = float(np.nanmin(r["qpos"][:, 2]))
            out[(mode, label)] = (mz, r[key], r["fell"])
            print(f"{mode:>7s} {npair:>5d} {label:10s} {mz:6.3f} {r[key]:7.3f} "
                  f"{r['mean_vx']:7.3f} {r['mean_vy']:7.3f} {str(r['fell']):>5s}")
    print("\n`full` 행과 나란히 비교하십시오. 차이가 크면 `HLCConfig.collision`을 "
          "'full'로 되돌리십시오.")
    return out


def deadzone_sweep(env=None, seconds=6.0, vals=(0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50),
                   video=True, fps=25):
    """★ 실측 ① — **미소 명령 구간에서 LLC가 실제로 얼마나 나쁜가.**

        from wtw_nav.llc import check
        check.deadzone_sweep()

    왜 재는가. `ActionConfig.deadzone = 0.2`는 "LLC 학습 시 ‖(vx,vy)‖<=0.2가 0으로
    치환됐다"는 **근거**만으로 잡은 값이고, 그 구간의 **실제 성능 저하 폭은 재본 적이
    없다.** 그런데 이 데드존 때문에 HLC가 낼 수 있는 전진속도에 구멍이 생긴다:

        vx=0 명령 -> 실제 +0.09 m/s (정지 편향)
        이를 상쇄할 vx≈-0.1 -> 데드존에 먹혀 0
        => "아주 천천히" 가 존재하지 않는다. 정지하려면 확 후진해야 한다.

    P1은 goal_radius 0.5 m라 무해하지만, P2의 외나무다리 끝·틈 직전 **cm 단위
    정지**에는 치명적이다.

    판정: 명령 대비 실제 속도가 단조 증가하고 오차가 크지 않다면 데드존을 좁혀도
    된다. 0.2 미만에서 추종이 무너지거나 낙상하면 현행 유지가 옳다.
    """
    env = env or build(verbose=False)
    print("미소 vx 명령 실측 — 데드존 0.2가 정말 필요한가")
    print(f"{'명령 vx':>8s} {'실측 vx':>9s} {'오차':>8s} {'실측 vy':>9s} "
          f"{'min_z':>6s} {'낙상':>5s}")
    print("=" * 52)
    out, traj = {}, {}
    for v in vals:
        r = rollout(env, P.make_commands(vx=float(v)), seconds=seconds)
        mz = float(np.nanmin(r["qpos"][:, 2]))
        err = r["mean_vx"] - v
        out[v] = (r["mean_vx"], r["mean_vy"], mz, r["fell"])
        traj[v] = r["qpos"]
        print(f"{v:8.2f} {r['mean_vx']:9.3f} {err:+8.3f} {r['mean_vy']:9.3f} "
              f"{mz:6.3f} {str(r['fell']):>5s}")

    # 판정: 데드존 안(0<v<=0.2)에서도 명령이 실제 속도를 움직이는가?
    inside = [v for v in vals if 0.0 < v <= 0.2]
    if len(inside) >= 2:
        lo, hi = inside[0], inside[-1]
        slope = (out[hi][0] - out[lo][0]) / (hi - lo)
        print(f"\n데드존 내부 기울기 d(실측)/d(명령) = {slope:.2f}  "
              f"(1.0이면 완전 추종, 0이면 무반응)")
        if slope > 0.5:
            print("  -> 미소 명령이 **먹힙니다.** `ActionConfig.deadzone`을 좁혀")
            print("     정밀 정지를 가능하게 할 여지가 있습니다.")
        else:
            print("  -> 미소 명령이 거의 무시됩니다. 데드존 0.2 유지가 옳습니다.")
            print("     정밀 정지는 다른 수단(코스 설계·LLC 재학습)이 필요합니다.")

    # 정지 명령에서의 편향 — HLC가 상쇄해야 할 양
    if 0.0 in out:
        print(f"정지 편향 = {out[0.0][0]:+.3f} m/s  "
              f"(이를 없애려면 vx≈{-out[0.0][0]:.2f} 명령이 필요)")

    # ⚠️ 지표만 믿지 말 것 — 전복한 로봇도 mean_vx가 그럴듯하게 나올 수 있다
    #    (pronk 체공률 오판 사례, llc_port_debug §8). 판정 장면을 눈으로 볼 것.
    if video:
        os.makedirs("checkpoints/llc_check", exist_ok=True)
        for v in [x for x in (0.0, 0.10, 0.20) if x in traj]:
            print(f"\n[영상] vx={v:.2f} 명령 (카메라는 로봇 추적)")
            try:
                frames = render(env["mj_model"], traj[v], fps=fps, track=True)
                _show(frames, fps,
                      save=f"checkpoints/llc_check/deadzone_vx{v:.2f}.mp4",
                      title=f"vx cmd={v:.2f} -> 실측 {out[v][0]:+.3f} m/s")
            except Exception as e:
                print(f"       ⚠️ 렌더 실패({type(e).__name__}: {e})")
                print("          Colab이면 mujoco import **전에** MUJOCO_GL=egl")
                break
    return out


def _candidate_pairs(mj_model) -> int:
    """MJX가 매 스텝 검사할 geom 쌍 수 (contype/conaffinity 비트마스크 기준)."""
    ct, ca = mj_model.geom_contype, mj_model.geom_conaffinity
    act = np.flatnonzero((ct != 0) | (ca != 0))
    return sum(1 for i in act for j in act
               if i < j and mj_model.geom_bodyid[i] != mj_model.geom_bodyid[j]
               and ((ct[i] & ca[j]) or (ct[j] & ca[i])))


def sweep_video(names=None, seconds=4.0, fps=25, out_dir="checkpoints/llc_check",
                env=None, show=None, video=True, keep_frames=False):
    """P0.5 게이트: 명령 차원별 극값 영상을 만들고 요약 지표를 낸다.

    Args:
        names: `SWEEPS`의 키 이름 하나 또는 목록. 생략하면 전부.
        show: 노트북 인라인 표시. `None`이면 **노트북에서 자동으로 켜진다**.
        video: `False`면 렌더링을 건너뛰고 수치 지표만 낸다 — 훨씬 빠르다.
        keep_frames: 반환 dict에 프레임 배열을 담는다. 기본 `False` —
            노트북이 배열을 통째로 찍어버리는 것을 막기 위해서다(영상은 이미 표시됨).
    """
    env = env or build()
    mj_model = env["mj_model"]
    if show is None:
        show = in_notebook()
    if isinstance(names, str):
        names = [names]
    if video:
        os.makedirs(out_dir, exist_ok=True)

    summary = {}
    for name in (names or list(SWEEPS)):
        desc, fn = SWEEPS[name]
        n = int(seconds / P.POLICY_DT)
        cmds = jnp.stack([fn(i * P.POLICY_DT) for i in range(n)])
        out = rollout(env, cmds, seconds=seconds)

        jm = jump_metrics(out["qpos"], mj_model)
        rec = dict(desc=desc, mean_vx=out["mean_vx"], mean_vy=out["mean_vy"],
                   min_z=float(np.nanmin(out["qpos"][:, 2])), fell=out["fell"],
                   path=None, **jm)
        print(f"  {name:16s} {desc:26s} vx={out['mean_vx']:+.3f} fell={str(out['fell']):5s}"
              f" 체공={jm.get('flight_frac', 0):5.1%}"
              f" 상승={jm.get('apex_rise', 0):+.3f}m"
              f" 체공이동={jm.get('air_dx', 0):+.3f}m")

        if video:
            try:
                frames = render(mj_model, out["qpos"], fps=fps)
                rec["path"] = f"{out_dir}/{name}.mp4"
                if keep_frames:
                    rec["frames"] = frames
                _show(frames, fps, save=rec["path"], title=f"{name}: {desc}", show=show)
            except RuntimeError as e:
                # 렌더링이 안 된다고 수치 지표까지 버릴 이유는 없다. 한 번만 알리고 계속.
                print(f"\n{e}\n")
                print("  → 렌더링을 끄고 수치 지표만 계속합니다.\n")
                video = False
        summary[name] = rec

    if video and not show:
        print(f"\n영상은 '{out_dir}/' 에 저장했습니다. 인라인으로 보려면 show=True.")
    print("\n★ 판정은 '체공률'이 아니라 '상승'과 '체공이동'으로 한다.")
    print("  다리를 접기만 해도 발은 뜬다 — 체공률만으로는 점프인지 알 수 없다.")
    print("  2026-07-28 실측: pronk 체공 20~30%인데 **상승 +0.001 m**(몸통이 안 뜸),")
    print("  체공이동 최대 0.13 m. → 이 체크포인트로는 틈 0.5~0.7 m 도약 불가.")
    return summary
