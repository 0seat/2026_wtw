"""P1 게이트: NavEnv가 학습에 쓸 수 있는 상태인지 검증한다.

    conda run -n mujoco_env python -m wtw_nav.envs.test_nav_env

가장 중요한 것은 **reset/step이 jit·vmap 안에서 도는가**다. 이게 안 되면 병렬 학습이
불가능하고, 그 사실을 학습을 돌려보고 나서야 알게 된다.
"""

from __future__ import annotations

import dataclasses
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from wtw_nav.configs import default_config
from wtw_nav.envs.nav_env import NavEnv
from wtw_nav.hlc import command_filter as cf
from wtw_nav.hlc import guidance
from wtw_nav.llc import policy as P

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """MJX 컴파일이 길어 진행 상황이 보여야 하므로 항상 flush 한다."""
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL':4s}  {name}" + (f"   {detail}" if detail else ""),
          flush=True)


def main() -> int:
    cfg = default_config()

    print("=" * 64)
    print("P1 게이트 — NavEnv")
    print("=" * 64)

    # -------------------------------------------------- 순수 함수 (env 없이)
    print("\n[1] command_filter")
    a0 = jnp.zeros(8)
    neutral = cf.neutral_command(cfg.action)
    cmd = cf.action_to_command(a0, neutral, cfg.action)
    check("15D 명령", cmd.shape == (15,), f"shape={cmd.shape}")
    # 저역필터를 거친 float32이므로 정확한 == 비교를 하면 안 된다 (실제로 여기서 한 번 틀렸다)
    check("duty=0.5 / roll=0 고정",
          abs(float(cmd[8]) - 0.5) < 1e-6 and abs(float(cmd[11])) < 1e-6,
          f"duty={float(cmd[8]):.6f} roll={float(cmd[11]):.6f}")
    check("gait = trot (0.5,0,0)",
          np.allclose(np.asarray(cmd[5:8]), [0.5, 0.0, 0.0], atol=1e-6),
          str(np.asarray(cmd[5:8])))
    check("stance_length=0.45, aux=0",
          abs(float(cmd[13]) - 0.45) < 1e-6 and abs(float(cmd[14])) < 1e-6,
          f"sl={float(cmd[13]):.6f} aux={float(cmd[14]):.6f}")

    # 마름모 제약: vx·yaw 동시 최대를 요구하면 축소돼야 한다
    a_max = jnp.array([1e3, 1e3, 0, 0, 0, 0, 0, 0], dtype=jnp.float32)
    c_max = cf.action_to_command(a_max, neutral, cfg.action)
    # 저역필터(α=0.3)를 되돌려 원 명령을 복원
    raw = (c_max - (1 - cfg.action.lowpass_alpha) * neutral) / cfg.action.lowpass_alpha
    s = abs(float(raw[0])) / cfg.action.diamond_vx + abs(float(raw[2])) / cfg.action.diamond_yaw
    check("vx-yaw L1 마름모 제약 (s<=1)", s <= 1.0 + 1e-3, f"s={s:.4f}")

    # 데드존 — 2026-07-29 실측으로 **기본 비활성**(0.0)이 됐다. 두 갈래를 다 본다.
    a_tiny = jnp.zeros(8).at[0].set(float(np.arctanh(2 * (0.05 - cfg.action.vx[0])
                                                     / (cfg.action.vx[1] - cfg.action.vx[0]) - 1)))
    # (a) 켜면 스냅한다 (되살릴 때를 대비해 경로 자체는 살아 있어야 한다)
    on = dataclasses.replace(cfg.action, deadzone=0.2, vx_bias=0.0)
    c_on = cf.action_to_command(a_tiny, jnp.zeros(15), on)
    check("데드존 ON: ‖v‖<=0.2 는 0으로 스냅", abs(float(c_on[0])) < 1e-6,
          f"vx={float(c_on[0]):.4f}")
    # (b) 기본(OFF)에서는 통과하고 편향만 빠진다. 여기서 0으로 스냅되면 **정밀 정지가
    #     불가능**해진다 — 편향을 상쇄할 vx≈-0.09가 데드존에 먹히기 때문.
    c_off = cf.action_to_command(a_tiny, jnp.zeros(15), cfg.action)
    raw_off = float(c_off[0]) / cfg.action.lowpass_alpha
    check("데드존 OFF(기본): 선형 통과 - vx_bias",
          abs(raw_off - (0.05 - cfg.action.vx_bias)) < 1e-3,
          f"vx={raw_off:+.4f} (기대 {0.05 - cfg.action.vx_bias:+.4f})")

    # action_for: 원하는 물리값 -> 액션 (수동 제어기·테스트용). 왕복이 맞아야 한다.
    # ⚠️ vx만 `vx_bias`가 빠져 나온다. 그게 의도다 — "명령 vx == 실제 vx"를 만드는
    #    보정이므로, 왕복 비교에서도 빼줘야 한다.
    want = dict(vx=1.0, yaw=-0.8, height=-0.10, footswing=0.20)
    a_want = cf.action_for(cfg.action, **want)
    c_want = cf.action_to_command(a_want, jnp.zeros(15), cfg.action)
    got = {"vx": float(c_want[0]), "yaw": float(c_want[2]),
           "height": float(c_want[3]), "footswing": float(c_want[9])}
    # 저역필터(prev=0, α=0.3)를 되돌려 비교
    got = {k: v / cfg.action.lowpass_alpha for k, v in got.items()}
    expect = dict(want, vx=want["vx"] - cfg.action.vx_bias)
    err = max(abs(got[k] - expect[k]) for k in want)
    check("action_for 왕복 (원하는 값 -> 액션 -> 명령)", err < 1e-3,
          f"max|err|={err:.2e}  {({k: round(v, 3) for k, v in got.items()})}")

    print("\n[2] guidance")
    q_id = jnp.array([1.0, 0.0, 0.0, 0.0])
    g = guidance.guidance_to_point(jnp.zeros(3), q_id, jnp.array([20.0, 0.0]), 20.0)
    check("정면 목표 -> (1,0,1)", np.allclose(np.asarray(g), [1, 0, 1], atol=1e-5),
          str(np.asarray(g).round(4)))
    yaw90 = jnp.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    g2 = guidance.guidance_to_point(jnp.zeros(3), yaw90, jnp.array([20.0, 0.0]), 20.0)
    check("로봇이 +90° 돌면 목표는 우측(-sin)",
          abs(float(g2[0])) < 1e-5 and float(g2[1]) < -0.99, str(np.asarray(g2).round(4)))

    # -------------------------------------------------- env
    print("\n[3] NavEnv 생성")
    t0 = time.time()
    env = NavEnv(cfg)
    check("생성 성공", True, f"{time.time()-t0:.1f}s")
    check("action_size == 8", env.action_size == 8, str(env.action_size))
    check("observation_size == 37", env.observation_size == 37, str(env.observation_size))

    print("\n[4] reset (jit)")
    t0 = time.time()
    reset = jax.jit(env.reset)
    s0 = reset(jax.random.PRNGKey(0))
    s0.obs.block_until_ready()
    check("jit reset 동작", True, f"{time.time()-t0:.1f}s (컴파일 포함)")
    check("obs 차원 37 (proprio10+유도3+명령8+라이다16)", s0.obs.shape == (37,), f"{s0.obs.shape}")
    check("obs 유한", bool(jnp.all(jnp.isfinite(s0.obs))))
    check("리셋 높이 z=0.34", abs(float(s0.pipeline_state.qpos[2]) - 0.34) < 1e-5,
          f"z={float(s0.pipeline_state.qpos[2]):.4f}")
    check("초기 거리 ≈ 코스 길이",
          abs(float(s0.info["dist"]) - cfg.course.length) < 1.0,
          f"dist={float(s0.info['dist']):.3f}")

    print("\n[5] ★ vmap reset (병렬 학습 가능성)")
    t0 = time.time()
    vreset = jax.jit(jax.vmap(env.reset))
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    sv = vreset(keys)
    sv.obs.block_until_ready()
    check("vmap reset 동작", sv.obs.shape == (8, 37), f"{sv.obs.shape}, {time.time()-t0:.1f}s")
    lat = np.asarray(sv.pipeline_state.qpos[:, 1])
    check("env마다 초기 횡오프셋이 다름 (랜덤화 작동)", float(np.std(lat)) > 1e-3,
          f"std={np.std(lat):.4f}")
    # ★ 출발 x 랜덤화가 실제로 걸렸는지. 이게 죽으면 지형 사다리 측정은 "지형 능력"이
    #   아니라 "고정 배치 암기"를 재게 된다 (`configs.CourseConfig.init_x`).
    sx = np.asarray(sv.pipeline_state.qpos[:, 0])
    check("env마다 출발 x가 다름 (위상 암기 차단)", float(np.std(sx)) > 1e-3,
          f"std={np.std(sx):.4f}, x=[{sx.min():+.2f}, {sx.max():+.2f}]")
    check("출발 x가 시작 발판(-2.0) 안", float(sx.min()) > -1.7, f"min={sx.min():+.3f}")

    print("\n[6] step (jit) — 컴파일 시간 확인")
    t0 = time.time()
    step = jax.jit(env.step)
    s1 = step(s0, jnp.zeros(8))
    s1.obs.block_until_ready()
    check("jit step 동작", True, f"{time.time()-t0:.1f}s (컴파일 포함)")
    check("obs 유한", bool(jnp.all(jnp.isfinite(s1.obs))))
    check("reward 유한", bool(jnp.isfinite(s1.reward)), f"r={float(s1.reward):+.4f}")
    check("done 스칼라 0/1", s1.done.shape == () and float(s1.done) in (0.0, 1.0))

    print("\n[7] 짧은 롤아웃 (5 s = 50 HLC 스텝, 정지 명령)")
    t0 = time.time()

    def body(carry, _):
        st = step(carry, jnp.zeros(8))
        return st, (st.reward, st.pipeline_state.qpos[2], st.done)

    sN, (rew, zs, dones) = jax.lax.scan(body, s0, None, length=50)
    zs = np.asarray(zs)
    check("50스텝 완주, qpos 유한", bool(np.all(np.isfinite(zs))),
          f"{time.time()-t0:.1f}s, z=[{zs.min():.3f}, {zs.max():.3f}]")
    check("낙상하지 않음", zs.min() > cfg.term.min_height, f"min_z={zs.min():.3f}")

    print("\n[8] 전진 명령 롤아웃 (vx 최대)")
    a_fwd = jnp.zeros(8).at[0].set(3.0)          # tanh 포화 -> vx 상한

    def body_f(carry, _):
        st = step(carry, a_fwd)
        return st, (st.pipeline_state.qpos[0], st.pipeline_state.qpos[2], st.done)

    sF, (xs, zf, df) = jax.lax.scan(body_f, s0, None, length=50)
    xs, zf = np.asarray(xs), np.asarray(zf)
    # 출발 x가 랜덤이므로 절대 x가 아니라 **변위**로 본다.
    x0 = float(s0.pipeline_state.qpos[0])
    check("전진함 (5 s 동안 +1 m 이상)", xs[-1] - x0 > 1.0,
          f"{x0:+.2f} -> {xs[-1]:.3f} m (Δ{xs[-1] - x0:+.3f})")
    check("전진 중 낙상 없음", zf.min() > cfg.term.min_height, f"min_z={zf.min():.3f}")

    print("\n[9] vmap step (배치 학습 경로)")
    t0 = time.time()
    vstep = jax.jit(jax.vmap(env.step))
    sv1 = vstep(sv, jnp.zeros((8, 8)))
    sv1.obs.block_until_ready()
    check("vmap step 동작", sv1.obs.shape == (8, 37),
          f"{sv1.obs.shape}, {time.time()-t0:.1f}s")
    check("배치 reward 유한", bool(jnp.all(jnp.isfinite(sv1.reward))))

    print("\n[10] brax 래퍼 호환 (학습 루프의 lax.scan 캐리 구조)")
    # brax PPO는 env state를 lax.scan 캐리로 쓴다. reset과 step의 metrics/info pytree
    # 구조가 다르면 "carry input and carry output must have the same pytree structure"로
    # 죽는다 — 실제로 metrics를 새 dict로 교체했다가 EvalWrapper가 넣는 'reward' 키를
    # 잃어 이 오류가 났다. 여기서 미리 잡는다.
    from brax.envs.wrappers import training as brax_training

    wrapped = brax_training.wrap(env, episode_length=env._max_steps, action_repeat=1)
    ws0 = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), 4))
    ws1 = jax.jit(wrapped.step)(ws0, jnp.zeros((4, 8)))
    for field in ("metrics", "info"):
        t0_ = jax.tree_util.tree_structure(getattr(ws0, field))
        t1_ = jax.tree_util.tree_structure(getattr(ws1, field))
        check(f"래퍼 적용 후 {field} pytree 구조 일치", t0_ == t1_,
              "" if t0_ == t1_ else f"\n      reset={t0_}\n      step ={t1_}")

    # EvalWrapper는 metrics에 'reward'를 끼워 넣는다 — 그 상황도 재현해 둔다
    from brax.envs.wrappers.training import EvalWrapper
    ev = EvalWrapper(wrapped)
    es0 = jax.jit(ev.reset)(jax.random.split(jax.random.PRNGKey(0), 4))
    es1 = jax.jit(ev.step)(es0, jnp.zeros((4, 8)))
    t0_ = jax.tree_util.tree_structure(es0.metrics)
    t1_ = jax.tree_util.tree_structure(es1.metrics)
    check("EvalWrapper metrics 구조 일치 ('reward' 보존)", t0_ == t1_,
          "" if t0_ == t1_ else f"\n      reset={t0_}\n      step ={t1_}")

    print("\n[11] 타임아웃이 truncation으로 처리되는가 (가치 부트스트랩)")
    # brax EpisodeWrapper: truncation = where(steps >= episode_length, 1 - state.done, 0)
    # env가 같은 스텝에 done=1을 세우면 truncation=0이 되어 brax가 시간 초과를 진짜
    # 종료로 오인하고 가치 부트스트랩을 0으로 자른다. 그러면 "아직 걷는 중"인 상태의
    # 가치가 0이 되어 오래 걷는 정책이 손해를 보고 학습이 붕괴한다.
    # 2026-07-29: 도달률 15.6%까지 갔다가 len이 300에 닿은 직후 단조 하강해 실패.
    # 실제로 300스텝을 굴리면 액션 0인 로봇이 30스텝에서 교착으로 끝나 버리므로,
    # 불변식을 직접 만든다: "마지막 스텝인데 낙상·교착·도달이 아닌" 상태를 합성해
    # env가 done을 세우지 않는지 본다. brax 공식상 done=0 이어야 truncation=1이 된다.
    s0 = jax.jit(env.reset)(jax.random.PRNGKey(0))
    info = dict(s0.info)
    info["step"] = jnp.asarray(env._max_steps - 1, jnp.float32)
    info["dist_at_window"] = info["dist"] + 100.0      # moved 크게 -> 교착 아님
    s_end = jax.jit(env.step)(s0.replace(info=info), jnp.zeros(env.action_size))
    d_end, m_end = float(s_end.done), float(s_end.metrics["dist"])
    check("타임아웃만으로는 env가 done을 세우지 않는다", d_end < 0.5,
          f"done={d_end:.0f}" + ("" if d_end < 0.5 else
          " — brax가 truncation=0으로 계산해 가치 부트스트랩을 잘라냅니다."
          " nav_env.step의 done에서 timeout을 빼십시오"))
    check("타임아웃 스텝에도 dist 지표가 기록된다", m_end > 0.0,
          f"dist={m_end:.2f}" + ("" if m_end > 0.0 else
          " — 타임아웃 에피소드의 최종 거리가 대시보드에서 0으로 집계됩니다"))

    n_fail = _results.count(False)
    print("\n" + "=" * 64)
    print(f"{len(_results)}/{len(_results)} PASS" if n_fail == 0
          else f"*** {n_fail}/{len(_results)} FAILED ***")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
