"""처리량 병목 분해 — 학습이 느릴 때 **추측하지 말고 이걸 먼저 돌린다**.

    python -m wtw_nav.bench            # 계층별 분해
    python -m wtw_nav.bench solver     # + 솔버 설정별 비용 비교

측정을 계층으로 쪼개는 이유는, "느리다"의 원인이 층마다 대처가 다르기 때문이다.

    L0  mjx.step 1회          <- 솔버 설정(iterations/ls_iterations)이 지배
    L1  LLC 1스텝 = 4×L0      <- + 정책 MLP 순전파
    L2  HLC 1스텝 = 5×L1      <- + 명령 필터·보상·종료 판정

⚠️ **HLC 1스텝은 물리 스텝 20회다.** `PPOConfig.num_timesteps`는 HLC 스텝을 세므로
   30M은 곧 6억 번의 `mjx.step`이다. 벽시계 시간을 가늠할 땐 항상 ×20 하십시오.
"""

from __future__ import annotations

import sys
import time

import jax
import jax.numpy as jnp

from wtw_nav.configs import HLCConfig, default_config
from wtw_nav.llc import policy as P

#: HLC 1스텝당 물리 스텝 수 = LLC decimation(4) × HLC decimation(5)
PHYS_PER_LLC = P.DECIMATION


def device_report() -> str:
    """어떤 하드웨어에서 도는지. GPU가 아니면 그 사실이 먼저다."""
    backend = jax.default_backend()
    devs = jax.devices()
    print(f"백엔드    : {backend}")
    print(f"디바이스  : {len(devs)}개 — {devs[0].device_kind if devs else '?'}")
    if backend == "cpu":
        print("\n  *** JAX가 CPU에서 돌고 있습니다. ***")
        print("      Colab: 런타임 > 런타임 유형 변경 > 하드웨어 가속기 > GPU 로 바꾸고")
        print("      런타임을 **재시작**한 뒤 처음부터 다시 실행하십시오.")
        print("      (`pip list | grep -i jax` 로 jaxlib가 cuda 빌드인지도 확인)")
    return backend


def _time(fn, *args, n: int = 10):
    """(컴파일+첫실행 시간, 정상상태 1회 시간)을 분리해서 반환."""
    t = time.time()
    out = jax.block_until_ready(fn(*args))
    t_compile = time.time() - t

    t = time.time()
    for _ in range(n):
        out = fn(*args)
    jax.block_until_ready(out)
    return t_compile, (time.time() - t) / n


def breakdown(cfg: HLCConfig | None = None, num_envs: int = 2048, n: int = 10) -> dict:
    """L0/L1/L2를 각각 재서 어느 층이 시간을 쓰는지 드러낸다."""
    from mujoco import mjx

    from wtw_nav.envs.nav_env import NavEnv

    cfg = cfg or default_config()
    phys_per_hlc = PHYS_PER_LLC * cfg.decimation

    print("\nenv 생성 중 (LLC 체크포인트 + MJX 모델)...", flush=True)
    t = time.time()
    env = NavEnv(cfg)
    print(f"  {time.time()-t:.1f}s", flush=True)
    print(f"솔버      : iterations={env.mj_model.opt.iterations} "
          f"ls_iterations={env.mj_model.opt.ls_iterations}")
    print(f"병렬 env  : {num_envs}")
    print(f"HLC 1스텝 = LLC {cfg.decimation} × 물리 {PHYS_PER_LLC} "
          f"= **물리 {phys_per_hlc}스텝**")

    reset = jax.jit(jax.vmap(env.reset))
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    t = time.time()
    state = jax.block_until_ready(reset(keys))
    print(f"\nreset (컴파일+실행) : {time.time()-t:6.1f}s")

    rows = []

    # ---- L0: 물리 1스텝만 ----
    f0 = jax.jit(jax.vmap(lambda d: mjx.step(env.mjx_model, d)))
    c0, s0 = _time(f0, state.pipeline_state, n=n)
    rows.append(("L0 mjx.step", 1, c0, s0))

    # ---- L1: LLC 1스텝 (물리 4 + 정책 MLP) ----
    cmd = jnp.broadcast_to(state.info["prev_cmd"][0], (num_envs, 15))
    f1 = jax.jit(jax.vmap(env._llc_step))
    args1 = (state.pipeline_state, state.info["obs_history"], state.info["gait"],
             state.info["last_actions"], state.info["last_last_actions"], cmd)
    c1, s1 = _time(f1, *args1, n=n)
    rows.append(("L1 LLC step", PHYS_PER_LLC, c1, s1))

    # ---- L2: HLC 1스텝 (LLC 5 + 보상/종료) ----
    act = jnp.zeros((num_envs, env.action_size))
    f2 = jax.jit(jax.vmap(env.step))
    c2, s2 = _time(f2, state, act, n=n)
    rows.append(("L2 HLC step", phys_per_hlc, c2, s2))

    print(f"\n{'층':<14s} {'물리스텝':>7s} {'컴파일':>9s} {'1회':>9s} "
          f"{'물리steps/s':>13s} {'예상비':>7s}")
    print("-" * 68)
    base = None
    for name, phys, tc, ts in rows:
        pps = num_envs * phys / ts
        if base is None:
            base = pps
        print(f"{name:<14s} {phys:>7d} {tc:>8.1f}s {ts*1e3:>8.1f}ms "
              f"{pps:>13,.0f} {pps/base:>6.2f}x")

    sps = num_envs / rows[-1][3]                      # HLC env-steps / s
    total = cfg.ppo.num_timesteps
    t_compile = sum(r[2] for r in rows) + 1e-9
    print(f"\nHLC 처리량 : {sps:,.0f} steps/s")
    print(f"컴파일 합계 : {t_compile/60:.1f}분  (한 번만 냅니다)")
    # ★★ 여기서 `total/sps`를 그대로 내놓으면 **항상 과소평가**한다 (2026-08-10 교정).
    #    이 함수가 재는 것은 `jax.vmap(env.step)` 하나뿐이고, 실제 학습에는 그 위에
    #    다음이 얹힌다:
    #      · PPO 갱신  — num_updates_per_batch × num_minibatches 회 그래디언트
    #      · brax eval — num_evals 회, 각각 episode_length 스텝을 따로 굴린다.
    #                    `PPOConfig.num_evals` 주석의 전례: 101분 중 18분이 eval
    #      · AutoReset/Episode 래퍼, 관측 정규화 — bench의 raw env.step에는 없다
    #    P4 미로 실측(2026-08-10): 예측 719 -> **실제 414 steps/s**, 즉 ×0.58.
    #    낙관적인 수를 보고 21시간짜리를 건 전례가 있으므로 **보수적으로 낸다.**
    eff = 0.58
    print(f"\n{total:,} 스텝 예상")
    print(f"  raw(env.step만)      : {total/sps/60:>5.0f}분")
    print(f"  **실제 예상**        : **{total/(sps*eff)/60:>5.0f}분** "
          f"+ 컴파일 {t_compile/60:.1f}분   (실측 계수 ×{eff})")
    print(f"  ※ 계수 근거: P4 미로 실측 719 -> 414 steps/s. PPO 갱신 ~30% + "
          f"eval ~18%.")
    print(f"  ※ 1시간에 끝내려면 num_timesteps <= {sps*eff*3600/1e6:.1f}M")
    return {"backend": jax.default_backend(), "sps": sps, "rows": rows,
            "t_compile": t_compile, "eta_min": total / (sps * eff) / 60,
            "eta_raw_min": total / sps / 60}


def solver_sweep(num_envs: int = 512, n: int = 10) -> None:
    """솔버 설정이 L0 비용을 얼마나 좌우하는지. MJX는 반복을 그래프에 펼친다.

    ⚠️ 여기서 싸다고 바로 채택하면 안 된다 — 4/8은 GPU에서 **발산**했다
    (`docs/03_results.md` §1.3). 비용을 알고 나서 정확도와 저울질할 것.
    """
    from mujoco import mjx

    print(f"\n솔버 설정별 L0 비용 ({num_envs} envs)")
    print(f"{'iters/ls':>10s} {'컴파일':>9s} {'1회':>9s} {'물리steps/s':>13s}")
    print("-" * 45)
    cfg = default_config()
    for it, ls in ((4, 8), (6, 12), (8, 16), (8, 8)):
        mj_model, mj_data, mjx_model = P.create_env(cfg.xml, iterations=it,
                                                    ls_iterations=ls)
        jidx = P._build_joint_index(mj_model)
        d0 = P.reset_data(mj_model, mj_data, mjx_model, jidx)
        d = jax.vmap(lambda _: d0)(jnp.arange(num_envs))
        f = jax.jit(jax.vmap(lambda x: mjx.step(mjx_model, x)))
        tc, ts = _time(f, d, n=n)
        print(f"{it:>5d}/{ls:<4d} {tc:>8.1f}s {ts*1e3:>8.1f}ms "
              f"{num_envs/ts:>13,.0f}")


def collision_sweep(num_envs: int = 512, n: int = 10) -> None:
    """충돌 필터별 L0 비용. **솔버보다 훨씬 큰 레버**다.

    실측(L4)에서 솔버를 4/8로 낮춰도 11%밖에 안 빨라졌다 —
    `mjx.step` 시간의 대부분은 솔버가 아니라 충돌 검사에 있다.
    """
    from mujoco import mjx

    cfg = default_config()
    print(f"\n충돌 필터별 L0 비용 ({num_envs} envs)")
    print(f"{'mode':>8s} {'충돌geom':>8s} {'컴파일':>9s} {'1회':>9s} "
          f"{'물리steps/s':>13s} {'배속':>6s}")
    print("-" * 60)
    base = None
    for mode in ("full", "world", "feet"):
        mj_model, mj_data, mjx_model = P.create_env(cfg.xml, collision=mode)
        jidx = P._build_joint_index(mj_model)
        n_geom = int(((mj_model.geom_contype != 0) |
                      (mj_model.geom_conaffinity != 0)).sum())
        d0 = P.reset_data(mj_model, mj_data, mjx_model, jidx)
        d = jax.vmap(lambda _: d0)(jnp.arange(num_envs))
        f = jax.jit(jax.vmap(lambda x: mjx.step(mjx_model, x)))
        tc, ts = _time(f, d, n=n)
        pps = num_envs / ts
        base = base or pps
        print(f"{mode:>8s} {n_geom:>8d} {tc:>8.1f}s {ts*1e3:>8.1f}ms "
              f"{pps:>13,.0f} {pps/base:>5.1f}x")
    print("\n  ⚠️ 빠르다고 바로 채택하지 말 것 — WTW는 `self_collisions=0`(자기충돌 켬)으로")
    print("     학습했습니다. `python -m wtw_nav.llc.check`로 추종 성능을 확인한 뒤 정하십시오.")


def main() -> int:
    backend = device_report()
    res = breakdown()
    if "solver" in sys.argv[1:]:
        solver_sweep()
    if "collision" in sys.argv[1:]:
        collision_sweep()

    ok = backend != "cpu" and res["eta_min"] < 240
    if not ok:
        print("\n*** 이 설정으로는 학습을 끝내기 어렵습니다. ***")
        if backend == "cpu":
            print("    GPU 런타임으로 전환하십시오.")
        else:
            print(f"    `PPOConfig.num_timesteps`를 "
                  f"{res['sps']*3600/1e6:.1f}M 이하로 줄이십시오 (=1시간).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
