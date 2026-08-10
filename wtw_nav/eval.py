"""학습된 HLC 정책의 평가 (P5).

    from wtw_nav import eval as E
    pol = E.load("checkpoints/hlc_p4_maze.pkl")
    E.report(pol)                       # 학습 미로 + 다른 seed + 큰 미로

★ **평가의 목적은 "잘 되나"가 아니라 "무엇을 배웠나"다.** 미로 하나는 MJX 모델
하나이므로(`terrain.maze` 주석) 학습 seed에서 잰 성공률은 **레이아웃 암기와 구분되지
않는다.** 그래서 이 모듈의 기본 동작은 항상 **학습에 쓰지 않은 미로**를 포함한다.

주의 — 관측 37D에는 절대좌표도 맵 크기도 없다(`envs.nav_env._obs`). 따라서 정책은
원리적으로 **크기·배치에 무관**해야 한다. n을 키웠을 때 성공률이 떨어지면 그것은
"일반화 실패"이기 전에 **유도벡터 `d_norm`을 먼저 의심할 자리**다 — 관측 중 유일하게
`_course_len`으로 정규화된 항이라 코스 길이에 따라 변화율이 달라진다
(`hlc.guidance` / `envs.nav_env._guidance`).

주의 — 실패를 **원인별로** 세지 않으면 대응을 못 정한다. 낙상·교착·시간초과는 각각
물리·유도·예산 문제이고 고치는 곳이 전부 다르다. `rollouts()`가 이를 분류한다.
"""

from __future__ import annotations

import pickle

import jax
import numpy as np

from wtw_nav.configs import HLCConfig, maze_config
from wtw_nav.envs.nav_env import NavEnv


def load(path: str = "checkpoints/hlc_p4_maze.pkl"):
    """체크포인트 -> `(정책함수, 학습설정, 로그, 스텝)`.

    주의 — 저장된 `params` 3-튜플은 `(normalizer, policy, value)`이고, brax의
    `make_inference_fn`에는 **앞의 둘만** 넘긴다. 셋을 그대로 넘기면 조용히
    틀린 정책이 나오는 것이 아니라 구조 오류로 죽는다 — 다행이다.

    주의 — 네트워크 크기는 저장된 `cfg`에서 읽는다. `default_config()`를 쓰면
    학습 때 값을 바꾼 실행에서 형상 불일치가 난다.
    """
    import functools

    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    with open(path, "rb") as f:
        ck = pickle.load(f)
    cfg: HLCConfig = ck["cfg"]

    env = NavEnv(cfg)
    net = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=cfg.ppo.policy_hidden,
        value_hidden_layer_sizes=cfg.ppo.value_hidden)
    make_policy = ppo_networks.make_inference_fn(net)
    # deterministic=True — 평가에서 표본 잡음을 섞으면 정책 능력이 아니라
    # 탐색 분산을 재게 된다.
    policy = make_policy((ck["params"][0], ck["params"][1]), deterministic=True)

    print(f"체크포인트 {path}: {ck['step']:,} 스텝, "
          f"학습 미로 {cfg.maze.n}×{cfg.maze.n} seed {cfg.maze.seed}")
    return functools.partial(_apply, jax.jit(policy)), cfg, ck.get("log"), ck["step"]


def _apply(policy_jit, obs, rng):
    return policy_jit(obs, rng)[0]


#: 실패 원인. **고치는 곳이 서로 다르므로 뭉뚱그리면 안 된다.**
#:   fell    물리·명령 문제      (LLC 한계 / 명령 범위)
#:   stuck   유도·인지 문제      (흐름장 / 라이다 / 코너)
#:   timeout 예산 문제           (`maze_config(corner_slack=...)`)
CAUSES = ("reached", "fell", "stuck", "timeout")


def rollouts(policy, cfg: HLCConfig, n: int = 20, env: NavEnv | None = None,
             seed0: int = 0) -> dict:
    """정책을 `n`개 시드로 돌려 **원인별로** 집계한다.

    주의 — env를 재사용하면 JIT 재컴파일이 없다. 미로가 바뀌면 새 env가 필요하다
    (미로 하나 = MJX 모델 하나).
    """
    env = env or NavEnv(cfg)
    step = jax.jit(env.step)
    reset = jax.jit(env.reset)

    rows = []
    for s in range(seed0, seed0 + n):
        rng = jax.random.PRNGKey(s)
        st = reset(rng)
        qpos, k = [], 0
        while k < env._max_steps:
            rng, sub = jax.random.split(rng)
            st = step(st, policy(st.obs, sub))
            qpos.append(np.asarray(st.pipeline_state.qpos))
            k += 1
            if float(st.done) > 0:
                break
        q = np.asarray(qpos)
        dist = float(st.info["dist"])
        reached = dist < cfg.course.goal_radius
        # 주의 — `done`은 낙상·교착에만 선다(타임아웃은 brax truncation이라 제외 —
        #    `nav_env.step` 주석). 그래서 done 여부로 셋을 가를 수 있다.
        if reached:
            cause = "reached"
        elif float(st.done) > 0:
            cause = "fell" if q[:, 2].min() < cfg.term.min_height else "stuck"
        else:
            cause = "timeout"
        rows.append(dict(seed=s, cause=cause, dist=dist, steps=k,
                         t=k * env._hlc_dt, min_z=float(q[:, 2].min()), qpos=q))
    return _summarize(rows, env)


def _summarize(rows, env) -> dict:
    n = len(rows)
    cnt = {c: sum(r["cause"] == c for r in rows) for c in CAUSES}
    ok = [r for r in rows if r["cause"] == "reached"]
    print(f"  {'원인':<8s} {'수':>3s} {'비율':>6s}")
    for c in CAUSES:
        print(f"  {c:<8s} {cnt[c]:>3d} {cnt[c]/n:>6.0%}")
    if ok:
        ts = np.array([r["t"] for r in ok])
        print(f"  도달 시간 {ts.mean():.1f} ± {ts.std():.1f} s "
              f"(최소 {ts.min():.1f}, 최대 {ts.max():.1f})")
    return dict(rate=cnt["reached"] / n, counts=cnt, n=n, rows=rows, env=env)


#: 평가 세트. **학습 미로(n=5, seed=0)를 첫 줄에 두고 나머지를 그 아래에서 읽는다.**
#: 첫 줄만 높고 나머지가 낮으면 그것이 곧 **레이아웃 암기**의 증거다.
SUITE = (("학습 미로", 5, 0), ("같은 크기·다른 배치", 5, 7),
         ("같은 크기·다른 배치", 5, 13), ("더 큰 미로", 8, 0), ("더 큰 미로", 11, 0))


def report(policy, suite=SUITE, n: int = 20, video: bool = True) -> dict:
    """★ 평가 본체 — 학습 미로 + 미학습 배치 + 더 큰 미로.

    주의 — 미로마다 MJX 모델이 새로 컴파일된다(수십 초). 다섯 줄이면 수 분이다.
    """
    out = {}
    for label, n_cell, seed in suite:
        print(f"\n=== {label}: {n_cell}×{n_cell} seed {seed} ===", flush=True)
        cfg = maze_config(n=n_cell, seed=seed)
        res = rollouts(policy, cfg, n=n)
        out[(n_cell, seed)] = res
        if video:
            save = f"eval_maze{n_cell}s{seed}.mp4"
            preview(res, save=save)

    base = out.get((5, 0))
    print(f"\n{'미로':<16s} {'도달률':>7s} {'학습 미로 대비':>14s}")
    print("-" * 40)
    for (n_cell, seed), r in out.items():
        rel = f"{r['rate'] - base['rate']:+.0%}" if base else "—"
        print(f"{f'{n_cell}x{n_cell} seed{seed}':<16s} {r['rate']:>7.0%} {rel:>14s}")
    if base:
        others = [r["rate"] for k, r in out.items() if k != (5, 0)]
        if others and base["rate"] - max(others) > 0.2:
            print("\n  주의 — **학습 미로에서만 잘합니다 — 레이아웃 암기입니다.**")
            print("     관측에 절대좌표가 없으므로 암기 경로는 라이다 패턴뿐입니다.")
            print("     대응: 학습 시 미로 seed를 배치로 섞거나(모델 여러 개), "
                  "`init_jitter`를 키우십시오.")
        elif others and min(others) > 0.6:
            print("\n  ✓ 미학습 배치·크기에서도 유지됩니다 — 크기 무관 설계가 "
                  "의도대로 작동합니다.")
    return out


def preview(res: dict, seed: int | None = None, save: str | None = None,
            fps: int = 10, distance: float = 8.0):
    """롤아웃 하나를 영상으로. 기본은 **실패한 것 중 첫 번째**를 고른다.

    ★ 성공 영상은 볼 것이 없다. 진단은 실패에서 나온다 — 어디서 막혔는지는
    숫자로 안 나오고, 지형 렌더링 결함도 영상으로만 드러났다(`llc.check.render`).
    """
    from wtw_nav.llc import check

    rows = res["rows"]
    pick = next((r for r in rows if r["cause"] != "reached"), rows[0]) \
        if seed is None else next(r for r in rows if r["seed"] == seed)
    env = res["env"]
    mm = env.maze_meta
    print(f"  영상: seed {pick['seed']}  {pick['cause']}  "
          f"{pick['t']:.1f} s  남은거리 {pick['dist']:.2f}")
    frames = check.render(env.mj_model, pick["qpos"], fps=fps, every=1,
                          distance=distance)
    tag = f"maze{mm['n']}x{mm['n']}s{mm['seed']}" if mm else "flat"
    check._show(frames, fps, save, title=f"{tag} seed{pick['seed']} "
                                         f"{pick['cause']}", show=True)


def curve(log, every: int = 1) -> None:
    """학습 로그를 표로. **어디서 수렴했는지**가 다음 실행의 예산을 정한다.

    P4 실측: 2.46M에서 96%, 나머지 3.4M(140분)이 산 것은 3%p뿐이었다.
    """
    if not log:
        print("  로그가 없습니다 (옛 체크포인트).")
        return
    # 주의 — `monitor.history`는 dict가 아니라 **`(step, metrics)` 튜플 목록**이다.
    print(f"{'steps':>10s} {'reached':>8s} {'dist':>7s} {'fell':>6s} {'len':>7s}")
    print("-" * 42)
    best, best_step = 0.0, 0
    for i, (step, m) in enumerate(log):
        rc = m.get("eval/episode_reached", 0.0)
        if rc > best:
            best, best_step = rc, step
        if i % every:
            continue
        print(f"{step:>10,d} {rc:>8.3f} {m.get('eval/episode_dist', 0):>7.3f} "
              f"{m.get('eval/episode_fell', 0):>6.3f} "
              f"{m.get('eval/avg_episode_length', 0):>7.1f}")

    # ★ **다음 실행의 예산은 여기서 정한다.** 최고점의 95%에 처음 닿은 지점 이후는
    #   대부분 낭비다 — P4에서 2.46M(96%) 이후 3.4M(140분)이 산 것은 3%p뿐이었다.
    knee = next((s for s, m in log
                 if m.get("eval/episode_reached", 0.0) >= 0.95 * best), None)
    if knee is not None and log:
        last = log[-1][0]
        print(f"\n  최고 {best:.1%} @ {best_step:,}   "
              f"95% 도달 @ **{knee:,}** ({knee / max(last, 1):.0%} 지점)")
        if knee < last * 0.6:
            print(f"  ⇒ 다음 실행은 num_timesteps를 **{knee * 1.3:,.0f}** 근처로 "
                  f"잡으십시오. 나머지는 {1 - best:.1%}를 못 줄이는 데 씁니다.")
