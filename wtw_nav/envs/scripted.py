"""수동(스크립트) 제어기 — **학습 전에 env가 풀 수 있는 문제인지** 확인한다.

    conda run -n mujoco_env python -m wtw_nav.envs.scripted

PPO를 돌리기 전에 이걸 먼저 통과시켜야 한다. 수동 제어기가 목표에 못 가면
성능 문제가 아니라 **보상·종료·유도벡터 중 하나가 틀린 것**이고, 그 사실을
30M 스텝을 태운 뒤에 알아내는 것은 낭비다.

제어 법칙은 아주 단순하다 — 유도 벡터만 보고:
    yaw ∝ sinφ        (목표 쪽으로 돌고)
    vx  ∝ max(cosφ,0) (목표를 향할 때만 전진)
P1은 평지에 장애물이 없으므로 이 정도면 도달해야 정상이다.
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

from wtw_nav.configs import ActionConfig, HLCConfig, default_config
from wtw_nav.envs.nav_env import NavEnv
from wtw_nav.hlc import command_filter as cf
from wtw_nav.hlc import guidance

#: obs 레이아웃에서 유도 벡터 (cosφ, sinφ, d_norm)의 위치
#: [proj_grav 3][lin_vel 3][ang_vel 3][z(AGL) 1][guidance 3][prev_cmd 8][lidar 16]
GUIDANCE_SLICE = slice(10, 13)


def _action_for_value(value, lo, hi):
    """물리값 -> (tanh 이전) 액션. `action_to_command`의 아핀 매핑을 되돌린다."""
    t = 2.0 * (value - lo) / (hi - lo) - 1.0
    return jnp.arctanh(jnp.clip(t, -0.999, 0.999))


def base_action(cfg: ActionConfig, **fixed) -> jnp.ndarray:
    """고정축의 **물리값** -> (tanh 이전) 8D 기준 액션. 지정 안 한 축은 범위 중앙.

    `vx`/`yaw`는 유도 법칙이 덮어쓰므로 넘겨도 무시된다.
    """
    return cf.action_for(cfg, **{k: v for k, v in fixed.items()
                                 if k not in ("vx", "yaw")})


#: 진로 유지기 기본 게인. `hlc.guidance.heading_hold`와 **같은 값**을 쓴다 —
#: 거기서 표류를 3.33 -> 0.22 m로 잡은 것이 검증된 조합이다.
HOLD_K_Y, HOLD_K_PSI, HOLD_PSI_MAX = 0.5, 1.5, 0.4


def guidance_controller(obs: jnp.ndarray, cfg: ActionConfig,
                        base_a: jnp.ndarray,
                        k_yaw: float = 2.0, vx_max: float = 1.0,
                        y: jnp.ndarray | float = 0.0,
                        psi: jnp.ndarray | float = 0.0,
                        hold: jnp.ndarray | float = 1.0) -> jnp.ndarray:
    """목표로 향하는 비례 제어기. jit 가능.

    요속 명령을 두 법칙의 가중합으로 낸다:

      ① **목표 방위 추종** (`hold=0`) — 유도벡터의 각 φ에 비례. 미로처럼 목표가
         옆이나 뒤에 있을 수 있는 경우에 맞다.
      ② **진로 유지** (`hold=1`) — `hlc.guidance.heading_hold`와 같은 종속 루프
         (횡오차 y -> 목표 방위 ψ_des -> 요속). 사다리처럼 목표가 y=0 정면에
         있는 코스에 맞다.

    ★ **왜 ①만으로는 안 되는가** (2026-08-06, 요철 영상에서 드러남). 코스가 21.9 m면
    y=0.7 m로 밀려도 목표 방위 오차는 4°뿐이라 보정이 거의 걸리지 않는다. 그런데
    LLC의 요속 편향(ω≈0.118 rad/s)은 **계통적**이라 같은 쪽으로만 쌓이고 y는 시간에
    대해 2차로 자란다. 결과적으로 로봇이 요철 띠(±1.0 m)를 벗어나 **옆의 평지를
    밟고 지나갔고**, 그것이 level 6.00으로 집계됐다 — 측정이 통째로 거짓이 될
    뻔했다. ②는 y를 직접 보므로 0.7 m에서 이미 ψ_des = -0.35 rad를 낸다.

    ⚠️ 이건 LLC를 봐주는 장치가 아니라 **배포 조건 그 자체**다. 실제 시스템에는
    10 Hz HLC가 요를 잡고 있고, 그것 없이 잰 값은 배포와 무관하다
    (`guidance.heading_hold` 주석). 그래서 게인도 거기와 같은 값을 쓴다.

    ⚠️ `hold`는 **traced 스칼라**다 (파이썬 bool이 아니다). 상수로 박으면 값마다
    재컴파일된다 — `base_a`와 같은 이유(`_episode_fn` 주석).

    ⚠️ `yaw ∝ sinφ`로 하면 **목표를 정확히 등졌을 때(φ=180°) sinφ=0이라 회전하지 않는**
    잘못된 평형점이 생긴다. 각도 자체(`atan2`)를 쓰면 그 지점에서 최대 회전이 나온다.

    Args:
        y: 현재 횡위치 (m, 월드). `hold>0`일 때만 쓰인다.
        psi: 현재 방위 (rad, 월드).
        hold: 0=목표 방위 추종, 1=진로 유지. 중간값은 선형 혼합.
    """
    cos_phi, sin_phi = obs[10], obs[11]
    phi = jnp.arctan2(sin_phi, cos_phi)          # [-π, π]

    yaw_bearing = k_yaw * phi
    psi_des = jnp.clip(-HOLD_K_Y * y, -HOLD_PSI_MAX, HOLD_PSI_MAX)
    yaw_hold = HOLD_K_PSI * (psi_des - psi)

    yaw_cmd = jnp.clip(hold * yaw_hold + (1.0 - hold) * yaw_bearing,
                       cfg.yaw[0], cfg.yaw[1])
    # 목표를 등지고 있을 때 전진하면 멀어진다 -> cosφ가 양수일 때만
    vx_cmd = jnp.clip(vx_max * jnp.maximum(cos_phi, 0.0), cfg.vx[0], cfg.vx[1])

    a = base_a.at[0].set(_action_for_value(vx_cmd, *cfg.vx))
    return a.at[1].set(_action_for_value(yaw_cmd, *cfg.yaw))


#: ★ 지형별 env 캐시. **모듈 안에 둔다** (2026-08-07).
#:
#: MJX 컴파일이 수 분이라 캐시는 필수인데, 노트북 전역(`ENVS = {}`)에 두면
#: `dev.reload_wtw()`가 지우지 못한다 — `reload_wtw`는 `sys.modules`만 비우기
#: 때문이다. 그러면 코드를 고치고 리로드해도 **옛 클래스로 만든 env가 그대로
#: 반환되어**, 고친 내용이 반영되지 않은 결과를 새 결과로 읽게 된다.
#: 모듈 전역에 두면 캐시가 코드와 **같은 수명**을 갖는다 — 수동 `clear()`가 필요 없다.
_ENV_CACHE: dict[tuple, NavEnv] = {}


def terrain_env(kind: str, values=None, timeout_s: float = 60.0,
                init_x: float = 1.2, reach: float = 110.0,
                x0: float | None = None) -> NavEnv:
    """지형 사다리 env 하나. 같은 인자면 재사용한다 (JIT 재컴파일 없음).

        env = scripted.terrain_env("slope")

    `values=None`이면 `TerrainConfig.PRESETS[kind]`.

    Args:
        x0: 첫 장애물까지 조주 거리. ★ **턱 실험의 위상 손잡이다**
            (`phase_sweep` 참조). `None`이면 설정 기본값(3.0).
            ⚠️ 값마다 **새 MJX 모델 = 새 컴파일**이다. 위상 8점이면 8회 컴파일이
            드는데, 이건 피할 수 없다 — 지형 geom 위치가 바뀌기 때문이다.
    """
    import dataclasses as dc

    vals = tuple(values) if values is not None else None
    key = (kind, vals, timeout_s, init_x, reach, x0)
    if key in _ENV_CACHE:
        return _ENV_CACHE[key]

    b = default_config()
    cfg = dc.replace(
        b,
        terrain=dc.replace(b.terrain, kind=kind,
                           values=vals or b.terrain.PRESETS[kind],
                           **({} if x0 is None else {"x0": float(x0)})),
        term=dc.replace(b.term, timeout_s=timeout_s),
        course=dc.replace(b.course, init_x=init_x),
        reward=dc.replace(b.reward, reach=reach))
    _ENV_CACHE[key] = NavEnv(cfg)
    return _ENV_CACHE[key]


def flat_env() -> NavEnv:
    """평지 기본 env (`body_clearance` 등 지형이 없어야 하는 측정용)."""
    key = ("__flat__",)
    if key not in _ENV_CACHE:
        _ENV_CACHE[key] = NavEnv(default_config())
    return _ENV_CACHE[key]


def _yaw_np(quat: np.ndarray) -> np.ndarray:
    """(N,4) wxyz -> yaw. 호스트 전용 (`guidance.yaw_from_quat`의 numpy 판)."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.arctan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))


def _episode_fn(env: NavEnv):
    """★ **env당 1회만 컴파일되는** 전체 에피소드 함수. `env`에 캐시한다.

    ⚠️ 예전에는 `rollout`이 호출마다 `jax.jit(env.step)`을 새로 만들었다
    (2026-08-02 수정). `jax.jit`의 캐시는 jit 객체마다 따로이므로 그러면
    **롤아웃마다 MJX가 재컴파일**된다. 시드 5개짜리 평지 점검에서는 견딜 만했지만
    `axis_sweep`은 (스윕값 × 시드)회를 돌므로 컴파일만 몇 시간이 된다.

    같은 이유로 제어기 파라미터(`base_a`, `k_yaw`, `vx_max`)를 **인자로** 받는다.
    파이썬 실수로 닫아 두면 값이 상수로 박혀 스윕 값마다 다시 컴파일된다.
    """
    fn = getattr(env, "_scripted_episode", None)
    if fn is None:
        def ep(rng, base_a, k_yaw, vx_max, hold):
            st0 = env.reset(rng)

            def body(st, _):
                # 진로 유지기는 y·ψ가 필요한데 관측 37D에는 **절대 좌표가 없다**
                # (있으면 정책이 위상을 암기한다, `_obs` 주석). 스크립트 제어기는
                # 측정 기구이므로 여기서는 상태를 직접 읽는다.
                q = st.pipeline_state.qpos
                a = guidance_controller(
                    st.obs, env.cfg.action, base_a, k_yaw, vx_max,
                    y=q[1], psi=guidance.yaw_from_quat(q[3:7]), hold=hold)
                st2 = env.step(st, a)
                # ★ **qpos를 통째로** 남긴다 (19D, 600스텝이면 46 KB — 무시 가능).
                #   x/y/z/yaw만 뽑던 것을 바꾼 이유가 둘 있다 (2026-08-06):
                #     ① 영상. 관절각이 없으면 렌더링을 할 수 없다. "왜 못 넘나"를
                #        숫자로만 보다가 경사에서 세 번 오진했다 — 눈으로 봤으면
                #        한 번에 끝났을 것들이다.
                #     ② 몸통 최고점(터널 통과 높이)은 관절각 없이는 못 구한다.
                #   yaw도 호스트에서 뽑는다(`_yaw_np`) — 스캔 출력이 좁을수록 좋다.
                return st2, (st2.pipeline_state.qpos,
                             st2.info["dist"], st2.done, st2.reward)

            return jax.lax.scan(body, st0, None, length=env._max_steps)[1]

        fn = jax.jit(ep)
        env._scripted_episode = fn
    return fn


def default_hold(env: NavEnv) -> float:
    """이 코스에서 진로 유지기를 켤 것인가 (`guidance_controller`의 `hold`).

    사다리·평지는 목표가 **y=0 정면**이므로 켠다. 미로는 목표가 옆이나 뒤에 있고
    흐름장이 경로를 주므로 목표 방위 추종이 맞다 — 켜면 벽으로 직진한다.
    """
    return 0.0 if env.maze_meta is not None else 1.0


def rollout(env: NavEnv, rng_seed: int = 0, k_yaw: float = 2.0,
            vx_max: float = 1.0, hold: float | None = None, **fixed):
    """한 에피소드를 스크립트 제어기로 돌린다.

    Args:
        hold: 진로 유지기 가중치. `None`이면 `default_hold(env)`.
            `0.0`으로 두면 2026-08-06 이전 동작(목표 방위 추종만)이 된다 —
            요철에서 옆 평지로 새던 그 동작이므로 비교용으로만 쓸 것.
        fixed: 나머지 6축(height, step_freq, footswing, pitch, stance_width)을
            범위 중앙 대신 지정한 **물리값**으로 고정한다. 예: `pitch=-0.2`.
            ★ **이것이 대조군 A다** — `axis_sweep` 참조.
    """
    if hold is None:
        hold = default_hold(env)
    qpos, dist, done, rew = _episode_fn(env)(
        jax.random.PRNGKey(rng_seed),
        base_action(env.cfg.action, **fixed),
        jnp.asarray(k_yaw, jnp.float32), jnp.asarray(vx_max, jnp.float32),
        jnp.asarray(hold, jnp.float32))

    qpos = np.asarray(qpos)
    dist, done, rew = map(np.asarray, (dist, done, rew))
    xs, ys, zs = qpos[:, 0], qpos[:, 1], qpos[:, 2]
    yaws = _yaw_np(qpos[:, 3:7])
    end = int(np.argmax(done > 0)) if done.max() > 0 else len(done) - 1
    # ★ `level` = 넘긴 장애물 수. PPO 실행의 `eval/episode_level`과 **같은 정의**로
    #   계산해야 대조군으로 쓸 수 있다 (`nav_env.step`의 level).
    clear_x = np.asarray(env._clear_x)
    level = int((xs[:end + 1].max() > clear_x).sum()) if clear_x.size else 0
    return dict(qpos=qpos,
                x=xs, y=ys, z=zs, yaw=yaws, dist=dist, done=done, reward=rew, end=end,
                reached=bool(dist[end] < env.cfg.course.goal_radius),
                level=level,
                min_z=float(zs[:end + 1].min()),
                ret=float(rew[:end + 1].sum()),
                t=float((end + 1) * env._hlc_dt))


def _summary(rate, lv, margin, rows, env, full: bool) -> dict:
    """`evaluate`의 반환값. `full=False`면 **배열을 담지 않는다** (위 주석 참조)."""
    out = {"rate": rate, "level": lv, "margin": margin}
    if full:
        out["rows"], out["env"] = rows, env
    return out


def _report_off_band(env: NavEnv, rows: list[dict]) -> float | None:
    """★ 요철 측정이 **오염됐는지** 잰다. 없으면 `None`.

    요철은 중앙 띠(`modules.ROUGH_HALF_W`)에만 깔려 있고 양옆은 평지다 —
    표류해서 발판 옆으로 떨어지는 것을 막기 위한 구조다(`PLATFORM_W` 주석).
    그런데 그 평지는 **요철을 피해 갈 수 있는 우회로**이기도 하다. 2026-08-06에
    실제로 그렇게 됐고, 영상으로 보기 전까지 level 6.00을 그대로 믿을 뻔했다.

    ⚠️ 넓이로 막을 수 없다 — 되먹임이 없으면 y는 시간에 대해 **2차로** 자란다.
    그래서 제어기 쪽에 진로 유지기를 넣었고(`guidance_controller`의 `hold`),
    이 함수는 그것이 실제로 듣고 있는지 매 실행 확인하는 계측이다.
    """
    from wtw_nav.terrain import modules as mod

    if getattr(getattr(env.cfg, "terrain", None), "kind", None) != "rough":
        return None
    hw = mod.ROUGH_HALF_W
    off = float(np.mean([np.mean(np.abs(r["y"][:r["end"] + 1]) > hw)
                         for r in rows]))
    ymax = max(float(np.abs(r["y"][:r["end"] + 1]).max()) for r in rows)
    print(f"요철 띠 이탈 {off:.0%}   |y|max {ymax:.2f} m  (띠 반폭 {hw} m)")
    if off > 0.10:
        print(f"\n  ✗ **이 측정은 오염됐습니다.** 로봇이 시간의 {off:.0%}를 요철 "
              f"바깥 평지에서 보냈습니다 — level이 세고 있는 것은 '요철을 "
              f"통과했다'가 아니라 '요철을 피했다'입니다.\n"
              f"     ① `rollout(..., hold=1.0)`인지 확인 (기본값이어야 함)\n"
              f"     ② 그래도 새면 `HOLD_K_Y`/`HOLD_PSI_MAX`를 올리십시오.\n"
              f"     ③ 띠를 넓히는 것은 최후수단입니다 — geom이 반폭에 비례해 "
              f"늘고, 원인(되먹임)은 그대로 남습니다.")
    return off


def evaluate(env: NavEnv | None = None, n: int = 5, cfg: HLCConfig | None = None,
             full: bool = False, **ctrl_kw) -> dict:
    """여러 시드로 돌려 도달률을 낸다.

    Args:
        full: `True`면 반환 dict에 `rows`(궤적·qpos 전체)와 `env`를 넣는다.
            ⚠️ 기본은 **False**다 — 이 함수가 셀 마지막 줄에 오면 Colab이 반환값을
            자동 출력하는데, `rows`에는 시드마다 (600, 19) qpos가 들어 있어 화면이
            수천 줄로 터진다. 2026-08-06과 08-07에 두 번 발생했다
            (`wtw_nav/dev.py`의 "Colab 셀 작성 규약" 규칙 5).
    """
    env = env or NavEnv(cfg or default_config())
    # ⚠️ 코스 길이는 `course.length`가 아니라 **지형이 정한다**(사다리·미로).
    #    옛 출력은 지형 실행에서도 10.0을 찍어 실제와 어긋났다 (2026-08-02).
    print(f"수동 제어기 평가 — 코스 {env._course_len:.2f} m, "
          f"타임아웃 {env.cfg.term.timeout_s} s ({env._max_steps} 스텝)"
          + (f", 고정축 {ctrl_kw}" if ctrl_kw else ""))
    print(f"{'seed':>4s} {'도달':>5s} {'level':>5s} {'남은거리':>8s} {'시간':>6s} "
          f"{'min_z':>6s} {'리턴':>8s}")
    print("-" * 52)

    rows = []
    for s in range(n):
        r = rollout(env, rng_seed=s, **ctrl_kw)
        rows.append(r)
        print(f"{s:>4d} {str(r['reached']):>5s} {r['level']:>5d} "
              f"{r['dist'][r['end']]:8.2f} "
              f"{r['t']:6.1f} {r['min_z']:6.3f} {r['ret']:8.1f}", flush=True)

    rate = float(np.mean([r["reached"] for r in rows]))
    lv = float(np.mean([r["level"] for r in rows]))
    print(f"\n도달률 {rate:.0%}  ({sum(r['reached'] for r in rows)}/{n}), "
          f"평균 level {lv:.2f}")

    _report_off_band(env, rows)
    _report_off_beam(env, rows)

    if rate < 1.0:
        # ⚠️ 지형 사다리에서는 **도달 실패가 정상이고 그것이 측정값이다.** 사다리는
        #    못 넘는 곳에서 멈추도록 설계됐으므로(`modules.ladder`) 여기서 "env를
        #    고치라"고 하면 정반대의 지시가 된다. 그 경고는 평지에서만 의미가 있다.
        if env.terrain_meta is not None:
            print(f"\n  지형 사다리이므로 도달 실패는 정상입니다 — 읽을 값은 "
                  f"**평균 level {lv:.2f}**이고, 이것이 대조군 A의 기준선입니다.")
        else:
            print("\n  ⚠️ 수동 제어기가 목표에 못 갔습니다. **PPO를 돌리기 전에** 원인을 찾으십시오.")
            print("     점검: ① 유도 벡터 부호 ② 종료 조건이 너무 이른가(낙상/교착)")
            print("           ③ 타임아웃이 코스 길이 대비 충분한가 ④ 액션 범위")
        return _summary(rate, lv, 0.0, rows, env, full)

    # 기준 제어기가 간신히 통과하면, 그보다 조금만 느린 정책은 전부 타임아웃되어
    # 도달 보너스를 한 번도 못 받는다 -> 학습이 엉뚱한 이유로 실패한다.
    worst_t = max(r["t"] for r in rows)
    margin = 1.0 - worst_t / env.cfg.term.timeout_s
    print(f"타임아웃 여유 {margin:.0%}  (최장 {worst_t:.1f} s / {env.cfg.term.timeout_s:.0f} s)")
    if margin < 0.3:
        print("\n  ⚠️ 여유가 부족합니다. 기준 제어기조차 간신히 통과하면 학습 초기에는")
        print("     도달 보너스를 거의 못 받아 학습이 어려워집니다.")
        # ⚠️ 미로에서 "코스를 줄이라"는 조언은 **실행 불가능하다** — 코스 길이가
        #    BFS 경로라 설정값이 아니다. `CourseConfig.length`는 평지에서만 쓰인다.
        #    (2026-08-10에 미로 실행이 이 문구를 찍어 엉뚱한 데를 보게 만들었다.)
        if env.maze_meta is not None:
            print(f"     미로는 코스 길이가 BFS 경로라 줄일 수 없습니다 -> "
                  f"`configs.maze_config(corner_slack=...)`를 올려 "
                  f"timeout_s를 {worst_t * 1.6:.0f} s 이상으로 만드십시오.")
        else:
            print(f"     `TerminationConfig.timeout_s`를 {worst_t * 1.6:.0f} s 이상으로 올리거나")
            print("     `CourseConfig.length`를 줄이십시오.")
    return _summary(rate, lv, margin, rows, env, full)


def axis_sweep(env: NavEnv, axis: str, values, n: int = 3, **ctrl_kw) -> dict:
    """★ 대조군 A — **한 축을 고정값으로 놓고 사다리를 태운다.**

        from wtw_nav.envs import scripted
        scripted.axis_sweep(env, "pitch", [-0.3, -0.15, 0.0, 0.15, 0.3])

    왜 PPO보다 이걸 먼저 돌리는가. gap에서 PPO 6.9M 스텝(5.7시간)의 결론은
    "고정 직진 명령과 동점"이었다(`terrain/limits.py`). 즉 **대조군이 없었다면
    그 실행에서 아무것도 배우지 못했을 것이다.** 이 함수는 그 대조군을 학습 전에,
    몇 분 만에 만든다.

    읽는 법:
      · 어떤 값도 `level`을 올리지 못한다 -> 그 축은 이 지형에 무력하다.
        PPO가 그 축을 굴려 이기면 그건 축이 아니라 **되먹임**(지형에 맞춰 값을
        바꾸는 것)이 산 것이므로, 그것이 곧 HLC가 필요하다는 증거가 된다.
      · 특정 고정값이 `level`을 올린다 -> **HLC 없이도 되는 부분**이다. PPO 결과는
        반드시 이 기준선 위에서 읽어야 한다.

    ⚠️ env를 재사용한다 (지형·모델이 같으므로 JIT 재컴파일이 없다). 지형을 바꾸면
    새 env를 만들어야 한다.
    """
    print(f"\n=== 축 스윕: {axis} — 지형 '{env.cfg.terrain.kind}', 시드 {n}개 ===")
    out = {}
    for v in values:
        res = evaluate(env=env, n=n, **{axis: v}, **ctrl_kw)
        out[v] = res["level"]
    print(f"\n--- {axis} 스윕 요약 (평균 level) ---")
    for v, lv in out.items():
        print(f"  {axis}={v:>6} -> level {lv:.2f}")
    best = max(out, key=out.get)
    print(f"  최고: {axis}={best} (level {out[best]:.2f}), "
          f"기준(중앙값 고정 없음) 대비 "
          f"{'개선 있음' if out[best] > min(out.values()) else '차이 없음'}")
    return out


#: `**ctrl_kw`로 넘어오는 명령 축 이름들. `video`의 인자와 겹치면 안 된다.
_AXES = ("vx", "yaw", "height", "step_freq", "footswing", "pitch", "stance_width")


def video(env: NavEnv, rng_seed: int = 0, save: str | None = None,
          fps: int = 10, px_w: int = 640, px_h: int = 480,
          distance: float = 3.5, return_data: bool = False, **ctrl_kw):
    """★ 롤아웃 하나를 **눈으로** 본다. 숫자만 보다 세 번 오진한 뒤 만들었다.

        scripted.video(env, pitch=0.3, save="slope_a0.mp4")

    ⚠️ 렌더 크기 인자가 `width`/`height`가 **아니라** `px_w`/`px_h`인 이유:
    `height`는 **WTW 명령 축 이름**이다. `width`/`height`로 두면
    `video(env, height=-0.22)`가 명령이 아니라 세로 픽셀 수로 흡수되어
    `MjrRect(0, 0, 640, -0.22)`로 죽는다 (2026-08-06에 실제로 발생). 게다가
    에러가 렌더러 생성 지점에서 나므로 GL 백엔드 문제로 오진하게 된다.

    ⚠️ 기본적으로 **아무것도 반환하지 않는다** — 노트북이 qpos/궤적 배열을 통째로
    찍어버리기 때문이다(`llc.check.preview`와 같은 이유, 2026-08-06에 실제로
    화면을 몇 천 줄 채웠다). 데이터가 필요하면 `return_data=True`.

    ⚠️ 프레임은 HLC 주기(10 Hz)로 기록된 것이므로 `every=1`이 **필수**다
    (`llc.check.render` 주석). 그래서 `fps`는 실시간 속도를 뜻한다 — 10이 등속이다.

    ⚠️ 종료(`done`) 이후 스캔은 계속 돌지만 그 구간의 물리는 의미가 없다
    (자동 리셋이 없다). 그래서 `end`에서 자른다.
    """
    from wtw_nav.llc import check

    bad = [k for k in ctrl_kw if k not in _AXES + ("k_yaw", "vx_max", "hold")]
    if bad:
        raise TypeError(f"알 수 없는 인자 {bad}. 명령 축은 {_AXES}이고, "
                        f"렌더 크기는 `px_w`/`px_h`입니다 (위 주석 참조).")

    r = rollout(env, rng_seed=rng_seed, **ctrl_kw)
    e = r["end"]
    print(f"  seed {rng_seed}  level {r['level']}  도달 {r['reached']}  "
          f"{r['t']:.1f} s")
    print(f"  x {r['x'][0]:.2f} -> 최대 {r['x'][:e + 1].max():.2f} "
          f"-> 최종 {r['x'][e]:.2f}   "
          f"y {r['y'][0]:+.2f} -> {r['y'][e]:+.2f}   "
          f"yaw {np.degrees(r['yaw'][0]):+.0f}° -> {np.degrees(r['yaw'][e]):+.0f}°")
    frames = check.render(env.mj_model, r["qpos"][:e + 1], fps=fps,
                          every=1, width=px_w, height=px_h,
                          distance=distance)
    # ⚠️ 미로를 'flat'으로 찍던 버그 (2026-08-10). `terrain.kind`는 미로에서 None이라
    #    "flat seed0 level0"이 떠서 지형이 안 만들어진 것처럼 보였다 — 실제로는
    #    미로가 정상이었다. 영상 제목이 틀리면 영상으로 하는 판정을 못 믿게 된다.
    if env.maze_meta is not None:
        mm = env.maze_meta
        tag = f"maze{mm['n']}x{mm['n']}s{mm['seed']}"
    else:
        tag = env.cfg.terrain.kind or "flat"
    check._show(frames, fps, save,
                title=f"{tag} seed{rng_seed} level{r['level']}", show=True)
    return r if return_data else None


def _geom_top(mj_model, mj_data) -> float:
    """현재 자세에서 **충돌하는 로봇 geom의 최고점** z (m).

    ⚠️ `geom_rbound`(경계구 반지름)를 쓰면 안 된다. 몸통 박스의 경계구는 반지름
    0.20 m라 실제 윗면(중심+0.057)보다 3.5배 높게 나온다 — 터널 높이를 그걸로
    정하면 실제로 통과 가능한 터널을 "불가능"으로 판정한다.
    회전을 반영한 **실제 z 방향 반경**을 형상별로 계산한다.
    """
    import mujoco

    m, d = mj_model, mj_data
    sel = (np.asarray(m.geom_bodyid) != 0) & (
        (np.asarray(m.geom_contype) != 0) | (np.asarray(m.geom_conaffinity) != 0))
    idx = np.flatnonzero(sel)
    if idx.size == 0:
        return float("nan")

    xmat = np.asarray(d.geom_xmat).reshape(-1, 3, 3)
    size = np.asarray(m.geom_size)
    gtype = np.asarray(m.geom_type)
    top = -np.inf
    for g in idx:
        t, s = int(gtype[g]), size[g]
        rz = np.abs(xmat[g][2])                    # 월드 z축이 본 geom 로컬 축
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            ext = s[0]
        elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            ext = rz[2] * s[1] + s[0]              # 축은 로컬 z, 반지름 s[0]
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            ext = float(rz @ s[:3])
        else:                                       # MESH 등 — 보수적으로
            ext = float(m.geom_rbound[g])
        top = max(top, float(d.geom_xpos[g][2]) + ext)
    return top


def body_clearance(env: NavEnv | None = None, values=None, n: int = 2,
                   settle_s: float = 2.0, **ctrl_kw) -> dict:
    """★ 실측 — **몸을 어디까지 낮출 수 있나** (= 터널 통과 높이).

    터널 지형을 실제로 만들 필요가 없다. WTW의 `body_height`는 상대 오프셋이라
    **명령값이 곧 통과 높이가 아니고**, 실제로 알아야 하는 것은 "그 명령으로
    걸을 때 로봇의 최고점이 얼마인가" 하나다. 평지에서 걸리면서 재면 된다.

    ⚠️ 정지 자세가 아니라 **걷는 중**에 잰다. 보행은 몸통을 상하로 흔들고,
    터널은 그 최대값에 걸린다. `settle_s` 이후 구간만 쓴다(초기 자세 안정화).

    ⚠️ 명령 하한(-0.22)이 곧 능력 하한이 아니다. 명령을 낮춰도 로봇이 따라오지
    못하면 실제 높이는 안 내려간다 — 그래서 **명령 대비 실측**을 같이 찍는다.
    """
    import mujoco

    env = env or NavEnv(default_config())
    if env.cfg.terrain.kind is not None:
        print("  ⚠️ 지형이 걸린 env입니다. 몸높이는 **평지**에서 재야 합니다 "
              "(지형 기복이 최고점에 섞입니다).")
    lo, hi = env.cfg.action.height
    if values is None:
        values = [round(v, 3) for v in np.linspace(lo, hi, 6)]

    m = env.mj_model
    d = mujoco.MjData(m)
    skip = int(settle_s / env._hlc_dt)

    print(f"\n=== 몸높이 실측 (터널 통과 높이) — height 명령 범위 "
          f"({lo}, {hi}), 시드 {n}개 ===")
    print(f"{'height명령':>10s} {'몸통z평균':>9s} {'최고점평균':>10s} "
          f"{'최고점최대':>10s} {'낙상':>5s}")
    print("-" * 52)

    out = {}
    for v in values:
        tops, trunks, fell = [], [], 0
        for s in range(n):
            r = rollout(env, rng_seed=s, height=v, **ctrl_kw)
            q = r["qpos"][skip:r["end"] + 1]
            if len(q) == 0:
                fell += 1
                continue
            for qi in q:
                d.qpos[:] = qi
                mujoco.mj_forward(m, d)
                tops.append(_geom_top(m, d))
            trunks.append(q[:, 2])
            if r["min_z"] < env.cfg.term.min_height:
                fell += 1
        if not tops:
            print(f"{v:10.3f} {'—':>9s} {'—':>10s} {'—':>10s} {fell:5d}")
            continue
        tops = np.asarray(tops)
        trunk = float(np.concatenate(trunks).mean())
        out[v] = dict(top_mean=float(tops.mean()), top_max=float(tops.max()),
                      trunk_mean=trunk, fell=fell)
        print(f"{v:10.3f} {trunk:9.3f} {tops.mean():10.3f} "
              f"{tops.max():10.3f} {fell:5d}")

    if out:
        best = min(out, key=lambda k: out[k]["top_max"])
        b = out[best]
        print(f"\n  최저 자세: height={best} -> 최고점 최대 **{b['top_max']:.3f} m**")
        print(f"  기준(중앙 명령) 대비 {out.get(values[len(values) // 2], b)['top_max'] - b['top_max']:+.3f} m")
        print(f"  ⇒ 터널 실측값 후보: **{b['top_max']:.3f} m** "
              f"(limits.py의 tunnel은 direction='min', margin=1.15 "
              f"-> 설계 하한 {b['top_max'] * 1.15:.3f} m)")
        print("  ※ 이 값은 천장에 **닿지 않고** 지나가는 높이입니다. 실제 터널을 "
              "만들어 확인하려면 `terrain.kind='tunnel'`로 사다리를 태우십시오.")
    return out


#: Go1 발 geom 이름 (menagerie `go1.xml`, `class="foot"`). 구 반지름 0.023 m.
FOOT_GEOMS = ("FL", "FR", "RL", "RR")
FOOT_R = 0.023
#: 발이 **딛고 있다**고 볼 높이 (지면 위 m). 발 반지름 + 여유.
#: 스윙 중인 발은 다리 밖으로 지나가도 되지만 **딛는 발**은 다리 위여야 한다 —
#: 그래서 외나무다리 요구폭은 스탠스 발만으로 정해진다.
STANCE_Z = 0.045


def _foot_xy(mj_model, mj_data) -> np.ndarray:
    """(4, 3) 발 geom 월드 좌표. 이름으로 찾는다 (인덱스는 모델마다 바뀐다)."""
    import mujoco

    ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, n)
           for n in FOOT_GEOMS]
    if any(i < 0 for i in ids):
        raise ValueError(f"발 geom을 찾지 못했습니다: {FOOT_GEOMS} -> {ids}. "
                         f"로봇 모델이 바뀌었으면 FOOT_GEOMS를 고치십시오.")
    return np.asarray(mj_data.geom_xpos)[ids]


def foot_track(env: NavEnv | None = None, values=None, n: int = 3,
               settle_s: float = 2.0, **ctrl_kw) -> dict:
    """★ 실측 — **발을 어디까지 좁힐 수 있나** (= 외나무다리 필요폭).

    ⚠️ 재는 것은 "외나무다리를 건널 수 있나"가 **아니다.** 그건 두 가지가 섞인
    질문이다 — 발 간격(LLC 기하)과 표류(항법). 이 함수는 **평지에서** 둘을
    분리해서 재고, 실지형 사다리는 그 예측을 확인하는 용도로만 쓴다.
    터널에서 이 절차가 정확히 맞았다 (`body_clearance` -> level 5.00/4.00 적중).

    세 수를 낸다:

        W_foot  = 2 × max|Δy|  — **몸 기준** 좌우 발 간격. 순수 LLC 기하.
                  ⚠️ `stance_width` **명령값이 아니다.** 2026-08-09 실측에서
                  명령 0.12 -> W_foot 0.275(2.29배)로 **포화**했다. 넓히는 쪽은
                  1:1로 듣는데(구간 이득 1.12) 좁히는 쪽은 1/3만 듣는다(0.37) —
                  중립 0.25에서 하한 0.12까지 끝까지 좁혀도 0.065 m(19%)뿐이다.
        floor   = W_foot + 2·FOOT_R — **기하 바닥.** 완벽히 중앙에 있을 때
                  필요한 다리 전폭. ★ 사다리 값의 기준은 이 수다.
        D       = max|y_body| — 표류. 순수 항법 (`hold=1` 기준).

    ⚠️ **`need = 2×(max|y_foot,world| + r)`는 폐기했다 (2026-08-09).** 월드 좌표에는
    표류가 섞여 있고 **표류는 코스 길이에 따라 자란다** — 평지 10 m에서 D=0.179였던
    것이 다리 사다리 21 m에서 0.350이 됐다. 즉 그 수는 로봇의 성질이 아니라
    "이 코스에서의 값"이었고, 그런 수를 설계에 쓰면 코스가 길어질 때마다 틀린다.
    기하(길이 무관)와 표류(길이 의존)를 **합치지 않고 따로 낸다.**

    ⚠️ D도 **넓힐수록 커진다** (0.179 -> 0.753). 외나무다리에서 `stance_width`를
    넓히는 것은 기하로 얻고 항법으로 잃는 게 아니라 **양쪽 다 손해**다.

    ⚠️ **스탠스 발만 센다** (`z < STANCE_Z`). 스윙 중인 발은 다리 밖으로 지나가도
    떨어지지 않는다 — 전부 세면 필요폭이 `footswing`만큼 부풀려진다.

    ⚠️ 걷는 중에 잰다. `settle_s` 이전은 초기 자세 안정화라 버린다.
    """
    import mujoco

    env = env or flat_env()
    if env.cfg.terrain.kind is not None:
        print("  ⚠️ 지형이 걸린 env입니다. 발 간격은 **평지**에서 재야 합니다 "
              "(지형에 걸려 발이 벌어지면 LLC 기하가 아니라 지형을 재게 됩니다).")
    lo, hi = env.cfg.action.stance_width
    if values is None:
        values = [round(v, 3) for v in np.linspace(lo, hi, 6)]

    m = env.mj_model
    d = mujoco.MjData(m)
    skip = int(settle_s / env._hlc_dt)

    print(f"\n=== 발 간격 실측 (외나무다리 필요폭) — stance_width 명령 범위 "
          f"({lo}, {hi}), 시드 {n}개 ===")
    print(f"{'sw명령':>7s} {'W_foot':>7s} {'추종률':>6s} {'이득':>5s} "
          f"{'기하바닥':>8s} {'표류 D':>7s} {'스탠스%':>7s} {'낙상':>4s}")
    print("-" * 60)

    out, prev = {}, None
    for v in values:
        dys, ymax, ywmax, ns, nt, fell = [], 0.0, 0.0, 0, 0, 0
        for s in range(n):
            r = rollout(env, rng_seed=s, stance_width=v, **ctrl_kw)
            q = r["qpos"][skip:r["end"] + 1]
            if r["min_z"] < env.cfg.term.min_height:
                fell += 1
            if len(q) == 0:
                continue
            ymax = max(ymax, float(np.abs(r["y"][skip:r["end"] + 1]).max()))
            for qi in q:
                d.qpos[:] = qi
                mujoco.mj_forward(m, d)
                f = _foot_xy(m, d)
                nt += 4
                st = f[f[:, 2] < STANCE_Z]              # 딛고 있는 발만
                ns += len(st)
                if len(st) == 0:
                    continue
                psi = float(_yaw_np(qi[None, 3:7])[0])
                # 몸 기준 횡오프셋 (요를 되돌린다)
                dx, dy = st[:, 0] - qi[0], st[:, 1] - qi[1]
                dys.append(np.abs(-np.sin(psi) * dx + np.cos(psi) * dy).max())
                ywmax = max(ywmax, float(np.abs(st[:, 1]).max()))
        if not dys:
            print(f"{v:7.3f} {'—':>7s} {'—':>6s} {'—':>5s} {'—':>8s} "
                  f"{'—':>7s} {'—':>7s} {fell:4d}")
            continue
        w = 2.0 * float(np.max(dys))
        floor = w + 2.0 * FOOT_R
        # 구간 이득 dW/dcmd. ★ 포화는 "명령을 줘도 안 좁아진다"가 아니라
        #   **"이득이 1에서 1/3로 떨어진다"**로 나타난다 — 비(ratio)만 보면 놓친다.
        gain = ((w - prev[1]) / (v - prev[0])) if prev else float("nan")
        prev = (v, w)
        out[v] = dict(w_foot=w, drift=ymax, floor=floor, gain=gain,
                      ratio=w / v, stance_frac=ns / max(nt, 1), fell=fell)
        print(f"{v:7.3f} {w:7.3f} {w / v:6.2f} "
              + (f"{gain:5.2f}" if gain == gain else f"{'—':>5s}")
              + f" {floor:8.3f} {ymax:7.3f} {ns / max(nt, 1):7.0%} {fell:4d}")

    if out:
        best = min(out, key=lambda k: out[k]["w_foot"])
        b = out[best]
        print(f"\n  최소 발 간격: stance_width={best} -> W_foot **{b['w_foot']:.3f} m** "
              f"(명령 대비 {b['ratio']:.2f}배)  ⇒ 기하 바닥 **{b['floor']:.3f} m**")
        gains = [r["gain"] for r in out.values() if r["gain"] == r["gain"]]
        if gains and gains[0] < 0.6 * gains[-1]:
            print(f"  ⚠️ **포화입니다.** 이득이 하한 쪽 {gains[0]:.2f}, 상한 쪽 "
                  f"{gains[-1]:.2f} — 넓히는 쪽은 1:1로 듣는데 좁히는 쪽은 "
                  f"{gains[0] / gains[-1]:.0%}만 듭니다. 명령 하한을 더 내려도 "
                  f"거의 얻는 것이 없습니다 (`body_height`와 같은 현상).")
        ds = [r["drift"] for r in out.values()]
        if ds[-1] > 1.5 * ds[0]:
            print(f"  ⚠️ 표류 D가 발 간격과 **같이** 커집니다 ({ds[0]:.2f} -> "
                  f"{ds[-1]:.2f}). 넓게 벌리는 것은 기하로 얻고 항법으로 잃는 게 "
                  f"아니라 **양쪽 다 손해**입니다.")
        # ★ 사다리는 **기하 바닥**을 기준으로 잡는다. 표류를 더해 만든 값은
        #   코스 길이에 따라 변해서 재현되지 않는다 (위 ⚠️ 참조).
        #   마지막 단이 곧 바닥이므로 "여기서 실패 = 순수 기하 한계"가 된다.
        rungs = ", ".join(f"{b['floor'] * k:.2f}"
                          for k in (2.5, 2.0, 1.7, 1.4, 1.2, 1.0))
        print(f"  ⇒ 사다리: scripted.terrain_env('beam', values=({rungs}))")
        print(f"     마지막 단 {b['floor']:.2f}가 기하 바닥입니다 — 거기서 실패하면 "
              f"기하 한계, 그 위에서 실패하면 **항법**이 한계입니다.")
        print(f"     표류 D={b['drift']:.2f}는 **따로** 읽으십시오. "
              f"코스 길이에 따라 자라므로 폭에 더하면 안 됩니다.")
    return out


def _report_off_beam(env: NavEnv, rows: list[dict]) -> dict | None:
    """★ 외나무다리 실패의 **원인을 판별**한다. 없으면 `None`.

    "건너지 못했다"만으로는 발 간격 탓인지 항법 탓인지 알 수 없다 — 그런데
    그 둘은 대응이 정반대다(전자는 LLC 재학습, 후자는 HLC 게인). 그래서 매 실행
    다음 셋을 같이 찍는다:

        max|y_body|      표류 (항법)
        max|y_foot,st|   딛는 발의 횡 최대 (기하 + 표류)
        이탈 스텝 비율   딛는 발이 다리 **밖**에 놓인 비율

    읽는 법은 `docs/03_results.md` §3·§7.2의 판별표.
    """
    import mujoco

    if getattr(getattr(env.cfg, "terrain", None), "kind", None) != "beam":
        return None
    vals = env.cfg.terrain.values
    if not vals:
        return None

    m = env.mj_model
    d = mujoco.MjData(m)
    # ⚠️ **다리 위에서만** 잰다 (2026-08-09). 전 구간 최대를 쓰면 다리 사이
    #    회복 구간에서 난 표류가 섞여, "다리가 좁아서"인지 "벌판에서 밀려서"인지
    #    구분이 안 된다. 첫 실측이 정확히 그렇게 오염됐다 — 평지 0.179 대 사다리
    #    0.350의 차이는 다리가 아니라 12 m 회복 구간이 만든 것이었다.
    ybody, yfoot, off, tot, on = 0.0, 0.0, 0, 0, 0
    ybody_all = 0.0
    for r in rows:
        q = r["qpos"][:r["end"] + 1]
        ybody_all = max(ybody_all, float(np.abs(r["y"][:r["end"] + 1]).max()))
        for qi in q:
            hw = _beam_halfwidth(env, float(qi[0]))
            if hw is None:                       # 다리 위가 아니다
                continue
            on += 1
            ybody = max(ybody, abs(float(qi[1])))
            d.qpos[:] = qi
            mujoco.mj_forward(m, d)
            st = _foot_xy(m, d)
            st = st[st[:, 2] < STANCE_Z]
            if len(st) == 0:
                continue
            tot += len(st)
            yfoot = max(yfoot, float(np.abs(st[:, 1]).max()))
            off += int((np.abs(st[:, 1]) > hw + FOOT_R).sum())

    if on == 0:
        print("  ⚠️ 다리 위에서 보낸 스텝이 없습니다 — 첫 다리에 닿기 전에 "
              "끝났습니다. 사다리 첫 단을 넓히거나 타임아웃을 늘리십시오.")
        return None
    frac = off / max(tot, 1)
    print(f"[다리 위에서만] 표류 max|y_body| {ybody:.3f} m   "
          f"딛는 발 max|y| {yfoot:.3f} m   다리 밖 착지 {frac:.1%}")
    print(f"  (참고: 전 구간 max|y_body| {ybody_all:.3f} m — 회복 구간 포함. "
          f"이 수로 판정하지 마십시오)")
    if frac > 0.02:
        print("  → 발이 다리 **밖**에 놓이고 있습니다. 폭이 부족하거나 "
              "착지 위상이 맞지 않는 것입니다.")
    elif ybody > 0.15:
        print("  → 발은 다리 위에 있는데 몸이 밀려 있습니다 — **항법**이 한계입니다 "
              "(`HOLD_K_Y`를 올려 재확인).")
    else:
        print("  → 다리 위에서는 안정적입니다. 실패했다면 **진입**(회복 구간에서 "
              "밀린 채 도착)이 원인입니다.")
    return dict(y_body=ybody, y_body_all=ybody_all, y_foot=yfoot, off=frac)


def _beam_halfwidth(env: NavEnv, x: float) -> float | None:
    """x 지점의 다리 반폭 (m). 다리 위가 아니면 `None`."""
    tm = env.terrain_meta
    if tm is None:
        return None
    for i, (xi, v) in enumerate(zip(tm["xs"], env.cfg.terrain.values)):
        if xi <= x <= xi + env.cfg.terrain.spacing * 0.6:
            return float(v) / 2.0
    return None


def phase_sweep(values=None, offsets: int = 8, n: int = 1, kind: str = "ledge",
                stride: float = 0.333, x0: float = 3.0,
                success: float = 0.8, **ctrl_kw) -> dict:
    """★ 실측 — **턱**. 최대 높이가 아니라 **높이별 성공 *확률*** 을 잰다.

    ⚠️ LLC는 `measure_heights=False`라 지형을 **원리적으로 못 본다.** 그래서 턱
    앞에 도착했을 때의 **보행 위상**(어느 발이 스윙 중이고 몇 %인지)이 제어
    불가능한 난수이고, 같은 명령·같은 높이여도 결과가 갈린다. "넘을 수 있는 최대
    높이"라는 질문 자체가 성립하지 않는다.

    위상은 스윕 대상이 아니라 **샘플링 대상**이다. 명령을 고정한 채 **첫 장애물
    거리 `x0`를 1 stride 안에서 균등하게** 흔들어 위상 0~360°를 덮는다:

        stride = vx / step_freq = 1.0 / 3.0 = 0.333 m

    기존 `rng_seed` 스윕은 초기 자세 잡음만 흔들어 위상을 덮지 못했다.

    ⚠️ `x0` 하나가 **MJX 모델 하나 = 컴파일 한 번**이다. offsets=8이면 8회 컴파일이
    들지만 피할 수 없다(지형 geom 위치가 바뀐다). 시드는 그 안에서 공짜다.

    ⚠️ 사다리는 **순차적**이라 i단을 실패하면 i+1단은 시도조차 못 한다. 즉
    "i단 성공률"은 독립 시행이 아니라 **생존곡선**이다. 그래서 둘을 같이 낸다:
        주변확률 = i단 통과 / 전체 실행       (미로 설계용 — 앞을 다 넘어야 하므로)
        조건부   = i단 통과 / i단 도달        (그 높이 자체의 능력)
    조건부는 표본이 급격히 줄므로 **도달 수를 같이 보고 판단할 것.**

    Args:
        success: 이 성공률 이상인 최대 단을 실측값으로 채택 (조건부 기준).
    """
    off = [x0 + k * stride / offsets for k in range(offsets)]
    env0 = terrain_env(kind, values=values, x0=off[0])
    vals = list(env0.cfg.terrain.values)
    cum = np.cumsum(vals) if kind == "ledge" else np.asarray(vals)

    print(f"\n=== 위상 스윕: '{kind}' — 위상 {offsets}점 × 시드 {n}개 "
          f"= {offsets * n}회, stride {stride} m ===")
    print(f"    x0 = {off[0]:.3f} … {off[-1]:.3f} m  "
          f"(위상 0° … {360 * (offsets - 1) / offsets:.0f}°)")
    if ctrl_kw:
        print(f"    고정축 {ctrl_kw}")

    levels = []
    for k, x in enumerate(off):
        env = terrain_env(kind, values=values, x0=x)
        row = [rollout(env, rng_seed=s, **ctrl_kw)["level"] for s in range(n)]
        levels += row
        print(f"  위상 {360 * k / offsets:3.0f}°  x0={x:.3f}  level {row}",
              flush=True)

    lv = np.asarray(levels)
    total = lv.size
    print(f"\n{'단':>3s} {'증분':>6s} {'누적':>6s} {'도달':>5s} {'통과':>5s} "
          f"{'주변':>6s} {'조건부':>7s}")
    print("-" * 46)
    out = {}
    for i, v in enumerate(vals):
        reached = int((lv >= i).sum())          # i단을 시도한 실행 수
        passed = int((lv > i).sum())
        cond = passed / reached if reached else float("nan")
        out[v] = dict(step=float(v), cum=float(cum[i]), reached=reached,
                      passed=passed, marginal=passed / total, cond=cond)
        print(f"{i:3d} {v:6.3f} {cum[i]:6.3f} {reached:5d} {passed:5d} "
              f"{passed / total:6.0%} "
              + (f"{cond:7.0%}" if reached else f"{'—':>7s}"))

    ok = [v for v, r in out.items()
          if r["reached"] >= max(3, total // 4) and r["cond"] >= success]
    if ok:
        best = max(ok)
        print(f"\n  ⇒ 실측값 후보: **{best:.3f} m** "
              f"(조건부 성공률 {out[best]['cond']:.0%} ≥ {success:.0%}, "
              f"도달 {out[best]['reached']}/{total})")
        print(f"     limits.py의 ledge는 direction='max', margin=0.7 "
              f"-> 설계 상한 {best * 0.7:.3f} m")
    else:
        print(f"\n  ⇒ 성공률 {success:.0%}를 넘는 단이 없습니다. 사다리 최저단"
              f"({vals[0]:.3f})부터 실패한다면 **LLC 명령만으로는 턱을 넘지 "
              f"못한다**는 뜻이고, 그것이 결론입니다 — 더 낮은 사다리로 한 번만 "
              f"확인하십시오.")
    spread = lv.max() - lv.min() if total else 0
    print(f"  ※ level 분산 {lv.min()}~{lv.max()} (폭 {spread}). "
          + ("폭이 0이면 위상이 결과를 바꾸지 못한 것이고, 그러면 이 스윕은 "
             "불필요했다는 뜻입니다 — 그것도 결과입니다."
             if spread == 0 else
             "폭이 크다는 것은 **위상이 지배 변수**라는 뜻이고, 평균 level로 "
             "기록하면 안 된다는 근거입니다."))
    return out


def main() -> int:
    res = evaluate(n=5)
    ok = res["rate"] >= 0.8
    print("\n" + ("PASS — env는 풀 수 있는 문제입니다. 학습으로 진행하십시오."
                  if ok else "*** FAIL — 학습 전에 env를 고쳐야 합니다 ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
