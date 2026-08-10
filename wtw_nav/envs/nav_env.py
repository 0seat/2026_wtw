"""HLC 학습 환경 — P1: 평지 직선 코스 + 유도 벡터 추종.

목적은 성능이 아니라 **인터페이스 검증**이다. 장애물이 없으므로 정책이 할 일은
"유도 벡터를 보고 vx/yaw를 내서 목표까지 간다"뿐이며, 여기서 성공률이 안 나오면
관측·보상·종료·LLC 연결 어딘가가 틀린 것이다.

구조 (`docs/02_hlc.md` §1):
    HLC 10 Hz  ──action 8D──▶ command_filter ──15D──▶ LLC 50 Hz (frozen, ×5) ──▶ MJX

주의 — JAX 제약: `reset`은 `jit`/`vmap` 안에서 돌아야 한다. `mujoco.MjData()`나
`mjx.put_data()`를 reset 안에서 호출하면 병렬 학습이 불가능하다 —
`__init__`에서 템플릿을 1회 만들고 `qpos`/`qvel`만 `replace` 한다.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco
from brax.envs.base import Env, State
from mujoco import mjx

from wtw_nav.configs import HLCConfig, default_config
from wtw_nav.hlc import command_filter as cf
from wtw_nav.hlc import guidance
from wtw_nav.hlc import sensors
from wtw_nav.llc import policy as P
from wtw_nav.terrain import maze, modules


class NavEnv(Env):
    """평지 직선 코스에서 목표점까지 가는 HLC 환경.

    브랙스의 `PipelineEnv`가 아니라 `Env`를 직접 상속한다 — 물리는 우리가 MJX로
    직접 굴리므로 브랙스 파이프라인(`sys`)이 필요 없다.
    """

    def __init__(self, cfg: HLCConfig | None = None):
        self.cfg = cfg or default_config()
        c = self.cfg

        self.policy_fn = P.load_policy(c.ckpt)

        # 지형 사다리를 쓰면 코스 길이는 설정값이 아니라 **지형이 정한다** —
        # 목표는 마지막 장애물 너머여야 "도달 = 전부 통과"가 성립한다.
        self.obstacle_xs: tuple[float, ...] = ()
        self.maze_meta: dict | None = None
        self.terrain_meta: dict | None = None
        if c.maze.enabled:
            world, self.maze_meta = maze.build(
                n=c.maze.n, seed=c.maze.seed, loop_prob=c.maze.loop_prob, xml=c.xml,
                max_geom_pairs=c.max_geom_pairs,
                max_contact_points=c.max_contact_points)
            # 코스 "길이"는 직선 거리가 아니라 **BFS 경로 길이**다. 타임아웃·보상
            # 규모가 전부 여기 물려 있으므로 유클리드 대각선(작다)을 쓰면 안 된다.
            self._course_len = float(self.maze_meta["path_len_m"])
            self._flow = jnp.asarray(self.maze_meta["flow"])
            self._dist_m = jnp.asarray(
                self.maze_meta["dist"].astype("float32") * self.maze_meta["pitch"])
            self._pitch = float(self.maze_meta["pitch"])
            self._n_cell = int(self.maze_meta["n"])
        elif c.terrain.kind is None:
            world, self._course_len = c.xml, c.course.length
        else:
            world, self.terrain_meta = modules.ladder(
                c.terrain.kind, c.terrain.values,
                x0=c.terrain.x0, spacing=c.terrain.spacing)
            self.obstacle_xs = tuple(self.terrain_meta["xs"])
            self._course_len = float(self.terrain_meta["end_x"])

        self.mj_model, mj_data, self.mjx_model = P.create_env(
            world, collision=c.collision)
        self.jidx = P._build_joint_index(self.mj_model)
        self._llc_step = P.make_step_fn(self.mjx_model, self.policy_fn, self.jidx)

        # 리셋 템플릿 — 이걸 1회 만들어 두고 reset에서는 qpos/qvel만 갈아끼운다.
        self._init_data = P.reset_data(self.mj_model, mj_data, self.mjx_model, self.jidx)
        self._init_qpos = jnp.asarray(P.wtw_init_qpos(self.mj_model, self.jidx))

        self._goal = jnp.asarray(self.maze_meta["goal_xy"], jnp.float32) \
            if self.maze_meta else jnp.array([self._course_len, 0.0])
        self._hlc_dt = P.POLICY_DT * c.decimation            # 0.02 * 5 = 0.1 s
        self._max_steps = int(c.term.timeout_s / self._hlc_dt)
        self._stuck_steps = max(1, int(c.term.stuck_window_s / self._hlc_dt))
        #: 장애물 판정선 — 이 x를 넘겼으면 그 장애물은 통과한 것으로 센다.
        #: 값은 `modules.ladder`가 장애물 **끝**에서 계산해 준다 (`CLEAR_MARGIN`).
        self._clear_x = jnp.asarray(self.terrain_meta["clear_xs"], jnp.float32) \
            if self.terrain_meta else jnp.zeros((0,), jnp.float32)
        #: 지형 윗면 높이 꺾은선. 없으면(평지·미로) 지면은 z=0이다.
        self._prof = (jnp.asarray(self.terrain_meta["prof_x"], jnp.float32),
                      jnp.asarray(self.terrain_meta["prof_z"], jnp.float32)) \
            if self.terrain_meta else None
        self._preflight()

    def _ground_z(self, x: jnp.ndarray) -> jnp.ndarray:
        """로봇 발밑 지형의 윗면 높이 (m). 평지·미로는 항상 0이다.

        ★ **낙상 판정을 지면 기준으로 만들기 위해 필요하다** (2026-08-02).
        `term.min_height`는 몸통 z의 **절대값**과 비교하는데, slope 사다리는
        지형이 3.6 m까지 올라간다 — 그러면 `qpos[2] < 0.15`가 전 구간에서
        거짓이 되어 **램프 위에서 주저앉아도 낙상으로 세지 않는다.** 남는 것은
        기울기 조건 하나뿐이고, 그러면 "넘어졌는데 level은 올라간 채 종료"가 된다.
        gap 사다리는 전 구간 z=0이라 이 문제가 드러나지 않았다.
        """
        if self._prof is None:
            return jnp.zeros(())
        return jnp.interp(x, self._prof[0], self._prof[1])

    def _preflight(self) -> None:
        """설정이 **원리적으로 불가능한 과제**를 만들고 있지 않은지 본다.

        P1에서 20 m 코스 + 30 s 타임아웃을 두고 6.5M 스텝을 태운 적이 있다 —
        정책 속도로는 32 s가 필요해 도달이 불가능했고, 보너스를 한 번도 못 받으니
        그것이 좋다는 것도 배울 수 없었다. 같은 사고를 조용히 반복하지 않는다.
        """
        c = self.cfg
        # 최악의 출발점(가장 뒤)을 기준으로 본다 — 평균으로 보면 절반의 에피소드가
        # 조용히 도달 불가능해지고, 그 절반은 reach 보너스를 영원히 못 본다.
        # 미로는 출발 흔들림이 셀 안(jitter)으로 제한된다 — 코스용 init_x를 그대로
        # 더하면 필요 시간을 과대평가해 쓸데없이 경고한다.
        back = c.maze.init_jitter if c.maze.enabled else c.course.init_x
        worst = self._course_len + back
        need = worst / 0.7                                 # 실측 순항 ~0.88 m/s
        if need > c.term.timeout_s:
            print(f"  주의 — 코스 {worst:.1f} m(최악 출발점)는 0.7 m/s로 {need:.0f}s가 "
                  f"필요한데 timeout은 {c.term.timeout_s:.0f}s입니다. "
                  f"도달이 불가능하면 reach 보너스를 한 번도 못 받습니다 "
                  f"-> term.timeout_s를 {need + 10:.0f} 이상으로 올리십시오.")
        total = c.reward.progress * self._course_len
        if c.reward.reach < 0.3 * total:
            print(f"  주의 — reach({c.reward.reach})가 progress 총합({total:.0f})의 "
                  f"{c.reward.reach / total:.0%}뿐입니다. 마무리 유인이 사라집니다 "
                  f"-> {0.5 * total:.0f} 근처로. (`envs.reward_audit` 재실행)")
        if self.maze_meta is not None:
            mm = self.maze_meta
            print(f"  미로 {mm['n']}×{mm['n']} (seed {mm['seed']}), "
                  f"피치 {mm['pitch']:.2f} m, 크기 {mm['n'] * mm['pitch']:.1f} m 사방")
            print(f"  출발 {mm['start_cell']} {tuple(round(v, 2) for v in mm['start_xy'])}"
                  f" -> 목표 {mm['goal_cell']} {tuple(round(v, 2) for v in mm['goal_xy'])}"
                  f",  BFS 경로 {int(mm['dist'][mm['start_cell']])} 홉 "
                  f"= {self._course_len:.1f} m")
            # 벽이 라이다 높이를 지나지 않으면 정책은 벽을 **볼 수 없다**. 그 상태로
            # 학습하면 "센서가 있는데 못 피한다"로 오진하게 된다.
            # ★ 처리량 경고. MJX는 충돌 후보 쌍을 컴파일 시점에 정적으로 펼쳐
            #   매 스텝 전부 계산한다(런타임 컬링 없음). 벽 geom 수가 그대로
            #   비용이고, P4 첫 실행에서 평지 3,096 HLC steps/s가 미로에서
            #   40 미만으로 떨어졌다. 학습을 걸기 전에 이 수를 보게 한다.
            wg = mm.get("wall_geoms")
            if wg:
                print(f"  벽 geom {wg}개 -> 정적 후보 쌍 ≈ {wg * 31:,} (평지는 ≈31)")
                modules.broadphase_report(self.mj_model)
                if not c.max_geom_pairs:
                    print(f"     주의 — `max_geom_pairs`가 꺼져 있습니다 — 후보 쌍 "
                          f"전부가 매 스텝 좁은단계를 돕니다. **학습 전에 "
                          f"`bench.breakdown(cfg)`로 처리량을 재십시오.**")
            from wtw_nav.terrain.maze import WALL_H
            if WALL_H < 0.45:
                print(f"  주의 — 벽 높이 {WALL_H} m가 라이다 높이(몸통 z≈0.34)에 너무 "
                      f"가깝습니다. 정책이 벽을 못 봅니다 -> maze.WALL_H를 올리십시오.")
            return
        if c.terrain.kind is not None:
            tm = self.terrain_meta
            print(f"  지형 '{c.terrain.kind}' 사다리 {len(c.terrain.values)}단, "
                  f"장애물 x={[round(x, 1) for x in self.obstacle_xs]}, "
                  f"목표 x={self._course_len:.2f}, 출발 x=±{c.course.init_x}")
            print(f"  통과 판정선 x={[round(v, 1) for v in tm['clear_xs']]}, "
                  f"최종 지형 높이 {tm['top_z']:.2f} m")
            # 라이다는 몸통 높이(≈0.34)에서 수평으로 쏜다 -> 무엇이 보이는지가
            # 지형 종류마다 다르고, 그것이 이 측정의 해석을 바꾼다.
            if c.terrain.kind in ("gap", "ledge", "rough"):
                print("  ※ 라이다(16D)는 이 지형을 **보지 못합니다** — 광선이 몸통 "
                      "높이에서 수평으로 나가므로 바닥의 틈도 낮은 턱도 그냥 "
                      "지나갑니다(`hlc/sensors.py`). 여기서 얻는 값은 '8D 명령이 "
                      "지형 정보 **없이** 낼 수 있는 상한'입니다.")
            else:
                print("  ※ 라이다(16D)가 이 지형을 부분적으로 봅니다 — 램프/벽면이 "
                      "몸통 높이를 지나므로 전방 빔에 잡힙니다. 즉 이 측정은 gap과 "
                      "달리 '**보고** 대응할 수 있을 때의 상한'입니다. 두 값을 "
                      "직접 비교하지 마십시오.")
            print("  ※ 출발 x를 랜덤화했으므로 유도벡터 d_norm으로 장애물 위치를 "
                  "역산할 수 없습니다. 몸통 높이 관측도 AGL이라 고도로도 역산되지 "
                  "않습니다(`_obs` 주석).")
            # 위상 암기가 깨지는 기준은 "출발점 분포 폭 vs 장애물 주기"다.
            span = 2.0 * c.course.init_x
            if span < c.terrain.spacing * 0.5:
                print(f"  주의 — 출발 분포 폭 {span:.1f} m가 장애물 주기 "
                      f"{c.terrain.spacing} m의 {span / c.terrain.spacing:.0%}뿐입니다. "
                      f"유도벡터 d_norm으로 위상을 역산할 여지가 남아 측정이 "
                      f"낙관 편향됩니다 -> course.init_x를 올리십시오.")

    # -------------------------------------------------- 유도 (코스 / 미로 공통)
    # 주의 — 분기는 **여기 파이썬 수준에서만** 한다. jit 안에서 갈라지면 두 경로가 다
    #    추적되고, 미로가 아닌 실행에서도 거리장 배열이 상수로 박힌다.
    def _guidance(self, data) -> jnp.ndarray:
        """(cos φ, sin φ, d_norm). 두 경로의 출력 형식이 같다 (`guidance` 모듈 주석)."""
        if self.maze_meta is None:
            return guidance.guidance_to_point(
                data.qpos[:3], data.qpos[3:7], self._goal, self._course_len)
        return guidance.guidance_field(
            data.qpos[:3], data.qpos[3:7], self._flow, self._dist_m,
            self._pitch, self._n_cell, self._course_len)

    def _remaining(self, data) -> jnp.ndarray:
        """목표까지 남은 거리. progress 보상의 퍼텐셜이고 도달 판정의 기준이다.

        미로에서는 **유클리드가 아니라 BFS 경로 길이**다 — 이유는
        `terrain.maze.distance_field` 주석.
        """
        if self.maze_meta is None:
            return guidance.progress_along(data.qpos[:3], self._goal)
        return guidance.field_remaining(
            data.qpos[:3], self._flow, self._dist_m, self._pitch, self._n_cell)

    # ------------------------------------------------------------------ 관측
    def _obs(self, data, prev_cmd) -> jnp.ndarray:
        """proprio(10) + 유도(3) + 직전 명령(8) + 라이다(16) = 37D.

        관절각은 넣지 않는다 — LLC 소관이며 보행 위상 과적합 위험 (`docs/02_hlc.md` §2).

        주의 — 몸통 높이는 **지면 기준(AGL)** 이다 (2026-08-02). 절대 z를 넣으면
        slope 사다리에서 z가 0 -> 3.9 m로 단조 증가하므로 **z가 곧 x의 대리변수**가
        된다. 그러면 정책은 지형이 아니라 고도계를 보고 램프를 예측하고,
        `course.init_x` 랜덤화가 지키려던 것(위상 암기 차단)이 무력화된다.
        평지·미로에서는 `_ground_z`가 0이라 이 변경이 아무 영향도 없다.

        주의 — 라이다는 **벽이 있는 코스에서만 정보를 준다.** 평지·지형 사다리에서는
        몸통 높이(0.34 m)를 지나는 것이 없어 전 빔이 1.0으로 상수다. 그건 버그가
        아니라 센서의 정의다(`hlc/sensors.py` 모듈 주석). 즉 이 16D는 `maze.py`가
        생기기 전까지 무용하며, 미리 넣어 둔 이유는 관측 차원이 바뀌면 정책을
        이어붙일 수 없어서다 — 미로 학습 직전에 바꾸면 그 전 실행이 전부 버려진다.
        """
        quat = data.qpos[3:7]
        return jnp.concatenate([
            P._projected_gravity(quat),                 # 3
            data.qvel[0:3],                             # 3 몸통 선속도(월드)
            data.qvel[3:6],                             # 3 몸통 각속도
            data.qpos[2:3] - self._ground_z(data.qpos[0]),   # 1 몸통 높이 (AGL)
            self._guidance(data),                       # 3
            cf.active_command(prev_cmd),                # 8
            sensors.lidar_2d(self.mjx_model, data),     # 16
        ])

    # ------------------------------------------------------------------ reset
    def reset(self, rng: jnp.ndarray) -> State:
        c = self.cfg
        rng, k_yaw, k_lat, k_x = jax.random.split(rng, 4)

        yaw = jax.random.uniform(k_yaw, minval=-c.course.init_yaw,
                                 maxval=c.course.init_yaw)
        lat = jax.random.uniform(k_lat, minval=-c.course.init_lateral,
                                 maxval=c.course.init_lateral)
        # ★ 출발 x 랜덤화 — 유도벡터 `d_norm`으로 장애물 위치를 역산하는 것을 막는다.
        #   (`configs.CourseConfig.init_x` 참조. 이게 없으면 사다리 측정값은
        #   "지형 능력"이 아니라 "고정 배치 암기"를 잰다.)
        x = jax.random.uniform(k_x, minval=-c.course.init_x,
                               maxval=c.course.init_x)

        if self.maze_meta is not None:
            # 미로에서는 출발 셀 **중심** 기준으로 흔든다. 코스용 ±1.2 m를 그대로
            # 쓰면 통로 폭 1.2 m를 넘겨 벽 속에서 시작한다.
            jit_ = c.maze.init_jitter
            cx, cy = self.maze_meta["start_xy"]
            x = cx + jnp.clip(x, -jit_, jit_)
            lat = cy + jnp.clip(lat, -jit_, jit_)

        qpos = self._init_qpos
        qpos = qpos.at[0].set(x)
        qpos = qpos.at[1].set(lat)
        # 출발점의 지면 높이만큼 띄운다. 평지·미로는 0이라 무영향이고, 지형
        # 사다리에서 출발 x가 램프에 걸릴 경우 발이 지형을 관통한 채 시작하는
        # 것을 막는다 (`llc.policy.wtw_init_qpos` 주석의 그 사고와 같은 부류).
        qpos = qpos.at[2].add(self._ground_z(x))
        qpos = qpos.at[3:7].set(jnp.array(
            [jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)]))

        data = self._init_data.replace(qpos=qpos,
                                       qvel=jnp.zeros_like(self._init_data.qvel))
        data = mjx.forward(self.mjx_model, data)

        obs_hist, gait, last_a, last_last_a = P.init_llc_state()
        prev_cmd = cf.neutral_command(c.action)

        info = {
            "rng": rng,
            "obs_history": obs_hist,
            "gait": gait,
            "last_actions": last_a,
            "last_last_actions": last_last_a,
            "prev_cmd": prev_cmd,
            "step": jnp.zeros(()),
            "dist": self._remaining(data),
            "dist_at_window": self._remaining(data),
            "reached": jnp.zeros(()),
        }
        # 주의 — brax `EvalWrapper`는 metrics를 **에피소드 전체에 걸쳐 합산**해
        #    `eval/episode_<key>`로 보고한다. 그래서 매 스텝 `dist`를 넣으면
        #    거리×스텝수(예: 1286.9)가 찍혀 읽을 수 없다. `dist*done`으로 넣으면
        #    종료 스텝에서 한 번만 더해져 합계 = **최종 거리**가 된다.
        # `nan`은 물리 발산 감지용. 0이 아니면 그 실행의 다른 지표를 믿지 말 것
        #   (step의 NaN 안전망 주석 참조).
        metrics = {"progress": jnp.zeros(()), "reached": jnp.zeros(()),
                   "fell": jnp.zeros(()), "dist": jnp.zeros(()),
                   "level": jnp.zeros(()), "nan": jnp.zeros(())}
        return State(data, self._obs(data, prev_cmd), jnp.zeros(()), jnp.zeros(()),
                     metrics, info)

    # ------------------------------------------------------------------ step
    def step(self, state: State, action: jnp.ndarray) -> State:
        c = self.cfg
        info = dict(state.info)

        cmd = cf.action_to_command(action, info["prev_cmd"], c.action)

        def llc_body(carry, _):
            d, h, g, la, lla = carry
            return self._llc_step(d, h, g, la, lla, cmd), None

        carry = (state.pipeline_state, info["obs_history"], info["gait"],
                 info["last_actions"], info["last_last_actions"])
        carry, _ = jax.lax.scan(llc_body, carry, None, length=c.decimation)
        data, obs_hist, gait, last_a, last_last_a = carry

        # ---- 보상 ----
        dist = self._remaining(data)
        progress = info["dist"] - dist                      # potential-based
        reached = (dist < c.course.goal_radius).astype(jnp.float32)

        proj_g_z = P._projected_gravity(data.qpos[3:7])[2]
        # ★ 지면 기준 높이(AGL). 절대 z를 쓰면 slope에서 낙상이 감지되지 않는다
        #   (`_ground_z` 주석).
        agl = data.qpos[2] - self._ground_z(data.qpos[0])
        fell = jnp.logical_or(agl < c.term.min_height,
                              proj_g_z > c.term.max_tilt).astype(jnp.float32)

        step_n = info["step"] + 1
        # 교착: stuck_window 마다 진행량을 확인
        window_end = jnp.mod(step_n, self._stuck_steps) < 0.5
        moved = info["dist_at_window"] - dist
        stuck = jnp.logical_and(window_end, moved < c.term.stuck_dist).astype(jnp.float32)

        d_cmd = cmd - info["prev_cmd"]
        reward = (c.reward.progress * progress
                  + c.reward.reach * reached
                  + c.reward.time
                  + c.reward.terminate * jnp.maximum(fell, stuck)
                  + c.reward.action_rate * jnp.sum(d_cmd * d_cmd))

        # 주의 — **타임아웃을 여기서 `done`에 넣지 말 것.** brax `EpisodeWrapper`는
        #     truncation = where(steps >= episode_length, 1 - state.done, 0)
        # 으로 계산한다. 우리가 같은 스텝에 done=1을 세우면 truncation=0이 되어
        # brax가 시간 초과를 **진짜 종료로 오인**하고 가치 부트스트랩을 0으로 자른다.
        # 그러면 "아직 걷는 중인데 300스텝에 걸린" 상태의 가치가 0으로 평가되고,
        # 오래 걷는 정책일수록 손해를 봐서 학습이 짧은 에피소드로 붕괴한다.
        # 2026-07-29 실측: dist 19.6 -> 2.28(도달률 15.6%)까지 갔다가 len이 300에
        # 닿은 직후부터 단조 하강해 dist 19.8로 되돌아갔다.
        # 시간 제한은 `train.py`가 `episode_length=env._max_steps`로 넘기므로
        # EpisodeWrapper가 truncation=1과 함께 올바르게 처리한다.
        done = jnp.clip(fell + stuck + reached, 0.0, 1.0)
        # ★ NaN 안전망 (2026-08-01 p2_gap 실행에서 6.88M 스텝에 학습이 죽었다).
        #   **NaN은 종료 조건을 전부 무력화한다** — `qpos[2] < 0.15`도 `proj_g > -0.5`도
        #   `moved < 0.1`도 피연산자가 NaN이면 False다. 그래서 한 번 발산한 env는
        #   done이 영영 서지 않고(그 실행의 `len`이 450에 붙박였다) 450스텝 내내
        #   NaN 보상을 뿜어 배치 전체의 그래디언트를 오염시킨다. 그 결과가
        #   파라미터 NaN -> brax `assert_is_replicated` 실패(NaN != NaN)였다.
        #   여기서 done을 세우면 AutoResetWrapper가 정상 상태로 되돌려 스스로 낫는다.
        bad = jnp.logical_not(jnp.logical_and(jnp.all(jnp.isfinite(data.qpos)),
                                              jnp.all(jnp.isfinite(data.qvel))))
        reward = jnp.where(bad, c.reward.terminate, reward)
        done = jnp.where(bad, 1.0, done)
        # 지표 기록용 "에피소드 마지막 스텝" 판정. `done`과 달리 타임아웃을 포함한다
        # (안 넣으면 시간 초과로 끝난 에피소드의 최종 거리가 0으로 집계된다).
        last = jnp.clip(done + (step_n >= self._max_steps).astype(jnp.float32),
                        0.0, 1.0)

        info.update(
            obs_history=obs_hist, gait=gait, last_actions=last_a,
            last_last_actions=last_last_a, prev_cmd=cmd, step=step_n, dist=dist,
            dist_at_window=jnp.where(window_end, dist, info["dist_at_window"]),
            reached=reached,
        )
        # 주의 — metrics를 **새 dict로 교체하지 말 것.** brax의 EvalWrapper가 `metrics['reward']`를
        #    끼워 넣는데, 교체하면 그 키가 사라져 `lax.scan` 캐리의 pytree 구조가 어긋난다
        #    ("carry input ... 5 children ... output ... 4 children, symmetric difference {'reward'}").
        #    brax 내장 env들도 새로 만들지 않고 update 한다.
        # ★ `level` = 넘긴 장애물 수. 이 지표 하나가 `terrain/limits.py`를 채운다 —
        #   사다리의 n번째까지 넘었다면 그 값이 곧 실측 한계다. 손 스윕이 필요 없는
        #   이유가 여기 있다 (`configs.TerrainConfig` 참조).
        level = jnp.sum((data.qpos[0] > self._clear_x).astype(jnp.float32))

        metrics = dict(state.metrics)
        # NaN을 지표에 흘리면 eval 합산이 통째로 NaN이 되어 **정상 env들의 측정까지
        # 못 읽게 된다.** 발산은 `nan` 지표로만 보고하고 나머지는 0으로 막는다.
        z = jnp.where(bad, 0.0, 1.0)
        metrics.update(progress=jnp.nan_to_num(progress) * z, reached=reached * z,
                       fell=fell * z,
                       dist=jnp.nan_to_num(dist) * last * z,   # 합산되므로 마지막 스텝에만
                       level=jnp.nan_to_num(level) * last * z,
                       nan=bad.astype(jnp.float32))
        # 관측도 막는다 — 안 하면 NaN이 obs normalizer의 러닝 통계에 들어가
        # **리셋 이후에도** 모든 env의 관측이 영구히 오염된다.
        obs = jnp.nan_to_num(self._obs(data, cmd), nan=0.0, posinf=0.0, neginf=0.0)
        return state.replace(pipeline_state=data, obs=obs,
                             reward=reward, done=done, metrics=metrics, info=info)

    # ------------------------------------------------------------------ 크기
    @property
    def action_size(self) -> int:
        return self.cfg.action.size          # 8

    @property
    def observation_size(self) -> int:
        return 10 + 3 + self.cfg.action.size + sensors.size()  # 37

    @property
    def backend(self) -> str:
        return "mjx"
