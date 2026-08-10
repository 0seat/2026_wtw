"""유도 벡터 — "어디로 가야 하는가"를 로봇 좌표계로 알려준다.

**훈련(코스 축)과 평가(BFS 거리장)가 동일한 출력 형식**을 내는 것이 핵심이다.
형식이 다르면 훈련↔배포 분포가 어긋난다 (`docs/02_hlc.md` §2).

출력: `(cos φ, sin φ, d_norm)`
  φ      : 로봇 요 기준 목표 방향 (전방이 0)
  d_norm : 남은 거리를 코스 길이로 정규화, [0, 1]로 클립
"""

from __future__ import annotations

import jax.numpy as jnp


def yaw_from_quat(quat: jnp.ndarray) -> jnp.ndarray:
    """MuJoCo 쿼터니언 (w,x,y,z) -> yaw."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def guidance_to_point(base_pos: jnp.ndarray, quat: jnp.ndarray,
                      goal_xy: jnp.ndarray, norm_dist: float) -> jnp.ndarray:
    """목표점 하나를 향하는 유도 벡터 (P1 직선 코스용).

    Args:
        base_pos: (3,) 몸통 위치
        quat: (4,) 몸통 자세 (w,x,y,z)
        goal_xy: (2,) 목표 지점
        norm_dist: 거리 정규화 기준 (보통 코스 길이)
    """
    d = goal_xy - base_pos[:2]
    dist = jnp.linalg.norm(d)
    # 목표에 거의 도달했을 때 방향이 발산하지 않도록
    heading = jnp.arctan2(d[1], jnp.where(dist > 1e-6, d[0], 1.0))
    phi = heading - yaw_from_quat(quat)
    return jnp.array([jnp.cos(phi), jnp.sin(phi),
                      jnp.clip(dist / norm_dist, 0.0, 1.0)])


def progress_along(base_pos: jnp.ndarray, goal_xy: jnp.ndarray) -> jnp.ndarray:
    """목표까지 남은 거리. progress 보상은 이 값의 감소량으로 준다(potential-based)."""
    return jnp.linalg.norm(goal_xy - base_pos[:2])


# --------------------------------------------------------------------------
# 미로 유도장 (P4)
# --------------------------------------------------------------------------
def _cell_of(xy: jnp.ndarray, pitch: float, n: int):
    """월드 좌표 -> 셀 인덱스. 미로 밖은 가장자리 셀로 클립한다."""
    ij = jnp.floor(xy / pitch).astype(jnp.int32)
    return jnp.clip(ij, 0, n - 1)


def field_remaining(base_pos: jnp.ndarray, flow: jnp.ndarray, dist_m: jnp.ndarray,
                    pitch: float, n: int) -> jnp.ndarray:
    """BFS 경로를 따라 목표까지 **남은 거리(m)**. progress 보상의 퍼텐셜이다.

    ★ 유클리드 거리를 쓰면 안 된다. 미로에는 "목표에 가까워지려면 일단 멀어져야
    하는" 구간이 반드시 있고, 유클리드 퍼텐셜은 그 구간을 벌해서 정책을 막다른
    골목 앞에 붙잡아 둔다 (`terrain/maze.distance_field` 주석).

    셀 안에서도 매끄럽도록 셀 홉 수에 **흐름 방향 투영**을 뺀다::

        φ(p) = hop(cell)·P - (p - center(cell)) · flow(cell)

    이러면 흐름을 따라 셀 경계를 넘을 때 φ가 연속이다(경계에서 앞 셀은 +P/2,
    다음 셀은 -P/2 를 빼므로 hop이 1 줄어드는 것과 정확히 상쇄된다). 홉만 쓰면
    퍼텐셜이 계단이라 보상이 2초에 한 번씩만 튄다.
    """
    xy = base_pos[:2]
    i, j = _cell_of(xy, pitch, n)
    f = flow[i, j]
    center = (jnp.array([i, j], dtype=xy.dtype) + 0.5) * pitch
    return dist_m[i, j] - jnp.dot(xy - center, f)


def guidance_field(base_pos: jnp.ndarray, quat: jnp.ndarray, flow: jnp.ndarray,
                   dist_m: jnp.ndarray, pitch: float, n: int,
                   norm_dist: float) -> jnp.ndarray:
    """미로용 유도 벡터. **`guidance_to_point`와 출력 형식이 같다** — (cos φ, sin φ, d_norm).

    형식이 같아야 훈련(직선 코스)과 평가(미로)의 관측 분포가 어긋나지 않는다
    (`docs/02_hlc.md` §2). 다른 것은 φ가 목표점 방향이 아니라 **거리장 흐름 방향**
    이라는 점뿐이다.
    """
    xy = base_pos[:2]
    i, j = _cell_of(xy, pitch, n)
    f = flow[i, j]
    heading = jnp.arctan2(f[1], jnp.where(jnp.abs(f[0]) + jnp.abs(f[1]) > 1e-6,
                                          f[0], 1.0))
    phi = heading - yaw_from_quat(quat)
    rem = field_remaining(base_pos, flow, dist_m, pitch, n)
    return jnp.array([jnp.cos(phi), jnp.sin(phi),
                      jnp.clip(rem / norm_dist, 0.0, 1.0)])


# --------------------------------------------------------------------------
# 진로 유지기
# --------------------------------------------------------------------------
def heading_hold(k_y: float = 0.5, k_psi: float = 1.5, psi_max: float = 0.4,
                 yaw_max: float = 0.6, y_ref: float = 0.0):
    """★ 최소 진로 유지기 — 명령의 `yaw`(2)만 되먹임으로 덮어쓴다.

    **왜 필요한가.** WTW의 명령 인터페이스는 순수 **속도 명령**이고, 그 안에는
    로봇이 어디에 있는지(y)도 어디를 보는지(ψ)도 들어오지 않는다. 즉 `vy=0,
    yaw=0`은 "제자리 유지"가 아니라 **"보정하지 않음"**이다. 그런데 이 LLC에는
    상수 편향이 있다(전진 명령 0에서 실측 +0.092 m/s). 요속에도 같은 크기의
    편향이 있으면 그것이 **두 번 적분**된다:

        ψ(t) ≈ ω_bias · t          (선형)
        y(t) ≈ vx · ω_bias · t²/2  (2차)

    2026-07-29 대조군 실측 x_end=5.83, |y|max=3.33 (8 s, vx 0.88)를 여기 대입하면
    ω_bias ≈ 0.12 rad/s다. 작은 값인데 8초면 3.3 m가 된다. **되먹임이 없는 한
    이것은 발산하며, 0을 명령한다고 잡히지 않는다.**

    그래서 이 유지기는 LLC를 봐주는 장치가 아니라 **배포 조건 그 자체**다.
    실제 시스템에는 10 Hz HLC가 요를 잡고 있고, 그것 없이 잰 값은 배포와 무관하다.
    열린 루프로 재겠다는 원칙이 측정을 파괴한 경위는 `docs/03_results.md` §3.5.

    ⚠️ **HLC가 배울 것을 대신 해주는 수준이 되면 안 된다.** 그래서 이 유지기는
    ① yaw 한 축만 건드리고 ② 게인을 낮게(`yaw_max` 0.6 << 학습범위 2.4) 잡는다.
    지형 돌파에 필요한 나머지 7축은 손대지 않는다.

    Args:
        k_y: 횡오차 -> 목표 방위 (rad/m)
        k_psi: 방위 오차 -> 요속 명령 (1/s)
        psi_max: 복귀 시 최대 지향 각도. 크게 두면 코스를 가로질러 지그재그한다.
        yaw_max: 요속 명령 포화. 학습 범위(±2.4)보다 훨씬 작게 둘 것.
        y_ref: 유지할 횡 위치.

    Returns:
        `(cmd15, mjx_data) -> cmd15` 순수 함수. `policy.make_closed_rollout_fn`에 넘긴다.
    """
    def fn(cmd: jnp.ndarray, data) -> jnp.ndarray:
        y = data.qpos[1]
        psi = yaw_from_quat(data.qpos[3:7])
        psi_des = jnp.clip(-k_y * (y - y_ref), -psi_max, psi_max)
        yaw = jnp.clip(k_psi * (psi_des - psi), -yaw_max, yaw_max)
        return cmd.at[2].set(yaw)

    return fn
