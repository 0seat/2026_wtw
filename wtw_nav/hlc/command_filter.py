"""HLC 액션(8D 연속) → LLC 명령(15D) 전개 + 안전화.

순서가 중요하다:
  1) tanh 아핀 매핑          — 각 차원을 학습 도달 범위의 진부분집합으로
  2) vx-yaw L1 마름모 제약   — 도달 영역이 상자가 아니다
  3) 미소 명령 데드존 스냅    — ‖(vx,vy)‖<=0.2 는 LLC 미학습 구간
  4) 1차 저역필터            — HLC 10 Hz의 계단 명령을 부드럽게
  5) 15D로 전개              — gait/duty/roll/stance_length/aux는 고정값

근거는 `docs/01_llc.md` §4.
"""

from __future__ import annotations

import jax.numpy as jnp

from wtw_nav.configs import ActionConfig
from wtw_nav.llc import policy as P


def action_to_command(action: jnp.ndarray, prev_cmd: jnp.ndarray,
                      cfg: ActionConfig) -> jnp.ndarray:
    """(8,) 액션 [-1,1] + 직전 15D 명령 -> 새 15D 명령.

    `prev_cmd`가 전부 0이면(리셋 직후) 저역필터가 첫 명령을 0쪽으로 끌어당기므로,
    호출부는 리셋 시 `neutral_command(cfg)`로 초기화할 것.
    """
    a = jnp.tanh(action)

    lo = jnp.array([r[0] for r in cfg.ranges])
    hi = jnp.array([r[1] for r in cfg.ranges])
    v = lo + (a + 1.0) * 0.5 * (hi - lo)          # [-1,1] -> [lo,hi]

    vx, yaw, vy = v[0], v[1], v[2]

    # (2) vx-yaw 마름모: 밖이면 원점 방향으로 균등 축소
    s = jnp.abs(vx) / cfg.diamond_vx + jnp.abs(yaw) / cfg.diamond_yaw
    scale = jnp.minimum(1.0, 1.0 / jnp.maximum(s, 1e-6))
    vx, yaw = vx * scale, yaw * scale

    # (3) 데드존 — 기본값 0.0(비활성). 2026-07-29 실측으로 불필요함이 확인됐다.
    #     ‖(vx,vy)‖<=0.2가 LLC 학습 시 0으로 치환된 것은 맞지만, **그 구간에서
    #     LLC는 완벽히 선형으로 반응한다**(기울기 0.99). `check.deadzone_sweep()`.
    speed = jnp.sqrt(vx * vx + vy * vy)
    keep = jnp.logical_or(speed > cfg.deadzone, cfg.deadzone <= 0.0)
    vx, vy = jnp.where(keep, vx, 0.0), jnp.where(keep, vy, 0.0)

    # (3b) 정지 편향 보정 — LLC는 vx=0 명령에서도 +0.092 m/s 전진한다(상수 오프셋).
    #      데드존이 있을 때는 이를 상쇄할 vx≈-0.09가 0으로 먹혀 **정지가 불가능**했다.
    #      여기서 빼면 "명령 vx == 실제 vx"가 되어 액션 0이 곧 정지가 된다.
    vx = vx - cfg.vx_bias
    # (3c) 요속 편향 보정. **기본 0.0 = 무효**이므로 재기 전까지 동작은 그대로다.
    #      켜는 근거와 정상상태 유도는 `ActionConfig.yaw_bias` 주석.
    yaw = yaw - cfg.yaw_bias

    cmd = _pack(vx, vy, yaw, v[3], v[4], v[5], v[6], v[7], cfg)

    # (4) 저역필터
    return cfg.lowpass_alpha * cmd + (1.0 - cfg.lowpass_alpha) * prev_cmd


def _pack(vx, vy, yaw, height, step_freq, footswing, pitch, stance_width,
          cfg: ActionConfig) -> jnp.ndarray:
    phase, offset, bound = P.GAITS[cfg.gait]
    return jnp.array([
        vx, vy, yaw,
        height, step_freq,
        phase, offset, bound,
        0.5,              # duty  — 상수로 학습됨, 조작 불가
        footswing,
        pitch,
        0.0,              # roll  — 상수로 학습됨, 조작 불가
        stance_width,
        0.45,             # stance_length
        0.0,              # aux
    ])


def action_for(cfg: ActionConfig, **desired) -> jnp.ndarray:
    """원하는 물리값 -> 그 값을 내는 (tanh 이전) 액션. `action_to_command`의 역함수.

        a = action_for(cfg, vx=1.0, yaw=-0.5)     # 나머지는 범위 중앙(=액션 0)

    수동 제어기와 테스트에 쓴다. 범위를 벗어난 값은 경계로 잘린다.
    이름은 액션 순서와 같다: vx, yaw, vy, height, step_freq, footswing, pitch, stance_width.
    """
    names = ("vx", "yaw", "vy", "height", "step_freq", "footswing", "pitch",
             "stance_width")
    out = []
    for name, (lo, hi) in zip(names, cfg.ranges):
        if name not in desired:
            out.append(0.0)                       # 액션 0 = 범위 중앙
            continue
        t = 2.0 * (float(desired[name]) - lo) / (hi - lo) - 1.0   # -> [-1,1]
        t = min(max(t, -0.999), 0.999)            # atanh 발산 방지
        out.append(float(jnp.arctanh(jnp.asarray(t))))
    return jnp.asarray(out, jnp.float32)


def neutral_command(cfg: ActionConfig) -> jnp.ndarray:
    """리셋 시 저역필터의 초기값. 정지·기본 자세."""
    return _pack(0.0, 0.0, 0.0, 0.0, 3.0, 0.08, 0.0, 0.25, cfg)


#: 관측에 넣을 명령 차원(HLC가 실제로 조작하는 8개)의 15D 인덱스
ACTIVE_CMD_IDX = jnp.array([0, 2, 1, 3, 4, 9, 10, 12])


def active_command(cmd15: jnp.ndarray) -> jnp.ndarray:
    """15D 명령에서 HLC가 조작하는 8개만 뽑는다 (관측용)."""
    return cmd15[ACTIVE_CMD_IDX]
