"""HLC 외부 감각 — 2D 라이다.

★ **왜 라이다 하나뿐인가** (2026-08-01 범위 재정의).

원래 설계는 센서 3종(height_scan 7×5, ceiling_scan 5, lidar 16 = 56D)이었다.
그 구성은 "지형의 높낮이를 재서 넘을 방법을 고른다"를 전제한다. 그런데 gap 사다리
실측(`terrain/limits.py`)에서 **8D 명령이 사줄 수 있는 지형 능력이 3 cm**로 나왔다.
3 cm면 넘을 수 있는 지형지물이 없다 — 높이를 재서 분류할 대상 자체가 없다.

그래서 이 로봇에게 지형은 **넘을 대상이 아니라 피할 대상**이고, 필요한 것은
"앞이 막혔나, 어디가 뚫렸나" 하나다. 그게 2D 라이다다.

⚠️ height_scan을 되살려야 하는 유일한 경우는 `limits.py`의 slope/ledge가 나중에
실측되어 **통과 가능한 지형이 생겼을 때**다. 그 전에 미리 만들면 쓰이지 않는
관측 차원 40개를 정책에 얹는 셈이다.

구현 메모 — `mjx.ray`를 쓴다(해석적 박스 교차를 직접 짜지 않는다):
  ① 물리와 센서가 **같은 모델**을 읽으므로 불일치가 원리적으로 없다.
  ② 미로 벽은 박스지만 회전(`geom_quat`)이 섞이는데, 직접 짜면 그게 버그 자리다
     (`terrain.modules.slope`의 램프가 이미 회전 박스다).
  ③ 비용이 문제로 확인되면 그때 해석적 버전으로 교체한다. 인터페이스는
     `lidar_2d` 하나이므로 교체 지점이 한 곳이다.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from mujoco import mjx

from wtw_nav.terrain.modules import TERRAIN_GROUP

#: 광선 수. 360°를 균등 분할하므로 16이면 22.5° 간격이다.
N_BEAMS = 16
#: 최대 사거리 (m). 이 밖은 전부 1.0(= 뚫림)으로 보고한다. 코스 폭·미로 셀 크기와
#: 맞춰야 한다 — 너무 길면 벽 뒤의 벽까지 보여 관측이 둔해지고, 너무 짧으면
#: 정책이 회피를 시작할 시간이 없다. 10 Hz · vx 최대 1.9 m/s면 3 m는 1.6초 앞이다.
MAX_RANGE = 3.0

#: `mjx.ray`의 geomgroup 마스크(길이 6). 지형 그룹만 켠다 — 안 그러면 광선이
#: 로봇 자기 다리에 맞는다 (`modules.TERRAIN_GROUP` 주석).
_GEOMGROUP = tuple(1 if i == TERRAIN_GROUP else 0 for i in range(6))


def _yaw_from_quat(quat: jnp.ndarray) -> jnp.ndarray:
    """몸통 +x축의 월드 방위각. 라이다는 **몸통 기준**이어야 한다.

    월드 기준으로 쏘면 정책이 "북쪽 3 m가 막힘"을 자기 조향과 연결하지 못한다.
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    fwd_x = 1.0 - 2.0 * (y * y + z * z)
    fwd_y = 2.0 * (x * y + w * z)
    return jnp.arctan2(fwd_y, fwd_x)


def beam_angles(n_beams: int = N_BEAMS) -> jnp.ndarray:
    """몸통 정면(0)부터 반시계로 균등 분할한 광선 각도. 정면이 항상 인덱스 0이다."""
    return jnp.linspace(0.0, 2.0 * jnp.pi, n_beams, endpoint=False)


def lidar_2d(mjx_model, data, n_beams: int = N_BEAMS,
             max_range: float = MAX_RANGE) -> jnp.ndarray:
    """몸통 중심에서 수평으로 쏜 `n_beams`개 광선의 정규화 거리.

    Returns:
        `(n_beams,)`, 값은 `[0, 1]`. **1.0 = 사거리 안에 아무것도 없음(뚫림)**,
        0에 가까울수록 코앞이 막혔다는 뜻이다. 부호를 이렇게 잡은 이유는
        미탐지와 원거리가 같은 값이어야 정책이 둘을 구분하려 들지 않기 때문이다.

    ⚠️ 광선 원점은 몸통 z를 그대로 쓴다(≈0.34 m). 벽은 그 높이를 지나야 보인다 —
       발판·턱처럼 낮게 깔린 것은 **원리적으로 안 보인다.** 그건 버그가 아니라
       이 센서의 정의다(위 모듈 주석: 넘을 대상이 아니라 피할 대상만 본다).
    """
    pnt = data.qpos[0:3]
    yaw = _yaw_from_quat(data.qpos[3:7])
    ang = yaw + beam_angles(n_beams)
    vecs = jnp.stack([jnp.cos(ang), jnp.sin(ang), jnp.zeros_like(ang)], axis=-1)

    def one(vec):
        dist, _ = mjx.ray(mjx_model, data, pnt, vec,
                          geomgroup=_GEOMGROUP, flg_static=True)
        return dist

    dist = jax.vmap(one)(vecs)
    # `mjx.ray`는 미탐지에 -1을 준다. 그대로 두면 "코앞이 막힘"보다 작은 값이 되어
    # 의미가 뒤집힌다.
    dist = jnp.where(dist < 0.0, max_range, jnp.minimum(dist, max_range))
    return dist / max_range


def size(n_beams: int = N_BEAMS) -> int:
    """관측 벡터에 더해지는 차원 수."""
    return n_beams


# --------------------------------------------------------------------------
# 진단
# --------------------------------------------------------------------------
def probe(env, xs, y: float = 0.0, yaw: float = 0.0):
    """★ **센서가 실제로 무엇을 보는지** 재본다. 학습 전에 반드시 한 번.

    지정한 x 위치들에 로봇을 놓고(자세는 리셋 자세, 지면 높이만큼 띄움) 라이다를
    쏜다. 물리를 굴리지 않으므로 몇 초면 끝난다.

    ⚠️ **관측에 들어 있다는 것과 정보를 준다는 것은 다르다.** 이 프로젝트에는
    ① 라이다가 로봇 자기 다리에 맞아 전 빔이 0.04 m(정규화 0.013)로 찍힌 사고와
    ② 평지·gap 사다리에서 전 빔이 1.0 상수여서 16D가 통째로 무용이었던 경우가
    둘 다 있다. 어느 쪽이든 그 상태로 학습하면 "센서가 있는데 못 피한다"로
    오진하게 된다. 그래서 주장 대신 이 함수를 돌린다.

    Args:
        env: `NavEnv` (덕 타이핑 — `mjx_model`/`_init_data`/`_init_qpos`/`_ground_z`)
        xs: 조사할 x 좌표들
    Returns:
        `(len(xs), n_beams)` 정규화 거리. 1.0 = 사거리 안에 아무것도 없음.
    """
    import jax

    qpos0 = env._init_qpos
    qvel0 = jnp.zeros_like(env._init_data.qvel)

    def one(x):
        q = qpos0.at[0].set(x).at[1].set(y)
        q = q.at[2].add(env._ground_z(x))
        q = q.at[3:7].set(jnp.array([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)]))
        d = mjx.forward(env.mjx_model, env._init_data.replace(qpos=q, qvel=qvel0))
        return lidar_2d(env.mjx_model, d)

    return jax.vmap(one)(jnp.asarray(xs, jnp.float32))


def probe_report(env, xs, **kw) -> None:
    """`probe`를 돌리고 사람이 읽을 표로 찍는다. 판정 문구까지 낸다."""
    import numpy as np

    r = np.asarray(probe(env, xs, **kw))
    xs = np.asarray(xs, float)
    gz = np.asarray([float(env._ground_z(jnp.asarray(x, jnp.float32))) for x in xs])

    #: 정면(0)과 좌우 22.5°(1, 15)가 진행 방향 정보를 담는다.
    print(f"{'x':>7s} {'지면z':>6s} {'몸통z':>6s} | "
          f"{'좌22°':>6s} {'정면':>6s} {'우22°':>6s} | {'최소':>6s} {'최소빔':>6s}")
    print("-" * 66)
    for i, x in enumerate(xs):
        b = r[i]
        print(f"{x:7.2f} {gz[i]:6.2f} {gz[i] + 0.34:6.2f} | "
              f"{b[1]:6.3f} {b[0]:6.3f} {b[15]:6.3f} | "
              f"{b.min():6.3f} {int(b.argmin()):6d}")

    #: ★ 이 지형들은 **원리적으로** 몸통 높이(≈0.34 m)를 지나지 않는다. 전 빔 1.0이
    #: 정상이므로 고장으로 보고하면 안 된다 (2026-08-06: 요철 실행에서 오경보가
    #: 나서 "지형이 안 만들어졌나"를 의심하게 만들었다).
    _BLIND = ("gap", "ledge", "rough")
    kind = getattr(getattr(getattr(env, "cfg", None), "terrain", None), "kind", None)

    print()
    if float(r.min()) > 0.999:
        if kind in _BLIND:
            print(f"  ✓ 전 빔이 1.0입니다 — 지형 '{kind}'에서는 **이것이 정상**입니다. "
                  "광선이 몸통 높이에서 수평으로 나가는데 이 지형은 바닥에 깔려 "
                  "있습니다(`lidar_2d` 주석). 관측 16D는 여기서 상수이므로, "
                  "이 측정이 주는 값은 '**보지 못하는 채로** 8D 명령이 낼 수 있는 "
                  "상한'입니다. 라이다를 의심할 필요가 없습니다.")
        else:
            print("  ✗ 전 빔이 1.0입니다 — **센서가 아무것도 못 봅니다.** 관측 16D가 "
                  "상수라 정책 입장에서는 없는 것과 같습니다. 지형 geom의 group이 "
                  f"{TERRAIN_GROUP}인지(`modules._finish`), 벽/램프가 몸통 높이"
                  "(≈0.34 m)를 지나는지 확인하십시오.")
    elif float(np.median(r)) < 0.05:
        print("  ✗ 대부분의 빔이 코앞(0에 가까움)입니다 — **광선이 로봇 자기 몸에 "
              "맞고 있습니다.** `modules.TERRAIN_GROUP`이 로봇 geom 그룹과 겹칩니다 "
              "(`assert_group_free` 참조).")
    else:
        seen = int((r.min(axis=1) < 0.999).sum())
        print(f"  ✓ 센서가 살아 있습니다 — {len(xs)}개 위치 중 {seen}곳에서 감지, "
              f"최소 {r.min():.3f} / 최대 {r.max():.3f}.")
        print("    정면 빔이 램프에 가까워질수록 줄어드는지 표에서 확인하십시오. "
              "줄지 않으면 보는 것은 램프가 아니라 다른 것입니다.")
