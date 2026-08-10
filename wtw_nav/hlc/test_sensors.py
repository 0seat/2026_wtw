"""`hlc/sensors.py` 게이트.

주의 — **음성 대조군이 이 파일의 존재 이유다.** 2026-07-29에 폐기된 손 스윕은 다리 폭을
2.00 -> 0.30 m로 6배 바꿔도 결과가 소수점 셋째 자리까지 같았다 — 즉 측정하려던 값이
결과에 관여한 적이 없었는데 그걸 몇 시간 뒤에야 알았다. 센서는 같은 실패를 하기
가장 쉬운 코드다(항상 그럴듯한 숫자를 뱉으므로). 그래서 여기서는 값이 맞는지뿐
아니라 **벽을 옮기면 값이 따라 움직이는지**를 반드시 확인한다.

    python -m wtw_nav.hlc.test_sensors
"""

from __future__ import annotations

import math
import sys

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from wtw_nav.hlc import sensors
from wtw_nav.terrain.modules import TERRAIN_GROUP

_N_PASS = 0
_N_FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _N_PASS, _N_FAIL
    if ok:
        _N_PASS += 1
        print(f"  PASS  {name}" + (f"   {detail}" if detail else ""))
    else:
        _N_FAIL += 1
        print(f"  FAIL  {name}" + (f"   {detail}" if detail else ""))


def _box_world(wall_x: float, wall_y: float = 0.0, half: float = 0.5):
    """자유 부유체 1개 + 벽 박스 1개짜리 최소 모델.

    Go1을 쓰지 않는다 — 벽까지의 정답 거리를 손으로 적을 수 있어야 하기 때문이다.
    """
    # 주의 — `<asset>`의 재질을 지우지 말 것. `mjx.ray`는 투명 geom을 거르려고
    #    `m.mat_rgba[m.geom_matid, 3]`을 인덱싱하는데, 재질이 하나도 없으면
    #    mat_rgba가 (0,4)라 게더가 범위를 벗어나 죽는다("Slice size ... out of range").
    #    menagerie scene.xml에는 재질이 있으므로 실모델에서는 안 나는 문제다.
    xml = f"""
    <mujoco>
      <asset><material name="dummy" rgba="1 1 1 1"/></asset>
      <worldbody>
        <geom name="wall" type="box" group="{TERRAIN_GROUP}"
              pos="{wall_x} {wall_y} 0.5" size="{half} {half} 0.5"/>
        <body name="probe" pos="0 0 0.34">
          <freejoint/>
          <geom name="probe_g" type="sphere" size="0.05" group="0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mx, dx = mjx.put_model(m), mjx.put_data(m, d)
    return mx, dx


def _set_pose(dx, x=0.0, y=0.0, yaw=0.0):
    q = dx.qpos.at[0].set(x).at[1].set(y).at[2].set(0.34)
    q = q.at[3:7].set(jnp.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]))
    return dx.replace(qpos=q)


def main() -> int:
    print("=" * 64)
    print("센서 게이트 — hlc/sensors.py (2D 라이다)")
    print("=" * 64)

    lidar = jax.jit(lambda mx, dx: sensors.lidar_2d(mx, dx))

    print("\n[1] 기하 — 정면 3 m 앞 벽 (반폭 0.5 -> 앞면 x=2.5)")
    mx, dx = _box_world(wall_x=3.0)
    dx = _set_pose(dx, 0.0, 0.0, 0.0)
    r = np.asarray(lidar(mx, dx))
    check("차원 == N_BEAMS", r.shape == (sensors.N_BEAMS,), f"{r.shape}")
    check("값이 전부 [0,1]", bool(np.all((r >= 0.0) & (r <= 1.0))),
          f"[{r.min():.3f}, {r.max():.3f}]")
    # 정면(빔 0)은 x=2.5에서 맞는다 -> 2.5/3.0
    check("정면 빔 = 2.5 m / 3.0", abs(r[0] - 2.5 / 3.0) < 1e-3,
          f"{r[0]:.4f} (기대 {2.5/3.0:.4f})")
    # 뒤쪽(빔 8 = 180°)은 아무것도 없다 -> 1.0
    check("후방 빔 = 미탐지 1.0", abs(r[sensors.N_BEAMS // 2] - 1.0) < 1e-6,
          f"{r[sensors.N_BEAMS//2]:.4f}")

    print("\n[2] ★ 음성 대조군 — 벽을 옮기면 값이 따라 움직이는가")
    reads = {}
    for wx in (1.5, 2.0, 3.0, 4.0):
        mx_i, dx_i = _box_world(wall_x=wx)
        reads[wx] = float(np.asarray(lidar(mx_i, _set_pose(dx_i)))[0])
    seq = [reads[w] for w in (1.5, 2.0, 3.0, 4.0)]
    check("벽이 멀어지면 정면 값이 단조 증가",
          all(a < b - 1e-4 for a, b in zip(seq, seq[1:])),
          " -> ".join(f"{v:.3f}" for v in seq))
    # 앞면 = wx - 0.5. 사거리 3 m 안이면 (wx-0.5)/3
    for wx in (1.5, 2.0, 3.0):
        check(f"  벽 x={wx} -> {(wx-0.5)/3.0:.3f}",
              abs(reads[wx] - (wx - 0.5) / 3.0) < 1e-3, f"{reads[wx]:.4f}")
    check("벽 x=4.0(앞면 3.5 m)은 사거리 밖 -> 1.0",
          abs(reads[4.0] - 1.0) < 1e-6, f"{reads[4.0]:.4f}")

    print("\n[3] 몸통 기준인가 — 로봇이 돌면 벽도 따라 돌아야 한다")
    mx, dx = _box_world(wall_x=3.0)
    r0 = np.asarray(lidar(mx, _set_pose(dx, yaw=0.0)))
    #  +90° 돌면 정면에 있던 벽은 **우측**(빔 -90° = 인덱스 12)으로 간다
    r90 = np.asarray(lidar(mx, _set_pose(dx, yaw=math.pi / 2)))
    q = sensors.N_BEAMS // 4
    check("yaw +90° 후 정면 빔은 뚫림", abs(r90[0] - 1.0) < 1e-6, f"{r90[0]:.4f}")
    check("yaw +90° 후 벽은 우측 빔(-90°)에서 보임",
          abs(r90[-q] - r0[0]) < 1e-3, f"{r90[-q]:.4f} vs 정면이던 {r0[0]:.4f}")

    print("\n[4] 자기 몸에 맞지 않는가 (그룹 필터)")
    # probe_g는 group 0이므로 지형 그룹 필터에 걸러져야 한다. 걸러지지 않으면
    # 원점에서 반지름 0.05에 맞아 전 빔이 ~0.017로 찍힌다.
    check("전 빔이 자기 반지름(0.05/3=0.017)보다 큼", bool(r0.min() > 0.1),
          f"min={r0.min():.4f}")

    print("\n[5] vmap — 학습 경로에서 배치로 돌아가는가")
    vl = jax.jit(jax.vmap(lambda d: sensors.lidar_2d(mx, d)))
    batch = jax.vmap(lambda x: _set_pose(dx, x=x))(jnp.linspace(-1.0, 1.0, 8))
    rb = np.asarray(vl(batch))
    check("vmap 동작", rb.shape == (8, sensors.N_BEAMS), f"{rb.shape}")
    check("env마다 값이 다름 (배치가 실제로 갈라짐)",
          float(np.std(rb[:, 0])) > 1e-3, f"std={np.std(rb[:,0]):.4f}")
    check("전진할수록 정면 거리 감소",
          bool(np.all(np.diff(rb[:, 0]) < 1e-6)),
          f"{rb[0,0]:.3f} -> {rb[-1,0]:.3f}")

    print("\n" + "=" * 64)
    print(f"{_N_PASS}/{_N_PASS + _N_FAIL} PASS")
    return 0 if _N_FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
