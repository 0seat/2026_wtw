"""격자 미로 + BFS 거리장 (P4). **범위 재정의 이후 이것이 연구의 본체다.**

`docs/decisions/0001-지형-돌파-폐기.md`: 8D 명령의 지형 능력이 3 cm로 실측되어 "지형 돌파"는
폐기됐다. 남은 것은 **막힌 곳을 피해 목표까지 가는 것**이고, 그 무대가 여기다.

구조 — 얇은 벽 격자 (thin-wall grid):

    벽면은 x = i·P (i = 0..n), 두께 WALL_T. 셀 i는 그 사이의 폭 CELL 구간이다.
        P = CELL + WALL_T          (피치)
        셀 (i,j) 중심 = ((i+0.5)P, (j+0.5)P)

    두꺼운 벽(점유 격자)이 아니라 얇은 벽인 이유: 통로 폭과 벽 두께를 따로 정할
    수 있어야 한다. 통로는 로봇 폭(0.3 m)과 표류(유지기 적용 시 |y|max 0.22 m)로
    정해지고, 벽 두께는 그것과 무관하다.

⚠️ **벽 높이는 라이다보다 높아야 한다.** 라이다는 몸통 z(≈0.34 m)에서 수평으로
   쏘므로(`hlc/sensors.py`) 그보다 낮은 벽은 **원리적으로 안 보인다**. 안 보이는
   벽에 부딪히는 정책을 학습시키면 "센서가 있는데도 못 피한다"로 오진하게 된다.

⚠️ **미로는 env마다 다르게 만들 수 없다.** MJX 모델 하나를 전 env가 공유하므로
   (`terrain.modules.ladder`의 ① 참조) 레이아웃은 `__init__`에서 하나 고정하고,
   랜덤화는 출발 셀·자세로 준다. 여러 레이아웃을 보려면 시드를 바꿔 **따로**
   학습·평가한다.
"""

from __future__ import annotations

import collections
import math

import mujoco
import numpy as np

from wtw_nav.terrain.modules import FRICTION, TERRAIN_GROUP

#: 통로 폭 (m). 로봇 폭 0.3 m + 유지기 적용 시 표류 |y|max 0.22 m. 1.2면 여유 3배.
#: 이보다 좁히면 실패가 "길 못 찾음"이 아니라 "벽 긁음"이 되어 지표가 흐려진다.
CELL = 1.2
#: 벽 두께 (m). 물리적 의미는 없고 얇을수록 유효 면적이 넓다. 너무 얇으면
#: 빠른 접촉에서 터널링이 나므로 발 반지름(0.02) 수준 아래로 내리지 말 것.
WALL_T = 0.2
#: 벽 높이 (m). ★ 라이다 높이(몸통 z ≈ 0.34)보다 확실히 높아야 한다. 모듈 주석 참조.
WALL_H = 0.6

PITCH = CELL + WALL_T


# --------------------------------------------------------------------- 생성
def generate(n: int, seed: int = 0, loop_prob: float = 0.1):
    """재귀 백트래커로 n×n 완전미로를 만든 뒤 벽 일부를 더 튼다.

    Args:
        loop_prob: 추가로 허무는 벽의 비율. 0이면 **완전미로**(순환 없음)라
            임의의 두 지점 사이 경로가 유일하다. 그러면 BFS 유도를 그대로 따라가는
            것 외에 다른 해가 없어 "길찾기"가 아니라 "선 따라가기" 과제가 된다.
            0.1이면 갈림길이 생겨 거리장이 실제로 의미를 갖는다.

    Returns:
        `(vw, hw)`.
          `vw[j, i]` = 셀 (i-1,j)와 (i,j) 사이 **수직** 벽 유무. shape (n, n+1)
          `hw[j, i]` = 셀 (i,j-1)와 (i,j) 사이 **수평** 벽 유무. shape (n+1, n)
        경계(i=0, i=n, j=0, j=n)는 항상 벽이다.
    """
    rng = np.random.default_rng(seed)
    vw = np.ones((n, n + 1), dtype=bool)
    hw = np.ones((n + 1, n), dtype=bool)

    seen = np.zeros((n, n), dtype=bool)
    stack = [(0, 0)]
    seen[0, 0] = True
    while stack:
        i, j = stack[-1]
        nbrs = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < n and 0 <= b < n and not seen[a, b]:
                nbrs.append((a, b, di, dj))
        if not nbrs:
            stack.pop()
            continue
        a, b, di, dj = nbrs[rng.integers(len(nbrs))]
        if di == 1:
            vw[j, i + 1] = False
        elif di == -1:
            vw[j, i] = False
        elif dj == 1:
            hw[j + 1, i] = False
        else:
            hw[j, i] = False
        seen[a, b] = True
        stack.append((a, b))

    # 순환 추가 — 내부 벽만 대상으로 한다(경계는 뚫으면 미로 밖으로 나간다).
    if loop_prob > 0:
        for j in range(n):
            for i in range(1, n):
                if vw[j, i] and rng.random() < loop_prob:
                    vw[j, i] = False
        for j in range(1, n):
            for i in range(n):
                if hw[j, i] and rng.random() < loop_prob:
                    hw[j, i] = False
    return vw, hw


def open_dirs(vw, hw, i: int, j: int):
    """셀 (i,j)에서 갈 수 있는 이웃들. BFS와 유도장이 같은 정의를 써야 한다."""
    out = []
    if not vw[j, i]:
        out.append((i - 1, j))
    if not vw[j, i + 1]:
        out.append((i + 1, j))
    if not hw[j, i]:
        out.append((i, j - 1))
    if not hw[j + 1, i]:
        out.append((i, j + 1))
    return out


# ----------------------------------------------------------------- 거리장
def distance_field(vw, hw, goal: tuple[int, int]) -> np.ndarray:
    """목표 셀로부터의 BFS 홉 수. 도달 불가 셀은 `-1`.

    ★ 이 값이 **보상의 퍼텐셜**이다. 유클리드 거리를 쓰면 안 된다 — 미로에서는
    목표에 가까워지려면 일단 멀어져야 하는 구간이 반드시 있고, 유클리드 퍼텐셜은
    그 구간을 벌한다. 그러면 정책은 막다른 골목 앞에서 벽에 붙어 진동한다.
    """
    n = vw.shape[0]
    dist = -np.ones((n, n), dtype=np.int32)
    dist[goal] = 0
    q = collections.deque([goal])
    while q:
        i, j = q.popleft()
        for a, b in open_dirs(vw, hw, i, j):
            if dist[a, b] < 0:
                dist[a, b] = dist[i, j] + 1
                q.append((a, b))
    return dist


def flow_field(vw, hw, dist: np.ndarray) -> np.ndarray:
    """각 셀에서 **거리가 줄어드는 이웃**을 향하는 단위 벡터. shape (n, n, 2).

    목표 셀과 고립 셀은 (0,0)이다.
    """
    n = dist.shape[0]
    flow = np.zeros((n, n, 2), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if dist[i, j] <= 0:
                continue
            best, bd = None, dist[i, j]
            for a, b in open_dirs(vw, hw, i, j):
                if 0 <= dist[a, b] < bd:
                    best, bd = (a, b), dist[a, b]
            if best is not None:
                flow[i, j] = (best[0] - i, best[1] - j)
    return flow


def cell_center(i, j) -> tuple[float, float]:
    return ((i + 0.5) * PITCH, (j + 0.5) * PITCH)


# -------------------------------------------------------------------- 빌드
def _runs(flags) -> list[tuple[int, int]]:
    """1인 구간을 **최대 길이로** 묶어 `(시작, 끝)` 목록으로. 끝은 포함이다."""
    out, s = [], None
    for k, v in enumerate(flags):
        if v and s is None:
            s = k
        elif not v and s is not None:
            out.append((s, k - 1)); s = None
    if s is not None:
        out.append((s, len(flags) - 1))
    return out


def _wall_boxes(vw, hw, n: int) -> list[tuple[str, float, float, float, float]]:
    """벽 슬롯을 **일직선 구간마다 박스 하나로** 합친다.

    ★ **처리량 문제다, 미관 문제가 아니다** (2026-08-09). 슬롯마다 geom을 하나씩
    놓으면 5×5 미로가 벽 geom 최대 60개가 되는데, MJX는 충돌 후보 쌍을 컴파일
    시점에 정적으로 펼쳐 **매 스텝 전부** 계산한다(런타임 컬링 없음). 즉 후보 쌍은
    로봇 geom × 월드 geom이므로 평지(바닥 1개) 대비 ~50배가 되고, 실제로
    P4 첫 학습에서 평지 3,096 HLC steps/s가 미로에서 40 미만으로 떨어졌다
    (`docs/01_llc.md` §8.4이 충돌을 처리량의 지배 요인으로 지목한 그대로다).
    라이다 `mjx.ray`도 geom 수에 선형이라 같은 방향으로 손해다.

    합쳐도 **형상은 완전히 같다** — 병합은 인접한 동일 직선 위의 박스들을 하나로
    바꿀 뿐이고, 아래에서 양 끝의 `half` 돌출을 그대로 유지하므로 모서리 구멍도
    생기지 않는다(그 돌출이 없으면 라이다가 교차점 틈으로 벽 뒤를 본다).

    바깥 둘레는 항상 이어져 있으므로 **네 변이 각각 박스 하나**가 된다.

    Returns: `(name, cx, cy, sx, sy)` 목록. `sx`/`sy`는 반길이다.
    """
    half = WALL_T / 2.0
    out = []
    # 세로벽: x = i·PITCH 고정, j 방향으로 이어진 구간을 묶는다.
    for i in range(n + 1):
        for j0, j1 in _runs([bool(vw[j, i]) for j in range(n)]):
            out.append((f"v{i}_{j0}_{j1}",
                        i * PITCH, (j0 + j1 + 1) / 2.0 * PITCH,
                        half, (j1 - j0 + 1) * PITCH / 2.0 + half))
    # 가로벽: y = j·PITCH 고정, i 방향.
    for j in range(n + 1):
        for i0, i1 in _runs([bool(hw[j, i]) for i in range(n)]):
            out.append((f"h{j}_{i0}_{i1}",
                        (i0 + i1 + 1) / 2.0 * PITCH, j * PITCH,
                        (i1 - i0 + 1) * PITCH / 2.0 + half, half))
    return out


def build(n: int = 5, seed: int = 0, loop_prob: float = 0.1,
          goal: tuple[int, int] | None = None,
          xml: str = "mujoco_menagerie/unitree_go1/scene.xml",
          max_geom_pairs: int = 0, max_contact_points: int = 0):
    """미로 MJX 모델과 유도에 필요한 배열들을 만든다.

    ⚠️ `terrain.modules._spec()`을 쓰지 않는다 — 그건 바닥 평면을 **지운다**(틈을
    뚫기 위해서였다). 미로에는 구멍이 없으므로 평면을 남겨야 한다.

    Returns:
        `(mj_model, meta)`. `meta`는 dict:
          n, pitch, vw, hw, dist(셀 홉), flow(n,n,2), goal_cell, goal_xy,
          start_cell, start_xy, path_len_m(출발 셀의 경로 길이, m)
    """
    vw, hw = generate(n, seed=seed, loop_prob=loop_prob)
    goal = goal or (n - 1, n - 1)
    dist = distance_field(vw, hw, goal)
    if int(dist.min()) < 0:
        raise ValueError("도달 불가 셀이 있습니다 — 생성기 버그입니다 "
                         f"(고립 {int((dist < 0).sum())}개)")
    flow = flow_field(vw, hw, dist)

    spec = mujoco.MjSpec.from_file(xml)

    def _wall(name, cx, cy, sx, sy):
        g = spec.worldbody.add_geom()
        g.name = name
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [sx, sy, WALL_H / 2.0]
        g.pos = [cx, cy, WALL_H / 2.0]
        g.condim = 3
        g.friction = list(FRICTION)
        g.group = TERRAIN_GROUP
        return g

    half = WALL_T / 2.0
    # 벽 조각은 피치 + 두께만큼 잡아 **모서리에서 겹치게** 한다. 딱 맞추면 교차점에
    # 두께만 한 구멍이 남고, 라이다 광선이 그 틈으로 빠져나가 벽 뒤를 본다.
    n_slot = int(vw.sum() + hw.sum())
    boxes = _wall_boxes(vw, hw, n)
    for name, cx, cy, sx, sy in boxes:
        _wall(name, cx, cy, sx, sy)
    n_geom = len(boxes)

    from wtw_nav.terrain.modules import set_broadphase
    set_broadphase(spec, max_geom_pairs, max_contact_points)

    m = spec.compile()
    # 벽 geom 수는 **정적** 후보 쌍을 정한다. `max_geom_pairs`를 켜면 그중 가까운
    # k쌍만 좁은단계를 도므로, 이 수가 커져도 비싼 부분은 늘지 않는다
    # (`configs.HLCConfig.max_geom_pairs`).
    print(f"  벽 geom {n_slot} -> **{n_geom}개** (직선 구간 병합, "
          f"{1 - n_geom / max(n_slot, 1):.0%} 감소)")
    from wtw_nav.terrain.modules import assert_group_free, drop_cylinder_collisions
    drop_cylinder_collisions(m)
    # `modules._finish`를 쓰지 않으므로(바닥을 남겨야 한다) 그룹 검사는 직접 한다.
    assert_group_free(m)

    start = (0, 0)
    meta = {
        "n": n, "pitch": PITCH, "vw": vw, "hw": hw,
        "dist": dist, "flow": flow,
        "goal_cell": goal, "goal_xy": cell_center(*goal),
        "start_cell": start, "start_xy": cell_center(*start),
        "path_len_m": float(dist[start]) * PITCH,
        "seed": seed,
        #: ★ 처리량 지표. 충돌 후보 쌍 = 로봇 geom(≈31) × 이 수.
        "wall_geoms": n_geom, "wall_slots": n_slot,
    }
    return m, meta


# --------------------------------------------------------------------- 표시
def render_ascii(vw, hw, dist=None) -> str:
    """미로를 글자로 그린다. `dist`를 주면 셀에 홉 수를 적는다.

    학습을 돌리기 전에 **눈으로 확인하기 위한** 것이다. 미로가 잘못 생성된 채로
    6시간을 태우는 것이 이 프로젝트에서 가장 비싼 실수였다.
    """
    n = vw.shape[0]
    out = []
    for j in range(n, -1, -1):
        row = ""
        for i in range(n):
            row += "+" + ("---" if j < n + 1 and hw[j, i] else "   ")
        out.append(row + "+")
        if j == 0:
            break
        jj = j - 1
        row = ""
        for i in range(n):
            row += ("|" if vw[jj, i] else " ")
            row += f"{dist[i, jj]:^3d}" if dist is not None else "   "
        row += "|" if vw[jj, n] else " "
        out.append(row)
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    vw, hw = generate(n, seed=seed)
    d = distance_field(vw, hw, (n - 1, n - 1))
    print(render_ascii(vw, hw, d))
    print(f"\n피치 {PITCH:.2f} m (통로 {CELL}, 벽 {WALL_T}), 크기 "
          f"{n * PITCH:.1f} × {n * PITCH:.1f} m")
    print(f"출발(0,0) -> 목표({n-1},{n-1}): {int(d[0, 0])} 홉 "
          f"= {int(d[0, 0]) * PITCH:.1f} m")
