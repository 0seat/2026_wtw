"""지형 모듈 — MJCF를 `MjSpec`으로 조립한다.

⚠️ **`<include>`로 만들지 말 것.** menagerie `go1.xml`은 `meshdir="assets"`(상대경로)를
선언하는데, MuJoCo는 이를 **주 모델 파일의 디렉터리** 기준으로 푼다. 다른 위치에서
절대경로로 include하면 `Error opening file '.../thigh_mirror.stl'`로 죽는다.
`MjSpec.from_file(scene.xml)` 후 worldbody를 편집하면 경로 문제가 없다 (mujoco>=3.2).

여기서 만드는 지형은 **측정용**이다. P2의 실제 지형 모듈은 이 위에 올린다.
"""

from __future__ import annotations

import mujoco

#: 기본 발판 두께·폭 (m). 두께는 로봇이 뚫고 떨어지지 않을 만큼만.
PLATFORM_T = 0.10
#: ⚠️ **넉넉해야 한다.** LLC는 직진 명령에도 완만히 선회한다 — vx=0.8에서 y가
#: 4초에 1.4 m 흘렀다. 무한 평면(menagerie scene)에서는 안 보이던 특성이라
#: 반폭 1.5 m로 잡았다가 대조군이 **발판 옆으로 떨어졌다**(z=-12 m).
#: 지형 실측이 "지형 때문에 실패"인지 "옆으로 새서 실패"인지 섞이면 안 된다.
PLATFORM_W = 6.0
#: menagerie floor와 같은 마찰 (go1.xml `<geom friction="0.6">`)
FRICTION = (0.6, 0.005, 0.0001)


def drop_cylinder_collisions(mj_model) -> int:
    """★ MJX는 **CYLINDER–BOX 충돌을 구현하지 않았다.** 실린더 충돌을 끈다.

    지형을 박스(또는 heightfield)로 만들면 `mjx.put_model`이
    `NotImplementedError: (mjGEOM_CYLINDER, mjGEOM_BOX) collisions not implemented`
    로 죽는다. MJX 지원표(`mjx._src.collision_driver._COLLISION_FUNC`)상
    CYLINDER는 **PLANE하고만** 붙고, BOX·HFIELD와는 붙지 않는다.

    다행히 Go1의 실린더 12개는 전부 **hip과 trunk**에 있고, 지형과 실제로 닿는
    부위는 전부 지원되는 형상이다:

        발       SPHERE  (calf)   — SPHERE–BOX  지원
        종아리   CAPSULE (calf)   — CAPSULE–BOX 지원
        허벅지   CAPSULE (thigh)  — CAPSULE–BOX 지원
        몸통     BOX/CAPSULE      — BOX–BOX     지원

    따라서 실린더를 꺼도 **지형 접촉 판정은 온전하다.** 평지 대조군으로 이 변경의
    영향을 확인할 것 (`measure.sanity()`).

    Returns: 비활성화한 geom 수
    """
    cyl = mj_model.geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER
    n = int(((mj_model.geom_contype != 0) | (mj_model.geom_conaffinity != 0))[cyl].sum())
    mj_model.geom_contype[cyl] = 0
    mj_model.geom_conaffinity[cyl] = 0
    return n


def _spec(xml: str = "mujoco_menagerie/unitree_go1/scene.xml"):
    """scene.xml을 열고 **바닥 평면을 제거한** spec을 준다.

    평면을 남기면 틈이 뚫리지 않는다 — 로봇이 허공을 밟고 걸어간다.
    """
    spec = mujoco.MjSpec.from_file(xml)
    for g in list(spec.worldbody.geoms):
        if g.name == "floor":
            spec.delete(g)
    return spec


def set_broadphase(spec, max_geom_pairs: int = 0, max_contact_points: int = 0) -> int:
    """★ MJX의 **근사 broadphase**를 켠다. 켠 항목 수를 돌려준다.

    `mjx._src.collision_driver`가 custom numeric `max_geom_pairs`를 읽어
    경계구 거리 `‖p2-p1‖ - (rbound1+rbound2)` 기준 **가장 가까운 k쌍만**
    좁은단계로 보낸다(`top_k`). 근거·주의는 `configs.HLCConfig.max_geom_pairs`.

    ⚠️ **컴파일 전 spec에 넣어야 한다.** 컴파일된 `mj_model`의 `numeric_data`는
    크기가 고정이라 나중에 추가할 수 없다.

    ⚠️ mujoco 버전에 따라 `add_numeric` 시그니처가 다를 수 있다. 조용히 넘어가면
    "켰다고 생각했는데 안 켜진" 상태로 측정하게 되므로 **예외를 그대로 올린다.**
    """
    n = 0
    for name, v in (("max_geom_pairs", max_geom_pairs),
                    ("max_contact_points", max_contact_points)):
        if v and v > 0:
            spec.add_numeric(name=name, data=[float(v)], size=1)
            n += 1
    return n


def broadphase_report(mj_model) -> dict:
    """모델에 broadphase numeric이 **실제로 들어갔는지** 읽어서 확인한다.

    설정만 하고 안 들어간 채 속도가 안 나오면 "MJX에 broadphase가 없다"는
    엉뚱한 결론으로 간다 — 실제로 그렇게 한 번 틀렸다.
    """
    import mujoco

    out = {}
    for k in ("max_geom_pairs", "max_contact_points"):
        i = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_NUMERIC, k)
        out[k] = int(mj_model.numeric_data[mj_model.numeric_adr[i]]) if i >= 0 else None
    # ⚠️ `None`은 **설정을 0으로 둬서 안 넣은 것**과 구분되지 않는다. 여기서는
    #    모델만 읽으므로 의도를 알 수 없다 -> 경고 대신 사실만 적는다.
    #    (2026-08-10에 `max_contact_points=0`인 정상 상태를 "미적용"으로 찍어
    #     문제인 줄 알게 만들었다.)
    on = {k: v for k, v in out.items() if v is not None}
    off = [k for k, v in out.items() if v is None]
    print(f"  broadphase 적용 {on or '없음'}"
          + (f" / 미설정 {off} (설정이 0이면 정상)" if off else ""))
    return out


def _slab(spec, name: str, x0: float, x1: float, top_z: float):
    """[x0, x1] 구간에 윗면이 `top_z`인 발판을 놓는다."""
    g = spec.worldbody.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [(x1 - x0) / 2.0, PLATFORM_W, PLATFORM_T / 2.0]
    g.pos = [(x0 + x1) / 2.0, 0.0, top_z - PLATFORM_T / 2.0]
    g.condim = 3
    g.friction = list(FRICTION)
    return g


#: ★ 지형 geom을 담는 geom group. `hlc/sensors.py`의 라이다가 **이 그룹만** 쏜다.
#: 안 나누면 광선이 로봇 자기 다리에 맞아 전 방향이 0.04 m로 찍힌다 (`mjx.ray`의
#: `bodyexclude`는 body 하나만 빼주는데 Go1은 trunk+다리 13개 body다).
#:
#: ⚠️ **3을 쓰면 안 된다.** menagerie Go1은 자기 충돌 geom 42개에 group 3을 쓴다
#: (실제로 3으로 뒀다가 라이다 16빔이 전부 로봇 자신에 맞았다, 2026-08-01).
#: scene.xml이 쓰는 그룹: 0(바닥) / 2(시각) / 3(충돌). 5는 비어 있다.
#: 모델을 바꾸면 `assert_group_free()`가 잡아 준다.
TERRAIN_GROUP = 5


def assert_group_free(mj_model) -> None:
    """로봇 모델이 `TERRAIN_GROUP`을 쓰고 있지 않은지 확인한다.

    이걸 조용히 넘기면 라이다가 지형 대신 자기 다리를 재고, 그 값은 그럴듯해서
    (0~1 사이 실수) 학습을 다 돌리고 나서야 이상하다는 것을 알게 된다.
    """
    import numpy as np
    used = np.asarray(mj_model.geom_group)[np.asarray(mj_model.geom_bodyid) != 0]
    if bool((used == TERRAIN_GROUP).any()):
        raise ValueError(
            f"로봇 geom이 지형 그룹 {TERRAIN_GROUP}을 쓰고 있습니다 — 라이다가 "
            f"자기 몸을 잰다. modules.TERRAIN_GROUP을 비어 있는 값으로 바꾸십시오 "
            f"(로봇이 쓰는 그룹: {sorted(set(used.tolist()))}).")


def _finish(spec):
    """컴파일 + MJX 미지원 충돌 제거 + 지형 geom 그룹 표시. 모든 빌더가 거친다."""
    m = spec.compile()
    drop_cylinder_collisions(m)
    # `_spec()`이 floor 평면을 지우므로 **worldbody(bodyid 0)에 남은 geom은 전부
    # 우리가 놓은 지형**이다. 빌더마다 group을 적어주는 대신 여기서 한 번에 칠하면
    # 새 빌더를 추가할 때 빠뜨릴 수 없다.
    m.geom_group[m.geom_bodyid == 0] = TERRAIN_GROUP
    assert_group_free(m)
    return m


def flat(length: float = 12.0):
    """대조군 — 틈도 턱도 없는 발판. 지형 자체의 부작용을 분리하기 위해 필요하다."""
    spec = _spec()
    _slab(spec, "ground", -2.0, length, 0.0)
    return _finish(spec)


#: ⚠️ 착지 발판 길이. **넉넉해야 한다.** 4 m로 뒀다가 vx=1.4 조합이 8초에
#: x=9.6까지 가서 **발판 끝을 지나 떨어졌고**, 그것이 "2 cm 턱에서 낙하"로
#: 집계됐다. 측정이 통째로 거짓이 될 뻔했다. seconds×vx_max 이상으로 잡을 것.
RUN_OUT = 16.0


def gap(width: float, x0: float = 3.0, run_out: float = RUN_OUT):
    """★ 실측 ② — 폭 `width` m의 **틈**. 착지 발판은 같은 높이.

    도약이 불가능함은 확정됐으므로(`docs/03_results.md` §2), 여기서 재는 것은
    **step-over** — 다리를 뻗어 반대편에 발을 얹고 넘어가는 것 — 의 한계 폭이다.
    이 값이 미로 설계의 틈 상한을 정한다.

    Args:
        width: 틈 폭 (m)
        x0: 틈 시작 x. 로봇은 x=0에서 출발하므로 가속 구간이 된다.
    """
    spec = _spec()
    _slab(spec, "near", -2.0, x0, 0.0)
    _slab(spec, "far", x0 + width, x0 + width + run_out, 0.0)
    return _finish(spec)


def ledge(height: float, x0: float = 3.0, run_out: float = RUN_OUT):
    """★ 실측 ③ — 높이 `height` m의 **턱**(위로 올라서기).

    LLC는 지형을 보지 못하므로(`measure_heights=False`) 발 착지점을 턱에 맞출 수
    없다. 넘을 수 있다면 그것은 `footswing`을 올려 발을 더 든 결과일 뿐이다.
    따라서 이 측정은 **HLC 명령만으로 감당 가능한 요철의 상한**을 준다.
    """
    spec = _spec()
    _slab(spec, "lower", -2.0, x0, 0.0)
    _slab(spec, "upper", x0, x0 + run_out, height)
    return _finish(spec)


#: 요철 타일 한 변 (m, x방향). 발끝 구(반경 ~0.02 m)보다 충분히 커야 발이 타일
#: **위에** 놓인다 — 타일이 발보다 작으면 재는 것이 지형이 아니라 접촉 이산화다.
ROUGH_TILE = 0.20
#: 요철을 까는 중앙 띠의 **반폭** (m). 전폭(`PLATFORM_W`=6)에 깔면 geom이 수천 개가
#: 되어 MJX가 느려진다. 양옆은 평평한 슬래브로 남겨 표류해도 떨어지지 않게 한다
#: (`PLATFORM_W` 주석의 그 사고).
#:
#: ⚠️ 2026-08-06에 로봇이 요철을 벗어나 **옆의 평지를 밟고 지나가는 것**이 영상에서
#: 확인됐다 (A0에서 27 s에 y가 +0.12 -> +0.70, 한 방향으로만). LLC의 고유 요속
#: 편향(ω≈0.118 rad/s, `limits.py` 주석)은 **계통적**이라 매번 같은 쪽으로 샌다.
#: 그대로 두면 "요철을 통과했다"가 아니라 **"요철을 피했다"를 level로 집계**한다.
#:
#: ★ **띠를 넓히지 않고 제어기를 고쳤다.** 2.0으로 넓혔다가 되돌렸다 — 넓히는 것은
#: geom을 2배로 늘리면서 원인(되먹임 부재)은 그대로 두는 증상 대처이고, 표류가
#: 시간에 대해 2차로 자라므로 어떤 폭도 충분하지 않다. `guidance.heading_hold`가
#: 이미 표류를 3.33 -> 0.22 m로 잡은 전례가 있어서 그 법칙을
#: `envs.scripted.guidance_controller`에 넣었다.
#: 그리고 넓이가 아니라 **이탈 비율을 매 실행 측정**한다 (`scripted.evaluate`).
ROUGH_HALF_W = 1.0
#: 좌우 레인 수. ★ **반드시 짝수여야 한다.**
#:
#: ⚠️ 처음에 3(좌·중·우)으로 뒀다가 검증에서 걸렸다 (2026-08-06). 홀수면 가운데
#: 레인이 y=0에 **걸터앉고**, 그 폭(2×1.0/3 = 0.667 m)이 `stance_width` 최대값
#: 0.42 m보다 넓어서 **좌우 발이 항상 같은 타일**에 놓인다 = 롤 교란이 0이다.
#: 요철의 절반(좌우 비대칭 착지)을 못 재게 되므로 측정이 반쪽이 된다.
#:
#: 짝수면 레인 **경계가 정확히 y=0**에 놓이므로, 발이 ±stance_width/2
#: (= ±0.06 ~ ±0.21 m) 어디에 있든 좌우가 서로 다른 타일을 밟는다. 레인 폭이
#: 발 간격보다 넓어도 이 성질은 유지되므로 개수를 늘려 geom을 낭비할 필요가 없다.
#: 레인 **폭**은 `2*ROUGH_HALF_W/ROUGH_LANES`이므로 반폭을 바꾸면 개수도 같이
#: 바꿔 폭 0.333 m를 유지한다 (반폭 1.0 -> 6개). 사다리 6단 전체 geom 약 450개다.
ROUGH_LANES = 6


def _rough_patch(spec, name: str, x0: float, x1: float, z: float,
                 amp: float, seed: int = 0):
    """[x0, x1] 구간 중앙 띠에 **윗면 높이가 z ± amp인 타일**을 깐다.

    ★ 자갈밭의 *능동적* 성질(돌이 발밑에서 굴러감)은 강체 시뮬레이터로 만들 수
    없으므로 **재현하지 않는다.** 여기서 재는 것은 정적 요철 하나다:
    **디딜 곳의 높이를 모르는 채로 걷기.**

    ⚠️ `amp`는 진폭이므로 **이웃 타일 사이의 단차는 최대 `2*amp`**다. `ledge`
    실측과 비교할 때 이 2배를 잊으면 안 된다.

    난수는 `seed`로 고정한다 — 지형이 실행마다 바뀌면 시드 간 `level` 차이가
    정책 분산인지 지형 분산인지 구분되지 않는다.
    """
    import numpy as np

    # 조용히 넘기면 롤 교란이 0인 채로 측정이 끝나고, 결과는 그럴듯해 보인다.
    if ROUGH_LANES % 2 != 0:
        raise ValueError(
            f"ROUGH_LANES는 짝수여야 합니다 (현재 {ROUGH_LANES}). 홀수면 레인이 "
            f"y=0에 걸터앉아 좌우 발이 같은 타일을 밟습니다 — 위 주석 참조.")
    rng = np.random.default_rng(seed)
    nx = max(1, int(round((x1 - x0) / ROUGH_TILE)))
    dx = (x1 - x0) / nx
    dy = 2.0 * ROUGH_HALF_W / ROUGH_LANES
    h = rng.uniform(-amp, amp, size=(nx, ROUGH_LANES))

    for i in range(nx):
        for j in range(ROUGH_LANES):
            g = spec.worldbody.add_geom()
            g.name = f"{name}_{i}_{j}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [dx / 2.0, dy / 2.0, PLATFORM_T / 2.0]
            g.pos = [x0 + (i + 0.5) * dx,
                     -ROUGH_HALF_W + (j + 0.5) * dy,
                     z + h[i, j] - PLATFORM_T / 2.0]
            g.condim = 3
            g.friction = list(FRICTION)

    # 양옆 — 표류해도 떨어지지 않도록 평평하게 남긴다.
    for side, sgn in (("l", +1.0), ("r", -1.0)):
        g = spec.worldbody.add_geom()
        g.name = f"{name}_flank_{side}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [(x1 - x0) / 2.0, (PLATFORM_W - ROUGH_HALF_W) / 2.0,
                  PLATFORM_T / 2.0]
        g.pos = [(x0 + x1) / 2.0,
                 sgn * (ROUGH_HALF_W + (PLATFORM_W - ROUGH_HALF_W) / 2.0),
                 z - PLATFORM_T / 2.0]
        g.condim = 3
        g.friction = list(FRICTION)
    return nx * ROUGH_LANES + 2


def rough_report(mj_model) -> None:
    """★ 요철 타일이 **실제로 깔렸는지** 모델에서 직접 읽는다.

    라이다는 이 지형을 원리적으로 못 보므로(`sensors.probe`) 센서로는 확인할 수
    없다. 그런데 지형이 없어도 로봇은 잘 걸어가고 level 6.00이 나온다 — 즉
    **실패가 성공처럼 보인다.** 그래서 모델을 직접 읽는 경로가 따로 필요하다.

    윗면 z의 폭이 `2 × 진폭`으로 나와야 정상이다.
    """
    import numpy as np

    tops: dict[str, list[float]] = {}
    for i in range(mj_model.ngeom):
        nm = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if nm.startswith("rough") and "flank" not in nm:
            tops.setdefault(nm.split("_")[0], []).append(
                float(mj_model.geom_pos[i][2] + mj_model.geom_size[i][2]))
    if not tops:
        print("  ✗ 요철 타일이 **하나도 없습니다.** terrain.kind가 'rough'인지 "
              "확인하십시오.")
        return
    print(f"{'단':>8s} {'타일수':>6s} {'윗면 z 최소':>11s} {'최대':>8s} "
          f"{'폭(=2×진폭)':>12s} {'평균|Δ|(=2a/3)':>15s}")
    print("-" * 68)
    for lvl in sorted(tops, key=lambda s: int(s[5:] or 0)):
        a = np.asarray(tops[lvl])
        span = a.max() - a.min()
        print(f"{lvl:>8s} {len(a):6d} {a.min():11.3f} {a.max():8.3f} "
              f"{span:12.3f} {span / 3.0:15.3f}")
    print(f"\n  ※ 옆 평지(flank)는 제외했습니다. `ROUGH_HALF_W`={ROUGH_HALF_W} m "
          f"바깥은 평평하며, 거기로 새면 측정이 오염됩니다 "
          f"(`envs.scripted._report_off_band`가 검사합니다).")


def rough(amp: float, x0: float = 3.0, run_out: float = RUN_OUT,
          length: float = 3.0, seed: int = 0):
    """★ 실측 ⑦ — 진폭 ±`amp` m의 **요철 구간** (단품).

    ⚠️ 라이다는 이 지형을 **원리적으로 보지 못한다** (광선이 몸통 높이 0.34 m
    수평, 요철은 ±0.06 m). 즉 이것은 HLC가 회피로 풀 수 있는 문제가 아니라
    **LLC의 강건성만 재는 측정**이다. 그래서 `axis_sweep`에서 어떤 축도 효과가
    없다면 그것은 "HLC가 필요하다"가 아니라 "LLC 재학습이 필요하다"는 뜻이다.
    """
    spec = _spec()
    _slab(spec, "approach", -2.0, x0, 0.0)
    _rough_patch(spec, "rough", x0, x0 + length, 0.0, amp, seed=seed)
    _slab(spec, "finish", x0 + length, x0 + length + run_out, 0.0)
    return _finish(spec)


#: `level`(넘긴 장애물 수) 판정 여유 (m). 장애물 **끝**에서 이만큼 더 가야 센다.
#: ⚠️ 예전에는 장애물 **시작** x + 0.8이었다 (2026-08-02 수정). gap은 폭이 최대
#: 0.30 m라 사실상 같은 값이었지만 slope는 램프 수평길이가 1.73~1.99 m라 그 규칙이
#: **오르는 도중을 통과로 집계**한다. `level`이 이 실험의 산출물이므로 그대로 두면
#: 결과 자체가 거짓이 된다.
CLEAR_MARGIN = 0.5


def ladder(kind: str, values, x0: float = 3.0, spacing: float = 3.0,
           run_out: float = 4.0, ramp_len: float = 2.0):
    """★ **난이도 사다리** — 같은 지형을 쉬운 값부터 어려운 값까지 코스 축에 늘어놓는다.

    이것이 P2의 커리큘럼 장치다. 왜 이 형태인가:

    ① **모델이 하나면 된다.** env마다 지형을 다르게 하려면 MJX 모델을 배치화하고
       `make_step_fn`까지 뜯어야 하는데, 난이도를 x축에 배열하면 모델 하나를
       전 env가 공유하면서도 정책은 전 난이도를 겪는다. 재컴파일도 없다.
    ② **커리큘럼이 저절로 된다.** 정책은 쉬운 것부터 만나고, 못 넘으면 거기서
       멈춘다. 레벨을 올려주는 스케줄러도, 성공률 임계값도 필요 없다.
    ③ **한계가 그대로 측정된다.** 도달 거리 x가 곧 "몇 번째까지 넘었나"이다.
       `limits.py`는 손 스윕이 아니라 이 값으로 채운다 — 그리고 그 탐색은 PPO가
       8축을 동시에 굴려서 하므로 **종속성 문제가 공짜로 풀린다.**

    ⚠️ 지형 종류는 섞지 않는다. 섞으면 "어디서 막혔나"가 "무엇 때문에 막혔나"를
    말해주지 못한다. 종류별로 따로 학습한다 (`docs/02_hlc.md` §8).

    Args:
        kind: gap / ledge / beam / tunnel / slope / rough
        values: 쉬운 것부터 어려운 것 순. `ledge`는 각 단의 **높이 증분**이다.
        spacing: 장애물 간 평지 구간. 자세를 회복할 여유가 있어야 한다.

    Returns:
        `(mj_model, meta)`. ★ 반환이 `(model, xs, end_x)` 3-튜플이었다가 dict로
        바뀌었다 (2026-08-02) — `maze.build`와 같은 형태다. 늘린 이유는 아래 둘이
        **없으면 slope 측정이 틀리기 때문**이다:

          xs        i번째 장애물이 **시작**하는 x (보고용)
          clear_xs  i번째를 넘긴 것으로 세는 x (= 장애물 끝 + `CLEAR_MARGIN`).
                    시작 x + 0.8이던 옛 규칙은 램프 중턱을 통과로 셌다.
          end_x     도달 판정 목표 (마지막 장애물을 넘어선 지점)
          prof_x/prof_z  지형 **윗면 높이** 꺾은선. `jnp.interp(x, ...)`로 발밑
                    지면 높이를 준다. 낙상 판정을 지면 기준으로 만드는 데 쓴다
                    (`NavEnv._ground_z` 주석 — slope에서는 이게 없으면 낙상이
                    아예 감지되지 않는다).
          top_z     마지막 지형 높이 (m)
    """
    import math

    spec = _spec()
    xs, clear_xs, z, x = [], [], 0.0, -2.0
    # 지형 윗면 프로파일. **단조 증가**여야 `jnp.interp`가 맞다.
    prof_x, prof_z = [-2.0], [0.0]

    if kind == "tunnel":
        # 지면은 통째로 이어지고 천장만 얹는다.
        for i, v in enumerate(values):
            xs.append(x0 + i * spacing)
        end_x = xs[-1] + TUNNEL_LEN + 1.0
        _slab(spec, "ground", -2.0, end_x + run_out, 0.0)
        for i, (xi, v) in enumerate(zip(xs, values)):
            c = spec.worldbody.add_geom()
            c.name = f"ceiling{i}"
            c.type = mujoco.mjtGeom.mjGEOM_BOX
            c.size = [TUNNEL_LEN / 2.0, PLATFORM_W, PLATFORM_T / 2.0]
            c.pos = [xi + TUNNEL_LEN / 2.0, 0.0, v + PLATFORM_T / 2.0]
            c.condim = 3
            c.friction = list(FRICTION)
        return _finish(spec), dict(
            kind=kind, xs=xs, end_x=end_x, top_z=0.0,
            clear_xs=[xi + TUNNEL_LEN + CLEAR_MARGIN for xi in xs],
            prof_x=[-2.0, end_x + run_out], prof_z=[0.0, 0.0])

    for i, v in enumerate(values):
        xi = x0 + i * spacing
        xs.append(xi)
        run_g = _slab(spec, f"run{i}", x, xi, z)   # 장애물 앞 평지
        if kind == "beam":
            # ★ 2026-08-09. 옛 코드는 다리 **사이**에 전폭 12 m(`PLATFORM_W`가
            #   반폭으로 쓰인다)를 깔았다. 다리마다 3 m짜리 **자유 표류 구간**이
            #   생기는 셈이라, 로봇이 거기서 밀린 채 다음 다리에 도착한다.
            #   그러면 사다리가 재는 것은 "얼마나 좁은 다리를 걷나"가 아니라
            #   "12 m 벌판에서 얼마나 직진하나"가 된다 — 실제로 실측
            #   `max|y_body|`가 평지 0.179에서 0.350으로 뛰었다.
            #   미로 통로 폭으로 좁혀 표류가 쌓일 여지를 없앤다.
            run_g.size = [(xi - x) / 2.0, BEAM_APPROACH_W / 2.0, PLATFORM_T / 2.0]
        if kind == "gap":
            x = xi + v
            # 틈 위에는 지면이 없다. 낙상 기준면은 **양쪽 발판 높이**를 유지하는
            # 것이 맞다 — 그래야 틈으로 빠지는 것이 그대로 감지된다.
            prof_x.append(x); prof_z.append(z)
        elif kind == "ledge":
            prof_x.append(xi - 1e-4); prof_z.append(z)
            z += v                                 # 계단처럼 누적된다
            x = xi
            prof_x.append(xi); prof_z.append(z)
        elif kind == "rough":
            # ⚠️ 프로파일(`prof_z`)에는 **명목 높이 z**를 넣는다. 요철은 ±amp로
            #    대칭이라 평균이 z이고, 낙상 판정(AGL < 0.15)의 오차는 amp뿐이다.
            #    타일별 높이를 넣으려면 prof가 단조 증가여야 한다는 전제가 깨진다.
            seg = spacing * 0.8
            _rough_patch(spec, f"rough{i}", xi, xi + seg, z, v, seed=i)
            x = xi + seg
            prof_x.append(x); prof_z.append(z)
        elif kind == "beam":
            b = _slab(spec, f"beam{i}", xi, xi + spacing * 0.6, z)
            b.size = [spacing * 0.3, v / 2.0, PLATFORM_T / 2.0]
            x = xi + spacing * 0.6
            prof_x.append(x); prof_z.append(z)
        elif kind == "slope":
            th = math.radians(v)
            ct, st, T = math.cos(th), math.sin(th), PLATFORM_T
            g = spec.worldbody.add_geom()
            g.name = f"ramp{i}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [ramp_len / 2.0, PLATFORM_W, T / 2.0]
            g.pos = [xi + ramp_len / 2.0 * ct + T / 2.0 * st, 0.0,
                     z + ramp_len / 2.0 * st - T / 2.0 * ct]
            g.quat = [math.cos(-th / 2.0), 0.0, math.sin(-th / 2.0), 0.0]
            g.condim = 3
            g.friction = list(FRICTION)
            prof_x.append(xi); prof_z.append(z)
            x, z = xi + ramp_len * ct, z + ramp_len * st
            prof_x.append(x); prof_z.append(z)
        else:
            raise ValueError(f"알 수 없는 지형 '{kind}'")
        clear_xs.append(x + CLEAR_MARGIN)

    end_x = x + 1.5                                # 마지막 장애물을 넘어선 지점
    _slab(spec, "finish", x, end_x + run_out, z)
    prof_x.append(end_x + run_out); prof_z.append(z)
    return _finish(spec), dict(
        kind=kind, xs=xs, clear_xs=clear_xs, end_x=end_x, top_z=z,
        prof_x=prof_x, prof_z=prof_z)


def beam(width: float, x0: float = 3.0, run_out: float = RUN_OUT):
    """★ 실측 ④ — 반폭 `width/2` m의 **외나무다리**.

    ⚠️ 여기서는 **옆으로 떨어지는 것이 곧 지형 실패**다. 다른 측정에서는
    `_off_platform`으로 걸러내던 그 현상이 측정 대상 자체가 된다.

    ★ **재는 것은 다리 폭이 아니라 발 간격이다** (2026-08-09 재정의). "건널 수
    있나"는 발 간격(LLC 기하)과 표류(항법)가 섞인 질문이라 어느 쪽을 고쳐야
    하는지 알려주지 못한다. 그래서 순서를 뒤집는다 —
    **`scripted.foot_track()`으로 평지에서 둘을 따로 재고**, 이 지형은 그
    예측을 확인하는 데 쓴다. 터널에서 같은 절차가 정확히 맞았다.
    사다리 `values`는 그 실측 뒤에 정한다 (`configs` PRESETS 주석).

    ⚠️ 옛 주석의 "**측정 중에는 표류를 보정하지 않는다**(열린 루프). 보정을 넣으면
    재는 것이 내 보정기의 성능이 된다"는 **철회한다** (2026-08-09). 근거가 경사
    A/B에서 반증됐다 — `hold` 0/1에서 종료 y가 -1.55 -> -0.07로 완전히 달라졌는데
    **level은 2.00/3.00으로 동일**했다. 즉 진로 유지기는 지형 한계를 가리지 않는다.
    반대로 열린 루프로 재면 LLC 능력이 아니라 **이미 아는 계통 편향**
    (ω≈0.118 rad/s)을 재게 되고, 그 편향은 배포 시 10 Hz HLC가 없앨 것이다.
    → **`hold=1`로 재고, 열린 루프는 대조군으로 1회만.**
    """
    spec = _spec()
    _slab(spec, "approach", -2.0, x0, 0.0)              # 넉넉한 조주 구간
    b = _slab(spec, "beam", x0, x0 + run_out, 0.0)
    b.size = [run_out / 2.0, width / 2.0, PLATFORM_T / 2.0]
    return _finish(spec)


#: 외나무다리 사다리에서 다리 **사이** 회복 구간의 전폭 (m). `ladder` 주석 참조.
#: 미로 통로 폭(2 m)과 같게 둔다 — 배포 조건과 어긋나면 잰 값을 못 쓴다.
#: ⚠️ 여기를 넓히면 "다리 폭"이 아니라 "직진 능력"을 재게 된다.
#: ⚠️ 좁히면 회복 구간에서 떨어지는 것이 다리 실패로 집계된다.
#: `envs.scripted._report_off_beam`이 다리 **위에서만** 표류를 재서 이를 분리한다.
BEAM_APPROACH_W = 2.0

#: 터널 천장 슬래브 길이 (m). 몸을 낮춘 채 **유지**해야 하는 거리.
TUNNEL_LEN = 2.0


def tunnel(clearance: float, x0: float = 3.0, run_out: float = RUN_OUT):
    """★ 실측 ⑤ — 천장 높이 `clearance` m의 **터널**.

    유일하게 `body_height`(-0.22~+0.13)가 주역인 지형이다. 다만 WTW의 height는
    **상대 오프셋**이라 절대 통과 높이는 실측해야만 안다.

    판정에 주의: 천장에 몸통이 닿아도 로봇은 멈추지 않고 **비벼서 밀고 나간다**.
    따라서 "지나갔다"만 보면 안 되고 접촉 유무와 자세 붕괴를 같이 본다.
    """
    spec = _spec()
    _slab(spec, "ground", -2.0, x0 + run_out, 0.0)
    c = spec.worldbody.add_geom()
    c.name = "ceiling"
    c.type = mujoco.mjtGeom.mjGEOM_BOX
    c.size = [TUNNEL_LEN / 2.0, PLATFORM_W, PLATFORM_T / 2.0]
    c.pos = [x0 + TUNNEL_LEN / 2.0, 0.0, clearance + PLATFORM_T / 2.0]
    c.condim = 3
    c.friction = list(FRICTION)
    return _finish(spec)


def slope(deg: float, x0: float = 3.0, ramp_len: float = 3.0,
          run_out: float = RUN_OUT):
    """★ 실측 ⑥ — 경사 `deg`도의 **오르막**.

    램프는 회전한 박스다. y축 둘레로 -θ 회전하면 로컬 +x가 (cosθ, 0, sinθ)로
    가므로 +x 방향 오르막이 된다. 램프 윗면의 **아래쪽 모서리**가 (x0, ·, 0)에
    오도록 중심을 역산한다 — 이걸 대충 놓으면 발판과 램프 사이에 보이지 않는
    턱이 생겨서 재는 것이 경사가 아니라 그 턱이 된다.
    """
    import math

    th = math.radians(deg)
    ct, st = math.cos(th), math.sin(th)
    L, T = ramp_len, PLATFORM_T

    spec = _spec()
    _slab(spec, "lower", -2.0, x0, 0.0)

    g = spec.worldbody.add_geom()
    g.name = "ramp"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [L / 2.0, PLATFORM_W, T / 2.0]
    g.pos = [x0 + L / 2.0 * ct + T / 2.0 * st, 0.0, L / 2.0 * st - T / 2.0 * ct]
    g.quat = [math.cos(-th / 2.0), 0.0, math.sin(-th / 2.0), 0.0]
    g.condim = 3
    g.friction = list(FRICTION)

    top_x, top_z = x0 + L * ct, L * st
    _slab(spec, "upper", top_x, top_x + run_out, top_z)
    return _finish(spec)
