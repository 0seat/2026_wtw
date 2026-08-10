"""★ 지형 능력 실측 — **P2 설계 전에 LLC의 한계를 숫자로 확정한다.**

    python -m wtw_nav.terrain.measure gap      # 실측 ② 틈 폭
    python -m wtw_nav.terrain.measure ledge    # 실측 ③ 턱 높이
    python -m wtw_nav.terrain.measure all      # 6종 전부 (A안의 입력)

노트북에서는 판정 장면이 **영상으로** 나온다 (카메라는 로봇을 추적)::

    from wtw_nav.terrain import measure
    measure.gap_sweep()                        # video=True 가 기본

왜 필요한가. 체크포인트 `parameters.pkl` 실측 결과 pretrain-v0 LLC는

    terrain_proportions     = [0,...,0, 1.0]   -> index 8
    terrain_noise_magnitude = 0.0              -> 그 지형의 높이 진폭
    measure_heights         = False            -> 지형 관측 없음

즉 **완전 평지에서, 지형을 전혀 보지 않고** 학습됐다. 발 착지점은 clock 위상으로만
결정되므로 **지형에 맞춰 발을 놓는 능력이 구조적으로 없다.** HLC가 줄 수 있는 것은
`footswing`·`body_height`·`step_freq` 같은 **전역 자세 조절**뿐이다.

주의 — **숫자만 믿지 말 것.** 통과 판정은 x·z 임계값으로 하는데, 이런 판정은 전에도
틀린 적이 있다 — pronk 체공률 17.3%가 실은 **전복한 로봇의 발이 공중에 있던 것**
이었고, 영상이 지표보다 옳았다(`docs/03_results.md` §2). 그래서 판정 장면을 반드시
눈으로 확인한다.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from wtw_nav.hlc import guidance as G
from wtw_nav.llc import check
from wtw_nav.llc import policy as P
from wtw_nav.terrain import modules

#: 넘기 시도 시 HLC가 쓸 만한 명령 조합. footswing이 유일한 실질 레버다.
ATTEMPTS = (
    ("기본", dict(vx=0.8, footswing=0.08, step_freq=3.0, height=0.0)),
    ("발높이", dict(vx=0.8, footswing=0.32, step_freq=3.0, height=0.0)),
    ("발↑느리게", dict(vx=0.8, footswing=0.32, step_freq=2.1, height=0.0)),
    ("발↑빠르게", dict(vx=1.4, footswing=0.32, step_freq=3.9, height=0.0)),
    ("발↑몸통↑", dict(vx=1.0, footswing=0.32, step_freq=3.0, height=0.13)),
)

#: 터널은 유일하게 `height`가 주역이다. footswing을 올리면 오히려 몸이 들린다.
TUNNEL_ATTEMPTS = (
    ("기본", dict(vx=0.8, height=0.0, footswing=0.08, step_freq=3.0)),
    ("몸통↓", dict(vx=0.8, height=-0.22, footswing=0.05, step_freq=3.0)),
    ("몸통↓느리게", dict(vx=0.5, height=-0.22, footswing=0.05, step_freq=2.1)),
    ("몸통↓넓게", dict(vx=0.8, height=-0.22, footswing=0.05, step_freq=3.0,
                       stance_width=0.42)),
)

#: 경사는 `pitch`(몸통 기울기)와 추진력의 조합이다.
SLOPE_ATTEMPTS = (
    ("기본", dict(vx=0.8, pitch=0.0, footswing=0.08, step_freq=3.0)),
    ("앞숙임", dict(vx=0.8, pitch=-0.35, footswing=0.08, step_freq=3.0)),
    ("앞숙임+발↑", dict(vx=0.8, pitch=-0.35, footswing=0.20, step_freq=3.0)),
    ("천천히", dict(vx=0.4, pitch=-0.35, footswing=0.20, step_freq=2.1)),
)

OUT_DIR = "checkpoints/terrain_measure"

_POLICY = None          # 체크포인트 로드는 한 번만 (torch.jit 로드가 느리다)


#: ★ 진로 유지기를 켠 채 측정한다. **기본값 True.**
#: 끄면 순수 열린 루프가 되는데, 그 조건에서는 표류가 8초에 3.3 m라 측정이
#: 지형이 아니라 표류를 재게 된다(2026-07-29). 근거는 `guidance.heading_hold`.
HOLD = True


def _env_for(mj_model, hold: bool | None = None):
    """지형 모델 하나에 대한 롤아웃 환경. **정책은 전역 캐시를 재사용**한다.

    `check.build()`를 지형마다 부르면 `load_policy`가 매번 torch 체크포인트를
    다시 읽어 수십 초를 버린다.
    """
    global _POLICY
    if _POLICY is None:
        _POLICY = P.load_policy(check.CKPT)
    import jax
    mj, mj_data, mjx_model = P.create_env(mj_model)
    jidx = P._build_joint_index(mj)
    hold = HOLD if hold is None else hold
    fn = (P.make_closed_rollout_fn(mjx_model, _POLICY, jidx, G.heading_hold())
          if hold else P.make_rollout_fn(mjx_model, _POLICY, jidx))
    return dict(policy_fn=_POLICY, mj_model=mj, mj_data=mj_data, mjx_model=mjx_model,
                jidx=jidx, rollout_fn=jax.jit(fn))


def _run(mj_model, cmd_kw, seconds=8.0, hold=None):
    """지형 모델 하나에 명령 하나로 롤아웃. 통과 판정용 요약 + 궤적을 낸다."""
    env = _env_for(mj_model, hold=hold)
    r = check.rollout(env, P.make_commands(**cmd_kw), seconds=seconds)
    q = np.asarray(r["qpos"])
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    return dict(x_end=float(x[-1]), x_max=float(np.nanmax(x)),
                y_end=float(y[-1]), y_absmax=float(np.nanmax(np.abs(y))),
                z_min=float(np.nanmin(z)), z_end=float(z[-1]),
                fell=bool(r["fell"]), mean_vx=float(r["mean_vx"]),
                qpos=q, mj_model=env["mj_model"])


def _video(res, title, save_name=None, fps=25):
    """판정 장면을 영상으로. 카메라는 로봇을 추적한다(`render(track=True)`)."""
    try:
        frames = check.render(res["mj_model"], res["qpos"], fps=fps, track=True)
    except Exception as e:
        print(f"       주의 — 렌더 실패({type(e).__name__}: {e})")
        print("          Colab이면 mujoco import **전에** os.environ['MUJOCO_GL']='egl'")
        return
    save = None
    if save_name:
        os.makedirs(OUT_DIR, exist_ok=True)
        save = f"{OUT_DIR}/{save_name}.mp4"
    check._show(frames, fps, save=save, title=title)


#: 지형별 (실패노트, 미달노트). `_judge`가 쓴다.
_NOTES = dict(gap=("낙하", "못 넘음"), ledge=("낙하", "못 올라감"),
              beam=("낙하", "못 나감"), tunnel=("자세붕괴", "막힘"),
              slope=("낙하", "못 올라감"))


def _judge(kind, value, r, x0=3.0, ramp_len=3.0):
    """★ 지형 통과 판정의 **단일 진실 공급원**.

    스윕과 축 선별이 서로 다른 판정을 쓰면 그 자체가 거짓 결론을 만든다.
    두 경로 모두 반드시 이 함수를 거친다.

    Returns: (ok, note)
    """
    import math

    # beam은 예외 — 옆으로 떨어지는 것이 곧 지형 실패다(측정 대상 자체).
    side = False if kind == "beam" else _off_platform(r)
    fail_note, short_note = _NOTES[kind]

    if kind == "tunnel":
        # 몸통이 낮아진 상태가 정상이므로 z 하한을 다르게 잡는다.
        fell = r["z_min"] < 0.10 or r["fell"]
        good = r["x_end"] > x0 + modules.TUNNEL_LEN + 1.0
    elif kind == "slope":
        th = math.radians(value)
        fell = r["z_min"] < -0.05 or r["fell"]
        good = (r["x_end"] > x0 + ramp_len * math.cos(th) + 0.3
                and r["z_end"] > ramp_len * math.sin(th) + 0.10)
    else:
        fell = r["z_min"] < -0.05 or r["fell"]
        good = {"gap": lambda: r["x_end"] > x0 + value + 0.5,
                "ledge": lambda: r["x_end"] > x0 + 0.8 and r["z_end"] > value + 0.15,
                "beam": lambda: r["x_end"] > x0 + 3.0}[kind]()

    note = ("옆으로 이탈" if side else fail_note if fell
            else "" if good else short_note)
    return (good and not fell and not side), note


def _build(kind, value, x0=3.0, ramp_len=3.0):
    """지형 이름 + 파라미터 -> MjModel."""
    if kind == "slope":
        return modules.slope(value, x0=x0, ramp_len=ramp_len)
    return dict(gap=modules.gap, ledge=modules.ledge, beam=modules.beam,
                tunnel=modules.tunnel)[kind](value, x0=x0)


def _off_platform(res) -> bool:
    """발판 **옆으로** 새서 떨어진 것인가. 지형 실패와 구분해야 한다.

    LLC는 직진 명령에도 완만히 선회하므로 y가 흐른다. 이걸 '틈을 못 넘었다'로
    집계하면 측정이 통째로 거짓이 된다.
    """
    return res["y_absmax"] > modules.PLATFORM_W - 0.5


# ---------------------------------------------------------------- 실측 ②
def gap_sweep(widths=(0.10, 0.15, 0.20, 0.25, 0.30, 0.40), x0=3.0, seconds=8.0,
              video=True):
    """실측 ② — 넘을 수 있는 **틈 폭**의 상한.

    판정: 틈 끝(x0+w)을 0.5 m 이상 지나가고, 떨어지지 않아야 통과.
    """
    print("=" * 70)
    print("실측 ② — 틈(gap) 통과 한계 폭")
    print("  도약은 불가능함이 확정됐으므로(llc_port_debug §8), 재는 것은 step-over다.")
    print("=" * 70)
    best, scenes = {}, {}
    for label, kw in ATTEMPTS:
        print(f"\n[{label}] {kw}")
        print(f"  {'폭 m':>6s} {'통과':>5s} {'x_end':>7s} {'z_min':>7s} "
              f"{'|y|max':>7s} {'비고':>10s}")
        ok_max, first_fail = 0.0, None
        for w in widths:
            r = _run(_build("gap", w, x0=x0), kw, seconds)
            ok, note = _judge("gap", w, r, x0)
            side = note == "옆으로 이탈"
            if ok:
                ok_max = max(ok_max, w); scenes[(label, "pass")] = (w, r)
            elif first_fail is None and not side:
                first_fail = w; scenes[(label, "fail")] = (w, r)
            print(f"  {w:6.2f} {('O' if ok else 'X'):>5s} {r['x_end']:7.2f} "
                  f"{r['z_min']:7.3f} {r['y_absmax']:7.2f} {note:>10s}")
        best[label] = ok_max
        print(f"  -> 최대 통과 폭 {ok_max:.2f} m")

    top = max(best.values())
    winners = [k for k, v in best.items() if v == top]
    print("\n" + "-" * 70)
    print(f"★ 틈 폭 상한 = **{top:.2f} m**   (최선 조합: {winners})")
    if top < 0.15:
        print("  주의 — 사실상 틈을 못 넘습니다. 미로에서 '틈'을 빼거나 LLC 재학습이 필요합니다.")
    print("  ※ 설계 문서의 0.5~0.7 m는 도약 전제였고, 도약은 불가능함이 확정됐습니다.")

    if video:
        w = winners[0]
        print("\n[영상] 판정이 옳은지 눈으로 확인하십시오 (카메라는 로봇 추적).")
        for kind, desc in (("pass", "최대 통과"), ("fail", "첫 실패")):
            if (w, kind) in scenes:
                width, r = scenes[(w, kind)]
                print(f"  · {desc} — {w}, 틈 {width:.2f} m")
                _video(r, f"gap {width:.2f}m [{w}] {desc}",
                       f"gap_{width:.2f}_{kind}")
    return best


# ---------------------------------------------------------------- 실측 ③
def ledge_sweep(heights=(0.02, 0.05, 0.08, 0.12, 0.16, 0.20), x0=3.0, seconds=8.0,
                video=True):
    """실측 ③ — 올라설 수 있는 **턱 높이**의 상한."""
    print("=" * 70)
    print("실측 ③ — 턱(ledge) 등반 한계 높이")
    print("  LLC는 지형을 못 보므로, 넘는다면 그것은 footswing으로 발을 더 든 결과다.")
    print("=" * 70)
    best, scenes = {}, {}
    for label, kw in ATTEMPTS:
        print(f"\n[{label}] {kw}")
        print(f"  {'높이 m':>7s} {'통과':>5s} {'x_end':>7s} {'z_end':>7s} "
              f"{'z_min':>7s} {'|y|max':>7s} {'비고':>10s}")
        ok_max, first_fail = 0.0, None
        for h in heights:
            # 주의 — '떨어짐'을 '못 올라감'과 섞지 말 것 — 발판 길이가 짧아 끝을 지나
            #    떨어진 것을 '2 cm 턱에서 낙하'로 집계한 적이 있다(RUN_OUT 참조).
            r = _run(_build("ledge", h, x0=x0), kw, seconds)
            ok, note = _judge("ledge", h, r, x0)
            side = note == "옆으로 이탈"
            if ok:
                ok_max = max(ok_max, h); scenes[(label, "pass")] = (h, r)
            elif first_fail is None and not side:
                first_fail = h; scenes[(label, "fail")] = (h, r)
            print(f"  {h:7.2f} {('O' if ok else 'X'):>5s} {r['x_end']:7.2f} "
                  f"{r['z_end']:7.3f} {r['z_min']:7.3f} {r['y_absmax']:7.2f} "
                  f"{note:>10s}")
        best[label] = ok_max
        print(f"  -> 최대 등반 높이 {ok_max:.2f} m")

    top = max(best.values())
    winners = [k for k, v in best.items() if v == top]
    print("\n" + "-" * 70)
    print(f"★ 턱 높이 상한 = **{top:.2f} m**   (최선 조합: {winners})")

    if video:
        w = winners[0]
        print("\n[영상] 판정이 옳은지 눈으로 확인하십시오 (카메라는 로봇 추적).")
        for kind, desc in (("pass", "최대 등반"), ("fail", "첫 실패")):
            if (w, kind) in scenes:
                height, r = scenes[(w, kind)]
                print(f"  · {desc} — {w}, 턱 {height:.2f} m")
                _video(r, f"ledge {height:.2f}m [{w}] {desc}",
                       f"ledge_{height:.2f}_{kind}")
    return best


# ---------------------------------------------------------------- 실측 ④
def beam_sweep(widths=(2.0, 1.2, 0.8, 0.6, 0.4, 0.3), x0=3.0, seconds=8.0,
               video=True):
    """실측 ④ — 건널 수 있는 **외나무다리 폭**의 하한.

    주의 — 다른 측정과 판정 규칙이 다르다. 여기서는 **옆으로 떨어지는 것이 곧 실패**다
    (`_off_platform`을 쓰지 않는다). LLC가 직진 명령에도 선회한다는 것을 이미
    알고 있으므로, 이 값은 사실상 "8초 동안 표류가 폭의 절반을 넘지 않는가"를 잰다.

    열린 루프로 잰다. yaw 보정을 넣으면 재는 것이 LLC가 아니라 내 보정기가 된다.
    """
    print("=" * 70)
    print("실측 ④ — 외나무다리(beam) 통과 한계 폭")
    print("  실패 모드는 낙하가 아니라 **표류**다. 보정 없이 잰다.")
    print("=" * 70)
    best, scenes = {}, {}
    for label, kw in ATTEMPTS:
        print(f"\n[{label}] {kw}")
        print(f"  {'폭 m':>6s} {'통과':>5s} {'x_end':>7s} {'z_min':>7s} "
              f"{'|y|max':>7s} {'비고':>10s}")
        ok_min, first_fail = None, None
        for w in widths:
            r = _run(_build("beam", w, x0=x0), kw, seconds)
            ok, note = _judge("beam", w, r, x0)
            if ok:
                ok_min = w if ok_min is None else min(ok_min, w)
                scenes[(label, "pass")] = (w, r)
            elif first_fail is None:
                first_fail = w
                scenes[(label, "fail")] = (w, r)
            print(f"  {w:6.2f} {('O' if ok else 'X'):>5s} {r['x_end']:7.2f} "
                  f"{r['z_min']:7.3f} {r['y_absmax']:7.2f} {note:>10s}")
        best[label] = ok_min
        print(f"  -> 최소 통과 폭 {'없음' if ok_min is None else f'{ok_min:.2f} m'}")

    vals = [v for v in best.values() if v is not None]
    print("\n" + "-" * 70)
    if not vals:
        print(f"★ 다리 폭 하한 = **통과 불가** (가장 넓은 {max(widths):.2f} m도 실패)")
    else:
        top = min(vals)
        print(f"★ 다리 폭 하한 = **{top:.2f} m**   "
              f"(최선 조합: {[k for k, v in best.items() if v == top]})")
        if top > 0.5:
            print("  주의 — 설계 문서의 BEAM 0.1~0.3 m는 달성 불가입니다.")

    if video:
        _best_scenes(best, scenes, "beam", "최소 통과", "첫 실패", "m", reverse=True)
    return best


# ---------------------------------------------------------------- 실측 ⑤
def tunnel_sweep(clearances=(0.50, 0.45, 0.40, 0.35, 0.30, 0.25), x0=3.0,
                 seconds=8.0, video=True):
    """실측 ⑤ — 통과 가능한 **천장 높이**의 하한.

    주의 — "지나갔다"만 보면 안 된다. 천장에 몸통이 닿아도 로봇은 멈추지 않고
    **비벼서 밀고 나간다**. 그래서 통과 여부와 함께 `vx`를 같이 본다 — 평지 대비
    속도가 크게 깎였으면 그건 통과가 아니라 마찰로 기어나온 것이다.
    """
    print("=" * 70)
    print("실측 ⑤ — 터널(tunnel) 통과 한계 천장 높이")
    print("  height는 **상대 오프셋**이라 절대 통과 높이는 실측해야만 안다.")
    print("=" * 70)
    best, scenes = {}, {}
    for label, kw in TUNNEL_ATTEMPTS:
        print(f"\n[{label}] {kw}")
        print(f"  {'천장 m':>7s} {'통과':>5s} {'x_end':>7s} {'z_end':>7s} "
              f"{'vx':>6s} {'|y|max':>7s} {'비고':>10s}")
        ok_min, first_fail = None, None
        for c in clearances:
            r = _run(_build("tunnel", c, x0=x0), kw, seconds)
            ok, note = _judge("tunnel", c, r, x0)
            side = note == "옆으로 이탈"
            if ok:
                ok_min = c if ok_min is None else min(ok_min, c)
                scenes[(label, "pass")] = (c, r)
            elif first_fail is None and not side:
                first_fail = c
                scenes[(label, "fail")] = (c, r)
            print(f"  {c:7.2f} {('O' if ok else 'X'):>5s} {r['x_end']:7.2f} "
                  f"{r['z_end']:7.3f} {r['mean_vx']:6.2f} {r['y_absmax']:7.2f} "
                  f"{note:>10s}")
        best[label] = ok_min
        print(f"  -> 최소 통과 높이 {'없음' if ok_min is None else f'{ok_min:.2f} m'}")

    vals = [v for v in best.values() if v is not None]
    print("\n" + "-" * 70)
    if not vals:
        print(f"★ 터널 높이 하한 = **통과 불가** (가장 높은 {max(clearances):.2f} m도 실패)")
    else:
        top = min(vals)
        print(f"★ 터널 높이 하한 = **{top:.2f} m**   "
              f"(최선 조합: {[k for k, v in best.items() if v == top]})")
    print("  ※ vx가 기본 조합 대비 크게 낮으면 '통과'가 아니라 비비고 나온 것입니다.")

    if video:
        _best_scenes(best, scenes, "tunnel", "최소 통과", "첫 실패", "m", reverse=True)
    return best


# ---------------------------------------------------------------- 실측 ⑥
def slope_sweep(degrees=(5.0, 10.0, 15.0, 20.0, 25.0, 30.0), x0=3.0,
                ramp_len=3.0, seconds=10.0, video=True):
    """실측 ⑥ — 오를 수 있는 **경사각**의 상한.

    턱과 달리 경사는 발 착지점을 맞출 필요가 없다 — 어디를 밟아도 지면이 있다.
    따라서 지형 인지 없는 LLC가 **원리적으로 가장 잘할 수 있는** 지형이며,
    여기서마저 값이 낮다면 그건 마찰·추진력 한계다.
    """
    import math

    print("=" * 70)
    print("실측 ⑥ — 경사(slope) 등반 한계 각도")
    print("  지형 인지가 필요 없는 유일한 지형 — LLC에 가장 유리한 조건이다.")
    print("=" * 70)
    best, scenes = {}, {}
    for label, kw in SLOPE_ATTEMPTS:
        print(f"\n[{label}] {kw}")
        print(f"  {'각도°':>6s} {'등반':>5s} {'x_end':>7s} {'z_end':>7s} "
              f"{'목표z':>6s} {'|y|max':>7s} {'비고':>10s}")
        ok_max = 0.0
        first_fail = None
        for d in degrees:
            top_z = ramp_len * math.sin(math.radians(d))      # 표시용 목표 높이
            r = _run(_build("slope", d, x0=x0, ramp_len=ramp_len), kw, seconds)
            ok, note = _judge("slope", d, r, x0, ramp_len)
            side = note == "옆으로 이탈"
            if ok:
                ok_max = max(ok_max, d)
                scenes[(label, "pass")] = (d, r)
            elif first_fail is None and not side:
                first_fail = d
                scenes[(label, "fail")] = (d, r)
            print(f"  {d:6.1f} {('O' if ok else 'X'):>5s} {r['x_end']:7.2f} "
                  f"{r['z_end']:7.3f} {top_z:6.2f} {r['y_absmax']:7.2f} {note:>10s}")
        best[label] = ok_max
        print(f"  -> 최대 등반 각도 {ok_max:.0f}°")

    top = max(best.values())
    print("\n" + "-" * 70)
    print(f"★ 경사 상한 = **{top:.0f}°**   "
          f"(최선 조합: {[k for k, v in best.items() if v == top]})")

    if video:
        _best_scenes(best, scenes, "slope", "최대 등반", "첫 실패", "°")
    return best


# ------------------------------------------------------- 축 선별 (조합 이전)
#: ★ HLC가 **실제로 조작 가능한 전 축**과 그 학습 범위.
#: `duty`(8)·`roll`(11)은 없다 — 커리큘럼 상한까지 폭이 0이라 이 가중치가 본 적 없다.
#: `stance_length`는 **있다** — `[0.35,0.45]` 전 구간이 학습됐고, HLC 설계에서
#: 고정한 것은 "지형 돌파에 불필요해서"라는 추측이었을 뿐 측정 결과가 아니다.
AXES = {
    "vx":            (0.4, 0.8, 1.2, 1.6),
    "step_freq":     (2.1, 2.6, 3.0, 3.4, 3.9),
    "footswing":     (0.05, 0.10, 0.20, 0.32),
    "height":        (-0.22, -0.10, 0.0, 0.13),
    "pitch":         (-0.35, -0.15, 0.0, 0.15, 0.35),
    "stance_width":  (0.12, 0.20, 0.25, 0.32, 0.42),
    "stance_length": (0.35, 0.40, 0.45),
}

#: 축 선별의 원점. 여기서 **한 축만** 움직인다.
BASE = dict(vx=0.8, step_freq=3.0, footswing=0.08, height=0.0, pitch=0.0,
            stance_width=0.25, stance_length=0.45)


def axis_screen(kind="gap", value=0.20, axes=None, x0=3.0, ramp_len=3.0,
                seconds=8.0):
    """★ **어느 명령 축이 이 지형에서 실제로 뭔가를 바꾸는가.**

    왜 조합 나열(`ATTEMPTS`)이 아니라 이건가. 조작 가능한 축이 7개이므로 조합은
    폭발하고, 손으로 고른 5개 조합은 **탐색이 아니라 내 선입견의 표본**이다.
    실제로 그 5개는 vx·footswing·step_freq·height 4축만 흔들었고 `stance_width`·
    `pitch`·`stance_length`는 건드리지도 않았다 — 그 상태로 "8D 전체가 3 cm"라고
    말한 것은 과장이었다(2026-07-29).

    그래서 순서를 바꾼다: **임계 근처 지형값 하나를 고정하고 축을 하나씩만 흔든다.**
    비용은 조합이 아니라 합(Σ|축| ≈ 30회)이고, 결과는 "어느 축이 살아있는가"라는
    답을 준다. 살아난 축만 골라 그 다음에 조합한다.

    주의 — `value`는 **현재 한계보다 조금 넘는 값**으로 줄 것. 이미 통과하는 지형에서는
    전 축이 O로 나와 아무것도 구별하지 못하고, 아무도 못 넘는 값에서는 전 축이 X다.
    틈이면 0.20(실측 상한 0.15), 턱이면 0.12 근처가 적당하다.

    Args:
        kind: gap / ledge / beam / tunnel / slope
        value: 고정할 지형 파라미터 (slope는 도(°))
        axes: 훑을 축 이름 목록. 기본은 `AXES` 전부.
    """
    print("=" * 72)
    print(f"축 선별 — {kind} {value}{'°' if kind == 'slope' else ' m'} 고정, "
          f"한 번에 한 축만")
    print(f"  원점: {BASE}")
    print("=" * 72)

    mj = _build(kind, value, x0=x0, ramp_len=ramp_len)   # 지형은 한 번만 만든다
    base_r = _run(mj, BASE, seconds)
    base_ok, base_note = _judge(kind, value, base_r, x0, ramp_len)
    print(f"\n[원점] {'O' if base_ok else 'X'} x_end={base_r['x_end']:.2f} "
          f"z_end={base_r['z_end']:.3f} z_min={base_r['z_min']:.3f} {base_note}")
    if base_ok:
        print("  주의 — 원점이 이미 통과합니다. `value`를 더 어렵게 주십시오 — "
              "전 축이 O로 나오면 아무것도 구별하지 못합니다.")

    live, dead = {}, []
    for name in (axes or AXES):
        print(f"\n[{name}]  ({'HLC 8D' if name in BASE else '?'})")
        print(f"  {'값':>7s} {'통과':>5s} {'x_end':>7s} {'z_end':>7s} "
              f"{'z_min':>7s} {'|y|max':>7s} {'비고':>10s}")
        xs, any_ok = [], False
        for v in AXES[name]:
            kw = dict(BASE, **{name: v})
            r = _run(mj, kw, seconds)
            ok, note = _judge(kind, value, r, x0, ramp_len)
            any_ok |= ok
            xs.append(r["x_end"])
            print(f"  {v:7.2f} {('O' if ok else 'X'):>5s} {r['x_end']:7.2f} "
                  f"{r['z_end']:7.3f} {r['z_min']:7.3f} {r['y_absmax']:7.2f} "
                  f"{note:>10s}")
        spread = max(xs) - min(xs)
        # 통과시키지 못해도 도달 거리를 크게 바꾸면 '살아있는' 축이다 —
        # 조합했을 때 임계를 넘길 여지가 있다.
        if any_ok or spread > 0.5:
            live[name] = (any_ok, spread)
            print(f"  -> 살아있음 (통과 {any_ok}, x_end 폭 {spread:.2f} m)")
        else:
            dead.append(name)
            print(f"  -> 무효 (x_end 폭 {spread:.2f} m — 이 지형에서 의미 없음)")

    print("\n" + "-" * 72)
    print(f"★ 살아있는 축: {sorted(live)}")
    print(f"  무효한 축  : {dead}")
    if live:
        print("  다음: 살아있는 축**만** 조합해 `ATTEMPTS`를 다시 짜고 스윕을 돌린다.")
    else:
        print("  주의 — 전 축 무효 = 이 지형은 명령으로 넘을 수 없다. limits에서 지울 것.")
    return live, dead


def _best_scenes(best, scenes, kind, pass_desc, fail_desc, unit, reverse=False):
    """최선 조합의 통과/실패 장면을 영상으로. 카메라는 로봇 추적."""
    vals = [v for v in best.values() if v is not None]
    if not vals:
        print("\n[영상] 통과 사례가 없어 실패 장면만 보여줍니다.")
    pick = (min(vals) if reverse else max(vals)) if vals else None
    who = next((k for k, v in best.items() if v == pick), next(iter(best)))
    print("\n[영상] 판정이 옳은지 눈으로 확인하십시오 (카메라는 로봇 추적).")
    for tag, desc in (("pass", pass_desc), ("fail", fail_desc)):
        if (who, tag) in scenes:
            val, r = scenes[(who, tag)]
            print(f"  · {desc} — {who}, {val:.2f}{unit}")
            _video(r, f"{kind} {val:.2f}{unit} [{who}] {desc}",
                   f"{kind}_{val:.2f}_{tag}")


# ---------------------------------------------------------------- 표류 대조
def drift_check(seconds=8.0, cmd_kw=None, video=False):
    """★ **진로 유지기가 표류를 실제로 잡는가.** 롤아웃 2회로 끝나는 관문.

    이걸 통과하지 못하면 지형 측정도 지형 커리큘럼도 의미가 없다 — 미로 통로가
    2 m인데 8초에 3.3 m를 옆으로 흘리면 지형에 닿기도 전에 벽에 부딪힌다.
    그러므로 GPU 시간을 더 쓰기 전에 **여기서 먼저 판정한다.**

    합격선: 횡 표류 |y|max < 0.30 m, 그리고 전진 거리가 열린 루프 대비 크게
    줄지 않을 것(유지기가 로봇을 세워버리면 표류는 0이지만 쓸모없다).
    """
    kw = cmd_kw or dict(BASE)
    mj = modules.flat()
    print("=" * 72)
    print("표류 대조 — 진로 유지기 유무 (평지, 지형 없음)")
    print(f"  명령: {kw}")
    print("=" * 72)

    out = {}
    for hold, name in ((False, "열린 루프"), (True, "유지기 ON")):
        r = _run(mj, kw, seconds, hold=hold)
        q = r["qpos"]
        psi = np.degrees(np.arctan2(
            2 * (q[:, 3] * q[:, 6] + q[:, 4] * q[:, 5]),
            1 - 2 * (q[:, 5] ** 2 + q[:, 6] ** 2)))
        out[hold] = r
        print(f"\n[{name}]  x_end={r['x_end']:6.2f}  |y|max={r['y_absmax']:6.2f}  "
              f"vx={r['mean_vx']:.3f}  ψ_end={psi[-1]:+7.1f}°  "
              f"|ψ|max={np.abs(psi).max():6.1f}°")
        if not hold:
            # 편향 추정: y ≈ vx·ω·t²/2  ->  ω = 2y/(vx·t²)
            w = 2 * r["y_absmax"] / max(r["mean_vx"] * seconds ** 2, 1e-6)
            print(f"       추정 요속 편향 ω ≈ {w:.3f} rad/s "
                  f"({np.degrees(w):.1f}°/s) — 되먹임이 없으면 t²로 자란다")

    op, cl = out[False], out[True]
    ok = cl["y_absmax"] < 0.30 and cl["x_end"] > 0.7 * op["x_end"]
    print("\n" + "-" * 72)
    print(f"표류 {op['y_absmax']:.2f} m -> {cl['y_absmax']:.2f} m "
          f"({op['y_absmax'] / max(cl['y_absmax'], 1e-6):.1f}배 감소), "
          f"전진 {op['x_end']:.2f} -> {cl['x_end']:.2f} m")
    print(f"★ {'통과 — 지형 커리큘럼으로 넘어갑니다.' if ok else '*** 실패 ***'}")
    if not ok:
        if cl["y_absmax"] >= 0.30:
            print("  표류가 안 잡힙니다. 게인(k_y, k_psi)을 올리거나, 이 LLC로는")
            print("  좁은 통로 자체가 불가능하다는 뜻입니다 — 후자면 통로를 넓히십시오.")
        else:
            print("  유지기가 로봇을 붙잡고 있습니다. yaw_max/psi_max를 낮추십시오.")
    if video:
        _video(cl, "유지기 ON — 평지", "drift_hold")
        _video(op, "열린 루프 — 평지", "drift_open")
    return ok


# ---------------------------------------------------------------- 대조군
def sanity(seconds=8.0, video=True):
    """대조군 — 평평한 발판에서 정상 보행하는가.

    이걸 통과해야 위 두 측정의 'X'를 지형 탓으로 돌릴 수 있다. 바닥 평면을 지우고
    박스로 갈아끼운 것, 실린더 충돌을 끈 것 자체가 원인일 수도 있기 때문이다.
    실제로 첫 시도는 발판 반폭이 1.5 m라 **로봇이 옆으로 떨어져** 실패했다.
    """
    r = _run(modules.flat(), dict(vx=0.8, footswing=0.08, step_freq=3.0, height=0.0),
             seconds)
    ok = r["x_end"] > 2.0 and not r["fell"] and r["z_min"] > 0.15
    print(f"[대조군] 평평한 발판: x_end={r['x_end']:.2f} |y|max={r['y_absmax']:.2f} "
          f"z_min={r['z_min']:.3f} vx={r['mean_vx']:.3f} -> "
          f"{'OK' if ok else '*** FAIL ***'}")
    if not ok:
        print("  주의 — 평지에서부터 실패합니다. 지형이 아니라 **발판 모델**이 문제입니다.")
        if r["y_absmax"] > modules.PLATFORM_W - 0.5:
            print(f"     원인: 옆으로 이탈 (|y|max={r['y_absmax']:.2f} vs "
                  f"반폭 {modules.PLATFORM_W}). `modules.PLATFORM_W`를 키우십시오.")
        else:
            print("     확인: 평면 제거 / 박스 마찰·두께 / 초기 z / 실린더 충돌 해제")
    if video:
        _video(r, "대조군 — 평평한 발판", "sanity")
    return ok


#: 이름 -> 스윕 함수. `main`과 `all_sweeps`가 공유한다.
SWEEPS = dict(gap=gap_sweep, ledge=ledge_sweep, beam=beam_sweep,
              tunnel=tunnel_sweep, slope=slope_sweep)


def all_sweeps(video=False):
    """★ 설계 문서 §4.1의 지형 6종을 전부 실측한다. **A안의 입력이다.**

    무엇을 지울지 정하려면 무엇이 불가능한지 알아야 한다. 결과는
    `wtw_nav/terrain/limits.py`에 상수로 박고, 그 파일이 미로 설계의 관문이 된다.

    video=False가 기본이다 — 6종 × 조합 × 값이면 영상이 수십 개다. 확정 직전에
    개별 스윕을 `video=True`로 다시 돌려 눈으로 확인할 것.
    """
    if not drift_check():
        print("\n*** 표류가 안 잡힙니다. 지형을 재봐야 표류를 재게 되므로 중단합니다. ***")
        return None
    if not sanity(video=False):
        print("\n*** 대조군 실패. 지형이 아니라 발판 모델 문제이므로 중단합니다. ***")
        return None
    out = {}
    for name, fn in SWEEPS.items():
        out[name] = fn(video=video)
    print("\n" + "=" * 70)
    print("전체 요약 — 이 숫자를 `terrain/limits.py`에 옮겨 적으십시오.")
    print("=" * 70)
    for name, best in out.items():
        vals = [v for v in best.values() if v is not None]
        print(f"  {name:8s} {'측정 실패' if not vals else f'{min(vals):.2f} ~ {max(vals):.2f}'}"
              f"   조합별: {best}")
    return out


def main() -> int:
    what = sys.argv[1:] or ["drift"]
    if what == ["all"]:
        return 0 if all_sweeps() is not None else 1
    if what == ["drift"]:
        return 0 if drift_check() else 1
    if not drift_check():
        return 1
    if not sanity(video=False):
        return 1
    for name in what:
        if name not in SWEEPS:
            print(f"알 수 없는 지형 '{name}'. 가능: {list(SWEEPS)} 또는 all")
            return 2
        SWEEPS[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
