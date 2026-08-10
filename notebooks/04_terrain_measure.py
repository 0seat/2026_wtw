"""04. 지형 능력 실측 — **A안의 입력**.

A안(2026-07-29): LLC는 고정하고 **불가능한 지형지물은 지운다.** 무엇을 지울지
정하려면 무엇이 불가능한지 숫자로 알아야 한다. 이 파일이 그 숫자를 만든다.
결과는 `wtw_nav/terrain/limits.py`에 손으로 옮겨 적고, 그 파일이 이후 모든
지형 생성의 관문이 된다(미측정 지형은 `validate()`가 거부한다).

Colab에서 (주의 — **mujoco import 전에** MUJOCO_GL을 잡아야 영상이 나온다)::

    from google.colab import drive; drive.mount('/content/drive')
    %cd "/content/drive/Othercomputers/BPC/D:/02_projects/2026_wtw"
    import os; os.environ["MUJOCO_GL"] = "egl"
    import sys; sys.path.append(".")

    from wtw_nav.terrain import measure
    measure.all_sweeps()                 # 6종 전부, 영상 없이 (~수 분)

    # 확정 직전 눈으로 확인 (판정 지표는 전에도 틀린 적이 있다 — llc_port_debug §8)
    measure.ledge_sweep(video=True)
    measure.slope_sweep(video=True)

한 방에::

    %run notebooks/04_terrain_measure.py

측정이 끝나면::

    from wtw_nav.terrain import limits
    limits.report()                      # 살아남은 지형 목록


★ 남은 2종 실측 셀 (2026-08-09) — `beam` / `ledge`
===================================================

설계 근거는 `docs/03_results.md` §3·§7.2. 셀 규약은 `wtw_nav/dev.py`.
**셀 A -> B 순서를 지킬 것** — B는 A가 잰 수로 사다리 값을 정한다(터널에서
평지 측정 -> 실지형 확인 절차가 정확히 맞았다).

[셀 A] 외나무다리 ① — **평지에서 발 간격을 잰다** (지형 없음, 컴파일 1회)::

    from wtw_nav.envs import scripted as S

    env_f = S.flat_env()
    FT = S.foot_track(env_f, n=3)
    S.video(env_f, stance_width=0.12, save="beamA_sw_min.mp4", distance=2.5)
    S.video(env_f, stance_width=0.42, save="beamA_sw_max.mp4", distance=2.5)
    print("두 영상의 발 간격이 눈에 띄게 다른지 보십시오 — 같아 보이면 포화입니다.")

[셀 B] 외나무다리 ② — **그 예측을 실지형으로 확인**한다.

주의 — 코드를 고쳤으면 셀 2(`reload_wtw()`)를 먼저 돌릴 것. `BEAM_APPROACH_W`가
지형 모델을 바꾸므로 캐시된 env를 쓰면 옛 지형으로 돌아간다::

    BEST = min(FT, key=lambda k: FT[k]["w_foot"])
    FLOOR = FT[BEST]["floor"]                       # = W_foot + 2×0.023
    VALS = tuple(round(FLOOR * k, 2) for k in (2.5, 2.0, 1.7, 1.4, 1.2, 1.0))
    print(f"sw={BEST}  W_foot={FT[BEST]['w_foot']:.3f}  기하바닥={FLOOR:.3f} -> {VALS}")

    env_b = S.terrain_env("beam", values=VALS, timeout_s=60.0)
    S.evaluate(env_b, n=5, stance_width=BEST)               # hold=1 (기본)
    S.evaluate(env_b, n=5, stance_width=BEST, hold=0.0)     # 열린 루프 대조군
    S.video(env_b, stance_width=BEST, save="beamB_hold1.mp4")
    S.video(env_b, stance_width=BEST, hold=0.0, save="beamB_hold0.mp4")
    print("판별표는 docs/03_results.md §3·§7.2. `evaluate`가 찍는 3수로 읽습니다.")

[셀 C] 턱 — **성공률**을 잰다 (주의 — 위상 8점 = MJX 컴파일 8회, 첫 실행이 길다)::

    PH0 = S.phase_sweep(offsets=8, n=1)                      # A0 = 바닥
    PH1 = S.phase_sweep(offsets=8, n=1, footswing=0.32)      # A1 = 발 최대로 들기
    env_l = S.terrain_env("ledge", x0=3.0)                   # 위상 0° (캐시 재사용)
    S.video(env_l, save="ledgeA0.mp4")
    S.video(env_l, footswing=0.32, save="ledge_fs032.mp4")
    print("두 표의 '조건부' 열을 비교하십시오. 같으면 footswing은 무력합니다.")

[셀 C2] A1이 효과 있을 때만 — 나머지 두 축::

    PH2 = S.phase_sweep(offsets=8, n=1, footswing=0.32, step_freq=3.9)
    PH3 = S.phase_sweep(offsets=8, n=1, footswing=0.32, height=0.13)
    print("올라가는 지형이므로 height는 +쪽(몸을 높임)만 의미가 있습니다.")
"""

import sys

from wtw_nav.terrain import limits, measure


def main():
    """주의 — 기본은 **표류 대조 하나만** 돌린다 (롤아웃 2회).

    스윕 5종을 전부 도는 것(`measure.all_sweeps()`)은 2026-07-29 실행에서
    **지형이 아니라 표류를 측정한다**는 것이 드러나 기본 경로에서 뺐다. 근거:
    다리 폭을 2.00 -> 0.30 m로 6배 바꿨는데 궤적이 소수점 셋째 자리까지 동일했다.
    """
    if not measure.drift_check():
        return 1
    print()
    limits.report()
    print("\n다음: 지형은 스윕이 아니라 **학습 커리큘럼**으로 잰다.")
    print("  손으로 짠 조합 스윕은 축이 7개라 조합이 폭발하는데다, 두 축이 동시에")
    print("  있어야 넘는 경우(종속성)를 원리적으로 못 찾는다. 그 탐색은 PPO가 하는 일이다.")
    return 0


if __name__ == "__main__":
    rc = main()
    if "IPython" not in sys.modules:
        sys.exit(rc)
