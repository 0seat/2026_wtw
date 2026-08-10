"""보상 설계 점검 — **학습을 돌리기 전에** 유인이 의도대로 서는지 본다.

    python -m wtw_nav.envs.reward_audit

JAX가 필요 없다(순수 산수). 몇 초면 끝나므로 보상 상수를 건드릴 때마다 돌릴 것.

왜 필요한가. 2026-07-29 P1 학습은 **잘 학습됐는데 실패**했다. 리턴 184.6 중
183.8이 progress였고, 도달 보너스(10)와 시간 페널티(-0.01×300=-3)가 반올림 오차
수준이었다. 정책은 얻을 수 있는 리턴의 97%를 이미 확보했으므로 **마지막 1.2 m를
마저 갈 이유가 없었다.** 도달률이 0.17에서 6.5M 스텝 동안 오르지 않았다.
이건 학습 문제가 아니라 설계 문제이고, 아래 표를 봤으면 1분에 알았을 일이다.
"""

from __future__ import annotations

import sys

from wtw_nav.configs import HLCConfig, default_config

def scenarios(cfg: HLCConfig) -> tuple[tuple[str, float, int, bool, bool], ...]:
    """(이름, 이동거리 m, 소요 스텝, 도달?, 나쁜종료?).

    ⚠️ 스텝 수를 상수로 박지 말 것 — `timeout_s`를 바꾸면 표가 조용히 틀려진다.
    """
    n_max = int(cfg.term.timeout_s / 0.1)          # HLC 10 Hz
    reach_d = cfg.course.length - cfg.course.goal_radius
    return (
        ("빠르게 도달 (절반 시간)", reach_d,          n_max // 2, True,  False),
        ("느리게 도달 (시간 임박)", reach_d,          n_max,      True,  False),
        ("1 m 앞에서 시간초과",     reach_d - 1.0,    n_max,      False, False),
        ("절반 가고 시간초과",      cfg.course.length / 2, n_max, False, False),
        ("제자리 교착",             0.0,              30,         False, True),
        ("즉시 낙상",               0.0,              10,         False, True),
    )


def returns(cfg: HLCConfig | None = None) -> list[tuple[str, float]]:
    cfg = cfg or default_config()
    r = cfg.reward
    return [(name, r.progress * dist + r.reach * reached
             + r.time * steps + r.terminate * bad)
            for name, dist, steps, reached, bad in scenarios(cfg)]


def main() -> int:
    cfg = default_config()
    rows = returns(cfg)
    print(f"보상 감사 — 코스 {cfg.course.length} m, 타임아웃 "
          f"{cfg.term.timeout_s} s ({int(cfg.term.timeout_s/0.1)} 스텝)")
    print(f"progress={cfg.reward.progress} reach={cfg.reward.reach} "
          f"time={cfg.reward.time} terminate={cfg.reward.terminate}")
    print(f"\n{'시나리오':24s} {'리턴':>9s}")
    print("-" * 36)
    for name, v in rows:
        print(f"{name:24s} {v:9.1f}")

    d = dict(rows)
    ok = True

    # ① 도달이 "거의 도달"보다 뚜렷하게 나아야 한다. 이 격차가 마무리 유인이다.
    gap = d["느리게 도달 (시간 임박)"] - d["1 m 앞에서 시간초과"]
    best = max(v for _, v in rows)
    print(f"\n① 도달 vs 1 m 앞 시간초과 : +{gap:.1f}  (최고 리턴의 {gap/abs(best):.0%})")
    if gap / abs(best) < 0.15:
        ok = False
        print("   ❌ 마무리 유인이 너무 작습니다. `reach`를 올리십시오.")
        print("      (2026-07-29: 11%였고 도달률이 0.17에서 안 올랐습니다)")

    # ② 빨리 끝내는 것이 이득이어야 한다. 아니면 정책이 시간을 다 쓴다.
    sp = d["빠르게 도달 (절반 시간)"] - d["느리게 도달 (시간 임박)"]
    print(f"② 빠른 도달 vs 느린 도달  : +{sp:.1f}  (최고 리턴의 {sp/abs(best):.0%})")
    if sp <= 0:
        ok = False
        print("   ❌ 서두를 이유가 없습니다. `time` 페널티를 키우십시오.")

    # ③ 순위가 뒤집히면 안 된다.
    order = [n for n, _ in sorted(rows, key=lambda kv: -kv[1])]
    want_first = "빠르게 도달 (절반 시간)"
    want_last = {"제자리 교착", "즉시 낙상"}
    print(f"③ 최상위={order[0]!r}  최하위 2개={set(order[-2:])}")
    if order[0] != want_first or set(order[-2:]) != want_last:
        ok = False
        print("   ❌ 순위가 의도와 다릅니다.")

    print("\n" + ("PASS — 보상이 의도대로 순위를 매깁니다."
                  if ok else "*** FAIL — 학습 전에 보상 상수를 고치십시오 ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
