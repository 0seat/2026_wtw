"""02. HLC 학습 실행기 (Colab).

로직은 `wtw_nav/train.py`, 모니터링은 `wtw_nav/monitor.py`에 있다. 이 파일은 호출만 한다.

Colab 셀 순서 (★ 순서가 중요하다)::

    # [셀 1] GL 백엔드 — 다른 어떤 import보다 먼저
    import os
    os.environ["MUJOCO_GL"] = "egl"

    from google.colab import drive; drive.mount('/content/drive')
    %cd "/content/drive/Othercomputers/BPC/D:/02_projects/2026_wtw"
    import sys; sys.path.append(".")
    from wtw_nav.dev import reload_wtw   # %autoreload 금지 (Colab 3.12에 imp 없음)

    # [셀 2] 코드가 반영됐는지 확인 (Drive 동기화 확인)
    from wtw_nav.llc import check
    check.version_check()

    # [셀 3] env 검증 — 학습 전에 반드시. 여기서 실패하면 학습은 시간 낭비다.
    !python -m wtw_nav.envs.test_nav_env

    # [셀 3b] 수동 제어기로 "풀 수 있는 문제인가" 확인. 여기서 도달 못 하면
    #         성능이 아니라 보상·종료·유도벡터가 틀린 것이다.
    !python -m wtw_nav.envs.scripted

    # [셀 4] 대시보드 — 학습 셀보다 **먼저** 띄워야 실시간으로 보인다
    %load_ext tensorboard
    %tensorboard --logdir runs

    # [셀 5] 학습. 노트북이면 인라인 라이브 플롯도 함께 갱신된다.
    from wtw_nav import train
    train.run()                      # ← 평지(P1). 지형 사다리는 아래 참조.

P1은 진단용으로 끝났다(`configs.PPOConfig` 주석). 지금 돌리는 것은 **P2 한계 측정**이다.

지형 사다리 실행 (P2) — 셀 5를 이걸로 대체::

    import dataclasses
    from wtw_nav.configs import default_config, TerrainConfig
    from wtw_nav import train

    KIND = "gap"
    cfg = default_config()
    cfg = dataclasses.replace(
        cfg,
        terrain=TerrainConfig(kind=KIND, values=TerrainConfig.PRESETS[KIND]),
        term=dataclasses.replace(cfg.term, timeout_s=45.0),
        reward=dataclasses.replace(cfg.reward, reach=100.0),
        ppo=dataclasses.replace(cfg.ppo, discounting=0.998, num_evals=25),
    )
    train.run(cfg, tag=f"p2_{KIND}", logdir=f"runs/p2_{KIND}")

주의 — **`tag`/`logdir`를 반드시 넘길 것.** 기본값이 `p1_flat`/`runs/hlc`라 그냥 부르면
P1 체크포인트를 덮어쓰고 텐서보드 로그가 섞인다.

주의 — **아래 세 값은 코스 길이에 물려 있다.** 코스 길이는 `terrain.values`가 정하므로
(`modules.ladder`의 `end_x`) 지형을 바꾸면 셋 다 다시 계산해야 한다. 자동화되어
있지 않다 — `NavEnv._preflight`가 어긋나면 경고를 찍으니 첫 출력을 확인할 것.

  gap 사다리 (0.05~0.30, spacing 3) -> 장애물 x=[3,6,9,12,15,18], end_x = 19.80 m
  출발 x가 ±1.2 m 랜덤이므로 **최악 21.0 m**로 잡아야 한다.

  timeout_s    최악 코스 / 0.7 m/s + 여유   21.0/0.7 = 30.0 s -> 45 s
  reach        progress 총합의 절반쯤       10 × 19.8 = 198 -> 100
  discounting  유효 지평 1/(1-γ) >= 에피소드 45 s = 450 스텝 -> 0.998 (지평 500)

★ **출발 x 랜덤화(`course.init_x`, 기본 ±1.2)가 이 측정의 타당성을 지탱한다.**
관측에 센서는 없지만 유도벡터의 `d_norm`이 들어가므로, 출발점이 고정이면 정책은
`d_norm`에서 자기 x를 역산해 **지형이 아니라 시계를 보고** 넘는다. 그러면 측정값은
"이 배치를 외운 결과"가 되어 미로로 옮겨가지 않는다. ±1.2는 장애물 주기 3 m의
80%를 덮고, 시작 발판이 x=-2.0부터라 뒤로 밀려도 발판을 벗어나지 않는다.
init_x를 0으로 두면 `_preflight`가 경고한다 — 무시하지 말 것.

읽을 지표는 `eval/episode_level` — 넘긴 장애물 수다. 이 값이 정체하는 지점이 곧
`terrain/limits.py`의 실측 한계이며, 그것이 이 실행의 산출물이다(정책이 아니다).

기준선(2026-07-31, 로컬 CPU 1 env): **고정 vx 최대 명령만으로 level 1** (x=6.26에서
두 번째 틈 0.10 m에 걸려 종료). PPO가 이보다 못 하면 학습이 아니라 설정이 문제다.

────────────────────────────────────────────────────────────────────────────
★ 경사(slope) 실험 (2026-08-02) — P3에서 유일하게 살아남은 항목
────────────────────────────────────────────────────────────────────────────

**왜 이것만 다시 재는가** (전부 `parameters.pkl` 실측, 2026-08-02 확인).
WTW `pretrain-v0`는 `terrain_proportions=[0,...,0,1.0]`이고 그 브랜치의 진폭
(`terrain_noise_magnitude`)이 0이라 **30×30 격자 전부가 평면 trimesh**였다 —
경사 브랜치(인덱스 0·1)는 한 번도 샘플되지 않았다. 반면 마찰은
`randomize_friction=True, [0.1, 3.0]`으로 학습됐고 latent가 온라인 추정하므로
LLC 소관이다. 요철도 진폭 0으로 미학습이지만 MJX box 격자 비용 때문에 제외했다.

**gap과 다른 점 둘.** ① 라이다가 램프를 **본다**(경사면이 몸통 높이 0.34 m를
가로지른다) ② 필요한 물리량(`pitch`로 몸통을 램프에 정렬)이 8D 명령 안에 **있다**.
gap에서 footswing이 무력했던 이유("발을 높이 들 뿐 멀리 보내지 않는다")가 여기엔
해당하지 않는다. 그래서 이 실험은 gap과 달리 양의 결과가 나올 수 있다.

주의 — **다만 LLC는 경사를 '간접적으로는' 이미 겪었다.** `randomize_gravity=True,
gravity_range=[-2,2]`이고 `_randomize_gravity`가 정규화된 기울어진 중력을 만들어
그것이 `projected_gravity`(LLC 관측 [0:3])의 기준이 된다. 지지면 기준으로
"기울어진 중력 + 평지" = "수직 중력 + 경사면"이므로, LLC가 겪은 **유효 경사는
평균 9.0° / p90 13.3° / 최대 19.7°**다. 따라서:

  · 10° 단은 **대조군 A가 그냥 넘을 것으로 예측**된다. 넘지 못하면 등가성 논증이
    틀린 것이고, 그때 의심할 것은 경사가 아니라 **램프 모서리**다(중력 랜덤화에는
    진입/이탈 모서리가 없다). 그 자체가 결과이므로 사다리 첫 단으로 남겨 둔다.
  · 정보가 있는 구간은 **15~30°**. HLC의 값어치는 여기서만 판정된다.
  · `level >= 2`는 HLC의 성과가 아니다. **판정 기준선은 대조군 A의 level이다.**

주의 — **상한은 마찰이 정한다.** 지형 μ=0.6이면 tan⁻¹(0.6)=30.96°가 물리적 상한이고
사다리 마지막 단 30°는 필요 μ 0.577로 여유가 4%뿐이다. 30°에서 멈추는 것은
"제어 실패"가 아니라 **마찰 상한**이며, 그렇게 해석이 확정되도록 일부러 마지막에
두었다 (`configs.TerrainConfig.PRESETS` 주석).

주의 — **대조군 A를 먼저 돌릴 것.** gap 실행의 결론이 "PPO 6.9M = 고정 명령"이었던
것을 기억할 것 — 대조군이 없었으면 5.7시간에서 아무것도 못 건졌다::

    from wtw_nav.configs import default_config, TerrainConfig
    from wtw_nav.envs.nav_env import NavEnv
    from wtw_nav.envs import scripted
    import dataclasses

    cfg = _slope_cfg()                       # 아래 정의
    env = NavEnv(cfg)                        # JIT 1회, 스윕 내내 재사용
    scripted.evaluate(env, n=3)              # A0: 전 축 중앙 고정
    scripted.axis_sweep(env, "pitch", [-0.3, -0.15, 0.0, 0.15, 0.3], n=3)
    scripted.axis_sweep(env, "height", [-0.2, -0.1, 0.0, 0.1], n=3)

`pitch` 스윕에서 이미 level이 오르면 **HLC 없이도 되는 부분**이고, PPO 결과는
반드시 그 기준선 위에서 읽어야 한다. 전 값이 동일하면 그 축은 고정값으로는
무력하다는 뜻이고, 그때 PPO가 이기면 그건 축이 아니라 **되먹임**(지형에 맞춰
값을 바꾸는 것)이 산 것 — 곧 HLC가 필요하다는 직접 증거가 된다.

★ **대조군 A 실측 (2026-08-02, Colab, 시드 3개).** B는 반드시 이 위에서 읽을 것::

    A0 (전 축 중앙)     level 2.00   (10°·15° 통과, 20° 램프에서 정지)
    pitch  -0.30 1.67 / -0.15 1.33 / 0.00 2.00 / +0.15 2.33 / +0.30 **3.00**
    height  -0.2~+0.1  전부 2.00      (완전 무효)
    footswing 0.08~0.30 전부 2.00      (완전 무효)

  ① **예측이 맞았다.** 중력 랜덤화의 유효 경사(평균 9.0°, p90 13.3°, p99 16.6°)와
     LLC 단독 능력(15° 통과, 20° 실패)이 거의 정확히 겹친다. LLC의 경사 능력은
     자기가 겪은 중력 기울기 분포와 같다.
  ② **pitch만 유효하다.** height·footswing은 값을 어떻게 줘도 level이 변하지
     않는다(소수점까지 동일). 8축 중 경사를 사는 축은 하나다.
  ③ 주의 — **최적 pitch가 상한에 붙은 채 단조 증가였다.** 그래서 `ActionConfig.pitch`를
     ±0.35 -> ±0.4(LLC 학습 범위 전체)로 넓혔다. 넓히기 전 이 표의 3.00은
     **능력이 아니라 우리가 그은 선**을 잰 값이다.
  ④ ★ **낙상이 0건이다.** 종료 스텝이 전부 30의 배수(=`stuck_steps`)였다 —
     15회 전부 `stuck`이고 `fell`은 한 번도 없었다. 즉 한계는 **안정성이 아니라
     추진력/견인력**이다. 로봇은 넘어지는 게 아니라 램프 앞에서 선다.
     -> 다음에 볼 축은 `vx_max`(대조군 제어기는 1.0인데 범위는 1.9까지다).

**B: PPO 실행** — 셀 5를 이걸로 대체::

    import dataclasses
    from wtw_nav.configs import default_config, TerrainConfig
    from wtw_nav import train

    def _slope_cfg():
        cfg = default_config()
        return dataclasses.replace(
            cfg,
            terrain=TerrainConfig(kind="slope",
                                  values=TerrainConfig.PRESETS["slope"]),
            term=dataclasses.replace(cfg.term, timeout_s=60.0),
            reward=dataclasses.replace(cfg.reward, reach=110.0),
            ppo=dataclasses.replace(cfg.ppo, num_timesteps=5_898_240,
                                    discounting=0.9985, num_evals=25),
        )

    train.run(_slope_cfg(), tag="p3_slope", logdir="runs/p3_slope")

값의 근거 (사다리 10~30°, ramp_len 2.0, spacing 3, x0 3.0):

  코스        장애물 x=[3,6,9,12,15,18], end_x = 21.23 m, 최종 높이 4.27 m
              통과 판정선 x=[5.5, 8.5, 11.4, 14.4, 17.3, 20.2]
  timeout_s   최악 (21.23+1.2)/0.7 = 32 s. 오르막은 더 느리므로 **60 s**(600스텝)
  reach       progress 총합 10×21.23 = 212의 절반 -> **110**
  discounting 유효 지평 1/(1-γ) >= 600스텝 -> **0.9985** (지평 667)
  num_timesteps  (num_evals-1)×batch×unroll×minibatch = 24×40,960의 배수로 잡아
              brax의 epoch 올림 낭비를 없앤다 -> 24×40,960×6 = **5,898,240**
              (gap 실행은 이걸 안 맞춰 6M 지정에 6.88M을 돌았다 = 45분 낭비)

주의 — 이 지형에서만 유효한 수정 셋 (2026-08-02, gap에서는 z=0이라 안 드러났다):
  · 낙상 판정이 **지면 기준(AGL)** 이 됐다. 절대 z를 쓰면 지형이 3.6 m까지
    올라가므로 `qpos[2] < 0.15`가 영원히 거짓이라 낙상이 감지되지 않았다.
  · `level` 판정선이 장애물 **끝** + 0.5 m로 바뀌었다. 옛 규칙(시작 + 0.8)은
    램프 중턱을 통과로 셌다 — 산출물 자체가 과대보고됐을 값이다.
  · 관측의 몸통 높이도 AGL이다. 절대 z는 x의 대리변수라 `init_x` 랜덤화가
    지키려던 위상 암기 차단이 무력화된다.


★★ P4 — 미로 (본 과제). Colab 셀 전체
=======================================

지형 6종 실측이 끝났다 (`limits.report()` = "전 지형 실측 완료"). 통과 가능 5종
(gap 0.05 / ledge 0.07 / slope 20° / tunnel 0.291 / rough 0.06), 제외 3종
(jump / slit / beam). 남은 것은 길찾기다.

주의 — **설정을 노트북에서 계산하지 않는다.** `configs.maze_config()`가 BFS 경로에서
timeout·reach·γ를 뽑는다. 슬로프의 상수(60 s / reach 110)를 미로에 복사했다가
21시간짜리를 걸었고, seed마다 경로가 달라 상수는 원리적으로 틀린다.

주의 — **`max_geom_pairs=32`가 기본으로 켜져 있다** (`maze_config`). 이게 없으면
78 steps/s = 21시간이다. 근거는 `docs/01_llc.md` §8.5.

---------------------------------------------------------------- 고정 셀 (3개)

[1] 설치 — 런타임당 1회::

    !pip -q install mujoco mujoco-mjx brax

[2] 마운트·경로·리로드 — 주의 — **코드를 고칠 때마다 이 셀부터**::

    import os, sys
    os.environ["MUJOCO_GL"] = "egl"          # ★ 다른 어떤 import보다 먼저
    from google.colab import drive; drive.mount('/content/drive')
    P = "/content/drive/Othercomputers/BPC/D:/02_projects/2026_wtw"
    os.chdir(P); sys.path.insert(0, P)
    from wtw_nav.dev import reload_wtw; reload_wtw()

[3] 공통 import — 주의 — **[2] 뒤에는 반드시 이 셀도 다시**::

    import dataclasses as dc
    from wtw_nav.configs import maze_config
    from wtw_nav.envs.nav_env import NavEnv
    from wtw_nav.envs import scripted as S
    from wtw_nav import bench, train
    cfg = maze_config(n=5, seed=0)

---------------------------------------------------------------- 실험 셀

[4] **게이트 ①: 물리가 옳은가.** `max_geom_pairs`는 근사다 — k가 실제 동시 접촉
    쌍보다 작으면 접촉이 조용히 누락되어 벽을 통과한다::

    env_ref = NavEnv(dc.replace(cfg, max_geom_pairs=0))   # 기준 (느림, 몇 분)
    r_ref = S.evaluate(env_ref, n=5)
    env = NavEnv(cfg)                                     # k=32
    r_bp = S.evaluate(env, n=5)
    print(f"기준 level {r_ref['level']:.2f} / k=32 level {r_bp['level']:.2f}")
    print("★ k=32가 **더 멀리 가면 불합격**입니다 — 벽을 통과한 것입니다.")

[5] **게이트 ②: env가 풀 수 있는 문제인가.** 학습 전 대조군::

    S.evaluate(env, n=5)
    S.video(env, save="p4_A.mp4", distance=8.0)
    print("도달률 0이면 학습 금지 — 흐름장/보상/종료를 먼저 보십시오.")

[6] **처리량 확인** (게이트 통과 후, 학습 직전)::

    bench.breakdown(cfg, num_envs=2048)

`broadphase 적용 {'max_geom_pairs': 32}` 줄과 ETA를 볼 것. ETA가 3시간을 넘으면
걸지 말고 원인부터 찾는다 (§8.3.2의 진단 3실험 — 벽 병합 / env 수 / geom 수).

[7] **학습** — 5.9M 스텝 ≈ 141분 + 컴파일::

    train.run(cfg, tag="p4_maze", logdir="runs/p4_maze")

곡선만 먼저 보고 싶으면 파일럿(1/6, ≈25분)::

    pilot = dc.replace(cfg, ppo=dc.replace(cfg.ppo, num_timesteps=24*40_960,
                                           num_evals=13))
    train.run(pilot, tag="p4_pilot", logdir="runs/p4_pilot")

[8] **평가 (P5) — ★ 다른 seed·다른 크기.** 학습 seed로 평가하면 암기를 못 잡는다.
    로직은 `wtw_nav/eval.py`에 있다 — 셀에는 호출만 둔다::

    from wtw_nav import eval as E
    pol, cfg_tr, log, step = E.load("checkpoints/hlc_p4_maze.pkl")
    E.curve(log)          # 어디서 수렴했나 -> 다음 실행 예산
    E.report(pol, n=20)   # 학습 미로 + seed 7·13 + n=8·11, 실패 영상 포함

★ **P4 결과 (2026-08-10)**: 도달률 **0.992**, 에피소드 91스텝(9.1 s)로 스크립트
제어기(14.4 s)보다 1.6배 빠름, 낙상 0.008. 5.9M 스텝 234분.
주의 — 도달률 95%에 **2.46M(43% 지점)** 에서 닿았다 — 다음 실행은 3.2M이면 충분하다.
상세는 `docs/03_results.md` §5.

★ 관측 37D에는 절대좌표도 맵 크기도 없으므로 **정책은 크기에 무관**해야 한다.
n=8에서 성공률이 떨어지면 먼저 의심할 것은 유도벡터의 `d_norm`이다 — 그것만
`_course_len`으로 정규화되어 있어 크기에 따라 변화율이 달라진다.

---------------------------------------------------------------- 규칙

`wtw_nav/dev.py`의 "Colab 셀 작성 규약" 7개를 따른다. 특히:
  · 코드 수정 -> [2] -> [3] -> 실험 셀 (env 캐시는 `reload_wtw`가 같이 지운다)
  · 배열 반환 함수를 셀 마지막 줄에 두지 말 것 (`evaluate`는 기본 `full=False`)
  · 영상을 같이 낼 것
"""

import os
import platform
import sys

if platform.system() == "Linux":
    os.environ.setdefault("MUJOCO_GL", "egl")


def main() -> int:
    from wtw_nav import train
    train.run()
    return 0


if __name__ == "__main__":
    rc = main()
    if "IPython" not in sys.modules:
        sys.exit(rc)
