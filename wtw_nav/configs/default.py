"""HLC 설정의 단일 진실 공급원.

여기 없는 상수를 코드에 흩어 두지 말 것 — `check.build()`가 solver 기본값을 중복
정의했다가 `create_env`만 고치고 실행은 옛 값으로 도는 사고가 실제로 있었다.

LLC 쪽 상수(명령 스케일·PD 게인·DOF 순서 등)는 `wtw_nav/llc/policy.py`가 소유한다.
이 파일은 **HLC 고유의 값**만 담는다.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ActionConfig:
    """HLC 액션(8D 연속) → LLC 명령 매핑.

    범위는 `curriculum/distribution.pkl`의 **실제 도달 영역**의 진부분집합이다
    (`docs/01_llc.md` §4). `scripts/train.py`를 근거로 쓰지 말 것.

    gait는 **trot 고정** — pronk는 도약을 만들지 못하면서 MJX에서 전복을 유발한다
    (`docs/03_results.md` §2).
    """

    # (하한, 상한). 순서 = 액션 벡터 순서
    vx: tuple[float, float] = (-1.9, 1.9)
    yaw: tuple[float, float] = (-2.4, 2.4)
    vy: tuple[float, float] = (-0.5, 0.5)
    height: tuple[float, float] = (-0.22, 0.13)
    step_freq: tuple[float, float] = (2.1, 3.9)
    footswing: tuple[float, float] = (0.05, 0.32)
    #: 주의 — 2026-08-02에 ±0.4로 넓혔다가 **되돌렸다.** 넓힌 근거는 "경사 스윕에서
    #: 최적 pitch가 상한(+0.3)에 붙은 채 단조 증가 중이니 지금 재는 것은 로봇의
    #: 능력이 아니라 우리가 그은 선"이었는데, 확장해서 재보니 **틀렸다**:
    #:     0.30 -> level 3.00 (3/3)   0.35 -> level 3.00 (3/3)
    #:     0.40 -> level 2.00 (시드 하나가 level 0으로 붕괴)
    #: 경계에 눌린 게 아니라 **0.3에서 이미 평탄역**이었고 0.4는 불안정하다.
    #: LLC 학습 범위(`limit_body_pitch = [-0.4, 0.4]`, `num_bins=1`로 균등 샘플)
    #: 안이긴 하지만, 0.35가 0.30과 동등하면서 유해 구간을 배제하므로 원래의
    #: 진부분집합 원칙이 그대로 맞았다. **3점만 보고 추세를 읽지 말 것.**
    pitch: tuple[float, float] = (-0.35, 0.35)
    stance_width: tuple[float, float] = (0.12, 0.42)

    #: vx-yaw 결합 제약 |vx|/VX_MAX + |yaw|/YAW_MAX <= 1 (도달 영역이 마름모꼴이라 필요)
    diamond_vx: float = 1.9
    diamond_yaw: float = 2.4

    #: ‖(vx,vy)‖ <= 이 값이면 0으로 스냅. **0.0 = 비활성(현재)**.
    #: 근거: LLC 학습 시 이 구간이 0으로 치환된 것은 사실이나, 2026-07-29 실측
    #: (`check.deadzone_sweep()`)에서 **LLC가 그 구간에서도 기울기 0.99로 선형
    #: 반응**했다. 반면 데드존을 켜두면 정지 편향(+0.092)을 상쇄할 vx≈-0.09가
    #: 0으로 먹혀 **정밀 정지가 원리적으로 불가능**해진다 — P2의 외나무다리 끝·
    #: 틈 직전에 치명적이다. 되살릴 이유가 생기면 근거를 여기 남길 것.
    deadzone: float = 0.0

    #: 정지 편향 보정 (m/s). LLC는 vx=0 명령에서 +0.092 m/s 전진하며, 이는
    #: 명령 크기와 무관한 **상수 오프셋**이다(0.00~0.50 전 구간에서 +0.081~+0.092).
    #: 명령에서 빼주면 "명령 vx == 실제 vx"가 되어 액션 0이 곧 정지가 된다.
    vx_bias: float = 0.092

    #: 요속 편향 보정 (rad/s). ★ **`vx_bias`의 요 짝인데 지금까지 없었다**
    #: (2026-08-09에 추가). LLC는 yaw=0 명령에서도 ω≈0.118 rad/s로 계속 돈다
    #: (`guidance.heading_hold` 주석의 유도: x_end 5.83 / |y|max 3.33 / 8 s).
    #:
    #: 주의 — **이것이 없으면 진로 유지기가 원리적으로 직진하지 못한다.**
    #: `heading_hold`는 적분항이 없는 순수 P 종속 루프(y -> ψ_des -> yaw)라
    #: **상수 외란을 제거할 수 없다**(type-0). 정상상태를 풀면
    #:     y_ss = ω_bias / (k_psi · k_y) = 0.118 / (1.5 × 0.5) = **0.157 m**
    #: 게인으로는 줄일 수만 있고 없앨 수 없다. 실제로 외나무다리 실측에서 다리
    #: 위 표류가 0.297 m로 나왔는데, 그 절반이 이 구조적 오차다 — 그리고 그
    #: 0.297은 실패한 다리의 **반폭 0.275보다 크다**. 즉 이 항이 없으면 재는 것이
    #: 다리 폭이 아니라 **내 제어기의 정상상태 오차**다.
    #:
    #: ★ 기본값은 **0.0(비활성)** 이다. 부호를 추측하면 편향이 2배가 되므로
    #: `envs.scripted.yaw_bias_estimate()`로 **재고 나서** 넣을 것.
    #: 편향이 상태 의존이면(속도·명령에 따라 변하면) 전방보정으로는 부족하고
    #: `heading_hold`에 적분항이 필요하다 — 그때는 여기 근거를 남길 것.
    yaw_bias: float = 0.0

    #: 명령 1차 저역필터 계수. new = a*cmd + (1-a)*prev
    lowpass_alpha: float = 0.3

    gait: str = "trot"

    @property
    def ranges(self) -> tuple[tuple[float, float], ...]:
        return (self.vx, self.yaw, self.vy, self.height,
                self.step_freq, self.footswing, self.pitch, self.stance_width)

    @property
    def size(self) -> int:
        return len(self.ranges)


@dataclasses.dataclass(frozen=True)
class CourseConfig:
    """P1 훈련 코스: 장애물 없는 평지 직선."""

    #: 주의 — **정책의 실제 속도로 타임아웃 안에 도달 가능해야 한다.** 2026-07-29:
    #: 20 m + 30 s로 두었더니 정책이 0.61 m/s로 정착해 19.5 m에 32 s가 필요했고,
    #: **도달이 원리적으로 불가능**했다. 보너스를 한 번도 못 받으니 그것이 좋다는
    #: 것도 배울 수 없다(탐색 실패). 10 m면 0.61 m/s로도 16 s — 여유 47%.
    #: 코스 길이는 P1(인터페이스 검증)의 본질이 아니므로 줄여도 목적을 해치지 않는다.
    length: float = 10.0          # 목표까지 거리 (m)
    goal_radius: float = 0.5      # 이 안에 들면 도달
    #: 초기조건 랜덤화 (A안, 소폭) — 훈련↔배포 분포 일치가 목적
    init_yaw: float = 0.30        # rad, ±
    init_lateral: float = 0.15    # m, ±
    init_vx: float = 0.0          # m/s, ± (P1은 정지 출발)
    #: ★ 출발 x 랜덤화 (m, ±). **지형 사다리의 측정 타당성이 여기 걸려 있다.**
    #: 관측에 센서는 없지만 유도벡터의 `d_norm`이 들어가므로, 출발점이 고정이면
    #: 정책은 `d_norm`에서 자기 x를 정확히 역산할 수 있다. 사다리의 장애물은
    #: x=[3,6,9,...]에 고정이므로 그러면 정책은 **지형이 아니라 시계를 보고** 넘고,
    #: 측정값은 "이 배치를 외운 결과"가 되어 미로로 옮겨가지 않는다.
    #: ±1.2 m는 장애물 간격(3 m)의 80%를 덮으므로 위상 암기를 깨기에 충분하고,
    #: 시작 발판이 x=-2.0부터라 뒤로 밀려도 발판을 벗어나지 않는다
    #: (`terrain.modules.ladder`의 첫 `_slab(x=-2.0, ...)`).
    init_x: float = 1.2


@dataclasses.dataclass(frozen=True)
class MazeConfig:
    """격자 미로 (P4). `enabled=False`면 지형/평지 코스를 쓴다.

    ★ 범위 재정의(2026-08-01, `docs/decisions/0001-지형-돌파-폐기.md`) 이후 **이것이 본 과제다.**
    지형 돌파는 8D 명령의 실측 능력이 3 cm라 폐기됐고, 남은 것은 길찾기다.

    주의 — 미로 하나는 MJX 모델 하나다 — env마다 다른 레이아웃을 줄 수 없다
    (`terrain.maze` 모듈 주석). 일반화를 보려면 `seed`를 바꿔 **따로** 평가한다.
    """

    enabled: bool = False
    #: 셀 수 (n×n). 피치 1.4 m이므로 5면 7 m, 8이면 11.2 m 사방이다.
    n: int = 5
    seed: int = 0
    #: 벽을 더 허무는 비율. 0이면 완전미로라 경로가 유일해서 "길찾기"가 아니라
    #: "선 따라가기"가 된다 (`terrain.maze.generate` 참조).
    loop_prob: float = 0.1
    #: 출발 셀 안에서의 위치 랜덤화 (m, ±). 통로 폭 1.2의 절반을 넘기면 벽에 낀 채
    #: 시작한다.
    init_jitter: float = 0.25


@dataclasses.dataclass(frozen=True)
class TerrainConfig:
    """지형 난이도 사다리 (`terrain.modules.ladder`). `kind=None`이면 평지(P1).

    ★ **커리큘럼 스케줄러가 없는 것이 의도다.** 난이도를 시간이 아니라 **코스 축**에
    배열하면 정책은 쉬운 것부터 만나고 못 넘는 곳에서 멈춘다. 레벨업 임계값도,
    모델 재빌드도, MJX 모델 배치화도 필요 없다. 그리고 **도달 거리가 곧 한계 측정**이라
    `terrain/limits.py`가 이 값으로 채워진다.

    주의 — **손 스윕으로 한계를 재려던 시도는 폐기됐다** (2026-07-29). 조작 축이 7개라
    조합이 폭발하고(4^7≈16k), 손으로 고른 조합은 "두 축이 동시에 있어야 넘는"
    종속성을 원리적으로 못 찾는다. 여기서는 PPO가 8축을 동시에 굴려 그 탐색을 한다.

    주의 — 종류를 섞지 말 것. 섞으면 "어디서 막혔나"가 "무엇 때문에 막혔나"를 못 준다.
    """

    kind: str | None = None
    #: 쉬운 것부터. `ledge`는 각 단의 **증분**(누적된다), `slope`는 도(°).
    #: 상한은 넉넉하게 둔다 — 정책이 못 넘으면 거기서 멈출 뿐 손해가 없고,
    #: 반대로 짧게 잡으면 진짜 한계를 영영 못 본다.
    values: tuple[float, ...] = ()
    spacing: float = 3.0          # 장애물 간 평지 (자세 회복 여유)
    x0: float = 3.0               # 첫 장애물까지 조주 거리

    #: 기본 사다리 — `kind`만 바꿔 쓸 때의 값.
    #:
    #: ★ `slope`는 2026-08-02에 `(5,10,15,20,25,30)`에서 바뀌었다. 근거 둘,
    #:   **둘 다 `parameters.pkl` 실측이다**:
    #:
    #:   ① 아래쪽이 낭비였다. `randomize_gravity=True, gravity_range=[-2,2]`이고
    #:      `_randomize_gravity`가 `gravity_vec = (U[-2,2]³ + [0,0,-9.8])/‖·‖`로
    #:      **정규화된 중력**을 만든다 — 그리고 그것이 LLC 70D 관측의 첫 3개
    #:      `projected_gravity`의 기준이다. 지지면 기준으로 "기울어진 중력 + 평지"와
    #:      "수직 중력 + 경사면"은 같은 상황이므로, LLC는 램프를 걸은 적은 없어도
    #:      **유효 경사 평균 9.0° / p90 13.3° / 최대 19.7°를 이미 겪었다**
    #:      (10° 이상이 전체의 40.6%). 5°·10° 단은 학습 분포 안이라 정보가 없다.
    #:
    #:   ② 위쪽이 마찰 상한에 걸린다. 지형 μ=0.6(`modules.FRICTION`, menagerie
    #:      floor와 동일)이면 정지마찰 상한은 tan⁻¹(0.6) = **30.96°**다. 30°는 필요
    #:      μ 0.577로 여유가 4%뿐이라, 거기서 미끄러지면 그것은 제어 능력이 아니라
    #:      **마찰**을 잰 것이다. 그래서 30°를 마지막 단으로 두어 **측정을 가둔다** —
    #:      30°에서 멈추면 해석이 확정된 실패다. μ를 올려 이 벽을 밀 수는 있지만
    #:      그러면 미로(같은 μ)와 조건이 달라진다.
    #:
    #:   주의 — 5°/10°에서 대조군 A가 죽으면 ①의 등가성 논증이 틀린 것이다. 그때
    #:      의심할 것은 경사 자체가 아니라 **램프 진입/이탈 모서리**다(중력
    #:      랜덤화에는 모서리가 없다).
    PRESETS = {
        "gap":    (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
        #: 턱 각 단의 **증분** (누적 0.55 m). 주의 — 2026-08-09에 재설계했다.
        #: 옛 값 `(0.03, 0.03, 0.03, 0.04, 0.04, 0.05)`는 **1~3단이 동일 난이도**라
        #: 사실상 {0.03, 0.04, 0.05} 세 점만 구분했고, 그 전 구간이 요철 실측이
        #: 이미 통과시킨 평균 단차 0.040 m 근처에 몰려 있었다 — 요철 1차 사다리와
        #: 같은 **천장 치기**를 반복하는 값이다.
        #: 등간격 + 요철 최대 단차(2a = 0.12 m)를 덮도록 다시 잡았다:
        "ledge":  (0.03, 0.05, 0.07, 0.10, 0.13, 0.17),
        #: 외나무다리 **전폭** m. 주의 — 이 값은 `scripted.foot_track()`의 실측
        #: (= 명령별 실제 발 간격 `W_foot`)이 나온 뒤 그 주변으로 다시 잡는다.
        #: 옛 값의 2.0 m는 요철 띠 반폭(1.0)의 2배로 **사실상 평지**였고, 0.3 m는
        #: `stance_width` 하한 0.12에 발 지름 0.046을 더한 0.166보다는 넓지만
        #: 표류 여유가 0이라 어느 쪽도 유용한 단이 아니다. 잠정값으로 두되
        #: 실측 셀은 `terrain_env("beam", values=...)`로 **덮어쓴다**.
        "beam":   (1.2, 0.9, 0.7, 0.55, 0.45, 0.35),
        "tunnel": (0.50, 0.45, 0.40, 0.35, 0.30, 0.25),
        "slope":  (10.0, 15.0, 20.0, 24.0, 27.0, 30.0),   # 도. 상한은 마찰이 정한다
        #: 요철 **진폭** ±m. 주의 — 이웃 타일 단차는 최대 **2배**다 (`modules._rough_patch`).
        #: 0.02는 단차 0.04로 gap 실측(0.05)과 같은 자릿수이고, 0.07은 단차 0.14로
        #: `ledge` 누적 상한(0.22)의 절반이다 — 즉 이 사다리는 "확실히 되는 곳"에서
        #: "확실히 안 되는 곳"까지를 6단에 담는다. 상한을 더 올리지 않는 이유는
        #: LLC가 `measure_heights=False`라 발 착지점을 지형에 맞출 수단이 아예
        #: 없기 때문이다 — 그 위는 재 봐야 전부 실패다.
        #: 주의 — **첫 사다리 (0.02~0.07)는 너무 쉬웠다** (2026-08-06). 대조군 A가
        #: level 5.33, 여러 고정값이 6.00(완주)으로 **천장을 쳤다** — 벽을 못 찾았다.
        #: 원인은 "단차 = 2×진폭"이라는 내 headline이 최악값이었다는 것이다. 이웃
        #: 타일 높이가 독립 U(-a,a)면 단차는 삼각분포라 **평균 |Δ| = 2a/3**이다:
        #:     a=0.07 -> 평균 0.047 m (gap 실측 0.05의 93%), 최대 0.140 m
        #: 즉 최고단조차 평균적으로는 gap 한계와 같은 수준이었다. 우연이 아니라
        #: 같은 자릿수의 문제를 재고 있었던 셈이고, 그래서 전부 통과했다.
        #: 평균 단차 기준으로 등간격이 되게 다시 잡는다 (0.04 -> 0.14 m):
        "rough":  (0.06, 0.09, 0.12, 0.15, 0.18, 0.21),
    }


@dataclasses.dataclass(frozen=True)
class RewardConfig:
    #: 주의 — 이 셋의 **상대 크기**가 정책 성격을 정한다. 2026-07-29 실패 기록:
    #: progress 10 / reach 10 / time -0.01 로 두었더니 리턴 184.6 중 183.8이
    #: progress였다 — 도달 보너스와 시간 페널티가 반올림 오차 수준이라
    #: **마무리할 이유도 서두를 이유도 없는** 정책이 나왔다. 20 m를 30초에 걸쳐
    #: 0.61 m/s로 느긋하게 가서 목표 1.2 m 앞에 멈췄고, 도달률이 0.17에서
    #: 6.5M 스텝 동안 오르지 않았다.
    progress: float = 10.0        # 유도 방향 진행량 (누적 = 10 × 이동거리)
    #: **progress 총합(= progress × 코스길이)의 절반쯤**으로 잡는다. 너무 작으면
    #: 마무리 유인이 사라지고(위 기록), 너무 크면 희소한 대형 보너스가 어드밴티지를
    #: 튀게 해 KL이 폭주한다. 코스 10 m -> 총합 95 -> reach 50.
    #: **코스 길이를 바꾸면 이 값도 바꾸고 `reward_audit`을 다시 돌릴 것.**
    reach: float = 50.0           # 도달 보너스
    #: 300스텝 = -15. 도달을 200스텝에 하면 300스텝보다 5점 이득 -> 속도에 값이 생긴다.
    time: float = -0.05           # 시간 페널티 (스텝당)
    terminate: float = -5.0       # 전복·추락·교착
    action_rate: float = -0.01    # 명령 급변 (|Δcmd|²)


@dataclasses.dataclass(frozen=True)
class TerminationConfig:
    min_height: float = 0.15      # 몸통 z 하한
    max_tilt: float = -0.5        # proj_gravity z 상한 (직립 -1, 이보다 크면 전복)
    #: P1 평지 20 m 기준. 주의 — 20 s로 두면 안 된다 — 수동 기준 제어기(vx 1.0)가
    #: **19.1 s**를 쓰므로 여유가 0.9 s뿐이고, 그보다 조금만 느린 정책은 전부
    #: 타임아웃되어 도달 보너스를 한 번도 못 받는다. 정책은 vx를 1.9까지 낼 수 있어
    #: 원리상 10.5 s면 가므로, 학습이 그걸 찾을 여지를 준다.
    timeout_s: float = 30.0
    #: 교착 판정: 이 시간 동안 진행이 stuck_dist 미만이면 종료
    stuck_window_s: float = 3.0
    stuck_dist: float = 0.1


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    #: 주의 — 이 값은 **HLC 스텝**을 센다. HLC 1스텝 = LLC 5 × 물리 4 = **물리 20스텝**이므로
    #: 10M은 곧 2억 번의 `mjx.step`이다. 벽시계 시간을 가늠할 땐 항상 ×20 할 것.
    #: P1(평지 직선 20 m)에서 정책이 배울 것은 "vx 최대, yaw ∝ φ" 하나뿐이라
    #: 30M은 과했다. 실제 소요는 `python -m wtw_nav.bench`로 먼저 재고 정하라.
    #: 주의 — **P1은 진단용 짧은 실행이다** (2026-07-29 결정). P1 정책은 P2로 이어붙일
    #: 수 없다 — P2 관측은 ~120D(센서 3종), P1은 21D라 입력 차원이 다르다. 따라서
    #: 도달률 95% 게이트를 채울 실익이 없고, `reached`가 **오르는 추세**만 확인하면
    #: 된다. 819k 스텝에서 이미 dist 19.6->2.8이었으므로 4M(48회 갱신)이면 보인다.
    #: 실제로 쓸 정책을 학습할 때(P2~)는 100회 이상으로 되돌릴 것.
    num_timesteps: int = 6_000_000
    num_envs: int = 2048
    #: 주의 — brax는 `batch_size × unroll_length × num_minibatches` 만큼의 env 스텝을
    #: 모아야 **정책을 한 번** 갱신한다. **이 값이 학습 속도의 병목이다** — 데이터가
    #: 아니라 갱신 횟수가 부족해서 실패한다. 2026-07-29 3회 실패 기록:
    #:   1024×20×32 = 655,360 -> 39회 갱신, 정점 후 붕괴
    #:    512×10×16 =  81,920 -> 48회 갱신, kl_mean 3.4로 폭주(가치함수 미수렴)
    #:    256× 5×16 =  20,480 -> 195회. **그런데 더 느려졌다** (1.35M에서 코스의
    #:                            21%만 주파, 이전 동일 스텝에서 91%였다)
    #:    256×10×16 =  40,960 -> 97회. **리턴이 53.2 -> 25.6으로 단조 하강.**
    #:    512×10× 8 =  40,960 -> 97회. 현재값 — 아래 참조.
    #:
    #: 주의 — **`unroll_length`로 갱신 횟수를 벌지 말 것.** brax GAE는 `unroll_length`
    #: 스텝만 보고 나머지는 `bootstrap_value`(가치함수)로 메운다. unroll 5 +
    #: discounting 0.997(유효 지평 333스텝)이면 어드밴티지가 거의 전부 가치함수
    #: 추정에 의존하는데, 초기에는 그게 가장 부정확하다. brax ant가 unroll 5를 쓰는
    #: 것은 γ=0.97(지평 33)이기 때문이다.
    #:
    #: ★ **`batch_size`로도 벌지 말 것 — `num_minibatches`로 벌어라.** (2026-07-29)
    #: brax에서 **그래디언트 1회의 표본 수 = `batch_size × unroll_length`**이고,
    #: `num_minibatches`는 그 그래디언트를 몇 번 밟는지만 정한다. 즉 `batch_size`를
    #: 줄이면 갱신 횟수는 늘지만 **그래디언트 잡음이 같은 비율로 커진다.** 6번째
    #: 실행에서 batch 512->256으로 미니배치가 5,120->2,560이 됐는데 lr은 3e-4
    #: 그대로였고, 리턴 자체가 단조 하강했다(보상 해킹이 아니라 최적화 실패).
    #: 지금 값은 **깨끗했던 미니배치 5,120을 복원하면서 갱신 횟수도 유지**한다:
    #:     512 × 10 × 8 = 40,960  (미니배치 5,120, 6M -> 146회 갱신)
    #: 이러면 "미니배치 축소" 가설과 "갱신 부족" 가설을 동시에 제거한다. 그래도
    #: 하강하면 남은 후보는 lr이므로 3e-4 -> 1.5e-4로 내릴 것.
    #: 제약: `batch_size × num_minibatches`가 `num_envs`의 배수여야 한다 (512×8=4096).
    batch_size: int = 512
    unroll_length: int = 10
    num_minibatches: int = 8
    num_updates_per_batch: int = 4
    learning_rate: float = 3e-4
    #: 주의 — 1e-2는 **너무 크다.** 실패한 실행에서 entropy_loss(-0.0536)가
    #: policy_loss(-0.00025)의 200배라 옵티마이저가 리턴이 아니라 엔트로피를
    #: 최대화했고, 정책 std가 0.8에 붙박인 채 성능이 단조 하강했다.
    #: 여전히 안 배우면 3e-4까지 더 낮출 것.
    entropy_cost: float = 1e-3
    #: 0.99는 유효 지평이 100스텝=10 s인데 코스 주파는 20~30 s다 — 가치함수가
    #: 목표를 못 본다. 0.997이면 지평 333스텝으로 에피소드 전체를 덮는다.
    discounting: float = 0.997
    #: 주의 — **`num_evals`는 예산을 덮어쓴다.** brax는 eval마다 최소 1회 학습을 하므로
    #: 실제 스텝 수 >= num_evals × env_step_per_training_step 이다. 40 × 655,360
    #: = 26.2M이라, `num_timesteps=10M`을 줬는데 25.5M을 돌았다(2026-07-29).
    #: eval 자체도 비쌌다 — 실패한 실행에서 101분 중 18분이 eval이었다.
    num_evals: int = 10
    #: ★ 그래디언트 노름 클리핑. **없으면 한 번의 이상치가 파라미터를 NaN으로 만든다.**
    #: 2026-08-01 p2_gap: 6.88M 스텝에서 파라미터가 NaN이 되어 5.7시간짜리 실행이
    #: 마지막 `assert_is_replicated`에서 죽었고 체크포인트도 못 남겼다. env 쪽
    #: NaN 안전망(`nav_env.step`)과 **둘 다** 필요하다 — 한쪽은 물리 발산을,
    #: 이쪽은 최적화 발산을 막는다.
    max_grad_norm: float | None = 1.0
    seed: int = 0
    #: MLP 3층. Critic도 동일 구조 별도 (docs/02_hlc.md §3)
    policy_hidden: tuple[int, ...] = (512, 256, 128)
    value_hidden: tuple[int, ...] = (512, 256, 128)


@dataclasses.dataclass(frozen=True)
class HLCConfig:
    action: ActionConfig = dataclasses.field(default_factory=ActionConfig)
    course: CourseConfig = dataclasses.field(default_factory=CourseConfig)
    terrain: TerrainConfig = dataclasses.field(default_factory=TerrainConfig)
    maze: MazeConfig = dataclasses.field(default_factory=MazeConfig)
    reward: RewardConfig = dataclasses.field(default_factory=RewardConfig)
    term: TerminationConfig = dataclasses.field(default_factory=TerminationConfig)
    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)

    #: HLC 10 Hz / LLC 50 Hz
    decimation: int = 5

    #: geom 충돌 필터. MJX 처리량을 가장 크게 좌우한다 — menagerie 원본("full")은
    #: 43개 geom이 서로 충돌해 매 스텝 ~900쌍을 계산하지만 평지에서 실제 접촉은 4개다.
    #: "world"는 자기충돌만 끄므로 다리 vs 지형(P2)은 보존된다.
    #: 자세한 근거와 선택지는 `llc.policy._apply_collision_filter` 참조.
    collision: str = "world"

    #: ★★ MJX의 **근사 broadphase** (2026-08-10에 발견). 0이면 끈다(기본).
    #:
    #: MJX는 후보 쌍을 컴파일 시점에 정적으로 펼치므로 로봇이 미로 반대편에 있어도
    #: 벽 전부와 좁은단계(narrow phase)를 돈다 — 그래서 미로에서 처리량이 평지
    #: 3,096 -> 78 HLC steps/s가 됐다. **그런데 MJX에 컬링 수단이 있다.**
    #: `collision_driver`가 custom numeric `max_geom_pairs`를 읽어
    #:     dist = ‖p2-p1‖ - (rbound1 + rbound2)
    #:     top_k(-dist, k=max_geom_pairs)
    #: 로 **경계구 거리 기준 가장 가까운 k쌍만** 좁은단계로 보낸다. 즉 거리 계산은
    #: 전부 하되(싸다) 비싼 CAPSULE-BOX·BOX-BOX는 k개만 돈다.
    #:
    #: ★ 이것이 `feet`/`maze` 같은 geom 축소와 다른 점: **맵 크기에 O(1)이다.**
    #: 미로를 5×5에서 11×11로 키워도 좁은단계 비용이 그대로다.
    #:
    #: 주의 — **근사다.** k가 실제 동시 접촉 쌍보다 작으면 접촉이 조용히 누락되어
    #: 로봇이 벽을 통과한다. 그래서 기본은 **끔**이고, 켤 때는 반드시 `world`와
    #: A/B로 궤적이 유지되는지 확인한다 (`full` -> `world` 전환과 같은 절차).
    #: 주의 — HFIELD·PLANE 쌍은 이 컬링을 **건너뛴다**(MJX 내부 `_MAX_NCON=8`).
    #:    즉 바닥 접촉은 영향을 받지 않는다 — 다행이다.
    #: 주의 — 캡은 **geom 종류 쌍마다** 적용된다(CAPSULE-BOX 따로, BOX-BOX 따로).
    max_geom_pairs: int = 0
    #: 솔버로 보내는 접촉점 수 상한 (condim 종류별). 0이면 끈다.
    #: 넘치면 **관통이 깊은 것부터** 남긴다. 쌍이 아니라 접촉점 쪽 상한이다.
    max_contact_points: int = 0

    ckpt: str = ("walk-these-ways/runs/gait-conditioned-agility/pretrain-v0"
                 "/train/025417.456545/checkpoints")
    xml: str = "mujoco_menagerie/unitree_go1/scene.xml"


def default_config() -> HLCConfig:
    return HLCConfig()


#: P4 미로 학습의 broadphase 캡. 2026-08-10 실측에서 `world`(전 geom 유지) 기준
#: 78 -> **708 HLC steps/s (9.1배)**, 5.9M 스텝 ETA 21시간 -> 141분.
#: 주의 — 근사이므로 A/B 게이트를 통과한 값만 쓸 것 (`HLCConfig.max_geom_pairs`).
MAZE_PAIRS = 32


def maze_config(n: int = 5, seed: int = 0, cruise: float = 0.7,
                corner_slack: float = 1.5, **over) -> HLCConfig:
    """★ P4 미로 설정 — **타임아웃·reach·감가를 BFS 경로에서 계산한다.**

        cfg = maze_config()                       # 5×5, seed 0
        cfg = maze_config(n=8, max_geom_pairs=0)  # 크게, broadphase 끄고

    주의 — 노트북에서 손으로 계산하지 말 것. 슬로프 실행의 상수(60 s / reach 110)를
    미로에 그대로 복사했다가 **21시간짜리 학습을 걸었다**(2026-08-09). 미로는
    seed마다 경로 길이가 다르므로 상수를 박으면 seed를 바꾸는 순간 틀린다.

    계산 규칙(근거는 `envs.nav_env._preflight`가 검사하는 것과 같다):
        timeout = (경로 + 출발흔들림) / cruise × corner_slack   ← 코너 감속 여유

    주의 — `corner_slack`은 1.3에서 **1.5로 올렸다** (2026-08-10). 1.3(5×5에서 21 s)로
    두니 스크립트 제어기가 14.1~15.7 s에 도달해 `evaluate`의 타임아웃 여유가
    25~30%로 기준(30%)에 걸렸다. 미로는 코스를 줄일 수 없으므로 여유는 여기서만
    만들 수 있다. 에피소드가 길어져도 `num_timesteps`는 env 스텝 수라 **총 학습
    시간은 그대로**다 — 에피소드 수가 줄 뿐이다.
        reach   = progress × 경로 × 0.5                          ← progress 총합의 절반
        gamma   = 1 - 1/(에피소드 스텝 × 1.1)                     ← 유효 지평 >= 에피소드

    주의 — 경로 길이를 알려면 미로를 만들어야 하는데(BFS), 모델까지 컴파일하면
    수십 초다. 여기서는 **격자만** 만들어 BFS를 돌린다(밀리초).

    Args:
        over: `HLCConfig` 필드를 그대로 덮어쓴다 (`ppo=`, `max_geom_pairs=` 등).
    """
    import dataclasses as dc

    from wtw_nav.terrain import maze as mz

    b = HLCConfig()
    vw, hw = mz.generate(n, seed=seed, loop_prob=b.maze.loop_prob)
    hops = int(mz.distance_field(vw, hw, (n - 1, n - 1))[0, 0])
    L = hops * mz.PITCH

    timeout = round((L + b.maze.init_jitter) / cruise * corner_slack)
    steps = int(timeout / (0.02 * b.decimation))

    # 주의 — **`num_timesteps`를 갱신 단위의 배수로 맞춘다.** brax는 eval 구간마다
    #    갱신 횟수를 **올림**하므로 어긋나면 지정보다 더 돈다 — gap 실행이 6M
    #    지정에 6.88M을 돌아 45분을 버렸다. 기본 6,000,000은 24×40,960의 배수가
    #    아니어서 미로에서도 같은 낭비가 난다(6.10 -> 7회로 올림).
    n_eval = 25
    per_update = b.ppo.batch_size * b.ppo.unroll_length * b.ppo.num_minibatches
    n_update = round(b.ppo.num_timesteps / per_update / (n_eval - 1))
    total = per_update * (n_eval - 1) * n_update

    cfg = dc.replace(
        b,
        maze=dc.replace(b.maze, enabled=True, n=n, seed=seed),
        term=dc.replace(b.term, timeout_s=float(timeout)),
        reward=dc.replace(b.reward, reach=round(b.reward.progress * L * 0.5)),
        ppo=dc.replace(b.ppo, discounting=round(1.0 - 1.0 / (steps * 1.1), 4),
                       num_evals=n_eval, num_timesteps=total),
        max_geom_pairs=MAZE_PAIRS)
    print(f"미로 {n}×{n} seed {seed}: BFS {hops}홉 = {L:.1f} m -> "
          f"timeout {timeout}s({steps}스텝)  reach {cfg.reward.reach}  "
          f"γ {cfg.ppo.discounting}  max_geom_pairs {cfg.max_geom_pairs}")
    print(f"  PPO {total:,} 스텝 = {per_update:,} × {n_eval - 1} eval구간 × "
          f"{n_update}회 (올림 낭비 0)")
    return dc.replace(cfg, **over) if over else cfg
