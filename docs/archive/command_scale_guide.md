# Walk These Ways (WTW) 15차원 명령(Command) 스케일 가이드

> 📖 **이식 전 과정은 [`wtw_porting_guide.md`](wtw_porting_guide.md)를 보십시오.**
> 이 문서는 **15D 명령 벡터의 레퍼런스**만 다룹니다. 관측 조립·DOF 순서·물리 설정·
> 리셋 자세·검증 절차 등 이식에 필요한 나머지는 전부 그쪽에 순서대로 정리돼 있습니다.
> (이 문서만 보고 작업하면 DOF 순서 같은 함정을 놓칩니다.)

이 문서는 WTW 하위 제어기(LLC)가 입력으로 받는 15차원 명령 벡터의 각 차원별 의미와 스케일(Scale) 값을 정리한 가이드입니다. 상위 제어기(HLC)가 정책을 학습하거나 명령을 내릴 때 반드시 이 스케일 값을 곱하여 신경망 관측(Observation)으로 넣어주어야 로봇이 정상적으로 작동합니다.

> ## ⚠️ 먼저 읽을 것: 인터페이스 15D ≠ 조작 가능한 15D
>
> **WTW의 명령 인터페이스는 15차원이 맞습니다.** 아래 표의 모든 슬롯이 실제로 존재합니다.
> 그러나 **우리가 보유한 공개 체크포인트 `pretrain-v0`가 실제로 학습한 차원은 그보다 적습니다.**
> 인터페이스에 슬롯이 있다는 것과 그 슬롯이 학습됐다는 것은 별개입니다.
>
> - `duty`(8), `roll`(11) — **상수로 고정된 채 학습됨. 조작해도 무효.**
> - `gait_phase/offset/bound`(5,6,7) — **`{0, 0.5}`로 양자화. 연속값 불가, 이산 4종만.**
>
> 논문·시연 영상에 나오는 duty 스윕은 **별도 config로 학습된 다른 정책**입니다.
> 상세는 §2의 "학습 상태" 열과 §5를 보십시오.

## 1. 15차원 커맨드 배열 (`commands_scale`)

모든 명령 벡터는 시뮬레이션 환경에 전달되기 직전, 아래의 `commands_scale` 배열과 요소별 곱(Element-wise multiplication)이 이루어집니다.

```python
import jax.numpy as jnp

# 반드시 순서대로 적용되어야 하는 스케일 배열
commands_scale = jnp.array([
    2.0, 2.0, 0.25, 2.0, 1.0, 
    1.0, 1.0, 1.0, 1.0, 0.15, 
    0.3, 0.3, 1.0, 1.0, 1.0
])
```

## 2. 각 차원별 의미 · 기본값 · **학습 상태**

"학습 상태"는 체크포인트에 동봉된 `parameters.pkl`(이 가중치를 만든 config 그 자체)에서 읽은 값입니다. 근거 파일과 재확인 방법은 §5.

| 인덱스 | 파라미터명 | 스케일 | 단위 | 설명 | Trotting 기본값 | **이 체크포인트의 학습 상태** |
|---|---|---|---|---|---|---|
| **0** | `vx` | 2.0 | m/s | 전후 선속도 (X축) | `1.0` (전진) / `0.0` (정지) | ✅ 도달 **±2.38** (yaw와 결합, §4) |
| **1** | `vy` | 2.0 | m/s | 좌우 게걸음 속도 (Y축) | `0.0` | ✅ `[-0.6, 0.6]` 전 구간 |
| **2** | `yaw_v` | 0.25 | rad/s | 제자리 회전 각속도 (Yaw) | `0.0` | ✅ 도달 **[-3.33, +2.86]** (vx와 결합) |
| **3** | `height` | 2.0 | m | 베이스(몸통) 높이 추종 | `0.0` (기본 높이) | ✅ `[-0.25, 0.15]` 전 구간 |
| **4** | `step_freq` | 1.0 | Hz | 발걸음 빈도(주파수) | `3.0` | ✅ `[2.0, 4.0]` 전 구간 |
| **5** | `gait_phase` | 1.0 | rad | 걸음걸이 위상차 (Phase) | **`0.5`** | ⚠️ **`{0, 0.5}` 양자화** |
| **6** | `gait_offset` | 1.0 | rad | 걸음걸이 오프셋 | **`0.0`** | ⚠️ **`{0, 0.5}` 양자화** |
| **7** | `gait_bound` | 1.0 | rad | 걸음걸이 바운드(Bound) 비율 | **`0.0`** | ⚠️ **`{0, 0.5}` 양자화** |
| **8** | `duration` | 1.0 | ratio | 체공(Flight) / 지지(Stance) 듀티 비율 | `0.5` | ❌ **상수 `0.5` 고정 — 조작 불가** |
| **9** | `foot_height` | 0.15 | m | 발을 들어 올리는 높이(Footswing) | `0.08` | ✅ `[0.03, 0.35]` 전 구간 |
| **10** | `pitch` | 0.3 | rad | 몸통 기울기 (Pitch 각도) | `0.0` | ✅ `[-0.4, 0.4]` 전 구간 |
| **11** | `roll` | 0.3 | rad | 몸통 기울기 (Roll 각도) | `0.0` | ❌ **상수 `0.0` 고정 — 조작 불가** |
| **12** | `stance_width` | 1.0 | m | 다리를 벌리는 너비 | `0.25` | ✅ `[0.10, 0.45]` 전 구간 |
| **13** | `stance_length`| 1.0 | m | 다리를 앞뒤로 벌리는 길이 | `0.45` | ✅ `[0.35, 0.45]` (HLC는 고정 사용) |
| **14** | `aux` | 1.0 | - | 보조 파라미터 (일반적으로 사용 안함) | `0.0` | — `[0, 0.01]` (사용 안 함) |

> 📌 **인덱스 5·6·7의 Trotting 기본값 정정**: 이전 판에는 `(0.5, 0.5, 0.5)`로 적혀 있었으나
> 이는 어떤 프리셋도 아닙니다. WTW 프리셋(`scripts/play.py:102`)은 다음과 같습니다:
> ```
> pronking [0, 0, 0]   trotting [0.5, 0, 0]   bounding [0, 0.5, 0]   pacing [0, 0, 0.5]
> ```
> **트로팅은 `(0.5, 0.0, 0.0)`입니다.**

> ## ⚠️ DOF 순서는 `FL, FR, RL, RR` 입니다
>
> `go1.urdf` 파일의 joint 등장 순서는 `FR, FL, RR, RL`이지만 **Isaac이 돌려주는 DOF 순서는
> 그것이 아닙니다.** 관측의 `dof_pos`/`dof_vel`, 12차원 행동, `clock_inputs`가 전부 이 순서를
> 따릅니다. 좌우를 바꿔 놓으면 로봇이 액션 포화(±10)와 함께 주저앉습니다 — 실제로 겪었고,
> 진단 기록은 `docs/llc_port_debug.md`에 있습니다.
>
> 이 순서로 놓아야 위 프리셋 이름이 `clock_inputs`의 실제 발 조합과 맞습니다:
> `offset=0.5` → FL·FR 대 RL·RR = 앞뒤쌍 = **bounding**,
> `bound=0.5` → FL·RL 대 FR·RR = 좌우쌍 = **pacing**.
>
> 참고: 관절의 **부호·영점·프레임 규약은 URDF와 MJCF가 완전히 동일**합니다
> (순기구학 대조 오차 `0.000000 m`). 다른 것은 순서뿐입니다.

## 3. 코드 적용 예시

HLC(상위 제어기)에서 15개의 인자를 생성했다면, JAX 환경에서 다음과 같이 조합하여 사용해야 합니다.

```python
# 1. HLC 또는 사용자 입력 (예: 앞으로 1m/s로 이동하는 트로팅)
commands = jnp.array([
    1.0, 0.0, 0.0,  # vx, vy, yaw_v
    0.0, 3.0,       # height, step_freq
    0.5, 0.0, 0.0,  # gait phase, offset, bound  ← 트로팅 프리셋 (0.5, 0, 0)
    0.5, 0.08,      # duration(고정), foot_height
    0.0, 0.0, 0.25, # pitch, roll(고정), stance_width
    0.45, 0.0       # stance_length, aux
])

# 2. 신경망 관측(Observation)에 주입하기 전 스케일링 수행
scaled_commands = commands * commands_scale

# 3. 전체 상태 벡터(State) 생성 시 이어 붙임
obs = jnp.concatenate([
    proj_gravity, 
    scaled_commands, # 여기에 반드시 스케일된 값이 들어가야 함!
    dof_pos_err, 
    dof_vel, 
    ...
])
```

## 4. HLC Action Space (확정)

15D 중 실제로 HLC가 조작할 수 있는 것은 **8차원 연속 + 1차원 이산**입니다.

| HLC 액션 | → 15D 인덱스 | 권장 범위 (도달 영역의 진부분집합) |
|---|---|---|
| `vx` | 0 | `[-1.9, 1.9]` ┐ 아래 결합 제약 적용 |
| `yaw_v` | 2 | `[-2.4, 2.4]` ┘ |
| `vy` | 1 | `[-0.5, 0.5]` |
| `height` | 3 | `[-0.22, 0.13]` |
| `step_freq` | 4 | `[2.1, 3.9]` |
| `footswing` | 9 | `[0.05, 0.32]` |
| `pitch` | 10 | `[-0.35, 0.35]` |
| `stance_width` | 12 | `[0.12, 0.42]` |
| `gait` (**이산**) | 5,6,7 | `trot (0.5,0,0)` \| `pronk (0,0,0)` |

**고정 차원**: `duty`(8)=0.5, `roll`(11)=0.0 — *학습되지 않아서* 고정.
`stance_length`(13)=0.45, `aux`(14)=0.0 — *지형 돌파에 불필요해서* 고정.
두 고정의 이유가 다르다는 점에 유의하십시오. 앞의 둘은 열고 싶어도 열 수 없습니다.

### 4.1 vx–yaw 결합 제약 (상자 클리핑으로는 불충분)

커리큘럼이 실제로 도달한 영역은 직사각형이 아니라 **마름모꼴**입니다. 직진이 빠를수록 회전 여유가 줄어듭니다 (`curriculum/distribution.pkl`, iteration 49500):

```
[trot]  vx ±2.38, yaw [-3.33,+2.86]      [pronk]  vx ±1.91, yaw ±1.91
  vx -2.381 ..........###........          vx -1.905 ........#####........
  vx -1.429 ......#########......          vx -0.952 ......#########......
  vx +0.000 ...##############....          vx +0.000 ......#########......
  vx +1.429 .....##########......          vx +1.429 .......########......
  vx +2.381 ........####.........          vx +1.905 ........#####........
        (행=vx, 열=yaw, # = 학습 도달)
```

```python
s = abs(vx) / VX_MAX + abs(yaw) / YAW_MAX
scale = jnp.minimum(1.0, 1.0 / jnp.maximum(s, 1e-6))   # 마름모 밖이면 원점 방향 축소
vx, yaw = vx * scale, yaw * scale
```
- trot 전용 단계: `VX_MAX=1.9`, `YAW_MAX=2.4`
- pronk 포함 단계: `VX_MAX=1.4`, `YAW_MAX=1.4` (pronk 마름모 내부에 완전히 포함)

### 4.2 미소 명령 데드존

`go1_gym/envs/base/legged_robot.py:822`가 `‖(vx,vy)‖ ≤ 0.2`인 명령을 **0으로 치환한 뒤** 학습했습니다. 따라서 `0 < ‖v‖ < 0.2` 구간은 LLC가 본 적이 없습니다. HLC 명령 매핑에서 이 밴드는 0으로 스냅하십시오.

### 4.3 gait 전환 주의

gait는 연속 차원이 아니므로 **저역필터를 적용하면 안 됩니다** (이산값이 뭉개져 미학습 중간값이 됩니다). 대신 HLC 10 Hz에서의 채터링을 막기 위해 **최소 유지 시간 히스테리시스**(예: 5 HLC step)를 겁니다.

## 5. 근거 및 재확인 방법

명령 범위는 **반드시 체크포인트에 동봉된 파일에서** 읽으십시오. 리포지토리의 `scripts/train.py`는 "현재 리포에 들어있는 스크립트"일 뿐, 우리 가중치를 만든 config라는 보장이 없습니다.

| 파일 (`runs/gait-conditioned-agility/pretrain-v0/train/025417.456545/`) | 알려주는 것 |
|---|---|
| `parameters.pkl` | 이 가중치 학습에 실제로 쓰인 `Cfg.commands` 전체 |
| `curriculum/distribution.pkl` | 커리큘럼이 **실제로 도달(unlock)한** 명령 영역 |

WTW의 명령 범위는 세 겹이며 혼동하면 안 됩니다:

| 표기 | 의미 | 예 (vx) |
|---|---|---|
| `lin_vel_x` | 커리큘럼 **시작** 영역 (seed) | `[-1.0, 1.0]` |
| `limit_vel_x` | 커리큘럼 **확장 상한** (도달 보장 아님) | `[-5.0, 5.0]` |
| `distribution.pkl`의 weights > 0 | **실제 도달 영역** ← 이것이 근거 | `±2.38` |

읽는 코드는 `hlc_plan.md` 부록 B에 있습니다 (`ml_logger`가 한 파일에 여러 객체를 append하므로 `pickle.load`를 EOF까지 반복해야 하고, CUDA 텐서라 `map_location='cpu'` 패치가 필요합니다).

* `duty`·`roll`이 고정인 근거: `parameters.pkl`의 `gait_duration_cmd_range=[0.5,0.5]`, `body_roll_range=[-0.0,0.0]`이며 커리큘럼 상한 `limit_*`도 동일하게 폭 0. 로컬 및 [상위 레포 master의 `scripts/train.py`](https://github.com/Improbable-AI/walk-these-ways/blob/master/scripts/train.py)와도 일치 (3중 확인).
* gait 양자화 근거: `parameters.pkl`의 `binary_phases: True` → `legged_robot.py:814-817`이 `round(2*x)/2 % 1` 적용.
* vx·yaw 외 차원은 `num_bins_* = 1`이라 커리큘럼 해상도가 없습니다 = 전 구간 균등 샘플. 그래서 height·footswing·pitch·stance_width는 `parameters.pkl`의 범위를 그대로 신뢰해도 됩니다.

## 6. LLC 추론 시 주의 — 모듈이 2개입니다

공식 `go1_gym_deploy/scripts/deploy_policy.py:59-66`이 보여주듯, 정책은 `body_latest.jit` 단독이 아닙니다:

```python
latent = adaptation_module.forward(obs_history)
action = body.forward(torch.cat((obs_history, latent), dim=-1))
```

`checkpoints/adaptation_module_latest.jit`를 함께 로드하지 않으면 입력 차원부터 맞지 않습니다.
