# 01 · 하위 제어기(LLC) — WTW 명령 사양과 MJX 이식

> **무엇** — WTW `pretrain-v0` 정책을 MuJoCo/MJX에서 구동하는 데 필요한 **모든** 사실.
> **독자** — LLC를 건드릴 때. 명령 범위·관측 조립·물리 설정·처리량을 찾아보는 레퍼런스.
> **소유** — 15D 명령 사양, 8D 액션 범위, DOF 순서, 관측 70D, MJX 설정, 함정 목록.
> 여기 없는 것: HLC 설계는 [02_hlc.md](02_hlc.md), 실측 경위·수치는 [03_results.md](03_results.md).

LLC는 **동결**이다. 이 문서의 내용은 재학습하지 않는 한 바뀌지 않는다.
구현체는 `wtw_nav/llc/policy.py`, 검증은 `wtw_nav/llc/test_policy.py`·`check.py`.

---

## 0. 시작하기 전에 — 가장 중요한 규칙

### 0.1 근거의 우선순위

명령 범위·스케일·설정을 알아낼 때 **참조 순서를 지켜야 한다.** 이걸 어겨서 실제로 헤맸다.

| 순위 | 출처 | 신뢰도 |
|---|---|---|
| 1 | **체크포인트 동봉 `parameters.pkl`** — 이 가중치를 만든 config 그 자체 | 절대 |
| 2 | **`curriculum/distribution.pkl`** — 커리큘럼이 실제로 도달한 명령 영역 | 절대 |
| 3 | 체크포인트 `.jit`의 `state_dict()` 텐서 shape | 절대 |
| 4 | `go1_gym/` 소스 코드 (관측 조립·보상 등 로직) | 높음 |
| 5 | ~~`scripts/train.py`~~ | **쓰지 말 것** — 리포에 들어있는 스크립트일 뿐, 이 가중치를 만들었다는 보장이 없다 |

### 0.2 세 겹의 명령 범위를 혼동하지 말 것

```
lin_vel_x   = [-1.0, 1.0]   커리큘럼 시작 영역(seed)      ← 이것을 "훈련 범위"로 오해하기 쉽다
limit_vel_x = [-5.0, 5.0]   커리큘럼 확장 상한(도달 보장 아님)
distribution.pkl weights>0  실제 도달 영역 = ±2.38        ← 액션 범위의 근거는 이것
```

> 📌 **앞선 오판 정정**: 초기 검토에서 `nav_env.py:39`의 vx scale `1.5`를 "LLC 훈련 상한
> 1.0 초과"로 지적했으나 **틀렸다.** 1.0은 커리큘럼 seed일 뿐이고 trot의 실제 도달 범위는
> ±2.38이다. **vx=1.5는 안전 구간 안이다.** 진짜 위험은 절대값이 아니라 vx와 yaw를
> 동시에 크게 주는 것(§4.3).

### 0.3 인터페이스에 슬롯이 있다 ≠ 그 슬롯이 학습됐다

WTW의 명령 인터페이스는 15D가 맞고 모든 슬롯이 실제로 존재한다. 그러나 **이 가중치는
그중 2개를 상수로 고정한 채 학습**됐다(§4.2). 논문·시연 영상에 나오는 duty 스윕은
**별도 config로 학습된 다른 정책**이다.

---

## 1. 준비물

```
walk-these-ways/runs/gait-conditioned-agility/pretrain-v0/train/025417.456545/
├── checkpoints/
│   ├── body_latest.jit               ← 필요
│   ├── adaptation_module_latest.jit  ← 필요 (빠뜨리면 입력 차원부터 안 맞는다)
│   └── ac_weights_last.pt            (참고용: 관측 정규화 버퍼가 없음을 확인하는 데 씀)
├── parameters.pkl                    ← 필요 (모든 상수의 근거)
└── curriculum/distribution.pkl       ← 필요 (명령 범위의 근거)

mujoco_menagerie/unitree_go1/scene.xml
walk-these-ways/resources/robots/go1/urdf/go1.urdf   (기구학 대조용)
```

### 1.1 pkl 읽는 법 (비자명)

`ml_logger`가 **한 파일에 객체를 계속 append**했고, 텐서가 **CUDA에 저장**되어 있다.

```python
import pickle, io, torch
torch.storage._load_from_bytes = lambda b: torch.load(
    io.BytesIO(b), map_location='cpu', weights_only=False)      # 이 패치 없으면 CPU에서 로드 실패

def load_all(path):
    objs = []
    with open(path, "rb") as f:
        while True:
            try: objs.append(pickle.load(f))
            except EOFError: break
    return objs                                                  # parameters.pkl은 4983개

cfg  = load_all(f"{RUN}/parameters.pkl")[0]["Cfg"]                # 중첩 경로 주의
dist = load_all(f"{RUN}/curriculum/distribution.pkl")[-1]["distribution"]
```

`parameters.pkl[0]`의 최상위 키는 `AC_Args / PPO_Args / RunnerArgs / Cfg`.
명령 설정은 **`["Cfg"]["commands"]`**. 명령 범위를 다시 검토해야 할 때의 전체
재확인 스니펫은 §14.

---

## 2. 정책 구조 — 모듈이 **2개**다

공식 `go1_gym_deploy/scripts/deploy_policy.py:59-66`:

```python
latent = adaptation_module.forward(obs_history)          # 2100 -> 2
action = body.forward(torch.cat((obs_history, latent), dim=-1))   # 2102 -> 12
```

`state_dict()` 실측:

| 모듈 | 층 | 활성화 |
|---|---|---|
| adaptation | 2100 → 256 → 128 → **2** | ELU (마지막 층 뒤에는 없음) |
| body | **2102** → 512 → 256 → 128 → 12 | ELU (마지막 층 뒤에는 없음) |

- `2102 = 2100(obs_history) + 2(latent)`. **body만 로드하면 입력 차원부터 안 맞는다.**
- 활성화는 `AC_Args.activation = 'elu'`.
- **관측 정규화 없음** — `ac_weights_last.pt`에 running mean/var 버퍼가 없다.
  (`std`(12,)는 행동 노이즈 std로 추론에는 안 쓴다.) 원시 관측을 그대로 넣는다.

> 병렬 학습을 하려면 torch JIT 호출이 아니라 **가중치를 뽑아 순수 JAX MLP로 재구현**해야 한다.
> torch 콜백은 `jit`/`vmap`/`scan` 안에 들어가지 않는다.

---

## 3. ★ DOF 순서 — 이 이식의 최대 함정

### **`FL, FR, RL, RR`** (각 다리 안에서는 hip, thigh, calf)

`go1.urdf` 파일의 joint **등장 순서는 `FR, FL, RR, RL`** 이지만 **그것이 아니다.**

근거 3중:
1. `legged_robot.py:878-902`가 `foot_indices[0]`을 **FL**, [1]=FR, [2]=RL, [3]=RR로 라벨링
2. 이 순서라야 `play.py:102`의 gait 프리셋 이름이 앞뒤가 맞는다 (§5.3 검산).
   `FR,FL,RR,RL`로 두면 두 이름이 서로 뒤바뀐 것처럼 보인다 — 그게 착시의 원인이었다
3. 폐루프 전수 조사 48조합(clock 슬롯쌍 6 × 다리순열 4 × 행동블록 2):
   `FL,FR,RL,RR`인 **12개만 보행 성공**(vx 0.49~0.85), 나머지 **36개 전부 실패**

**틀리면**: 좌우가 뒤바뀐 관측·행동이 **양의 되먹임**을 만들어, 정책이 자세를 바로잡으려 할수록
반대로 밀리며 액션이 ±10으로 포화하고 로봇이 주저앉는다. 실제 증상:

```
min_z    ≈ 0.055~0.059   (몸통이 바닥까지 내려앉음)
mean_vx  ≈ 0.0~0.09 m/s  (목표 0.6~0.9)
|a|max   = 10.00         (clip_actions 포화)
max|pg_y| ≈ 1.0          (최종적으로 옆으로 넘어짐)
```

⚠️ **로봇은 스텝을 밟긴 했다**(접지 비율 0.25~0.57). 몸통이 0.14~0.28로 크게 바운스하다
붕괴했고, 몸통 회전을 고정하면 넘어지진 않지만 전진도 못 했다(`vx=0.022`).
"정책이 죽었다"가 아니라 **"정책이 자기 자신과 싸운다"** 로 보이는 것이 이 고장의 특징이다.

> 📌 **관절의 부호·영점·프레임 규약은 URDF와 MJCF가 완전히 동일하다.**
> URDF 순기구학을 직접 구현해 MuJoCo 발 위치와 대조한 결과 **오차 0.000000 m**
> (default/zeros/랜덤 3종 × 4개 다리). **부호를 의심하는 데 시간을 쓰지 말 것.**
> 틀린 것은 **순서뿐**이었다.

관절 인덱스는 하드코딩하지 말고 **이름으로 모델에서 유도**한다 (`policy._build_joint_index`).

수정 후 검증(18개 명령, 5 s)과 배제된 가설 전체 목록은 [03_results.md §1](03_results.md).

---

## 4. 명령 15D — ★ 이 절이 명령 사양의 단일 출처다

### 4.1 스케일 (`legged_robot.py:1196-1203`)

`Cfg.obs_scales`에서 조립된다. 관측에 넣기 전 **반드시 원소별로 곱한다.**

```python
COMMANDS_SCALE = [2.0, 2.0, 0.25, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.15, 0.3, 0.3, 1.0, 1.0, 1.0]
#                 vx   vy  yaw   h    f   ph  off  bd  duty  fs  pitch roll  sw   sl  aux
```

### 4.2 각 차원의 의미 · 기본값 · 실제 학습 상태

"학습 상태"는 체크포인트에 동봉된 `parameters.pkl`(이 가중치를 만든 config 그 자체)에서
읽은 값이다.

| idx | 이름 | 스케일 | 단위 | 설명 | trot 기본값 | **이 체크포인트의 학습 상태** | HLC |
|---|---|---|---|---|---|---|---|
| 0 | `vx` | 2.0 | m/s | 전후 선속도 | `1.0`/`0.0` | ✅ 커리큘럼 도달 **±2.38** (yaw와 결합, §4.3) | ✅ |
| 1 | `vy` | 2.0 | m/s | 좌우 게걸음 | `0.0` | ✅ `[-0.6, 0.6]` 전 구간 (bin 1개 = 균등) | ✅ |
| 2 | `yaw_v` | 0.25 | rad/s | 제자리 회전 | `0.0` | ✅ 도달 **[-3.33, +2.86]** (vx와 결합) | ✅ |
| 3 | `height` | 2.0 | m | 몸통 높이 추종 | `0.0` | ✅ `[-0.25, 0.15]` 전 구간 | ✅ |
| 4 | `step_freq` | 1.0 | Hz | 발걸음 빈도 | `3.0` | ✅ `[2.0, 4.0]` 전 구간 | ✅ |
| 5 | `gait_phase` | 1.0 | rad | 걸음 위상차 | **`0.5`** | ⚠️ **`{0, 0.5}` 양자화** (§5.2) | 이산 |
| 6 | `gait_offset` | 1.0 | rad | 걸음 오프셋 | **`0.0`** | ⚠️ **`{0, 0.5}` 양자화** | 이산 |
| 7 | `gait_bound` | 1.0 | rad | 걸음 바운드 | **`0.0`** | ⚠️ **`{0, 0.5}` 양자화** | 이산 |
| 8 | `duty` (duration) | 1.0 | ratio | 체공/지지 비율 | `0.5` | ❌ **`[0.5, 0.5]` 상수 — 조작 불가** | ❌ |
| 9 | `footswing` (`foot_height`) | 0.15 | m | 발 들어올림 높이 | `0.08` | ✅ `[0.03, 0.35]` 전 구간 | ✅ |
| 10 | `pitch` | 0.3 | rad | 몸통 피치 | `0.0` | ✅ `[-0.4, 0.4]` 전 구간 | ✅ |
| 11 | `roll` | 0.3 | rad | 몸통 롤 | `0.0` | ❌ **`[-0.0, 0.0]` 상수 — 조작 불가** | ❌ |
| 12 | `stance_width` | 1.0 | m | 다리 벌림 너비 | `0.25` | ✅ `[0.10, 0.45]` 전 구간 | ✅ |
| 13 | `stance_length` | 1.0 | m | 다리 앞뒤 길이 | `0.45` | ✅ `[0.35, 0.45]` (고정 사용 0.45) | ⛔ |
| 14 | `aux` | 1.0 | - | 보조(compliance) | `0.0` | — `[0, 0.01]` (사용 안 함) | ⛔ |

**`duty`·`roll`이 조작 불가인 근거.** `parameters.pkl`의 `gait_duration_cmd_range=[0.5,0.5]`,
`body_roll_range=[-0.0,0.0]`이며 커리큘럼 상한 `limit_*`도 **둘 다 폭 0**이다. 학습 내내
상수만 관측했으므로 해당 입력 채널은 사실상 죽어 있다. 로컬 및
[상위 레포 master의 `scripts/train.py`](https://github.com/Improbable-AI/walk-these-ways/blob/master/scripts/train.py)와도
일치 (3중 확인).

**vx·yaw 외 차원은 `num_bins_* = 1`** 이라 커리큘럼 해상도가 없다 = 전 구간 균등 샘플.
그래서 height·footswing·pitch·stance_width는 `parameters.pkl`의 범위를 그대로 신뢰해도 된다.

> 📌 **트로팅 기본값 정정**: 한때 idx 5·6·7을 `(0.5, 0.5, 0.5)`로 적었으나 이는 어떤
> 프리셋도 아니다(`{0,0.5}³` 꼭짓점이긴 하나 4개 프리셋에 속하지 않아 학습 중 거의
> 나오지 않은 조합). **트로팅은 `(0.5, 0.0, 0.0)`이다.** 옛 `nav_env.py:30`의
> `base_commands`가 이 오류를 갖고 있었고 P1에서 수정했다.

### 4.3 vx–yaw 결합 제약 (상자가 아니라 마름모)

`distribution.pkl` iteration 49500, 21×21 격자 (행=vx, 열=yaw, `#`=학습 도달):

```
[trot]  104/441 unlock                    [pronk]  70/441 unlock
  vx -2.381 ..........###........           vx -1.905 ........#####........
  vx -1.905 ........######.......           vx -1.429 .......#######.......
  vx -1.429 ......#########......           vx -0.952 ......#########......
  vx -0.476 ...##############....           vx  0.000 ......#########......
  vx +0.000 ...##############....           vx +0.952 ......#########......
  vx +0.952 ....############.....           vx +1.429 .......########......
  vx +1.905 .......######........           vx +1.905 ........#####........
  vx +2.381 ........####.........
  marginal: vx±2.38, yaw[-3.33,+2.86]      marginal: vx±1.91, yaw±1.91
```

**마름모꼴** — 직진이 빠를수록 회전 여유가 줄어든다. 상자형 클리핑으로는 표현할 수 없다.
액션을 상자로 매핑한 뒤 **L1 결합 제약을 한 번 더 건다**:

```python
s = abs(vx) / VX_MAX + abs(yaw) / YAW_MAX
scale = jnp.minimum(1.0, 1.0 / jnp.maximum(s, 1e-6))   # 마름모 밖이면 원점 방향 축소
vx, yaw = vx * scale, yaw * scale
```

- trot 전용 단계: `VX_MAX = 1.9`, `YAW_MAX = 2.4`
- pronk 포함 단계: `VX_MAX = 1.4`, `YAW_MAX = 1.4` (pronk 마름모 내부에 완전히 포함)

현재는 **gait가 trot 고정**(§5.5)이므로 trot 값만 쓴다.

### 4.4 미소 명령 데드존

`go1_gym/envs/base/legged_robot.py:822`가 `‖(vx,vy)‖ ≤ 0.2`인 명령을 **0으로 치환한 뒤**
학습했다. 따라서 `0 < ‖v‖ < 0.2` 구간은 LLC가 본 적이 없다 → **0으로 스냅**할 것.

### 4.5 ★ HLC 액션 스페이스 (확정 8D)

15D 중 실제로 HLC가 조작하는 것은 **8차원 연속**이다. 범위는 도달 영역의
**진부분집합**으로 잡는다(경계 외삽 회피). 연속 차원은 tanh∈[-1,1] → 아핀 매핑.

```
a[0] vx           → cmd[0]   [-1.9, 1.9]  ┐ §4.3 L1 결합 제약 적용
a[1] yaw_v        → cmd[2]   [-2.4, 2.4]  ┘   + ‖(vx,vy)‖≤0.2 데드존 스냅
a[2] vy           → cmd[1]   [-0.5, 0.5]
a[3] height       → cmd[3]   [-0.22, 0.13]
a[4] step_freq    → cmd[4]   [2.1, 3.9]
a[5] footswing    → cmd[9]   [0.05, 0.32]
a[6] pitch        → cmd[10]  [-0.35, 0.35]
a[7] stance_width → cmd[12]  [0.12, 0.42]
고정: cmd[5:8]=trot(0.5,0,0), cmd[8]=0.5(duty), cmd[11]=0.0(roll),
      cmd[13]=0.45(stance_length), cmd[14]=0.0(aux)
```

⚠️ **고정 5차원의 이유가 두 종류라는 점에 유의.** `duty`·`roll`은 *학습되지 않아서*
고정이고 열고 싶어도 열 수 없다. `stance_length`·`aux`는 *지형 돌파에 불필요해서*
고정이므로 필요하면 열 수 있다. `gait`는 *실측 결과 이득이 없어서* 고정이다(§5.5).

구현은 `wtw_nav/hlc/command_filter.py`, 상수는 `wtw_nav/configs/default.py`가 소유한다.
적용 순서가 중요하다 — ① tanh 아핀 매핑 ② L1 마름모 제약 ③ 데드존 스냅
④ 저역필터(α≈0.3) ⑤ 고정 차원 전개.

---

## 5. gait 파라미터 (idx 5, 6, 7)

### 5.1 프리셋

```python
GAITS = {"pronk": (0, 0, 0), "trot": (0.5, 0, 0), "bound": (0, 0.5, 0), "pace": (0, 0, 0.5)}
```
`scripts/play.py:102` 기준.

### 5.2 `{0, 0.5}`로 양자화된다

`parameters.pkl`의 `binary_phases: True` → `legged_robot.py:814-817`:
```python
commands[5:8] = (torch.round(2 * commands[5:8])) / 2.0 % 1
```
따라서 `(phase, offset, bound)`는 `{0, 0.5}³`의 꼭짓점에서만 학습됐고 실제로는 4개
프리셋만 사용됐다(`legged_robot.py:765-781`의 gaitwise curriculum).
**중간값(예: phase=0.25)은 LLC가 한 번도 본 적 없다** → trot↔pronk 연속 보간은 분포 이탈.

### 5.3 ⚠️ 필드명과 gait 이름이 어긋난다

DOF 순서 FL,FR,RL,RR로 전개해 검산하면:

| 명령 | 필드명 | 실제 결과 gait |
|---|---|---|
| `commands[6]=0.5` | `gait_offset` | FL·FR 대 RL·RR = 앞뒤쌍 = **bound** |
| `commands[7]=0.5` | `gait_bound` | FL·RL 대 FR·RR = 좌우쌍 = **pace** |

`play.py:102`의 `bounding=[0,0.5,0]`, `pacing=[0,0,0.5]`와 일치한다.
**필드명을 믿고 매핑하면 pace/bound가 뒤바뀐다.**
(`legged_robot.py:774-781`의 생성 로직은 정반대로 적혀 있어 한때 충돌로 보였다.
trot/pronk는 양쪽 일치하므로 실사용에는 영향이 없었다.)

### 5.4 clock_inputs 계산 (`_step_contact_targets`)

```python
gait_indices = remainder(gait_indices + dt * step_freq, 1.0)     # dt = 0.02 (정책 주기)
feet = [gait + phase + offset + bound,   # FL
        gait + offset,                    # FR
        gait + bound,                     # RL
        gait + phase]                     # RR
x = remainder(feet, 1.0)
warped = where(x < duty, x * (0.5/duty), 0.5 + (x - duty) * (0.5/(1-duty)))
clock_inputs = sin(2*pi*warped)
```
`duty=0.5`(이 체크포인트)에서 워핑은 **항등**이다.

### 5.5 gait는 trot 고정 — 이산 액션을 폐기한 이유

원안은 gait를 9번째 이산 액션(trot/pronk)으로 열고, 채터링 방지를 위해 **최소 유지 시간
히스테리시스**(예: 5 HLC step)를 거는 것이었다. 연속 차원이 아니므로 **저역필터를 걸면
안 되고**(이산값이 뭉개져 미학습 중간값이 된다), Brax PPO가 연속 액션 전제라
"스칼라 1개 + 하드 임계 → 필요시 Gumbel-softmax 승격" 경로를 계획했다.

**폐기됐다.** P0.5 실측에서 pronk가 몸통을 띄우지 못하고(정점 상승 +0.001 m) MJX에서는
전복까지 유발했다. 이득 없이 실패만 만드는 선택지다.
→ [decisions/0003-gait-trot-고정.md](decisions/0003-gait-trot-고정.md), 수치는 [03_results.md §2](03_results.md).

---

## 6. 관측 70D — 순서가 전부다

`legged_robot.py:302-338` (`observe_command=True`, `observe_two_prev_actions=True`,
`observe_clock_inputs=True`, `observe_vel=False`):

```
[  0:  3]  projected_gravity                                 = R^T @ [0,0,-1]
[  3: 18]  commands * COMMANDS_SCALE                         (15)
[ 18: 30]  (dof_pos - default_dof_pos) * 1.0                 (12)  obs_scales.dof_pos = 1.0
[ 30: 42]  dof_vel * 0.05                                    (12)  obs_scales.dof_vel = 0.05
[ 42: 54]  actions                                           (12)  방금 실행된 a_t (클리핑 후)
[ 54: 66]  last_actions                                      (12)  그 직전 a_{t-1}
[ 66: 70]  clock_inputs                                      (4)
```
합 3+15+12+12+12+12+4 = **70**. 마지막에 `clip(obs, -100, 100)`.

2차 출처: 배포 쪽 조립 순서가 `go1_gym_deploy/envs/lcm_agent.py`에 있다 (관측 스케일 배열
`:56-60`, 명령 인덱스 해석 `:239-247`). `durations = commands[:, 8]`로 우리 인덱스 맵과 일치한다.

- `projected_gravity`: MuJoCo 쿼터니언 `(w,x,y,z)` 기준
  `[-2(xz-wy), -2(yz+wx), -(1-2(x²+y²))]`.
  MuJoCo `mju_quat2Mat`과 대조해 오차 3.3e-16 확인.
- `actions`/`last_actions`는 **스케일 전 원시 행동**(클리핑만 적용).

### 6.1 히스토리 버퍼

```python
obs_history = concat([obs_history[70:], obs])     # 오래된 것이 앞, 최신이 뒤. 총 2100
```
`history_wrapper.py:23`, 30 스텝. 리셋 시 0으로 채우고 **마지막 칸에만** 리셋 관측을 넣는다.

---

## 7. 행동 → 관절 목표각 (`_compute_torques:919-925`)

```python
actions = clip(raw_action, -10, 10)                  # clip_actions = 10.0
scaled = actions * 0.25                              # action_scale
scaled[[0, 3, 6, 9]] *= 0.5                          # hip_scale_reduction, hip 관절 4개
joint_pos_target = scaled + default_dof_pos
```

`default_dof_pos` (DOF 순서 FL,FR,RL,RR):
```
FL: ( 0.1, 0.8, -1.5)    FR: (-0.1, 0.8, -1.5)
RL: ( 0.1, 1.0, -1.5)    RR: (-0.1, 1.0, -1.5)
```
(앞다리 thigh 0.8, 뒷다리 1.0 — 비대칭이 정상이다.)

---

## 8. 물리 모델 설정

### 8.1 WTW 학습 조건 (`Cfg.control`, `Cfg.sim`)

| 항목 | 값 |
|---|---|
| `sim.dt` | 0.005 |
| `control.decimation` | 4 → 정책 주기 **0.02 s (50 Hz)** |
| `control.stiffness` | Kp = **20** |
| `control.damping` | Kd = **0.5** |
| `control.control_type` | `actuator_net` (학습된 액추에이터 신경망) |

> ⚠️ **actuator_net을 PD로 근사한다** (= sim-to-sim 격차 ①). 다만 실측해보니 격차가 작다 —
> `resources/actuator_nets/unitree_go1.pt`를 직접 질의하면
> **유효 Kp = 18.7~20.1, 유효 Kd = 0.75~0.85**. 즉 Kp=20/Kd=0.5 PD가 타당한 근사다.
> (한때 이 격차를 붕괴의 원인으로 의심했으나 측정으로 배제했다.)

### 8.2 MuJoCo position 액추에이터로 PD 만들기

```python
mj_model.actuator_gainprm[:, 0] =  kp
mj_model.actuator_biasprm[:, 1] = -kp
mj_model.actuator_biasprm[:, 2] = -kd      # force = kp*(ctrl - q) - kd*qd
mj_model.opt.timestep = 0.005
```
구동 관절의 `dof_damping`/`dof_frictionloss`는 **0으로** 둔다(속도 항이 Kd만 남도록).
free joint dof는 건드리지 말 것.

### 8.3 ★ MJX 전용 설정 (없으면 실행 불가)

menagerie `go1.xml`은 정확도 우선이라 MJX에서 **컴파일만 수십 분** 걸린다 — 느린 게
아니라 아예 끝나지 않는 것처럼 보인다. MJX는 솔버 반복을 컴파일 그래프에 그대로
펼치기 때문이다. (= sim-to-sim 격차 ②)

| 항목 | menagerie 기본 | MJX용 |
|---|---|---|
| `opt.iterations` | **100** | **8** |
| `opt.ls_iterations` | **50** | **16** |
| `opt.cone` | elliptic | pyramidal |
| `opt.impratio` | 100 | 1 |
| 발 `geom_condim` | 6 | 3 |
| `opt.integrator` | Euler | **Euler 유지** |

`create_env(mjx_friendly=True)`(기본값)가 위 표의 오른쪽으로 바꾼다.

- **`iterations`를 8 미만으로 낮추지 말 것.** 4/8이면 MJX **GPU에서 보행이 무너진다**
  (vx 0.8 → 0.218, 낙상). 8/16과 16/32는 결과가 동일 = 수렴이므로 더 올릴 이유도 없다.
  **판정 기준은 "돌아가느냐"가 아니라 "반복을 늘려도 결과가 안 변하느냐".**
  하드웨어 의존성 실측은 [03_results.md §1.2](03_results.md).
- **`implicitfast`를 쓰지 말 것.** mujoco_playground의 `implicitfast + iterations=1`
  조합은 이 모델·게인에서 발산한다(z가 112 m로 튐).
- ⚠️ **solver 설정의 단일 출처는 `policy.create_env`다.** 한때 `check.build()`가
  `iterations=4, ls_iterations=8`을 자기 기본값으로 들고 있어서, `create_env`만 8/16으로
  고쳤는데도 실행은 계속 4/8로 돌았다. `build()`는 이제 `None`을 기본으로 두고
  `create_env`에 위임한다. `check.version_check()`도 **시그니처가 아니라 실제 생성된
  모델의 `opt.iterations`** 를 확인한다 — 시그니처만 보면 이 버그를 못 잡는다.

### 8.4 ★★ 충돌 필터 — MJX 처리량의 지배 요인

솔버 설정보다 이쪽이 훨씬 크다. 측정치는 [03_results.md §4](03_results.md).

- **`collision="world"`(기본)를 쓴다.** 자기충돌을 해제해 후보 쌍 843 → 42로 줄인다.
- 비트마스크: world geom `contype=1/conaff=1`, robot geom `contype=2/conaff=1`
  → robot–robot은 `(2&1)=0`으로 걸러지고 robot–world만 남는다
- **`feet`을 쓰지 말 것.** `world` 대비 이득이 15%뿐인데 다리 vs 지형 접촉을 잃어
  좁은 통로·턱에서 다리가 지형을 통과한다.
- **정합성**: WTW는 `self_collisions = 0`(=켜짐)으로 학습했으므로 `world`는 원본과의
  편차다. 그러나 A/B 결과 `full`과 **완전히 동일**했다. `check.collision_ab()`로 재확인 가능.
- 구현은 `policy._apply_collision_filter`, 설정은 `HLCConfig.collision`.

### 8.5 ★★ 월드 geom이 많으면 broadphase — `max_geom_pairs`

§8.4는 "충돌이 지배 요인"까지만 말한다. 그 다음 질문 — **월드 geom이 많아지면?** —
의 답이 이것이고, 이걸 모르는 채로 21시간짜리 학습을 걸었다.

MuJoCo C는 매 스텝 broadphase로 먼 쌍을 걸러내지만 **MJX는 기본적으로 걸러내지 않는다.**
로봇이 미로 반대편에 있어도 벽 전부와 계산한다. 기능은 있고 **꺼져 있었다**:

> "While MuJoCo handles broadphase culling out of the box, MJX-JAX requires
> additional parameters." — `doc/mjx.rst`

`mjx/_src/collision_driver.py`가 custom numeric을 읽는다:

```python
max_geom_pairs = _numeric(m, 'max_geom_pairs')
dist = ‖p2 - p1‖ - (rbound1 + rbound2)
_, idx = jax.lax.top_k(-dist, k=max_geom_pairs)     # 가장 가까운 k쌍만 좁은단계로
```

거리 계산은 전부 하되(벡터 연산, 쌈) 비싼 좁은단계는 k개만 돈다.
**미로에서 9.1배(78 → 708 HLC steps/s)이고, 맵 크기에 O(1)이다** — 벽이 13 → 70개가
되어도 좁은단계가 그대로다. 채택값 **k = 32**.

**주의사항**

- `<custom><numeric name="max_geom_pairs" data="32"/></custom>` — **컴파일 전 spec**에
  넣어야 한다. 컴파일된 `numeric_data`는 크기 고정이라 나중에 못 넣는다.
  구현은 `terrain.modules.set_broadphase`, 확인은 `broadphase_report`.
- 캡은 **geom 종류 쌍마다** 적용된다. 로봇에 SPHERE·CAPSULE·BOX가 다 있으므로
  벽(BOX)과의 조합이 셋이고, 실제 좁은단계는 대략 `k × 3`이다.
- **HFIELD·PLANE 쌍은 컬링을 건너뛴다**(MJX 내부 `_MAX_NCON=8`) — 바닥 접촉은
  영향받지 않는다.
- ⚠️ **근사다.** k가 실제 동시 접촉 쌍보다 작으면 접촉이 조용히 누락되어 벽을
  통과한다. 기본값은 0(끔)이고, 켤 때는 `world`와 A/B로 궤적이 유지되는지 확인한다
  (`full` → `world` 전환을 정당화한 것과 같은 절차). k=32의 A/B 결과는
  [03_results.md §5](03_results.md).
- `max_contact_points`도 있다(솔버로 보내는 접촉점 상한, condim 종류별). 넘치면
  **관통이 깊은 것부터** 남긴다. 아직 안 써 봤다.
- **★ geom 축소는 채택하지 않았다.** 근거는
  [decisions/0005-broadphase-채택-geom축소-거부.md](decisions/0005-broadphase-채택-geom축소-거부.md).

**MuJoCo Warp**: bench 로그의 `Failed to import warp`는 별도 백엔드
(`mjx.put_model(m, impl='warp')`)가 없다는 뜻이다. 문서상 "mitigates performance
issues around scaling the number of contacts and constraints"라 이 문제에 직접
해당하지만, `max_geom_pairs`로 충분해져서 **시도하지 않았다.** 필요해지면 여기부터.

### 8.6 토크 한계 (참고)

Isaac은 URDF `effort = 33.5` Nm로 클리핑하고 menagerie는 23.7/35.55다.
**A/B 결과 차이 없음** — 두 설정의 결과가 완전히 동일했다. 굳이 맞출 필요 없다.

---

## 9. ★ 리셋 자세

`mj_resetData`는 **모델 기본값**을 준다 — Go1에서는 관절이 전부 0(다리를 편 자세)에
z=0.445이고 **발이 지면을 4 mm 관통한다(ncon=4)**.

```
[A] mj_resetData        z=0.445, 관절 0,       발 -0.004 관통, ncon=4   ← 학습 분포 밖
[C] WTW init_state      z=0.340, 관절 default, 발 +0.015 여유, ncon=0   ← 정답
```

WTW 학습 리셋 자세는 `Cfg.init_state`: `base pos = [0,0,0.34]`, `quat = [1,0,0,0]`,
`joints = default_joint_angles`(= `default_dof_pos`) → 발이 지면 위 1.5~2.5 cm.

**틀리면**: 관절 0인 상태의 `dof_pos_err`가 최대 1.5 rad로 학습 분포 밖 → 액션이
`clip_actions=10`까지 → `×0.25` = **±2.5 rad 목표각 오프셋** → **NaN 발산**.
(PD로 default 자세만 유지하면 두 초기 자세 모두 안정적이므로, 발산은 초기 자세 자체가
아니라 **초기 자세가 정책을 분포 밖으로 밀어낸 결과**다.)

`mj_forward`를 돌린 뒤 `mjx.put_data` 할 것(파생량 일관성). 구현은 `policy.reset_data(...)`.
**`nav_env.reset`도 반드시 이 자세를 쓴다.**

---

## 10. 스텝 루프 — 연산 순서

`legged_robot.py`의 `step` → `post_physics_step` 순서를 그대로 따른다:

```
1. action = clip(policy(obs_history), -10, 10)
2. target = action*0.25 (hip은 ×0.5) + default_dof_pos   -> ctrl
3. 물리 4 서브스텝 (ctrl 고정)
4. gait_indices += 0.02 * step_freq                       ← 관측 계산 前
5. obs 조립 (actions=a_t, last_actions=a_{t-1}), clip(±100)
6. obs_history = concat([obs_history[70:], obs])
7. 행동 시프트: last_last <- last, last <- a_t
```

성능 주의:
- **롤아웃 전체를 `lax.scan` 하나로** 돌릴 것. Python 루프로 스텝마다 호출하고
  중간에 `np.asarray(...)`로 값을 꺼내면 매 스텝 host 동기화가 걸려 수십 분이 된다.
  (컴파일 1회·디스패치 1회·동기화 1회 → `policy.make_rollout_fn()`)
- 렌더링은 롤아웃 중이 아니라 **끝난 뒤 로그된 qpos에서 일반 MuJoCo로 재생**
  (`check.render()`).
- `init_llc_state()`는 dtype을 명시할 것 — weak-typed 스칼라를 넘기면 재컴파일이 한 번 더 생긴다.

---

## 11. 검증 절차 (게이트)

### 11.1 로컬 (JAX 불필요) — `python -m wtw_nav.llc.test_policy`

29개 항목. 상수가 `parameters.pkl` 유래인지 대조, MLP forward vs torch,
adaptation module 누락 시 실패하는지 **음성 대조**, 관절 매핑, 70D 조립,
`proj_gravity == R^T·[0,0,-1]` 교차검증, 리셋 자세 `ncon == 0`, DOF 순서.

```bash
conda run -n mujoco_env python -m wtw_nav.llc.test_policy
```

jax/mjx가 있으면 `policy.py`를 직접 실행하고, 없으면 numpy 미러로 같은 수식을 검증하므로
**어느 환경에서도 돌아간다.**

### 11.2 Colab (JAX/MJX) — `check.run_gate(env)`

| # | 항목 | 통과값 |
|---|---|---|
| 1 | torch vs JAX | `max\|diff\| = 6.1e-05` (임계 1e-3; float32 누적 오차는 백엔드마다 1e-5~1e-4) |
| 2 | 리셋 자세 | z=0.34, ncon=0 |
| 3 | 스모크 10스텝 | 발산 없음 |
| 4 | trot vx=0.8, 5 s | **vx = 0.871**, min_z 0.155, 무낙상 |
| 5 | 정지 명령 | 편향 +0.091 m/s (§12) |
| 6 | 후진/게걸음/회전 | −0.430 / +0.480 / +1.058 ← 관절 순서 회귀를 잡는다 |

⚠️ 속도는 **반드시 몸통 좌표계로** 잰다(`check.body_velocity`, `mju_quat2Mat` 대조 오차
8.9e-16). 명령이 몸통 기준이므로 월드 x 속도로 재면 방향을 튼 만큼 과소평가된다.

---

## 12. 알려진 잔여 특성

| 특성 | 수치 | 대응 |
|---|---|---|
| **정지 시 전진 편향** | vx=0 명령에서 **+0.09 m/s** (선형, 엔진 무관) | 조정 시도 모두 추종 성능을 악화시켜 **현재 설정 유지**. HLC가 폐루프로 보정하되, **"멈춰"로는 안 멈춘다** |
| **도약 불가** | 72개 명령 조합 탐색, 체공 중 수평이동 최대 **8 cm**, 정점 상승 +0.001 m | WTW 공개 코드에 도약 메커니즘 자체가 없다(설정 필드는 죽은 코드) |
| **MJX에서 pronk 전복** | `pg_z = +0.878` (뒤집힘) | pronk를 HLC 액션에서 제외, gait는 trot 고정 |
| **직진 명령에서 횡방향 표류** | vx=0.8·yaw=0으로 8초에 전진 5.84 m, **횡 3.32 m** (초기 yaw 랜덤화 없이). 요속 편향 **ω ≈ 0.118 rad/s** | menagerie `scene.xml`은 **무한 평면**이라 이 특성이 드러나지 않는다. 유한한 발판·통로에서는 **옆으로 떨어진다** |

**횡방향 표류의 세 가지 대응** — 이것이 `beam`·`slit`을 지형에서 제외하게 만든 원인이다:
① 지형 실측용 발판은 반폭 6 m 이상(`terrain.modules.PLATFORM_W`)
② 판정에 `_off_platform()` 검사를 넣어 "지형 실패"와 "옆으로 이탈"을 구분
③ 좁은 통로에서는 HLC가 **상시 yaw 보정**을 해야 한다

정지 편향의 조정 시도 표, 도약 탐색 전체, 표류의 정상상태 오차 분석은
[03_results.md](03_results.md).

> 📌 게이트 [5]의 기준 0.15 m/s는 실측 0.09에 여유를 둔 값 — 이보다 커지면 회귀 신호다.

---

## 13. 함정 목록 (한 줄 요약)

1. **DOF 순서는 `FL,FR,RL,RR`** — URDF 파일 순서 아님. 틀리면 양의 되먹임으로 붕괴 (§3)
2. **부호·영점은 URDF=MJCF 동일** — 순기구학 대조 오차 0. 여기 시간 쓰지 말 것 (§3)
3. **adaptation module 필수** — body 입력은 2102 = 2100 + latent 2 (§2)
4. **`mj_resetData` ≠ 학습 리셋 자세** — z=0.34 + default 각도로 덮어쓸 것 (§9)
5. **MJX solver ≥ 8/16** — 미수렴 구간에서는 결과가 하드웨어 의존적 (§8.3)
6. **`implicitfast` 금지** — 발산 (§8.3)
7. **롤아웃은 `lax.scan` 하나로** — Python 루프 + 중간 동기화는 수십 분 (§10)
8. **`duty`·`roll`은 조작 불가** — 상수로 학습됨 (§4.2)
9. **gait는 `{0,0.5}` 양자화** — 연속 보간 불가 (§5.2)
10. **`gait_offset`→bound, `gait_bound`→pace** — 필드명과 gait 이름이 어긋남 (§5.3)
11. **명령 범위 근거는 `distribution.pkl`** — `scripts/train.py` 쓰지 말 것 (§0.1)
12. **vx–yaw는 마름모** — 상자 클리핑으로 부족 (§4.3)
13. **`‖(vx,vy)‖≤0.2`는 데드존** — 0으로 스냅 (§4.4)
14. **속도 판정은 몸통 좌표계** (§11.2)
15. **`MUJOCO_GL`은 mujoco import 前에** 설정 — 나중에 바꿔도 소용없음.
    GPU면 `egl`, CPU 런타임이면 `osmesa` (`policy.default_gl()`이 자동 판별)
16. **느리면 솔버가 아니라 충돌을 볼 것** — 자기충돌 해제로 7.1배. 솔버 4배는 11%뿐 (§8.4)
16b. **월드 geom이 많으면 `max_geom_pairs`** — MJX에도 broadphase가 있고 기본이 꺼짐이다.
    미로에서 9.1배이고 **맵 크기에 O(1)**. geom을 깎지 말 것 (§8.5)
17. **`num_timesteps`는 HLC 스텝** — HLC 1스텝 = LLC 5 × 물리 4 = **물리 20스텝**.
    벽시계 시간 추정은 항상 ×20. `python -m wtw_nav.bench`로 먼저 잴 것
18. **brax `EvalWrapper`는 metrics를 에피소드 전체 합산** — 매 스텝 거리를 넣으면
    거리×스텝수가 찍힌다. 최종값이 필요하면 `값 * done`으로 넣을 것
19. **타임아웃을 env의 `done`에 넣지 말 것** — brax가 truncation=0으로 계산해
    가치 부트스트랩을 잘라내고 학습이 붕괴한다 ([02_hlc.md §5.2](02_hlc.md))
20. **지형 모델을 `<include>`로 만들지 말 것** — `go1.xml`의 `meshdir="assets"`는
    **주 모델 파일 디렉터리** 기준으로 풀린다. 다른 위치에서 절대경로로 include하면
    메시 로딩이 실패한다. `MjSpec.from_file` 후 worldbody 편집이 정답
21. **MJX는 CYLINDER–BOX / CYLINDER–HFIELD 충돌이 없다** — 지형을 박스로 만들면
    `put_model`이 `NotImplementedError`로 죽는다. Go1의 실린더 12개는 hip·trunk에만
    있고 지형 접촉부(발 SPHERE, 종아리·허벅지 CAPSULE, 몸통 BOX)는 전부 지원되므로
    `terrain.modules.drop_cylinder_collisions()`로 끄면 된다.
    지원표: `mjx._src.collision_driver._COLLISION_FUNC`

---

## 14. 부록 — 명령 범위 근거 재확인 스니펫

액션 범위를 다시 검토해야 할 때 아래를 재실행한다. **`scripts/train.py`를 근거로 쓰지 말 것.**

```python
import pickle, io, torch, numpy as np

# 가중치가 CUDA 텐서로 저장되어 있어 CPU 머신에서는 이 패치가 없으면 로드 실패한다
torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location='cpu',
                                                      weights_only=False)

RUN = "walk-these-ways/runs/gait-conditioned-agility/pretrain-v0/train/025417.456545"

def load_all(p):              # ml_logger는 한 파일에 객체를 계속 append한다 (4983개)
    objs = []
    with open(p, "rb") as f:
        while True:
            try: objs.append(pickle.load(f))
            except EOFError: break
    return objs

# --- 명령 범위: 중첩 경로는 /Cfg/commands (최상위 키는 AC_Args/PPO_Args/RunnerArgs/Cfg) ---
cmds = load_all(f"{RUN}/parameters.pkl")[0]["Cfg"]["commands"]
cmds["gait_duration_cmd_range"]   # [0.5, 0.5]  → duty 상수
cmds["body_roll_range"]           # [-0.0, 0.0] → roll 상수
cmds["binary_phases"]             # True        → gait {0, 0.5} 양자화

# --- 실제 도달 영역: 행=vx, 열=yaw, weights>0 = 학습 도달 ---
dist = load_all(f"{RUN}/curriculum/distribution.pkl")[-1]["distribution"]
M   = dist["weights_trot"].reshape(21, 21)
g   = dist["grid_trot"]
vx  = np.round(g[0].reshape(21, 21)[:, 0], 3)      # bin 중심 (np.linspace 쓰지 말 것)
yaw = np.round(g[2].reshape(21, 21)[0, :], 3)
vx[M.sum(axis=1) > 0].min(), vx[M.sum(axis=1) > 0].max()     # -2.381, 2.381
yaw[M.sum(axis=0) > 0].min(), yaw[M.sum(axis=0) > 0].max()   # -3.333, 2.857
```

**재현 확인 완료 (2026-07-28)**: 위 절차로 duty·roll 폭 0, `binary_phases=True`,
trot vx ±2.381 / yaw [−3.333, +2.857], pronk vx·yaw ±1.905가 모두 재현됨.
- 21×21 격자는 `limit_vel_x`/`limit_vel_yaw` = `[-5, 5]`를 21등분 (중심 간격 0.476).
- vx·yaw 외의 차원은 `num_bins_* = 1` → 커리큘럼 해상도 없음 = 전 구간 균등 샘플.
