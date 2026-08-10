# 2026_wtw — Go1 재난 미로 내비게이션

Unitree Go1이 미로를 자율 주행한다. **Walk These Ways(WTW)** 사전학습 보행 정책을
MJX로 이식해 **동결**하고, 그 위에 PPO로 **상위 제어기(HLC)** 를 얹은 계층 구조다.
Colab + MJX, **Sim-only**.

```
BFS 거리장(플래너) ──▶ HLC 10 Hz, 8D 명령 (PPO 학습) ──▶ LLC 50 Hz (WTW, 동결) ──▶ MJX 200 Hz
                                                          HLC 1스텝 = LLC 5 = 물리 20
```

---

## 지금 어디까지 왔나

| # | 단계 | 게이트 | 상태 |
|---|---|---|---|
| P0 | LLC 이식 | torch↔JAX < 1e-4, vx=0.8 추종 | 완료 2026-07-28 (`vx 0.871`) |
| P0.5 | 명령 스윕 · 도약 판정 | pronk 4足 체공 | 완료 — **도약 불가 확정** |
| P1 | 평지 코스 + 유도 벡터 | 도달률 > 95% | 완료 |
| P2·P3 | 지형 돌파 | — | **폐기** ([0001](decisions/0001-지형-돌파-폐기.md)) |
| P4 | 미로 주행 학습 | 도달률 > 70% | 완료 — **99.2%** (2026-08-10, 234분) |
| P5 | 일반화 판정 | 미학습 배치·큰 미로에서 유지 | 코드 완료, **실행 대기** |

**다음 작업** — Colab에서 P5를 돌린다. 학습 미로의 99.2%가 **레이아웃 암기와 구분되지
않기** 때문이다.

```python
from wtw_nav import eval as E
pol, cfg_tr, log, step = E.load("checkpoints/hlc_p4_maze.pkl")
E.curve(log)          # 수렴 지점 → 다음 실행 예산 (P4는 43%에서 이미 95% 도달)
E.report(pol, n=20)   # 학습 미로 + seed 7·13 + n=8·11, 실패 원인 분류 + 영상
```

**열려 있는 큰 결정** — LLC를 MJX에서 재학습할 것인가([0004](decisions/0004-llc-재학습-보류.md)).
P4/P5 결과를 보고 판단한다.

---

## 문서 지도

순서대로 읽으면 손댈 수 있는 상태가 된다.

| | 무엇 | 언제 여나 |
|---|---|---|
| **[01_llc.md](01_llc.md)** | 15D 명령 사양 · WTW→MJX 이식 · MJX 설정 · 함정 21개 | LLC를 건드릴 때. **찾아보는 레퍼런스** |
| **[02_hlc.md](02_hlc.md)** | 관측 · 행동 · 보상 · 종료 · env 파이프라인 · PPO | HLC를 학습시키거나 고칠 때. **훑어 읽는 설계서** |
| **[03_results.md](03_results.md)** | 실측 대장 — 지형 한계 · 처리량 · 학습 결과 | "그 숫자 어디서 나왔지"를 확인할 때 |
| [decisions/](decisions/) | 뒤집힌 판단 5건 | "왜 이렇게 됐지"를 확인할 때 (통독 대상 아님) |
| [archive/](archive/) | 재구성 이전 원본 7개 | 이 재구성에서 무언가 빠졌다고 의심될 때 |

### ★ 한 사실 = 한 소유 파일

같은 내용을 두 곳에 적지 않는다. 아래 표가 어디에 적을지를 정한다.

| 사실 | 소유 |
|---|---|
| 15D 스케일 · 8D 액션 범위 · 마름모 · 양자화 · 데드존 | **01_llc.md §4, §5** |
| DOF 순서 · 관측 70D · 리셋 자세 · MJX 설정 · broadphase 사용법 | **01_llc.md** |
| 관측 37D · 보상식 · PPO 설정 · brax 함정 | **02_hlc.md** |
| 측정 수치와 그 경위 | **03_results.md** (덧붙이기만) |
| **지형 한계값 자체** | **`wtw_nav/terrain/limits.py`** — 문서는 인용만 |
| HLC 상수(범위·게인·보상·PPO) | **`wtw_nav/configs/default.py`** |
| LLC 상수(스케일·PD·DOF) | **`wtw_nav/llc/policy.py`** |
| 현재 상태 · 다음 작업 | **이 파일** |
| 설치 · 실행 절차 · 저장소 구조 | **[루트 README](../README.md)** |

★ 실행되는 값은 코드가 소유한다 — **코드가 문서보다 덜 낡기** 때문이다.
문서의 숫자를 코드로 옮기지 말 것. 반대로, 실측이 끝날 때마다 `limits.py`를 갱신하고
문서는 참조한다.

---

## 코드 구조

```
2026_wtw/
├── docs/                  # 이 문서들
├── wtw_nav/               # 프로젝트 본체 (import 가능한 패키지)
├── notebooks/             # Colab 진입점 (로직 없음)
├── walk-these-ways/       # 외부: WTW 원본 + 사전학습 가중치  (수정 금지)
├── mujoco_menagerie/      # 외부: Go1 모델                    (수정 금지)
├── checkpoints/           # 학습 산출물 (Drive 아래여야 한다)
└── runs/                  # TensorBoard 로그
```

### `wtw_nav/`

| 모듈 | 역할 |
|---|---|
| `configs/default.py` | **모든 HLC 상수의 단일 진실 공급원.** 액션 범위·마름모·데드존·저역필터·보상·종료·PPO. `maze_config()`가 미로 크기에서 타임아웃·에피소드 길이·`num_timesteps`를 역산 |
| `llc/policy.py` | 동결 LLC 로드·추론. torch JIT이 아니라 **순수 JAX MLP 재구현**(병렬 학습용). `load_policy` / `create_env` / `make_step_fn` / `make_rollout_fn` / `get_obs` / `make_commands` |
| `llc/check.py` | P0/P0.5 게이트 (**Colab, JAX 필요**). `build` / `run_gate` / `preview` / `sweep_video` / `render` / `collision_ab` / `version_check` |
| `llc/test_policy.py` | P0 정적 게이트 (**로컬, JAX 불필요**). 29항목. jax가 없으면 numpy 미러로 같은 수식을 검증 |
| `hlc/command_filter.py` | 8D 액션 → 15D 명령. 순서 중요: 아핀 → L1 마름모 → 데드존 → 저역필터 → 고정 차원 |
| `hlc/guidance.py` | 유도 벡터 `(cosφ, sinφ, d_norm)`. 훈련·평가가 **같은 형식**을 내는 것이 핵심 |
| `hlc/sensors.py` | 2D 라이다 16D (해석적 ray–AABB 질의) |
| `terrain/modules.py` | 지형 모듈 → box geom. `ladder` / `set_broadphase` / `drop_cylinder_collisions` |
| `terrain/maze.py` | 미로 생성 + BFS 거리장. 벽 직선 병합(33→13 geom) |
| `terrain/limits.py` | **지형 한계값의 소유자.** `report()` / `validate()` — 미측정 지형은 거부한다 |
| `terrain/measure.py` | 지형 실측 도구 |
| `envs/nav_env.py` | HLC 학습 env. `brax.envs.base.Env` 직접 상속 (물리를 MJX로 직접 굴리므로 brax `sys` 불필요) |
| `envs/scripted.py` | **수동 제어기 — PPO 전에 env가 풀리는지 확인.** `evaluate` / `video` / `foot_track` / `phase_sweep` / `body_clearance` |
| `envs/test_nav_env.py` | P1 게이트. vmap 가능성 + brax 래퍼 pytree 구조 |
| `envs/reward_audit.py` | 보상 유인의 **상대 크기** 검사 (JAX 불필요, 몇 초) |
| `train.py` | PPO 실행기. `_check_ppo_config()`가 갱신 횟수·`num_evals`를 사전 경고 |
| `eval.py` | P5 평가. `load` / `rollouts` / `report` / `preview` / `curve` |
| `bench.py` | 처리량·ETA 측정. `breakdown` |
| `monitor.py` | 콘솔 표 + TensorBoard + 노트북 인라인 라이브 플롯 |
| `dev.py` | `reload_wtw()` — Colab 모듈 캐시 제거 |

### `notebooks/`

`01_llc_check.py`(LLC 검증) / `02_train_hlc.py`(학습, 장시간) /
`03_eval_hlc.ipynb`(평가) / `04_terrain_measure.py`(지형 실측)

주의 — 파일명이 숫자로 시작하므로 **`import`할 수 없다** (`from notebooks import 01_llc_check`
→ `SyntaxError`). 로직은 전부 `wtw_nav/` 안에 있으니 그쪽을 import 한다.

### 외부 리소스

- `walk-these-ways/runs/gait-conditioned-agility/pretrain-v0/train/025417.456545/` —
  `checkpoints/`(body + adaptation + ac_weights), `parameters.pkl`,
  `curriculum/distribution.pkl`. **명령 범위의 유일한 정답 근거**
- `mujoco_menagerie/unitree_go1/scene.xml`

---

## Colab 워크플로 — 구조가 강제하는 규칙 다섯

```
로컬(Windows, CPU)          Google Drive 동기화          Colab (T4/A100)
  wtw_nav/**/*.py  ─────────────────────────────────▶  drive.mount()
  코드 편집·1-env 형상 검증                              os.chdir(project_path)
                                                        sys.path.append(cwd)
  notebooks/*.py ───────────────────────────────────▶  import wtw_nav.*
                                                        → 학습 / 렌더링
```

1. **루트 스크립트 금지.** 모든 로직은 `wtw_nav/` 안의 import 가능한 모듈에 둔다.
   노트북은 마운트·설정·호출만 한다.
   ★ 과거 루트 `wtw_mjx_core.py` / `train_hlc.py` / `run_mjx_sweep.py` 방식이 **파일 유실
   사고**를 냈다. `.pyc`만 남았는데 CPython 3.12로 컴파일된 것을 로컬 3.10/3.11에서는
   완전 역마샬할 수 없어, marshal 스트림에서 심볼·상수 문자열만 긁어 구조를 회수해야 했다
   (`MJX_TO_WTW` / `WTW_TO_MJX` / `DEFAULT_DOF_POS_WTW` / `COMMANDS_SCALE`). Colab에서
   `sys.path` 취급도 불안정하다.
2. 주의 — **`%autoreload` 금지.** IPython autoreload는 `imp`를 import 하는데 `imp`는 Python
   3.12에서 제거됐고 Colab은 3.12다 ([colabtools#5758](https://github.com/googlecolab/colabtools/issues/5758)).
   대신:
   ```python
   from wtw_nav.dev import reload_wtw
   reload_wtw()                      # wtw_nav.* 모듈 캐시 제거
   from wtw_nav.llc import check     # 다시 import 하면 새 코드
   ```
   ★ **기존에 만들어 둔 객체는 옛 클래스에 묶여 있으니 다시 생성해야 한다**
   (`cfg`를 재생성하지 않아 `TypeError: unexpected keyword argument`가 난 적이 있다).
   Drive 동기화 확인은 `check.version_check()`.
3. **체크포인트는 반드시 Drive에.** Colab `/content`는 런타임 종료 시 소실된다.
   과거 파라미터 유실이 실제로 발생한 지점.
4. **JIT 재컴파일 최소화.** 지형 랜덤화 시 geom 개수를 고정하고 위치·크기만 바꾸는 원칙은
   성능이 아니라 **세션 시간 보호** 차원에서 필수다.
5. **노트북 3분할 유지.** 학습(`02`)과 평가(`03`)를 한 노트북에 두지 않는다 —
   학습 셀 재실행 비용이 너무 크다.

> 설치와 실행 절차(로컬 검증 3종 → Colab 학습 → 평가)는 **[루트 README](../README.md)**
> 가 소유한다. 여기는 그 절차가 왜 그런 모양인지를 다룬다.

### 렌더링

영상은 노트북에 자동 인라인 표시되고 `checkpoints/llc_check/*.mp4`로도 저장된다.
mp4 저장은 ffmpeg이 필요하지만 없어도 **표시는 계속**되며, 동영상 표시가 막히면
정지 프레임으로 대체된다.

주의 — 화면이 검으면 GL 백엔드 문제다. **`MUJOCO_GL`은 mujoco import 前에 정해져야 한다** —
나중에 바꿔도 소용없다. GPU 런타임이면 `egl`, CPU 런타임이면 `osmesa`이고
`policy.default_gl()`이 `/dev/nvidia*` 유무로 자동 판별한다.

TensorBoard는 **학습 셀보다 먼저** 띄워야 실시간으로 보인다:
```python
%load_ext tensorboard
%tensorboard --logdir runs
```
진행 상황은 **eval 시점에만** 갱신되므로 `PPOConfig.num_evals`가 점의 개수를 결정한다.

---

## 알려진 이슈

1. **`wtw_nav/eval.py`는 아직 실행된 적이 없다** — 정적 검증만 거쳤다. `load()`의 brax
   API 사용(`make_ppo_networks` 인자, `(normalizer, policy)` 슬라이스)이 첫 실행에서
   깨질 가능성이 가장 높다.
2. **`notebooks/03_eval_hlc.ipynb`는 스켈레톤**이다. 로직은 `wtw_nav/eval.py`에 있다.
3. **`ActionConfig.yaw_bias`는 배선만 돼 있고 측정된 적이 없다**(기본 0.0 = 무효).
   `beam` 재개의 전제 조건이다 → [0002](decisions/0002-beam-slit-제외.md).
4. **로컬 `mujoco_env`에 JAX가 없다** (mujoco 3.1.0 + torch만). 로컬은 정적 검증까지,
   MJX 실행은 Colab에서.
5. `hlc/fsm.py`는 없다 — 지형 돌파 폐기로 전환할 모드가 사라졌다.
   `sensors.py`의 높이스캔·클리어런스도 같은 이유로 미구현이다.
