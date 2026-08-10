# 2026_wtw — Go1 재난 미로 내비게이션 (WTW → MJX 계층 제어)

Unitree Go1이 미로를 자율 주행한다. **Walk These Ways(WTW)** 사전학습 보행 정책을
MJX로 이식해 **동결**하고, 그 위에 PPO로 **상위 제어기(HLC)** 를 얹은 계층 구조다.

```
BFS 거리장(플래너) ──▶ HLC 10 Hz, 8D 명령 (PPO) ──▶ LLC 50 Hz (WTW, 동결) ──▶ MJX 200 Hz
                                                     HLC 1스텝 = LLC 5 = 물리 20
```

| | |
|---|---|
| 환경 | Colab + MJX (MuJoCo XLA), **Sim-only** |
| 현재 | **P4 미로 도달률 99.2%** (5×5, 5.9M 스텝, 234분) — 스크립트 제어기보다 1.6배 빠름 |
| 다음 | P5 일반화 판정 (미학습 배치·큰 미로) |

> 📖 **설계·측정 문서는 [`docs/`](docs/)에 있다.** 순서대로 읽으면 손댈 수 있는 상태가 된다:
> [문서 지도](docs/README.md) → [01_llc](docs/01_llc.md) → [02_hlc](docs/02_hlc.md) → [03_results](docs/03_results.md)

---

## 설치

외부 저장소 둘은 **이 저장소에 포함되지 않는다**(용량·라이선스). 프로젝트 루트에 직접 클론한다.

```bash
git clone https://github.com/improbable-ai/walk-these-ways.git
```
```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

⚠️ **사전학습 가중치는 WTW 저장소 안에 들어 있다.** 클론하면
`walk-these-ways/runs/gait-conditioned-agility/pretrain-v0/train/025417.456545/`에
`body_latest.jit` · `adaptation_module_latest.jit` · `parameters.pkl` ·
`curriculum/distribution.pkl`이 함께 온다. 넷 다 필요하다 —
`parameters.pkl`과 `distribution.pkl`은 **명령 범위의 유일한 정답 근거**다
([01_llc.md §0.1](docs/01_llc.md)).

**로컬(CPU)** — 형상·정확성 검증만. 학습은 못 한다.
```bash
conda create -n mujoco_env python=3.11 && conda activate mujoco_env && pip install mujoco torch numpy
```

**Colab(GPU)** — 학습·렌더링.
```bash
pip -q install mujoco mujoco-mjx brax
```

---

## 사용법

### 1. 로컬에서 먼저 — JAX 없이 도는 검증 3종

**학습을 돌리기 전에 이 순서로** 확인한다. 뒤로 갈수록 비싸다.

```bash
conda run -n mujoco_env python -m wtw_nav.llc.test_policy
```
LLC 상수·관측 70D 조립·**DOF 순서**·리셋 자세. 29항목. jax가 없으면 numpy 미러로 같은
수식을 검증하므로 어느 환경에서도 돈다.

```bash
conda run -n mujoco_env python -m wtw_nav.envs.test_nav_env
```
`reset`/`step`의 **vmap 가능성**과 brax 래퍼 pytree 구조. 둘 다 학습을 돌리기 전엔
드러나지 않는 고장이다.

```bash
conda run -n mujoco_env python -m wtw_nav.envs.reward_audit
```
보상 유인의 **상대 크기**. 도달 vs 거의 도달 격차, 속도 유인, 순위 역전을 검사한다.
**보상 상수를 건드릴 때마다 돌릴 것.**

임포트가 깨지면 (`partially initialized module 'jax'` 등) 먼저 이것부터 —
대개 **커널 재시작**이 답이다:
```bash
conda run -n mujoco_env python -c "from wtw_nav.llc.test_policy import env_report; env_report()"
```

### 2. Colab에서 — 마운트 후 노트북 실행

노트북은 **마운트·설정·호출만** 한다. 로직은 전부 `wtw_nav/` 안에 있다.

```python
from google.colab import drive; drive.mount('/content/drive')
import os, sys
P = "/content/drive/MyDrive/.../2026_wtw"      # 프로젝트 경로
os.chdir(P); sys.path.insert(0, P)
```

| 노트북 | 하는 일 |
|---|---|
| `notebooks/01_llc_check.py` | P0/P0.5 LLC 게이트 + 명령 스윕 영상 |
| `notebooks/02_train_hlc.py` | HLC PPO 학습 (장시간) |
| `notebooks/03_eval_hlc.ipynb` | 평가·시각화 |
| `notebooks/04_terrain_measure.py` | 지형 한계 실측 |

⚠️ 파일명이 숫자로 시작해 **`import`할 수 없다.** `%run notebooks/01_llc_check.py`로 돌리거나
`wtw_nav.*`를 직접 import 한다.

### 3. 학습

**돌리기 전에 비용을 잰다.** 이 프로젝트는 벤치 없이 학습을 걸었다가 21시간짜리를
만든 적이 있다.

```python
from wtw_nav import bench
bench.breakdown()          # ⚠️ raw env.step만 잰다. 실제는 ×0.58 (PPO 갱신 30% + eval 18%)
```

```python
from wtw_nav import train
from wtw_nav.configs.default import maze_config
cfg = maze_config(n=5, seed=0)     # 미로 크기에서 타임아웃·에피소드 길이·num_timesteps 역산
train.run(cfg)
```

`train._check_ppo_config()`가 갱신 횟수와 `num_evals`를 **학습 시작 전에** 경고한다.
체크포인트는 `checkpoints/`(= Drive 아래)에 저장된다 — Colab `/content`는 런타임 종료 시
소실된다.

TensorBoard는 **학습 셀보다 먼저** 띄워야 실시간으로 보인다:
```python
%load_ext tensorboard
%tensorboard --logdir runs
```

### 4. 평가

```python
from wtw_nav import eval as E
pol, cfg_tr, log, step = E.load("checkpoints/hlc_p4_maze.pkl")
E.curve(log)          # 수렴 지점 -> 다음 실행 예산 (P4는 43% 지점에서 이미 95% 도달)
E.report(pol, n=20)   # 학습 미로 + 미학습 배치(seed 7·13) + 큰 미로(n=8·11)
```

★ **미로 하나는 MJX 모델 하나이므로 학습 seed의 성적은 레이아웃 암기와 구분되지 않는다.**
`report()`가 미학습 배치·큰 미로를 함께 돌려 이를 판정하고, 실패를
`reached / fell / stuck / timeout`으로 분류해 실패 영상을 낸다.

### 5. 지형 한계 확인

```python
from wtw_nav.terrain import limits
limits.report()
```
실측값은 **`limits.py`가 소유한다** — 문서가 아니라. `validate()`는 미측정 지형을
조용히 잘라내지 않고 **거부한다.**

---

## 저장소 구조

```
2026_wtw/
├── docs/                  설계·측정 문서 (여기부터 읽는다)
│   ├── README.md          문서 지도 · SSOT 소유권 · 현재 상태 · Colab 규칙
│   ├── 01_llc.md          15D 명령 사양 · WTW→MJX 이식 · 함정 21개
│   ├── 02_hlc.md          관측 · 보상 · env 파이프라인 · PPO
│   ├── 03_results.md      실측 대장
│   ├── decisions/         뒤집힌 판단 5건 (ADR)
│   └── archive/           재구성 이전 원본
├── wtw_nav/               프로젝트 본체 (import 가능한 패키지)
│   ├── configs/           모든 HLC 상수의 단일 진실 공급원
│   ├── llc/               동결 LLC 로드·추론 + P0 게이트
│   ├── hlc/               명령 필터 · 유도 벡터 · 라이다
│   ├── terrain/           지형 모듈 · 미로 · 한계값
│   ├── envs/              학습 env · 수동 제어기 · 게이트
│   ├── train.py eval.py bench.py monitor.py dev.py
├── notebooks/             Colab 진입점 (로직 없음)
├── checkpoints/           학습 산출물 (P4 결과만 추적)
├── walk-these-ways/       외부 — 직접 클론 (미추적)
└── mujoco_menagerie/      외부 — 직접 클론 (미추적)
```

**루트 스크립트를 두지 않는다.** 모든 로직은 `wtw_nav/` 안의 import 가능한 모듈에 둔다.
과거 루트에 있던 `wtw_mjx_core.py` / `train_hlc.py` / `run_mjx_sweep.py`가 **유실 사고**를
냈고, Colab에서 `sys.path` 취급도 불안정하다.

⚠️ **`walk-these-ways/`와 `mujoco_menagerie/`는 수정하지 않는다.** 외부 읽기 전용이다.

⚠️ **Colab에서 `%autoreload`를 쓰지 말 것** — autoreload가 import 하는 `imp`는 Python
3.12에서 제거됐고 Colab은 3.12다. 대신 `from wtw_nav.dev import reload_wtw; reload_wtw()`
후 다시 import 한다. **기존 객체는 옛 클래스에 묶여 있으니 재생성해야 한다.**

---

## 라이선스·귀속

- [Walk These Ways](https://github.com/improbable-ai/walk-these-ways) (Improbable AI) —
  사전학습 정책과 명령 인터페이스의 출처
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Google DeepMind) —
  Unitree Go1 모델

둘 다 이 저장소에 포함되지 않으며 각자의 라이선스를 따른다.
