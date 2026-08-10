# 2026_wtw 프로젝트 구조 안내 (Project Structure)

이 문서는 본 프로젝트의 핵심 디렉토리 및 파일들의 역할과 관계를 설명합니다. 이 프로젝트는 **Walk These Ways (WTW)** 강화학습 정책을 **MJX (MuJoCo XLA)** 환경으로 이식(Porting)하고, 상위 제어기(HLC)를 훈련하여 로봇(Unitree Go1)이 재난 환경 미로를 자율 주행하도록 만드는 계층적 제어 아키텍처입니다.

---

## 🧭 구조의 전제 — Colab 워크플로

현재의 `wtw_nav/` 패키지 + `notebooks/` 구조는 **Colab에서의 학습 실행을 위해 의도적으로 채택**된 것입니다. 로컬 PC는 CPU 전용이라 형상 검증까지만 담당하고, 실제 학습은 Colab GPU에서 돌립니다.

```
로컬 (Windows, CPU)          Google Drive 동기화          Colab (A100/T4)
  wtw_nav/**/*.py  ─────────────────────────────────────▶  drive.mount()
  코드 편집·1-env 정확성 검증                                os.chdir(project_path)
                                                            sys.path.append(cwd)
  notebooks/*.ipynb ──────────────────────────────────────▶ import wtw_nav.envs.nav_env
                                                            → 학습 실행 / 렌더링
```

이 구조가 강제하는 규칙:

1. **루트 스크립트 금지.** 모든 로직은 `wtw_nav/` 안의 import 가능한 모듈에 둡니다. 노트북은 마운트·설정·호출만 합니다. (과거 루트에 있던 `wtw_mjx_core.py` / `train_hlc.py` / `run_mjx_sweep.py` 방식은 파일 유실 사고를 냈고 Colab에서 `sys.path` 취급도 불안정했습니다.)
2. **셀 재실행 안전성.** ⚠️ **`%autoreload`를 쓰지 마십시오** — IPython의 autoreload는 `imp`를 import 하는데 `imp`는 Python 3.12에서 제거됐고 Colab은 3.12라 `ModuleNotFoundError: No module named 'imp'`가 납니다 ([colabtools#5758](https://github.com/googlecolab/colabtools/issues/5758)). 대신:
   ```python
   from wtw_nav.dev import reload_wtw
   reload_wtw()                      # wtw_nav.* 모듈 캐시 제거
   from wtw_nav.llc import check     # 다시 import 하면 새 코드
   ```
   기존에 만들어 둔 객체(예: `env = check.build()`)는 옛 클래스에 묶여 있으니 다시 생성해야 합니다. Drive 동기화 확인은 `check.version_check()`가 파일 mtime과 실제 반영값을 찍어 줍니다.
3. **체크포인트는 반드시 Drive에.** Colab `/content`는 런타임 종료 시 소실됩니다. `checkpoints/`를 프로젝트 폴더(= Drive) 아래로 고정합니다.
4. **JIT 재컴파일 최소화.** Colab 세션 시간이 유한하므로, 지형 랜덤화 시 geom 개수를 고정하고 위치·크기만 바꾸는 원칙은 성능이 아니라 **세션 시간 보호** 차원에서 필수입니다.
5. **노트북 3분할 유지.** 학습(`02`)과 평가(`03`)를 한 노트북에 두지 않습니다 — 학습 셀 재실행 비용이 너무 큽니다.

---

## 📂 디렉토리 구조

```
2026_wtw/
├── docs/                       # 설계·참조 문서
├── wtw_nav/                    # 프로젝트 본체 (import 가능한 패키지)
│   ├── configs/                # (미구현) 단일 진실 공급원
│   ├── llc/                    # 하위 제어기 래퍼
│   ├── hlc/                    # 상위 제어기 구성요소
│   ├── terrain/                # 지형·미로 생성
│   └── envs/                   # RL 환경
├── notebooks/                  # Colab 진입점
├── walk-these-ways/            # 외부: WTW 원본 + 사전학습 가중치
├── mujoco_menagerie/           # 외부: Go1 로봇 모델
└── checkpoints/                # 학습 산출물 (학습 시 생성)
```

### `docs/`
* **`hlc_design.md`** — 상위 제어기 마스터 설계 문서 (**what/why**). 역할 분리, 센서 구성, 보상 설계, 로드맵.
* **`hlc_plan.md`** — 구현 계획서 (**how/order**). 현황 진단, 모듈 인터페이스, 단계별 검증 게이트, 리스크. 부록 B에 명령 범위 재확인 스니펫.
* **`command_scale_guide.md`** — 15차원 명령 벡터의 의미·스케일·**실제 학습 상태**. HLC Action Space의 근거 문서.
* **`project_structure.md`** — (본 문서).

### `wtw_nav/llc/` — 하위 제어기 (LLC)
* **`policy.py`** — WTW 사전학습 가중치를 로드해 JAX/MJX에서 구동하는 래퍼. 병렬 학습이 가능하도록 torch JIT 호출이 아니라 **순수 JAX MLP로 재구현**되어 있습니다.
  * `load_policy(ckpt_dir)` — **`body_latest.jit` + `adaptation_module_latest.jit` 둘 다** 로드. `latent = adaptation(obs_history)` → `body(concat([obs_history, latent]))`.
  * `create_env(xml_path)` — WTW 학습 조건에 맞춰 PD 게인(Kp=20, Kd=0.5)·timestep(0.005)을 설정. **`mjx_friendly=True`(기본)가 menagerie XML의 solver iteration 100→1, ls 50→4, elliptic→pyramidal cone, 발 condim 6→3으로 바꿉니다** — 이게 없으면 MJX 컴파일이 수십 분 걸립니다.
  * `make_rollout_fn(...)` — T 스텝 전체를 `lax.scan` 하나로. **Python for 루프로 스텝을 반복 호출하지 마십시오** (스텝마다 dispatch + host 동기화 → 수십 분).
  * `get_obs(...)` — WTW의 **70D** 관측 조립. `make_step_fn(...)` — LLC 1스텝(50 Hz).
  * `make_commands(...)` — 15D 명령 조립. duty·roll은 학습되지 않았으므로 인자로 노출하지 않습니다.
* **`test_policy.py`** — P0 게이트(**로컬**). 상수가 체크포인트에서 유래하는지·MLP가 torch와 일치하는지·관절 매핑·70D 조립·리셋 자세를 검증합니다. jax/mjx가 있으면 `policy.py`를 직접 실행하고, 없으면 numpy 미러로 같은 수식을 검증하므로 **어느 환경에서도 돌아갑니다**.
  ```bash
  conda run -n mujoco_env python -m wtw_nav.llc.test_policy
  ```
  * `env_report()` — jax/mujoco/mjx 임포트 가능 여부와 버전을 점검합니다. `partially initialized module 'jax'` 같은 오류가 나면 먼저 이걸 돌리십시오 (대개 **커널 재시작**이 답입니다 — pip install 후에는 항상 재시작).
    ```bash
    conda run -n mujoco_env python -c "from wtw_nav.llc.test_policy import env_report; env_report()"
    ```
* **`check.py`** — P0/P0.5 게이트(**Colab, JAX 필요**). `build()` / `run_gate()` / `preview()` / `sweep_video()`.
  ```python
  from wtw_nav.llc import check, policy as P
  env = check.build()                                        # JIT 컴파일 1회, 재사용
  check.run_gate(env)                                        # P0
  check.preview(env, P.make_commands(vx=0.8, gait="trot"))   # 영상 하나 빠르게
  check.sweep_video(env=env)                                 # P0.5 전체
  ```
  * 영상은 노트북에서 **자동 인라인 표시**되고 `checkpoints/llc_check/*.mp4`로도 저장됩니다. mp4 저장은 ffmpeg이 필요하지만, 없어도 **표시는 계속**되며 동영상 표시가 막히면 정지 프레임으로 대체됩니다.
  * 화면이 검으면 GL 백엔드 문제입니다 — Colab은 `MUJOCO_GL=egl`이어야 하고 mujoco import 전에 정해져야 하는데, 이 패키지가 자동 설정합니다.

### `wtw_nav/configs/`
* **`default.py`** — **모든 HLC 상수의 단일 진실 공급원**. 액션 범위·마름모 계수·데드존·저역필터·코스·보상·종료·PPO 하이퍼파라미터. 여기 없는 상수를 코드에 흩어 두지 마십시오 (기본값을 두 곳에 두었다가 한 곳만 고쳐 실행이 옛 값으로 돈 사고가 있었습니다). LLC 상수는 `llc/policy.py`가 소유합니다.

### `wtw_nav/hlc/` — 상위 제어기
* **`command_filter.py`** — HLC 액션(8D) → 15D 명령. ① tanh 아핀 매핑 ② vx–yaw L1 마름모 제약 ③ 미소 명령 데드존 스냅 ④ 저역필터 ⑤ 고정 차원 전개(gait=trot, duty=0.5, roll=0). 순서가 중요합니다.
* **`guidance.py`** — 유도 벡터 `(cosφ, sinφ, d_norm)`. **훈련(코스 축)과 평가(BFS 거리장)가 같은 출력 형식**을 내는 것이 핵심입니다.
* `sensors.py` — (미구현) 라이다 / 하향 높이 스캔 / 상향 클리어런스. 지형이 box 파라미터로 생성되므로 **해석적 질의**로 구현 (물리·센서 불일치 원천 차단).
* `fsm.py` — (미구현) 지형 모드 판별 + 모드별 명령 제약 (Phase 3 이후).

### `wtw_nav/terrain/` — 지형 생성 (미구현)
* `modules.py` — gap / step / beam / tunnel / slope / slit을 box geom으로 조립.
* `course.py` — 훈련용 직선 코스 + 초기조건 랜덤화.
* `maze.py` — 평가용 미로 + BFS 거리장 (0.1 m 셀, 생성기 의미 정보로 통과 가능성 마킹).

### `wtw_nav/train.py`
PPO 학습 실행기. 노트북은 `train.run()`을 호출만 합니다. 체크포인트는 `checkpoints/`(Drive 아래)에 저장됩니다 — Colab `/content`는 런타임 종료 시 소실됩니다.

### `wtw_nav/monitor.py`
학습 진행 모니터링. **콘솔 표 + TensorBoard + 노트북 인라인 라이브 플롯**을 함께 냅니다. 의존성은 전부 선택적이라 없으면 해당 출력만 건너뜁니다.

```python
%load_ext tensorboard
%tensorboard --logdir runs      # ← 학습 셀보다 먼저 띄워야 실시간으로 보입니다
```

진행 상황은 **eval 시점에만** 갱신되므로 `PPOConfig.num_evals`가 점의 개수를 결정합니다(기본 40).

### `wtw_nav/envs/`
* **`test_nav_env.py`** — P1 게이트. **`reset`/`step`이 `jit`·`vmap` 안에서 도는지**, 그리고 **brax 래퍼를 씌웠을 때 pytree 구조가 유지되는지**를 확인합니다(둘 다 학습을 돌려보기 전엔 드러나지 않습니다).
  ```bash
  conda run -n mujoco_env python -m wtw_nav.envs.test_nav_env
  ```
* **`scripted.py`** — 수동 제어기. **PPO를 돌리기 전에 env가 풀 수 있는 문제인지** 확인합니다. 유도 벡터만 보고 목표로 향하는 비례 제어기가 도달하지 못하면 성능이 아니라 보상·종료·유도벡터가 잘못된 것이고, 그걸 30M 스텝 태운 뒤 알아내는 건 낭비입니다.
  ```bash
  conda run -n mujoco_env python -m wtw_nav.envs.scripted
  ```
  도달률뿐 아니라 **타임아웃 여유**도 냅니다. 기준 제어기가 간신히 통과하면 그보다 조금만 느린 정책은 전부 타임아웃되어 도달 보너스를 못 받으므로, 여유 30% 미만이면 경고합니다(실제로 이 검사로 `timeout_s` 20 s → 30 s 조정을 잡았습니다).
* **`nav_env.py`** — HLC 훈련용 Brax/JAX 환경.
  * HLC(10 Hz) 액션 8D → 15D WTW 명령으로 전개 → LLC(50 Hz)를 `lax.scan`으로 5회 호출.
  * 관측 21D = proprio(10) + 유도 벡터(3) + 직전 명령(8). **관절각은 넣지 않습니다** — LLC 소관이며 보행 위상 과적합 위험.
  * 보상 = progress(potential-based) + 도달 + 시간 페널티 + 종료 페널티 + 명령 급변 페널티.
  * `brax.envs.base.Env`를 직접 상속합니다 — 물리를 MJX로 직접 굴리므로 브랙스 파이프라인(`sys`)이 필요 없습니다.
  * ⚠️ **`reset`은 `jit`/`vmap` 안에서 돌아야 합니다.** `__init__`에서 리셋 템플릿을 1회 만들고 `reset`에서는 `qpos`/`qvel`만 `replace` 합니다. `mujoco.MjData()`나 `mjx.put_data()`를 `reset` 안에서 부르면 병렬 학습이 불가능해집니다.

### `notebooks/` — Colab 진입점
* **`01_llc_check.py`** — `wtw_nav.llc.check`를 순서대로 호출하는 얇은 실행기. `%run notebooks/01_llc_check.py`로 한 번에 돌립니다.
  * ⚠️ 파일명이 숫자로 시작하므로 **`import`할 수 없습니다** (`from notebooks import 01_llc_check` → `SyntaxError`). 로직은 전부 `wtw_nav/llc/check.py`에 있으니 그쪽을 import 하십시오. 노트북에 로직을 두지 않는 규칙(위 §구조의 전제 1번)이 이래서 필요합니다.
* **`02_train_hlc.py`** — Brax PPO로 HLC 학습 (병렬 2048 env). 장시간 실행.
* **`03_eval_hlc.ipynb`** — 학습된 HLC 평가·시각화 (성공률, 도달시간, 낙상 원인).

### 외부 서브모듈 및 의존 리소스
* **`walk-these-ways/`** — WTW 원본 저장소. 사전학습 가중치는
  `runs/gait-conditioned-agility/pretrain-v0/train/025417.456545/checkpoints/`에 있습니다
  (`body_latest.jit`, `adaptation_module_latest.jit`, `ac_weights_last.pt`).
  같은 디렉토리의 `parameters.pkl`·`curriculum/distribution.pkl`이 **명령 범위의 유일한 정답 근거**입니다.
* **`mujoco_menagerie/`** — Go1 모델 (`unitree_go1/scene.xml`).
* **`checkpoints/`** — HLC 학습 결과(PPO 파라미터) 저장 위치. Drive 아래에 두어야 소실되지 않습니다.

---

## ⚠️ 알려진 이슈 (현재 상태)

1. **`envs/nav_env.py`는 Phase 1 스켈레톤입니다.** 액션 4D, 센서 없음, 그리고 `reset()`이 `jit`/`vmap` 안에서 동작하지 않아 병렬 학습이 불가능합니다. gait 값도 `(0.5, 0.5, 0.5)`로 잘못되어 있습니다(트로팅은 `(0.5, 0.0, 0.0)`). → P1에서 `llc/policy.py`를 쓰도록 재작성 예정.
2. **`wtw_nav/configs/`가 아직 없습니다.** 명령 범위·PD 게인 등 상수가 현재 `llc/policy.py`에 모여 있습니다. HLC 관련 상수가 추가되는 P1 시점에 분리합니다.
3. **로컬 `mujoco_env`에 JAX가 없습니다** (mujoco 3.1.0 + torch만). 따라서 로컬에서는 `test_policy.py`의 정적 검증까지만 가능하고, JAX 실행·MJX 롤아웃은 Colab에서 확인해야 합니다.
4. `notebooks/03_eval_hlc.ipynb`는 스켈레톤 상태입니다.

---

## 🔄 워크플로우 요약

1. **로컬 (Windows, CPU)** — 에디터에서 `wtw_nav/` 모듈을 수정하고, `notebooks/01_llc_check.py`로 1-env 구동·형상을 검증합니다. CPU라 학습은 하지 않습니다.
2. **동기화** — Google Drive 동기화 완료를 기다립니다.
3. **Colab (GPU)** — 노트북에서 `drive.mount()` → `os.chdir(project_path)` → `sys.path.append` 후 `import wtw_nav.*`. `02`로 학습, `03`으로 평가합니다. 결과물은 Drive 아래 `checkpoints/`에 저장합니다.
