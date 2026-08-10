"""HLC PPO 학습 실행기 (Brax).

노트북은 이 모듈을 호출만 한다 (`docs/README.md` (Colab 워크플로 1번)).

    from wtw_nav import train
    make_inference_fn, params, metrics = train.run()

⚠️ 체크포인트는 반드시 Drive 아래(프로젝트 폴더)에 저장한다 — Colab `/content`는
런타임 종료 시 소실된다.
"""

from __future__ import annotations

import functools
import os
import pickle
import time

from wtw_nav.configs import HLCConfig, default_config
from wtw_nav.envs.nav_env import NavEnv


def _check_ppo_config(p) -> None:
    """brax PPO의 나눗셈 제약을 **미리** 검사한다.

    안 하면 env 생성·컴파일에 몇 분 쓴 뒤에야 assert로 죽는다.
    """
    import jax

    n_dev = jax.local_device_count()
    problems = []
    if p.num_envs % n_dev:
        problems.append(f"num_envs({p.num_envs}) % device_count({n_dev}) != 0")
    if (p.batch_size * p.num_minibatches) % p.num_envs:
        problems.append(
            f"batch_size*num_minibatches({p.batch_size * p.num_minibatches}) "
            f"% num_envs({p.num_envs}) != 0")
    if p.num_evals < 1:
        problems.append("num_evals >= 1 이어야 함")
    if problems:
        raise ValueError("PPOConfig 제약 위반:\n  - " + "\n  - ".join(problems))

    # ---- 여기부터는 죽이지 않고 경고만. 학습이 조용히 실패하는 원인들이다. ----
    # brax는 이만큼 모아야 정책을 **한 번** 갱신한다.
    per_update = p.batch_size * p.unroll_length * p.num_minibatches
    n_updates = p.num_timesteps // per_update
    floor = p.num_evals * per_update
    print(f"PPO: 갱신 1회당 {per_update:,} env steps -> 총 {n_updates}회 갱신")
    if n_updates < 100:
        print(f"  ⚠️ 갱신 {n_updates}회는 너무 적습니다. PPO는 수백 회가 필요합니다.\n"
              f"     batch_size/unroll_length/num_minibatches를 줄이거나 "
              f"num_timesteps를 늘리십시오.\n"
              f"     (2026-07-29: 39회로 돌렸다가 정점 후 단조 하강으로 실패)")
    # ★ brax는 epoch당 스텝 수를 **올림**한다. 그래서 예산을 넘겨 돈다:
    #     per_epoch = ceil(num_timesteps / (num_evals_after_init × per_update))
    #     실제 총량 = num_evals_after_init × per_epoch × per_update
    #   2026-08-01 p2_gap: 6M을 줬는데 6,881,280(114.7%)을 돌았다 — 45분 낭비.
    after_init = max(p.num_evals - 1, 1)
    per_epoch = -(-p.num_timesteps // (after_init * per_update))     # ceil
    actual = after_init * per_epoch * per_update
    if actual > p.num_timesteps * 1.02:
        clean = after_init * per_epoch * per_update
        print(f"  ⚠️ 실제로는 {actual:,} 스텝을 돕니다 "
              f"(지정 {p.num_timesteps:,}의 {actual / p.num_timesteps:.1%}). "
              f"brax가 epoch를 올림해서입니다.\n"
              f"     예산을 정확히 맞추려면 num_timesteps를 "
              f"(num_evals-1) × {per_update:,} 의 배수로: 예 {clean:,}")
    if floor > p.num_timesteps:
        print(f"  ⚠️ **num_timesteps가 무시됩니다.** eval마다 최소 1회 학습하므로 실제로는\n"
              f"     >= num_evals × {per_update:,} = {floor:,} 스텝을 돕니다 "
              f"(지정값 {p.num_timesteps:,}).\n"
              f"     num_evals를 {max(1, p.num_timesteps // per_update)} 이하로 낮추십시오.")


def run(cfg: HLCConfig | None = None, out_dir: str = "checkpoints",
        tag: str = "p1_flat", logdir: str | None = "runs/hlc", plot: bool | None = None):
    """PPO 학습을 돌리고 파라미터를 저장한다.

    Args:
        logdir: TensorBoard 로그 경로. `None`이면 비활성.
        plot: 노트북 인라인 라이브 플롯. `None`이면 노트북에서 자동으로 켜진다.
    """
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo

    from wtw_nav.monitor import ProgressMonitor

    cfg = cfg or default_config()
    p = cfg.ppo
    _check_ppo_config(p)
    env = NavEnv(cfg)

    print(f"env: obs={env.observation_size} act={env.action_size} "
          f"HLC dt={env._hlc_dt:.3f}s max_steps={env._max_steps}")
    print(f"PPO: {p.num_timesteps:,} steps, {p.num_envs} envs, {p.num_evals} evals "
          f"(-> eval 간격 {p.num_timesteps // max(p.num_evals,1):,} steps)")

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=p.policy_hidden,
        value_hidden_layer_sizes=p.value_hidden,
    )

    t0 = time.time()
    mon = ProgressMonitor(logdir=logdir, total_steps=p.num_timesteps, plot=plot)
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/hlc_{tag}.pkl"

    # ★ eval마다 저장한다. **끝에 한 번만 저장하면 안 된다** — 2026-08-01 p2_gap은
    #   5.7시간을 돌고 마지막 `assert_is_replicated`에서 죽어 체크포인트를 통째로
    #   잃었다. 측정값은 TensorBoard에 남았지만 정책은 남지 않았다.
    # brax는 `policy_params_fn(current_step, make_policy, params)`로 부른다.
    # 마지막 인자가 params이며, 그 3-튜플은 `ppo.train`이 최종 반환하는 것과 **동일**
    # 하다(normalizer, policy, value) — 중간 저장본과 최종 저장본의 형식이 같다.
    def _save(step, *args):
        with open(path, "wb") as f:
            pickle.dump({"params": args[-1], "cfg": cfg, "log": mon.history,
                         "step": int(step)}, f)

    # 첫 줄이 나오기까지 MJX 컴파일로 수 분이 걸린다. 안내가 없으면 멈춘 줄 안다.
    print("\n[train] JIT 컴파일 중 — 첫 eval 행이 나오기까지 수 분 걸립니다. "
          "이후 행은 eval 간격마다 갱신됩니다.", flush=True)
    print(f"[train] eval마다 {path} 에 저장합니다 (중간에 죽어도 남습니다).",
          flush=True)

    try:
        make_inference_fn, params, metrics = ppo.train(
            environment=env,
            num_timesteps=p.num_timesteps,
            num_evals=p.num_evals,
            episode_length=env._max_steps,
            num_envs=p.num_envs,
            batch_size=p.batch_size,
            unroll_length=p.unroll_length,
            num_minibatches=p.num_minibatches,
            num_updates_per_batch=p.num_updates_per_batch,
            learning_rate=p.learning_rate,
            entropy_cost=p.entropy_cost,
            discounting=p.discounting,
            max_grad_norm=p.max_grad_norm,
            normalize_observations=True,
            action_repeat=1,
            seed=p.seed,
            network_factory=network_factory,
            progress_fn=mon,
            policy_params_fn=_save,
        )
    except Exception as e:
        # 로그만이라도 남긴다 — 한계 측정의 산출물은 정책이 아니라 이 숫자들이다.
        with open(f"{out_dir}/hlc_{tag}_log.pkl", "wb") as f:
            pickle.dump({"cfg": cfg, "log": mon.history, "error": repr(e)}, f)
        mon.close()
        print(f"\n⚠️ 학습이 {(time.time()-t0)/60:.1f}분에서 죽었습니다: "
              f"{type(e).__name__}: {e}")
        print(f"   마지막 체크포인트 {path}, 로그 {out_dir}/hlc_{tag}_log.pkl 는 남았습니다.")
        print("   `eval/episode_nan`이 0이 아니면 물리 발산입니다 "
              "(`nav_env.step`의 NaN 안전망 주석).")
        raise
    mon.close()

    _save(p.num_timesteps, params)
    print(f"\n학습 완료 {(time.time()-t0)/60:.1f}분 → {path}")
    return make_inference_fn, params, metrics
