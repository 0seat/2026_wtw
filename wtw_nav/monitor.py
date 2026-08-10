"""학습 진행 모니터링 — 콘솔 표 + TensorBoard + 노트북 인라인 라이브 플롯.

Brax PPO는 `progress_fn(num_steps, metrics)`를 eval 시점마다 부른다. 그 사이 간격이
길기 때문에(`num_evals`로 조절), 진행이 보이려면 eval 횟수를 충분히 잡아야 한다.

의존성은 전부 선택적이다 — 없으면 해당 출력만 건너뛴다.
  tensorboardX : brax 설치 시 함께 들어온다
  matplotlib   : Colab 기본 제공
"""

from __future__ import annotations

import time


def _is_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


class ProgressMonitor:
    """`brax ppo.train(progress_fn=...)`에 넘길 콜백.

        mon = ProgressMonitor(logdir="runs/p1", total_steps=cfg.ppo.num_timesteps)
        ppo.train(..., progress_fn=mon)
        mon.close()

    Colab에서 TensorBoard를 보려면 **학습 셀보다 먼저** 아래를 실행해 두십시오::

        %load_ext tensorboard
        %tensorboard --logdir runs
    """

    #: 인라인 플롯에 그릴 지표 (있는 것만 그린다)
    #: 주의 — 라벨은 **ASCII만** — Colab 기본 폰트(DejaVu Sans)에 한글 글리프가 없어
    #: 매 프레임 `UserWarning: Glyph ... missing from font(s)`가 쏟아진다.
    #: 콘솔 표는 한글이어도 무방하다(폰트와 무관).
    PLOT_KEYS = (
        ("eval/episode_reward", "reward"),
        ("eval/episode_dist", "dist to goal (m)"),
        ("eval/episode_reached", "reached"),
        ("eval/episode_fell", "fell"),
        ("eval/episode_nan", "nan (physics blowup)"),
        # ★ 지형 사다리에서 넘긴 장애물 수. 평지(P1)에서는 항상 0이라 자동으로
        #   빠진다(`keys` 필터). 이 값이 `terrain/limits.py`를 채우는 지표다.
        ("eval/episode_level", "level cleared"),
    )

    def __init__(self, logdir: str | None = "runs/hlc", total_steps: int | None = None,
                 plot: bool | None = None, print_table: bool = True):
        self.total = total_steps
        self.print_table = print_table
        self.plot = _is_notebook() if plot is None else plot
        self.history: list[tuple[int, dict]] = []
        self.t0 = time.time()
        self._header_done = False

        self.writer = None
        if logdir:
            try:
                from tensorboardX import SummaryWriter
                self.writer = SummaryWriter(logdir)
                print(f"[monitor] TensorBoard -> {logdir}   "
                      f"(%load_ext tensorboard; %tensorboard --logdir {logdir.split('/')[0]})")
            except Exception as e:
                print(f"[monitor] TensorBoard 비활성 ({type(e).__name__}: {e})")

        self._fig = None
        self._disp = None          # IPython display handle (제자리 갱신용)

    # ------------------------------------------------------------------ 콜백
    def __call__(self, num_steps: int, metrics: dict) -> None:
        m = {k: float(v) for k, v in metrics.items()
             if isinstance(v, (int, float)) or hasattr(v, "item")}
        self.history.append((int(num_steps), m))

        if self.writer is not None:
            for k, v in m.items():
                self.writer.add_scalar(k, v, num_steps)
            self.writer.flush()

        if self.print_table:
            self._print_row(num_steps, m)
        if self.plot:
            self._draw()

    # ------------------------------------------------------------------ 표
    def _print_row(self, num_steps: int, m: dict) -> None:
        # ★ `level`을 반드시 넣는다. 2026-08-01 p2_gap 5.7시간 실행에서 이게 표에
        #   없어 정작 산출물인 "넘긴 장애물 수"를 실행 내내 볼 수 없었고, dist에서
        #   역산해야 했다. `nan`은 물리 발산 감시용 — 0이 아니면 그 행은 못 믿는다.
        cols = [("reward", "eval/episode_reward"),
                ("level", "eval/episode_level"),
                ("dist", "eval/episode_dist"),
                ("reached", "eval/episode_reached"),
                ("fell", "eval/episode_fell"),
                ("nan", "eval/episode_nan"),
                ("len", "eval/avg_episode_length")]
        cols = [(lbl, k) for lbl, k in cols if k in m]
        if not self._header_done:
            head = f"{'steps':>12s} {'진행':>6s} " + " ".join(f"{l:>9s}" for l, _ in cols)
            print(head + f" {'경과':>8s}")
            print("-" * (len(head) + 9), flush=True)
            self._header_done = True
        pct = f"{100*num_steps/self.total:5.1f}%" if self.total else "     -"
        vals = " ".join(f"{m[k]:9.3f}" for _, k in cols)
        el = time.time() - self.t0
        print(f"{num_steps:>12,} {pct} {vals} {el/60:7.1f}m", flush=True)

    # ------------------------------------------------------------------ 플롯
    def _draw(self) -> None:
        try:
            import matplotlib.pyplot as plt
            from IPython import display
        except Exception:
            self.plot = False
            return

        keys = [(k, lbl) for k, lbl in self.PLOT_KEYS
                if any(k in m for _, m in self.history)]
        if not keys:
            return

        if self._fig is None:
            self._fig, self._axes = plt.subplots(
                1, len(keys), figsize=(4.2 * len(keys), 3.2))
            if len(keys) == 1:
                self._axes = [self._axes]
            # 인라인 백엔드가 셀 종료 시 한 번 더 자동 표시하는 것을 막는다.
            plt.close(self._fig)

        xs = [s for s, _ in self.history]
        for ax, (k, lbl) in zip(self._axes, keys):
            ys = [m.get(k, float("nan")) for _, m in self.history]
            ax.clear()
            ax.plot(xs, ys, marker="o", ms=3)
            ax.set_title(lbl, fontsize=10)
            ax.set_xlabel("env steps")
            ax.grid(alpha=0.3)
        self._fig.tight_layout()

        # 주의 — `clear_output()`을 쓰면 **셀 출력 전체**가 지워져 진행 표까지 사라진다.
        # 그러면 학습이 멈춘 것처럼 보인다. 그림 하나만 제자리 갱신할 것.
        if self._disp is None:
            self._disp = display.display(self._fig, display_id=True)
        else:
            self._disp.update(self._fig)

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()

    def to_dict(self) -> dict:
        return {"history": self.history, "elapsed_s": time.time() - self.t0}
