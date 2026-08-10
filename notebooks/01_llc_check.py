"""01. LLC 검증 — P0 / P0.5 게이트 실행기.

로직은 전부 `wtw_nav/llc/check.py`에 있다 (파일명이 숫자로 시작하면 import가 불가능하고,
프로젝트 규칙상 노트북에는 로직을 두지 않는다 — `docs/README.md` 참조).

Colab에서는 이 파일을 실행하지 말고 **모듈을 직접 import** 하는 쪽이 편하다::

    from google.colab import drive; drive.mount('/content/drive')
    %cd "/content/drive/Othercomputers/BPC/D:/02_projects/2026_wtw"
    import sys; sys.path.append(".")
    from wtw_nav.dev import reload_wtw   # %autoreload 금지 (Colab 3.12에 imp 없음)

    from wtw_nav.llc import check
    env = check.build()             # 정책 + MJX 모델 (JIT 컴파일 1회)
    check.run_gate(env)             # P0  게이트
    check.sweep_video(env=env, show=True)   # P0.5 게이트 — 영상까지

한 방에 돌리려면::

    %run notebooks/01_llc_check.py

로컬(CPU, JAX 없음)에서는 정적 검증만 가능하다::

    conda run -n mujoco_env python -m wtw_nav.llc.test_policy
"""

import sys

from wtw_nav.llc import check


def main():
    env = check.build()
    ok = check.run_gate(env)
    if not ok:
        print("\nP0 미통과 — P0.5로 넘어가지 않습니다.")
        return 1
    print()
    check.sweep_video(env=env)
    return 0


if __name__ == "__main__":
    rc = main()
    # %run 으로 부를 때 SystemExit이 커널을 어지럽히지 않도록 CLI에서만 종료코드를 낸다.
    if "IPython" not in sys.modules:
        sys.exit(rc)
