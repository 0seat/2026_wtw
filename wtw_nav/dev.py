"""노트북용 개발 편의 함수.

⚠️ **Colab에서 `%autoreload`를 쓰지 마십시오.** IPython의 autoreload 확장은 `imp`를
import 하는데, `imp`는 Python 3.12에서 제거됐고 Colab은 3.12다 →
`ModuleNotFoundError: No module named 'imp'`.
(참고: https://github.com/googlecolab/colabtools/issues/5758)

대신 `reload_wtw()`를 쓴다 — `sys.modules`에서 `wtw_nav.*`를 지워 다음 import 때
디스크에서 다시 읽게 한다. 커널 재시작이 필요 없고 3.12에서도 동작한다.


★★ Colab 셀 작성 규약 (2026-08-06 확립)
=========================================

**고정 셀 3개 + 실험 셀 N개.** 고정 셀은 건드리지 않고, 실험은 항상 새 셀에 쓴다.

    [셀 1] pip 설치
    [셀 2] 마운트 + 경로 + `reload_wtw()` + `check.version_check()`
    [셀 3] 공통 준비 — import, `ENVS` 캐시, 헬퍼 정의
    [셀 4~] 실험 하나당 한 셀

규칙 1 — **`importlib.reload`를 쓰지 않는다.**
    `reload_wtw()`가 `sys.modules`의 `wtw_nav.*`를 통째로 지우므로 셀 2 하나로
    끝난다. 개별 모듈을 reload 하면 모듈 A가 옛 B를, 모듈 C가 새 B를 참조하는
    상태가 생겨 "고쳤는데 왜 그대로냐"로 몇 시간을 날린다.

규칙 2 — **코드를 고쳤으면 `ENVS.clear()`를 같이 한다.**
    `reload_wtw()`는 클래스를 무효화하지만 **이미 만든 객체는 옛 클래스에 묶여
    있다.** env를 캐시해 두는 이유(MJX 컴파일 수 분)와 정면으로 부딪히므로,
    수정 후에는 반드시 비워야 한다. 표준 순서:
        코드 수정 -> [셀 2] -> `ENVS.clear()` -> [셀 3] -> 실험 셀
    수정과 실험을 번갈아 하면 재컴파일만 반복한다. 수정을 몰아서 할 것.

규칙 3 — **env는 지형 종류별로 하나, 딕셔너리에 캐시한다.**
    `axis_sweep`은 (스윕값 × 시드)회를 도는데 env가 같으면 컴파일이 1회다
    (`envs.scripted._episode_fn` 주석). 셀을 다시 눌러도 재사용되도록
    `ENVS = globals().get("ENVS", {})` 형태로 둔다.

규칙 4 — **셀은 합친다. 한 셀이 하나의 질문에 답해야 한다.**
    "env 만들기" / "돌리기" / "그리기"로 쪼개면 셀 사이 상태 의존이 늘고
    재실행 순서를 틀리기 쉽다. `지형 준비 -> 센서 확인 -> 대조군 -> 스윕 -> 영상`이
    한 셀에 들어가는 편이 낫다.

규칙 5 — **배열을 반환하는 함수를 셀 마지막 줄에 두지 않는다.**
    Colab이 마지막 표현식을 자동 출력하므로 `qpos`(19D × 600스텝)나 궤적 배열이
    화면을 수천 줄 채운다. 실제로 `scripted.video`가 그렇게 터졌다 (2026-08-06).
    진단 함수는 **표를 print 하고 `None`을 반환**하도록 만든다
    (`llc.check.preview`, `envs.scripted.video`의 `return_data` 인자).

규칙 6 — **영상을 같이 낸다.** 숫자만 보고 경사 실패 원인을 세 번 오진했고
    (범위 제약 -> 추진 부족 -> 진입 실패, 전부 틀림), 정작 지형이 화면에 안 보이던
    렌더링 결함은 영상을 찍고 나서야 드러났다. `scripted.video(env, ...)`.

규칙 7 — **`os.environ["MUJOCO_GL"]`은 셀 2에서 한 번만.**
    mujoco가 import 된 뒤에 바꾸면 아무 효과가 없다. 아래 셀에서 반복해 봐야
    "설정했으니 됐겠지"라는 착각만 만든다.
"""

from __future__ import annotations

import sys


def reload_wtw(verbose: bool = True) -> list[str]:
    """`wtw_nav.*` 모듈 캐시를 비운다. **호출 후 반드시 다시 import 할 것.**

        from wtw_nav.dev import reload_wtw
        reload_wtw()
        from wtw_nav.llc import check      # 이제 새 코드
        from wtw_nav import train

    ⚠️ 이미 만들어 둔 객체(예: `env = check.build()`)는 **옛 클래스에 묶여 있다.**
    코드를 고쳤다면 객체도 다시 만들어야 한다.
    """
    names = sorted(n for n in list(sys.modules)
                   if n == "wtw_nav" or n.startswith("wtw_nav."))
    for n in names:
        del sys.modules[n]
    if verbose:
        print(f"[dev] {len(names)}개 모듈 캐시 제거. 이제 다시 import 하십시오.")
        print("      (기존 객체는 옛 코드에 묶여 있으니 다시 생성할 것)")
    return names


def notebook_setup(project_dir: str | None = None, gl: str | None = None) -> None:
    """Colab 첫 셀에서 부를 초기화. **mujoco를 import 하기 전에** 호출해야 한다.

        # [셀 1]
        import os, sys
        os.environ["MUJOCO_GL"] = "egl"          # ← 이 줄이 가장 먼저
        from google.colab import drive; drive.mount('/content/drive')
        sys.path.append("/content/drive/.../2026_wtw")
        import os; os.chdir("/content/drive/.../2026_wtw")

    이 함수는 그 뒤에 남은 것들(GL 확인, 경로 점검)을 해 준다.
    """
    import os
    import platform

    if platform.system() == "Linux":
        if gl is None:
            # ★ GPU가 없으면 egl은 실패한다 — `llc.check.default_gl` 주석 참조.
            #   여기서 mujoco를 import 하지 않도록 같은 판정을 직접 한다.
            import glob
            gl = "egl" if glob.glob("/dev/nvidia*") else "osmesa"
        os.environ.setdefault("MUJOCO_GL", gl)

    if project_dir:
        os.chdir(project_dir)
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)

    print(f"cwd = {os.getcwd()}")
    print(f"MUJOCO_GL = {os.environ.get('MUJOCO_GL')!r}")
    if "mujoco" in sys.modules:
        print("  ⚠️ mujoco가 이미 import 되어 있습니다 — MUJOCO_GL 변경이 반영되지 않습니다.\n"
              "     렌더링이 필요하면 런타임을 재시작하고 이 셀을 가장 먼저 실행하십시오.")
    for p in ("walk-these-ways", "mujoco_menagerie", "wtw_nav"):
        print(f"  {'OK  ' if os.path.isdir(p) else 'MISS'} {p}")
