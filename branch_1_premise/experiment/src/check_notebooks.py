"""업로드 전 문법 게이트 (RUNBOOK 함정 15).

R3에서 `print("\n...")`의 이스케이프가 실제 개행으로 납작해져 마지막 셀이
SyntaxError로 죽었다 — GPU 4시간을 다 쓰고 나서. 셀은 실행 직전에 컴파일되므로
문법 오류는 노트북을 올리기 전에 여기서 전부 잡는다.

    python src/check_notebooks.py *.ipynb
"""
import json, sys, glob

MAGIC = ("!", "%")


def check(path):
    nb = json.load(open(path, encoding="utf-8"))
    bad = []
    codes = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for i, cell in enumerate(codes, 1):
        lines = "".join(cell["source"]).split("\n")
        # `!pip` / `%cd` 는 IPython 전용이라 compile()이 못 읽는다 — 가린다
        src = "\n".join("pass" if l.lstrip().startswith(MAGIC) else l for l in lines)
        try:
            compile(src, f"{path}:cell{i}", "exec")
        except SyntaxError as e:
            bad.append(f"  cell {i} line {e.lineno}: {e.msg}")
    return bad


def main(argv):
    paths = []
    for a in argv or ["*.ipynb"]:
        paths.extend(sorted(glob.glob(a)))
    failed = 0
    for p in paths:
        bad = check(p)
        if bad:
            failed += 1
            print(f"FAIL {p}")
            print("\n".join(bad))
        else:
            print(f"OK   {p}")
    if failed:
        print(f"\n{failed}개 노트북에 문법 오류 — 업로드하지 마라")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
