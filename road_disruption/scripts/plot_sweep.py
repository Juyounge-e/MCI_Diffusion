"""
sweep 결과 CSV → 꺾은선 그래프.

주의: road 환경(matplotlib 3.11.1)에서는 savefig 가 종료코드 127 로 죽는다
(ASCII 제목에서도 재현됨 — freetype/DLL 문제로 보임). MCI 환경으로 돌린다.

    C:/Users/user00/anaconda3/envs/MCI/python.exe ^
        road_disruption/scripts/plot_sweep.py --label artcenter --metric pdr
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roaddis.cli import slug
from roaddis.paths import DATA_DIR, data_subdir

# 지표 이름 → 축 라벨. sim_src/main.py 가 내보내는 5개가 전부다.
LABELS = {
    "pdr":    "PDR (예방가능사망률)",
    "pdrWOG": "PDR (Green 제외)",
    "time":   "총 소요시간 (초)",
    "rew":    "Reward (생존확률 합계)",
    "rewWOG": "Reward (Green 제외)",
}

# 값이 클수록 좋은 지표 — 기준선 초과가 '악화'가 아니라 '개선'이다
HIGHER_IS_BETTER = {"rew", "rewWOG"}

# dataviz 검증기 통과 색 (대비 ≥3:1, 색맹 분리 ΔE 13.0, 일반시야 ΔE 16.3)
C_LINE = "#2a78d6"      # 기준선과 동일한 구간
C_HOT = "#d03b3b"       # 기준선 초과
C_BASE = "#6d7d8a"      # 기준선 참조선
C_INK = "#0f1720"
C_GRID = "#dde4ea"


def use_korean_font():
    """한글 라벨이 □ 로 나오지 않게 폰트를 지정한다 (Windows 기본: Malgun Gothic)."""
    from matplotlib import font_manager, rcParams
    have = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Malgun Gothic", "NanumGothic", "Nanum Gothic",
                 "AppleGothic", "Dotum", "Gulim"):
        if cand in have:
            rcParams["font.family"] = cand
            break
    else:
        print("  [주의] 한글 폰트를 찾지 못했습니다. 라벨이 깨질 수 있습니다.")
    rcParams["axes.unicode_minus"] = False       # 마이너스가 □ 로 나오는 것 방지


def load(label: str, csv_path=None) -> tuple:
    p = Path(csv_path) if csv_path else (
        DATA_DIR / "sweep" / f"{slug(label)}_results.csv")
    if not p.exists():
        sys.exit(f"파일 없음: {p}\n  run_cut_sweep.py 를 먼저 실행하세요.")
    rows = list(csv.DictReader(p.read_text(encoding="utf-8-sig").splitlines()))
    ok = [r for r in rows if r.get("status") == "ok"]
    base = next((r for r in ok if r["way_id"] == "baseline"), None)
    cuts = [r for r in ok if r["way_id"] != "baseline"]
    skipped = len(rows) - len(ok)
    if skipped:
        print(f"  [주의] status 가 ok 가 아닌 행 {skipped}개는 제외했습니다")
    return base, cuts, p


def main():
    ap = argparse.ArgumentParser(description="sweep 결과 → PNG 꺾은선")
    ap.add_argument("--label", required=True, help="run_cut_sweep 의 --label")
    ap.add_argument("--csv", default=None, help="CSV 경로 직접 지정")
    ap.add_argument("--metric", default="pdr", choices=list(LABELS),
                    help="그릴 지표 (기본 pdr)")
    ap.add_argument("--top", type=int, default=3, help="이름을 붙일 상위 개수")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")                        # 화면 없는 환경(SSH)에서 필수
    use_korean_font()
    import matplotlib.pyplot as plt

    base, cuts, src = load(args.label, args.csv)
    if base is None:
        sys.exit("기준선(baseline) 행이 없습니다. --skip-baseline 없이 다시 돌리세요.")

    m = args.metric
    bv = float(base[m])
    xs = np.arange(1, len(cuts) + 1)
    ys = np.array([float(r[m]) for r in cuts])
    names = [r["name"] or "(무명)" for r in cuts]

    # '악화' 방향은 지표마다 다르다. rew/rewWOG 는 클수록 좋으므로
    # 기준선보다 낮은 쪽이 악화다. pdr/time 은 반대.
    worse_is_up = m not in HIGHER_IS_BETTER
    hot = (ys > bv + 1e-12) if worse_is_up else (ys < bv - 1e-12)
    worse_word = "초과" if worse_is_up else "미달"

    print(f"기록: {src.name}")
    print(f"지표 {m}  기준선 {bv:.6f}  최소 {ys.min():.6f}  최대 {ys.max():.6f}")
    print(f"기준선 {worse_word}(악화) {int(hot.sum())}개 / {len(ys)}개")

    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.grid(axis="y", color=C_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)

    # 기준선 — 설명은 왼쪽 끝에 둔다. 오른쪽은 강조점이 몰려 겹친다.
    ax.axhline(bv, color=C_BASE, linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
    ax.annotate(f"기준선 {bv:.4f} (차단 없음)",
                xy=(1, bv), xytext=(4, -16), textcoords="offset points",
                ha="left", va="top", fontsize=9, color=C_BASE)

    # 선 + 점
    ax.plot(xs, ys, color=C_LINE, linewidth=1.8, zorder=3)
    ax.scatter(xs[~hot], ys[~hot], s=22, color=C_LINE,
               edgecolor="white", linewidth=0.8, zorder=4,
               label="기준선과 같거나 나음")
    if hot.any():
        ax.scatter(xs[hot], ys[hot], s=76, color=C_HOT,
                   edgecolor="white", linewidth=1.4, zorder=5,
                   label=f"기준선 {worse_word} (악화)")

    # 상위 N개 직접 라벨 — 위아래 번갈아 놓고, 오른쪽 끝에서는 왼쪽으로 뒤집는다
    order = (np.argsort(-ys) if worse_is_up else np.argsort(ys))[:args.top]
    for k, i in enumerate(order):
        if not hot[i]:
            continue
        dy = 13 if k % 2 == 0 else -19
        right_edge = xs[i] > len(xs) * 0.72
        ax.annotate(f"{names[i]}  {ys[i]:.4f}",
                    xy=(xs[i], ys[i]),
                    xytext=(-8 if right_edge else 8, dy),
                    textcoords="offset points", fontsize=9,
                    ha="right" if right_edge else "left",
                    color=C_HOT, fontweight="bold")

    # 가로축은 CSV 행 순서다. 이어서 돌린(resume) 행은 뒤에 붙으므로
    # 길이순이나 지리적 순서와 일치하지 않는다.
    ax.set_xlabel("삭제한 way(csv 순서)", fontsize=10, color=C_INK)
    ax.set_ylabel(LABELS[m], fontsize=10, color=C_INK)
    ax.set_title(f"도로 하나씩 삭제했을 때의 {LABELS[m]}   ·   way {len(ys)}개",
                 fontsize=12.5, fontweight="bold", color=C_INK, pad=12, loc="left")
    ax.tick_params(colors=C_BASE, labelsize=9)
    ax.set_xlim(0.4, len(xs) + 0.6)

    # 라벨이 붙는 쪽(악화 방향)에 여백을 더 준다
    lo, hi = min(ys.min(), bv), max(ys.max(), bv)
    span = max(hi - lo, abs(bv) * 0.02, 1e-9)
    pad_lo, pad_hi = (0.12, 0.30) if worse_is_up else (0.30, 0.12)
    ax.set_ylim(lo - span * pad_lo, hi + span * pad_hi)

    leg = ax.legend(loc="upper right" if not worse_is_up else "lower right",
                    frameon=True, fontsize=9,
                    framealpha=0.95, edgecolor=C_GRID)
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    out = Path(args.out) if args.out else (
        data_subdir("plots") / f"{slug(args.label)}_{m}.png")
    fig.savefig(out, dpi=args.dpi, facecolor="white")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
