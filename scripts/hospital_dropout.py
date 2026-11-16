"""병원 하나 제외(사용 불가) 실험 — 생성된 시나리오 폴더를 후처리한다. 시뮬레이터 코드는 건드리지 않는다.

흐름 (저장소 루트 Desktop/MCI_Diffusion 에서 실행. config 경로가 ./MCI_ADV2/... 상대경로라서):
  1) base    원본 시나리오를 <exp>/<point>/base 로 복사 (결과 경로만 바꿈)
  2) (사용자) base 를 --trace 로 시뮬레이션
  3) drops   base trace 에서 실제 환자를 받은 병원을 찾아, 하나씩 뺀 drop_hXX 폴더 생성
  4) (사용자) drop_hXX 전부 시뮬레이션
  5) collect base 대비 지표 변화를 summary.csv 로 정리

이름 규칙 (<exp> = --exp, 예: exp_hosdrop_hos08 = 시나리오가 고른 병원 8곳 그대로):
  scenarios/<exp>/<point>/base/              <point> = 원본 폴더명 (lat.._lon..)
  scenarios/<exp>/<point>/drop_h06/          06 = base hospital_info.csv 의 Index (원래 번호)
  scenarios/<exp>/<point>/manifest.csv       생성/건너뜀 기록
  results/<exp>/<point>/<variant>/results_<variant>_stat.txt
  results/<exp>/<point>/summary.csv

빼는 방식: hospital_info.csv 행 삭제 + Hos2Hos 행·열 삭제 + uav_info 재매핑, 번호는 0부터 다시 매김.
남은 병원으로만 운영한다(다른 병원으로 보충하지 않음). 빼고 나서 상급(종별코드 1) 또는
그 외 병원이 0곳이 되면 시뮬레이션이 깨지므로 만들지 않고 manifest 에 skipped 로 남긴다.
"""
import argparse
import gzip
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SIM = Path("MCI_ADV2")
OUT_NAME = None  # main() 에서 --exp 로 정한다
STAT_ROWS = ["rew", "time", "pdr", "rewWOG", "pdrWOG", "red", "yellow", "green", "black"]
N_RULE_TOKENS = 6  # "START, RedOnly, Red OnlyAMB, Yellow OnlyAMB" 공백 분리 토큰 수


def read_csv(p):
    return pd.read_csv(p, encoding="utf-8-sig")


def write_csv(df, p, index=False):
    df.to_csv(p, index=index, encoding="utf-8-sig")


def point_dirs(point):
    scen = SIM / "scenarios" / OUT_NAME / point
    res = SIM / "results" / OUT_NAME / point
    return scen, res


def rewrite_config(src_dir, dst_dir, variant, res_root):
    """config yaml 의 시나리오 경로·결과 경로를 새 폴더로 바꿔 dst_dir 에 쓴다."""
    src_cfg = next(src_dir.glob("config_*.yaml"))
    text = src_cfg.read_text(encoding="utf-8")
    m = re.search(r'info_path: "(\./MCI_ADV2/scenarios/[^"]+?)/patient_info\.csv"', text)
    if not m:
        raise ValueError(f"patient info_path 를 찾지 못함: {src_cfg}")
    text = text.replace(m.group(1), "./" + dst_dir.as_posix())
    text = re.sub(r'output_path: "[^"]*"', f'output_path: "./{res_root.as_posix()}"', text)
    text = re.sub(r'exp_indicator: "[^"]*"', f'exp_indicator: "{variant}"', text)
    (dst_dir / f"config_{variant}.yaml").write_text(text, encoding="utf-8")


def cmd_base(a):
    src = Path(a.src)
    point = src.name
    scen, res = point_dirs(point)
    dst = scen / "base"
    if dst.exists():
        raise SystemExit(f"이미 있음: {dst}  (다시 만들려면 지우고 실행)")
    shutil.copytree(src, dst)
    for old in dst.glob("config_*.yaml"):
        old.unlink()
    rewrite_config(src, dst, "base", res)
    print(f"base 생성: {dst}")
    print(f"config  : {dst / 'config_base.yaml'}")


def used_hospitals(trace_path):
    """trace 에서 병원별 이송 시작·수용·전원 횟수를 센다 (반복 전체 합)."""
    with gzip.open(trace_path, "rt", encoding="utf-8") as f:
        traces = json.load(f)
    rows = []
    for events in traces.values():
        for ev in events:
            if ev["event"] == "transport_start":
                rows.append((ev["hospital_id"], "sent"))
            elif ev["event"] == "hospital_arrival" and ev.get("admitted"):
                rows.append((ev["hospital_id"], "admitted"))
            elif ev["event"] == "diversion":
                rows.append((ev["from_hospital"], "diverted_out"))
                rows.append((ev["to_hospital"], "diverted_in"))
    df = pd.DataFrame(rows, columns=["idx", "kind"])
    cnt = df.groupby(["idx", "kind"]).size().unstack(fill_value=0)
    for c in ["sent", "admitted", "diverted_out", "diverted_in"]:
        if c not in cnt:
            cnt[c] = 0
    return cnt, len(traces)


def drop_matrix(df, h):
    labels = df.columns[1:]
    keep = [i for i in range(len(labels)) if i != h]
    mat = df.iloc[:, 1:].to_numpy()[np.ix_(keep, keep)]
    out = pd.DataFrame(mat, columns=[str(i) for i in range(len(keep))])
    out.insert(0, "", range(len(keep)))
    return out


def make_drop(base, dst, h, res_root, variant):
    shutil.copytree(base, dst)
    for old in dst.glob("config_*.yaml"):
        old.unlink()

    hos = read_csv(base / "hospital_info.csv")
    hos = hos.drop(index=h).reset_index(drop=True)
    hos["Index"] = range(len(hos))
    write_csv(hos, dst / "hospital_info.csv")

    for name in ["distance_Hos2Hos_euc.csv", "distance_Hos2Hos_road.csv"]:
        write_csv(drop_matrix(read_csv(base / name), h), dst / name)

    uav = read_csv(base / "uav_info.csv")
    if len(uav):
        uav = uav[uav["hospital_idx"] != h].copy()
        uav["hospital_idx"] = uav["hospital_idx"].where(uav["hospital_idx"] < h, uav["hospital_idx"] - 1)
        uav = uav.reset_index(drop=True)
        uav["Index"] = range(len(uav))
    write_csv(uav, dst / "uav_info.csv")

    rewrite_config(base, dst, variant, res_root)


def cmd_drops(a):
    scen, res = point_dirs(a.point)
    base = scen / "base"
    trace = res / "base" / "trace_base.json.gz"
    if not trace.exists():
        raise SystemExit(f"base trace 없음: {trace}\n먼저 base 를 --trace 로 시뮬레이션하세요.")
    cnt, n_iter = used_hospitals(trace)
    hos = read_csv(base / "hospital_info.csv")

    rows = []
    for h, r in hos.iterrows():
        c = cnt.loc[h] if h in cnt.index else pd.Series(0, index=cnt.columns)
        variant = f"drop_h{h:02d}"
        rec = {"variant": variant, "dropped_index": h, "name": r["요양기관명"], "tier_code": r["종별코드"],
               "road_dist_km": r["road_dist"], "sent": int(c["sent"]), "admitted": int(c["admitted"]),
               "diverted_out": int(c["diverted_out"]), "diverted_in": int(c["diverted_in"]),
               "iters": n_iter}
        rest = hos.drop(index=h)
        if c["admitted"] == 0:
            rec["status"] = "skipped_unused"
        elif (rest["종별코드"] == 1).sum() == 0:
            rec["status"] = "skipped_no_tier3_left"
        elif (rest["종별코드"] != 1).sum() == 0:
            rec["status"] = "skipped_no_tier2_left"
        else:
            dst = scen / variant
            if dst.exists():
                shutil.rmtree(dst)
            make_drop(base, dst, h, res, variant)
            rec["status"] = "created"
        rows.append(rec)

    man = pd.DataFrame(rows)
    write_csv(man, scen / "manifest.csv")
    print(man[["variant", "name", "tier_code", "sent", "admitted", "status"]].to_string(index=False))
    print(f"\nmanifest: {scen / 'manifest.csv'}")


def read_iters(res_dir, variant):
    """results_<variant>.txt → {지표: 반복별 값 배열}."""
    p = res_dir / variant / f"results_{variant}.txt"
    out = {}
    for name, line in zip(STAT_ROWS, p.read_text(encoding="utf-8").splitlines()):
        out[name] = np.array(line.split()[N_RULE_TOKENS:], dtype=float)
    return out


def paired(d):
    """반복별 차이의 평균과 95% CI 반폭 (같은 시드라 짝지어 비교)."""
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))


def cmd_collect(a):
    scen, res = point_dirs(a.point)
    man = read_csv(scen / "manifest.csv")
    base = read_iters(res, "base")
    rows = [{"variant": "base", "status": "base",
             **{f"{k}": base[k].mean() for k in ["pdr", "time", "rew"]},
             "pdr_ci": paired(base["pdr"] - base["pdr"].mean())[1]}]
    for _, m in man.iterrows():
        rec = m.to_dict()
        if m["status"] == "created":
            if not (res / m["variant"] / f"results_{m['variant']}.txt").exists():
                rec["status"] = "not_run"
            else:
                v = read_iters(res, m["variant"])
                for k in ["pdr", "time", "rew"]:
                    rec[k] = v[k].mean()
                    rec[f"d_{k}"], rec[f"d_{k}_ci"] = paired(v[k] - base[k])
                rec["pdr_significant"] = abs(rec["d_pdr"]) > rec["d_pdr_ci"]
        rows.append(rec)
    out = pd.DataFrame(rows)
    write_csv(out, res / "summary.csv")
    cols = [c for c in ["variant", "name", "tier_code", "admitted", "status", "pdr", "d_pdr", "d_pdr_ci",
                        "pdr_significant", "d_time"] if c in out]
    print(out[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nsummary: {res / 'summary.csv'}")


SEVERITY = {0: "red", 1: "yellow", 2: "green", 3: "black"}


def cmd_flows(a):
    """variant 별로 환자가 최종 수용된 병원(원래 번호 기준)을 중증도별 반복당 평균 인원으로 정리한다."""
    scen, res = point_dirs(a.point)
    hos = read_csv(scen / "base" / "hospital_info.csv")
    variants = ["base"] + sorted(p.name for p in scen.glob("drop_h*"))
    rows = []
    for v in variants:
        trace = res / v / f"trace_{v}.json.gz"
        if not trace.exists():
            print(f"trace 없음, 건너뜀: {trace}")
            continue
        h = None if v == "base" else int(v[len("drop_h"):])
        with gzip.open(trace, "rt", encoding="utf-8") as f:
            traces = json.load(f)
        for events in traces.values():
            for ev in events:
                if ev["event"] == "hospital_arrival" and ev.get("admitted"):
                    i = ev["hospital_id"]
                    orig = i if h is None or i < h else i + 1  # drop 시나리오는 h 뒤 번호가 하나씩 당겨져 있다
                    rows.append((v, orig, SEVERITY[ev["severity"]], 1 / len(traces)))
    df = pd.DataFrame(rows, columns=["variant", "orig_index", "severity", "n"])
    tab = df.pivot_table(index=["variant", "orig_index"], columns="severity", values="n",
                         aggfunc="sum", fill_value=0)
    tab = tab.reindex(columns=[c for c in SEVERITY.values() if c in tab.columns])
    tab["total"] = tab.sum(axis=1)
    tab = tab.reset_index()
    tab.insert(2, "name", tab["orig_index"].map(hos["요양기관명"]))
    tab.insert(3, "tier_code", tab["orig_index"].map(hos["종별코드"]))
    write_csv(tab, res / "flows.csv")
    for v, g in tab.groupby("variant", sort=False):
        print(f"\n== {v}  (반복당 평균 인원)")
        print(g.drop(columns="variant").to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"\nflows: {res / 'flows.csv'}")


def short_name(n):
    return str(n).split()[-1]


def cmd_plot(a):
    """summary.csv → dpdr.png (병원별 제외 시 pdr 변화), flows.csv → flows_table.png (실험별 수용 인원표)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    scen, res = point_dirs(a.point)
    summ = read_csv(res / "summary.csv")
    base_pdr = summ.loc[summ["variant"] == "base", "pdr"].iloc[0]
    d = summ[summ["status"] == "created"].dropna(subset=["d_pdr"]).sort_values("d_pdr")

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(d) + 1.4))
    ys = np.arange(len(d))
    sig = d["pdr_significant"].astype(bool).to_numpy()
    colors = np.where(sig, "#c0392b", "#9aa0a6")
    ax.axvline(0, color="#555", lw=1)
    ax.hlines(ys, d["d_pdr"] - d["d_pdr_ci"], d["d_pdr"] + d["d_pdr_ci"], colors=colors, lw=2)
    ax.scatter(d["d_pdr"], ys, c=colors, s=40, zorder=3)
    for y, (_, r) in zip(ys, d.iterrows()):
        ax.text(r["d_pdr"] + r["d_pdr_ci"] + 0.004, y, f"{r['d_pdr']:+.3f}", va="center", fontsize=9)
    ax.set_yticks(ys, [f"h{int(r.dropped_index):02d} {short_name(r['name'])}" for _, r in d.iterrows()])
    ax.set_xlabel("Δpdr (제외 - base), 30회 평균 ± 30회 95% CI")
    ax.set_title(f"병원 하나 제외 시 pdr 변화  (기준 pdr {base_pdr:.3f})\n{a.point}", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    xmax = (d["d_pdr"] + d["d_pdr_ci"]).max()
    ax.set_xlim(right=xmax + 0.04)
    fig.tight_layout()
    fig.savefig(res / "dpdr.png", dpi=150)
    plt.close(fig)

    flows = read_csv(res / "flows.csv")
    variants = ["base"] + sorted(v for v in flows["variant"].unique() if v != "base")
    cols = ["red", "yellow", "green", "black", "total"]
    n = len(variants)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.3 * nrow))
    for ax, v in zip(axes.flat, variants):
        g = flows[flows["variant"] == v].sort_values("orig_index")
        cell = [[f"{int(r.orig_index)} {short_name(r['name'])}"] + [f"{r[c]:.1f}" for c in cols] for _, r in g.iterrows()]
        cell.append(["합계"] + [f"{g[c].sum():.1f}" for c in cols])
        ax.axis("off")
        t = ax.table(cellText=cell, colLabels=["병원", "Red", "Yellow", "Green", "Black", "합계"],
                     loc="upper center", cellLoc="right", colLoc="center",
                     colWidths=[0.3] + [0.14] * len(cols))
        t.auto_set_font_size(False)
        t.set_fontsize(9)
        t.scale(1, 1.25)
        for (i, j), c in t.get_celld().items():
            if j == 0:
                c.set_text_props(ha="left")
            if i == 0 or i == len(cell):
                c.set_text_props(weight="bold")
        if v == "base":
            title = "base (제외 없음)"
        else:
            h = int(v[len("drop_h"):])
            title = f"{v}  — {short_name(summ.loc[summ['variant'] == v, 'name'].iloc[0])} 제외"
        ax.set_title(title, fontsize=10, loc="left")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(f"실험별 병원 수용 인원 (30회 반복의 1회당 평균)  {a.point}", fontsize=11)
    fig.tight_layout()
    fig.savefig(res / "flows_table.png", dpi=150)
    plt.close(fig)

    # 꺾은선: flows_table 을 그대로 그림으로. 칸 = 실험, x = 중증도 + 합계, 선 = 병원(색 고정)
    hosp = sorted(flows["orig_index"].unique())
    names = flows.drop_duplicates("orig_index").set_index("orig_index")["name"]
    xt = [f"{h} {short_name(names[h])}" for h in hosp]
    hcolor = {h: f"C{i}" for i, h in enumerate(hosp)}
    xlab = ["Red", "Yellow", "Green", "Black", "합계"]
    ymax = flows["total"].max() * 1.08
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.2 * nrow), sharey=True)
    for ax, v in zip(axes.flat, variants):
        g = flows[flows["variant"] == v].set_index("orig_index")
        for h in hosp:
            if h not in g.index:
                continue
            ax.plot(range(len(cols)), [g.loc[h, c] for c in cols], marker="o", ms=4, lw=1.6,
                    color=hcolor[h], label=f"{h} {short_name(names[h])}")
        ax.set_xticks(range(len(cols)), xlab)
        ax.set_ylim(0, ymax)
        ax.grid(alpha=0.3)
        title = "base (제외 없음)" if v == "base" else f"{v} — {short_name(names[int(v[len('drop_h'):])])} 제외"
        ax.set_title(title, fontsize=10, loc="left")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("수용 인원 (30회 평균)")
    handles = [plt.Line2D([], [], color=hcolor[h], marker="o", lw=1.6) for h in hosp]
    fig.legend(handles, xt, loc="lower center", ncol=len(hosp), fontsize=8, frameon=False)
    fig.suptitle(f"실험별 병원 수용 인원 (30회 반복의 1회당 평균)  {a.point}", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(res / "flows_lines.png", dpi=150)
    plt.close(fig)

    # 히트맵: 행 = 제외 실험, 열 = 병원 + 미도착, 값 = base 대비 수용 인원 변화
    drops = [v for v in variants if v != "base"]
    base = flows[flows["variant"] == "base"].set_index("orig_index")
    blocks = [("total", "합계")]
    fig, axes = plt.subplots(1, len(blocks), figsize=(0.75 * len(hosp) + 4, 0.55 * len(drops) + 2.2), squeeze=False)
    for ax, (c, label) in zip(axes[0], blocks):
        mat = np.full((len(drops), len(hosp) + 1), np.nan)
        for i, v in enumerate(drops):
            g = flows[flows["variant"] == v].set_index("orig_index")[c]
            dropped = int(v[len("drop_h"):])
            for j, h in enumerate(hosp):
                if h != dropped:
                    mat[i, j] = g.get(h, 0.0) - base[c].get(h, 0.0)
            mat[i, -1] = base[c].sum() - g.sum()  # 시뮬레이션 끝까지 병원에 못 간 인원
        lim = np.nanmax(np.abs(mat)) or 1
        ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                txt = "제외" if np.isnan(val) else f"{val:+.1f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if not np.isnan(val) and abs(val) > 0.6 * lim else "black")
        ax.set_xticks(range(len(hosp) + 1), xt + ["미도착"], fontsize=8, rotation=30, ha="right")
        ax.set_yticks(range(len(drops)), [f"{v}\n{short_name(summ.loc[summ['variant'] == v, 'name'].iloc[0])} 제외" for v in drops], fontsize=8)
        ax.set_title(f"{label}: base 대비 수용 인원 변화 (30회 평균)", fontsize=10, loc="left")
    fig.suptitle(f"병원을 빼면 환자가 어디로 옮겨 가나  {a.point}", fontsize=11)
    fig.tight_layout()
    fig.savefig(res / "flows_heatmap.png", dpi=150)
    plt.close(fig)

    for f in ["dpdr.png", "flows_table.png", "flows_lines.png", "flows_heatmap.png"]:
        print(f"plot: {res / f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("base")
    s.add_argument("--src", required=True, help="원본 시나리오 폴더 (MCI_ADV2/scenarios/.../lat.._lon..)")
    s.set_defaults(fn=cmd_base)
    for name, fn in [("drops", cmd_drops), ("collect", cmd_collect), ("flows", cmd_flows), ("plot", cmd_plot)]:
        s = sub.add_parser(name)
        s.add_argument("--point", required=True, help="원본 폴더명 (lat.._lon..)")
        s.set_defaults(fn=fn)
    for s in sub.choices.values():
        s.add_argument("--exp", required=True, help="실험 폴더 이름 (예: exp_hosdrop_hos08)")
    a = p.parse_args()
    global OUT_NAME
    OUT_NAME = a.exp
    a.fn(a)


if __name__ == "__main__":
    main()
