# -*- coding: utf-8 -*-
"""National version of viz_daejeon_4x4.py for any generated model."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
ASPECT = 1.3


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def parse_interval(value):
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    if len(numbers) < 2:
        raise ValueError(f"Cannot parse pdr_q interval: {value}")
    return float(numbers[0]), float(numbers[1])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--folder_template", default="cond_national_N{N}_pdr{p}")
    parser.add_argument("--csv_name", default="samples.csv")
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_list", nargs="+", type=int, default=[15, 25, 45])
    parser.add_argument(
        "--pdr_list", nargs="+", type=float, default=[0.01, 0.08, 0.13]
    )
    parser.add_argument("--shp", default="MCI_ADV2/scenarios/ctprvn.shp")
    parser.add_argument(
        "--summary", default="notebooks/outputs/analysis/national_all_summary.csv"
    )
    parser.add_argument("--point_size", type=float, default=5.0)
    parser.add_argument("--export_scale", type=float, default=3.0)
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = resolve(args.input_root)
    output = resolve(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    # q-bin handling is identical to viz_daejeon_4x4.py.
    summary = pd.read_csv(resolve(args.summary))
    intervals = summary["pdr_q"].apply(parse_interval)
    bins = np.unique(
        np.asarray([intervals.iloc[0][0]] + [item[1] for item in intervals], dtype=float)
    )
    bins.sort()
    bins[0], bins[-1] = -np.inf, np.inf
    labels = summary["pdr_q"].astype(str).tolist()
    colorbar_labels = [
        f"Q{i + 1} [{low:.3f}, {high:.3f}]"
        for i, (low, high) in enumerate(intervals)
    ]
    nq = len(labels)

    # All national administrative boundaries, including Jeju.
    boundary = (
        gpd.read_file(resolve(args.shp), encoding="cp949")
        .set_crs(epsg=5179, allow_override=True)
        .to_crs(epsg=4326)
    )
    if "CTP_ENG_NM" in boundary.columns:
        boundary = boundary[
            boundary["CTP_ENG_NM"].astype(str).str.lower() != "jeju-do"
        ].copy()
    elif "CTP_KOR_NM" in boundary.columns:
        boundary = boundary[
            ~boundary["CTP_KOR_NM"].astype(str).str.contains("제주", na=False)
        ].copy()
    boundary["geometry"] = boundary.geometry.simplify(0.002)
    bx, by = [], []
    for geometry in boundary.geometry:
        polygons = geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry]
        for polygon in polygons:
            x, y = polygon.exterior.xy
            bx.extend(list(x) + [None])
            by.extend(list(y) + [None])

    # Same mainland-focused view used by viz_roadloss_national.py.
    view = dict(x=[124.5, 130.5], y=[33.8, 38.7])

    n_rows, n_cols = len(args.n_list), len(args.pdr_list)
    titles = [f"N={n}, PDR={p:g}" for n in args.n_list for p in args.pdr_list]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=0.015,
        vertical_spacing=0.025,
    )

    shown_colorbar = False
    missing = []
    for row_index, n_value in enumerate(args.n_list):
        for col_index, p_value in enumerate(args.pdr_list):
            row, col = row_index + 1, col_index + 1
            subplot_index = row_index * n_cols + col_index + 1
            xaxis = "x" if subplot_index == 1 else f"x{subplot_index}"
            yaxis = "y" if subplot_index == 1 else f"y{subplot_index}"

            fig.add_trace(
                go.Scatter(
                    x=bx,
                    y=by,
                    mode="lines",
                    xaxis=xaxis,
                    yaxis=yaxis,
                    line=dict(color="black", width=1.5),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

            folder = args.folder_template.format(N=n_value, p=f"{p_value:g}")
            csv_path = input_root / folder / args.csv_name
            if csv_path.exists():
                frame = pd.read_csv(csv_path)
                required = {"lat", "lon", "pdr"}
                absent = required.difference(frame.columns)
                if absent:
                    raise ValueError(f"{csv_path}: missing columns {sorted(absent)}")
                q_index = pd.cut(
                    frame["pdr"], bins=bins, labels=False, include_lowest=True
                ).astype(float)
                unique_coordinates = frame[["lat", "lon"]].drop_duplicates().shape[0]
                point_size = args.point_size
                fig.add_trace(
                    go.Scatter(
                        x=frame["lon"],
                        y=frame["lat"],
                        mode="markers",
                        xaxis=xaxis,
                        yaxis=yaxis,
                        marker=dict(
                            size=point_size,
                            color=q_index,
                            colorscale="RdBu_r",
                            cmin=0,
                            cmax=nq - 1,
                            opacity=0.85,
                            line=dict(width=0.5, color="rgba(70,70,70,0.6)"),
                            showscale=not shown_colorbar,
                            colorbar=dict(
                                title=dict(
                                    text="PDR quantile bin",
                                    font=dict(
                                        family="Times New Roman",
                                        size=15,
                                        color="black",
                                    ),
                                ),
                                tickmode="array",
                                tickvals=list(range(nq)),
                                ticktext=colorbar_labels,
                                tickfont=dict(
                                    family="Times New Roman",
                                    size=13,
                                    color="black",
                                ),
                                len=0.85,
                                x=1.01,
                            ),
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )
                shown_colorbar = True
                q_values = sorted(q_index.dropna().astype(int).unique().tolist())
                if len(q_values) == 1:
                    q_text = f"Q{q_values[0] + 1}"
                else:
                    q_text = ",".join(f"Q{value + 1}" for value in q_values)
                fig.layout.annotations[subplot_index - 1].text = (
                    f"n={n_value} | pdr={p_value:g} | {q_text}"
                )
                print(
                    f"[color-check] n={n_value} pdr={p_value:g} "
                    f"q-bin={q_text} n={len(frame)} unique={unique_coordinates}"
                )
            else:
                missing.append(str(csv_path))
                fig.layout.annotations[subplot_index - 1].text = (
                    f"N={n_value}, pdr={p_value:g} (missing)"
                )

            fig.update_xaxes(
                range=view["x"], row=row, col=col, showticklabels=False
            )
            fig.update_yaxes(
                range=view["y"],
                scaleanchor=xaxis,
                scaleratio=ASPECT,
                row=row,
                col=col,
                showticklabels=False,
            )

    for annotation in fig.layout.annotations:
        annotation.font = dict(
            family="Times New Roman",
            size=25,
            color="black",
        )

    fig.update_layout(
        width=370 * n_cols + 170,
        height=390 * n_rows + 40,
        template="plotly_white",
        font=dict(
            family="Times New Roman",
            size=14,
            color="black",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=8, r=175, t=42, b=8),
    )
    fig.write_image(output, scale=args.export_scale)
    print(f"[saved] {output}")
    if missing:
        print(f"[warning] missing {len(missing)} CSV files")
        for path in missing:
            print(f"  {path}")


if __name__ == "__main__":
    main()
