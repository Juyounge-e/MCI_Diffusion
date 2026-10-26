"""Dispatch Light — shared visual theme for every page under ``src/vis_src``.

Single source of truth. ``MCI_Streamlit.py`` and each script in ``pages/`` call
:func:`inject_theme` once, right after ``st.set_page_config``.

Where the theme actually lives
-----------------------------
Most of it is **not** here: it is in ``.streamlit/config.toml``. Streamlit's
advanced theme options (1.44+) reach places CSS cannot — the dataframe canvas,
slider tracks, alert boxes, code blocks, badges, chart colour order and the
sidebar as its own sub-theme. Anything expressible there belongs there, because
config survives Streamlit DOM changes while ``data-testid`` selectors do not.

What stays in this file:

* the webfont import (config can name a family, it cannot fetch one),
* heading decoration (rule under h1, accent tick on h2/h3),
* tabular figures — every number in the app is monospaced and fixed-width so
  columns never shift between reruns,
* the sidebar nav active state,
* small helpers (:func:`page_header`, :func:`kpi_row`, :func:`triage_badges`)
  so pages stop hand-rolling markup,
* plotly / altair templates built from the same palette, registered as the
  library default so the ~20 existing chart call sites need no edits.

Design notes
------------
* Light ground, white panels, deep-navy sidebar. Daylight-readable; the
  situation-room register comes from the sidebar and the accent, not from
  painting everything black.
* Labels and prose are sans. Mono is reserved for figures, IDs and clock
  values — that is the whole "instrument" signal, and it costs nothing legible.
* Blue is the only interaction colour. Red / Yellow / Green / Black belong to
  START triage data (:data:`TRIAGE`) and are never used as decoration.
* Nothing here hides a Streamlit control that does something: main menu,
  sidebar collapse and page nav all stay. Only the deploy button goes.
"""

from __future__ import annotations

import streamlit as st

__all__ = [
    "PALETTE",
    "TRIAGE",
    "CHART_SEQUENCE",
    "inject_theme",
    "page_header",
    "kpi_row",
    "triage_badges",
    "section",
]


#: Console palette. Exported so charts drawn in Python (altair / plotly /
#: folium) match the CSS without hard-coding hexes at each call site.
#: Mirrored by ``.streamlit/config.toml`` — change both together.
PALETTE = {
    "ground": "#F4F7FA",       # app background
    "panel": "#FFFFFF",        # raised surface: cards, metrics, inputs
    "sunken": "#EFF3F7",       # code blocks, wells
    "sidebar": "#12212E",      # deep navy — situation-room register
    "sidebar_alt": "#1B2E3E",  # sidebar widget surfaces
    "sidebar_text": "#DCE7EF",
    "sidebar_muted": "#8798A6",
    "sidebar_accent": "#4D9BF0",  # 어두운 바닥용 밝은 블루 = theme.sidebar.primaryColor
    "rule": "#DCE4EC",         # hairline borders
    "rule_hi": "#C3D0DC",      # borders that need to read as interactive
    "text": "#16232E",         # primary text
    "muted": "#5B6B78",        # labels, captions, axis text
    "accent": "#0B63CE",       # interaction / selection — the only accent
    "accent_soft": "#E8F1FC",  # accent wash for hover / selected rows
    "ok": "#17805A",
    "warn": "#E08A00",
    "bad": "#D93B30",
}

#: START triage colours. Data only — never used for chrome, buttons or accents.
TRIAGE = {
    "red": "#E23B3B",
    "yellow": "#E0A32B",
    "green": "#2E9E52",
    "black": "#5B6B78",
}

#: Categorical series order for every chart library. Same list as
#: ``chartCategoricalColors`` in config.toml.
CHART_SEQUENCE = [
    "#0B63CE", "#17805A", "#E08A00", "#D93B30",
    "#6B4FD8", "#0F8FA8", "#B5651D", "#5B6B78",
]

#: Native badge colour for each triage class. ``st.badge`` picks these up from
#: the semantic colour families set in config.toml.
_TRIAGE_BADGE_COLOR = {"red": "red", "yellow": "yellow", "green": "green", "black": "gray"}

_SANS = (
    "'IBM Plex Sans KR', 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, "
    "'Segoe UI', Roboto, 'Malgun Gothic', sans-serif"
)
_MONO = "'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace"

# NOTE: @import must be the first thing in the stylesheet or the browser drops
# it. Keep it at the top of _CSS.
_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=IBM+Plex+Mono:wght@400;500;600;700&"
    "family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap');"
)

_CSS = f"""<style>
{_FONT_IMPORT}

:root {{
  --dc-ground:   {PALETTE["ground"]};
  --dc-panel:    {PALETTE["panel"]};
  --dc-sunken:   {PALETTE["sunken"]};
  --dc-rule:     {PALETTE["rule"]};
  --dc-rule-hi:  {PALETTE["rule_hi"]};
  --dc-text:     {PALETTE["text"]};
  --dc-muted:    {PALETTE["muted"]};
  --dc-accent:   {PALETTE["accent"]};
  --dc-accent-soft: {PALETTE["accent_soft"]};
  --dc-ok:       {PALETTE["ok"]};
  --dc-warn:     {PALETTE["warn"]};
  --dc-bad:      {PALETTE["bad"]};

  --dc-sidebar:       {PALETTE["sidebar"]};
  --dc-sidebar-alt:   {PALETTE["sidebar_alt"]};
  --dc-sidebar-text:  {PALETTE["sidebar_text"]};
  --dc-sidebar-muted: {PALETTE["sidebar_muted"]};
  --dc-sidebar-accent: {PALETTE["sidebar_accent"]};

  --dc-triage-red:    {TRIAGE["red"]};
  --dc-triage-yellow: {TRIAGE["yellow"]};
  --dc-triage-green:  {TRIAGE["green"]};
  --dc-triage-black:  {TRIAGE["black"]};

  --dc-sans: {_SANS};
  --dc-mono: {_MONO};
}}

/* ══ preserved from the original block ══════════════════════════
   Multiselect must be allowed to fill its column; without this the
   rule-picker chips wrap into an unusable narrow strip.            */
.stMultiSelect [data-baseweb="select"] {{ max-width: 100% !important; }}

/* ══ 1. chrome ══════════════════════════════════════════════════
   Colours come from config.toml. Only the sticky header treatment and
   the vertical rhythm live here. The header keeps its place and height:
   when the sidebar is collapsed its re-open control renders inside it,
   so hiding or shrinking it strands the sidebar shut.               */
header[data-testid="stHeader"] {{
  background: color-mix(in srgb, var(--dc-ground) 82%, transparent) !important;
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--dc-rule);
}}
[data-testid="stAppDeployButton"] {{ display: none !important; }}
footer {{ display: none !important; }}

/* The main menu is kept — it carries Rerun / Clear cache / Settings —
   but toned down so it stops reading as the loudest thing on screen. */
#MainMenu, [data-testid="stMainMenu"] {{ opacity: .5; transition: opacity .15s; }}
#MainMenu:hover, [data-testid="stMainMenu"]:hover {{ opacity: 1; }}

/* The 3.75rem header is sticky with a -40px bottom margin, so it only
   contributes ~20px of flow. Anything under ~2.5rem here puts the page
   title behind the toolbar. */
.block-container, [data-testid="stMainBlockContainer"] {{
  padding-top: 2.9rem !important;
  padding-bottom: 3rem !important;
}}

/* ══ 2. type ════════════════════════════════════════════════════
   Sizes/weights/families are config (headingFontSizes, headingFont).
   Decoration only here. No uppercase, no mono headings: the console
   read comes from the figures, not from shouting the labels.        */
h1 {{
  letter-spacing: -.022em;
  border-bottom: 1px solid var(--dc-rule);
  padding-bottom: .5rem;
  margin-bottom: .15rem !important;
}}
h2, h3 {{
  letter-spacing: -.012em;
  padding-bottom: .3rem;
  margin-bottom: .75rem !important;
  border-bottom: 1px solid var(--dc-rule);
}}
h2::before, h3::before {{
  content: "";
  display: inline-block;
  width: 3px; height: .8em;
  border-radius: 2px;
  background: var(--dc-accent);
  margin-right: .5em;
  vertical-align: -.06em;
}}

/* ══ 3. figures — mono + fixed-width, everywhere a number appears ══ */
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"],
[data-testid="stMarkdownContainer"] table td,
[data-testid="stMarkdownContainer"] table th,
.dc-fig {{
  font-family: var(--dc-mono) !important;
  font-variant-numeric: tabular-nums;
}}
[data-testid="stMetricValue"] {{
  font-weight: 600 !important;
  letter-spacing: -.03em;
}}
[data-testid="stMetricLabel"] p {{
  font-size: .78rem !important;
  color: var(--dc-muted) !important;
  letter-spacing: .01em;
}}
[data-testid="stMetricDelta"] {{ font-size: .76rem !important; }}

/* Markdown tables are plain DOM, so they get the full treatment
   (st.dataframe is a canvas and is themed through config instead). */
[data-testid="stMarkdownContainer"] table {{
  border-collapse: collapse;
  font-size: .82rem;
  width: 100%;
}}
[data-testid="stMarkdownContainer"] th {{
  font-family: var(--dc-sans) !important;
  font-size: .74rem;
  color: var(--dc-muted);
  font-weight: 600;
  border-bottom: 1px solid var(--dc-rule-hi);
  padding: .4rem .65rem;
  text-align: left;
  white-space: nowrap;
}}
[data-testid="stMarkdownContainer"] td {{
  border-top: 1px solid var(--dc-rule);
  padding: .36rem .65rem;
}}
[data-testid="stMarkdownContainer"] tbody tr:hover td {{ background: var(--dc-accent-soft); }}

/* ══ 4. sidebar — deep navy, colours from [theme.sidebar] ════════
   Only the nav needs help: Streamlit's derived hover/active wash is
   tuned for light backgrounds and disappears on the navy.           */
[data-testid="stSidebarNav"] {{
  border-bottom: 1px solid var(--dc-sidebar-alt);
  padding-bottom: .45rem;
  margin-bottom: .35rem;
}}
[data-testid="stSidebarNavLink"] {{
  border-left: 2px solid transparent;
  padding-left: .75rem !important;
}}
[data-testid="stSidebarNavLink"] span {{ font-size: .86rem !important; }}
[data-testid="stSidebarNavLink"]:hover {{
  background: color-mix(in srgb, var(--dc-sidebar-accent) 10%, transparent) !important;
  border-left-color: var(--dc-sidebar-muted);
}}
[data-testid="stSidebarNavLink"][aria-current="page"] {{
  background: color-mix(in srgb, var(--dc-sidebar-accent) 17%, transparent) !important;
  border-left-color: var(--dc-sidebar-accent);
}}
[data-testid="stSidebarNavLink"][aria-current="page"] span {{
  color: color-mix(in srgb, var(--dc-sidebar-accent) 78%, white) !important;
  font-weight: 600;
}}
/* Safety net: captions and widget labels are derived colours, and a
   derived grey that works on white vanishes on the navy. */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label p {{ color: var(--dc-sidebar-muted) !important; }}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
  color: var(--dc-sidebar-text) !important;
  border-bottom-color: var(--dc-sidebar-alt) !important;
}}
[data-testid="stSidebar"] h1::before,
[data-testid="stSidebar"] h2::before,
[data-testid="stSidebar"] h3::before {{ background: var(--dc-sidebar-accent); }}
[data-testid="stSidebar"] hr {{ border-color: var(--dc-sidebar-alt) !important; }}

/* ══ 5. tabs — quiet underline, generous hit area ════════════════ */
.stTabs [data-baseweb="tab-list"] {{
  gap: .15rem;
  border-bottom: 1px solid var(--dc-rule);
}}
.stTabs [data-baseweb="tab"] {{
  padding: .55rem 1rem !important;
  font-weight: 500;
}}
.stTabs [data-baseweb="tab"]:hover {{ background: var(--dc-accent-soft); }}
.stTabs [aria-selected="true"] {{ font-weight: 600; }}

/* ══ 6. console eyebrow / status strip (page_header) ═════════════ */
.dc-sub {{
  font-family: var(--dc-sans);
  font-size: .78rem;
  color: var(--dc-muted);
  font-variant-numeric: tabular-nums;
  margin: .5rem 0 1.15rem;
  display: flex;
  flex-wrap: wrap;
  gap: .25rem .9rem;
  align-items: baseline;
}}
.dc-sub b {{ color: var(--dc-text); font-weight: 600; }}
.dc-sub .dc-k {{
  font-family: var(--dc-sans);
  font-size: .7rem;
  color: var(--dc-muted);
  margin-right: .3rem;
}}
/* Live pip — the one place a dot earns its keep. */
.dc-live {{
  display: inline-flex; align-items: center; gap: .35rem;
  font-size: .72rem; color: var(--dc-ok); font-family: var(--dc-sans);
}}
.dc-live::before {{
  content: ""; width: 6px; height: 6px; border-radius: 50%;
  background: var(--dc-ok);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--dc-ok) 18%, transparent);
}}

/* ══ 7. containers ══════════════════════════════════════════════ */
[data-testid="stExpander"] summary {{ font-weight: 600; font-size: .88rem; }}
[data-testid="stExpander"] summary:hover {{ color: var(--dc-accent); }}
hr {{ border-color: var(--dc-rule) !important; margin: 1.15rem 0 !important; }}

/* Alerts keep their semantic fill (config) and gain a weight bar so the
   severity is readable without reading the text. */
.stAlert, [data-testid="stAlert"] {{ border-left-width: 3px !important; }}

/* ══ 8. embedded frames ═════════════════════════════════════════
   folium / plotly render into iframes; only the shell is ours.     */
iframe[title="streamlit_folium.st_folium"],
[data-testid="stIFrame"] {{
  border: 1px solid var(--dc-rule);
  border-radius: .3rem;
  background: var(--dc-panel);
}}

::-webkit-scrollbar {{ width: 9px; height: 9px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--dc-rule-hi); border-radius: 6px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--dc-muted); }}

/* 지도와 결과가 주인공인 작업 영역: 불필요한 좌우 여백을 줄인다. */
[data-testid="stMainBlockContainer"] {{
  max-width: 1800px;
  padding-left: clamp(1rem, 3vw, 2.5rem) !important;
  padding-right: clamp(1rem, 3vw, 2.5rem) !important;
}}
.st-key-main_view {{
  padding: .35rem 0 .65rem;
  margin-bottom: .3rem;
  border-bottom: 1px solid var(--dc-rule);
}}
.st-key-main_view button {{ min-height: 2.6rem; font-weight: 500; }}
.st-key-main_view [role="radiogroup"] {{ flex-wrap: nowrap; overflow-x: auto; }}
.dc-sub {{ margin-bottom: .4rem; overflow-wrap: anywhere; }}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding-top: 1rem; }}
[data-testid="stMain"] [data-testid="stExpander"] {{ background: var(--dc-panel); }}
button:focus-visible {{ outline: 2px solid var(--dc-accent); outline-offset: 2px; }}
@media (max-width: 1100px) {{
  .st-key-main_view button {{ min-width: 0; padding-left: .4rem; padding-right: .4rem; }}
  .st-key-main_view button [data-testid="stMarkdownContainer"] p {{ font-size: .8rem; }}
  .st-key-main_view span:has(> [data-testid="stIconMaterial"]) {{ display: none; }}
}}
@media (max-width: 700px) {{
  .dc-sub {{ gap: .25rem .6rem; }}
}}

@media (prefers-reduced-motion: reduce) {{
  * {{ transition: none !important; animation: none !important; }}
}}
</style>"""


# ── chart libraries ─────────────────────────────────────────────────
# Registered as each library's default, so the existing ~20 plotly/altair call
# sites inherit the palette without edits.
#
# Note on layering: ``st.plotly_chart`` / ``st.altair_chart`` default to
# ``theme="streamlit"``, which lays Streamlit's own chart theme over the figure.
# That theme reads ``chartCategoricalColors`` from config.toml — the same list as
# :data:`CHART_SEQUENCE` — so the series colours match either way, and these
# templates additionally cover fonts, gridlines and anything rendered outside
# Streamlit (``fig.write_image``, notebooks). Pass ``theme=None`` at a call site
# that must use these templates verbatim.
#
# Both registrations are wrapped: a library version bump must never take the
# dashboard down over styling.

_AXIS = {"gridcolor": PALETTE["rule"], "linecolor": PALETTE["rule_hi"],
         "zerolinecolor": PALETTE["rule_hi"], "tickcolor": PALETTE["rule_hi"]}


def _register_plotly_theme() -> None:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except Exception:
        return
    if "mci" in pio.templates:
        pio.templates.default = "plotly_white+mci"
        return
    pio.templates["mci"] = go.layout.Template(
        layout=go.Layout(
            colorway=CHART_SEQUENCE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=PALETTE["panel"],
            font=dict(family=_SANS, size=12, color=PALETTE["text"]),
            title=dict(font=dict(size=14, color=PALETTE["text"])),
            # Figures are mono so tick columns stay aligned across frames.
            xaxis=dict(tickfont=dict(family=_MONO, size=11, color=PALETTE["muted"]), **_AXIS),
            yaxis=dict(tickfont=dict(family=_MONO, size=11, color=PALETTE["muted"]), **_AXIS),
            legend=dict(font=dict(size=11, color=PALETTE["muted"]),
                        bgcolor="rgba(0,0,0,0)", borderwidth=0),
            hoverlabel=dict(font=dict(family=_MONO, size=11),
                            bgcolor=PALETTE["panel"], bordercolor=PALETTE["rule_hi"]),
            margin=dict(l=56, r=20, t=44, b=44),
            colorscale=dict(sequential=[
                [0.0, "#EAF2FB"], [0.25, "#AFCCEF"], [0.5, "#639BDD"],
                [0.75, "#2266BE"], [1.0, "#073A78"],
            ]),
        )
    )
    pio.templates.default = "plotly_white+mci"


def _register_altair_theme() -> None:
    try:
        import altair as alt
    except Exception:
        return
    cfg = {
        "config": {
            "background": PALETTE["panel"],
            "view": {"stroke": PALETTE["rule"], "continuousWidth": 360, "continuousHeight": 240},
            "font": _SANS,
            "axis": {
                "labelFont": _MONO, "labelFontSize": 11, "labelColor": PALETTE["muted"],
                "titleFont": _SANS, "titleFontSize": 11, "titleColor": PALETTE["muted"],
                "titleFontWeight": 600,
                "gridColor": PALETTE["rule"], "domainColor": PALETTE["rule_hi"],
                "tickColor": PALETTE["rule_hi"],
            },
            "legend": {
                "labelFont": _SANS, "labelFontSize": 11, "labelColor": PALETTE["muted"],
                "titleFont": _SANS, "titleFontSize": 11, "titleColor": PALETTE["muted"],
            },
            "title": {"font": _SANS, "fontSize": 14, "fontWeight": 600,
                      "color": PALETTE["text"], "anchor": "start"},
            "range": {"category": CHART_SEQUENCE,
                      "heatmap": ["#EAF2FB", "#AFCCEF", "#639BDD", "#2266BE", "#073A78"]},
            "mark": {"color": PALETTE["accent"]},
            "point": {"filled": True, "size": 55},
            "line": {"strokeWidth": 2},
            "bar": {"cornerRadiusEnd": 2},
        }
    }
    try:  # altair >= 5.5 / 6.x
        alt.theme.register("mci", enable=True)(lambda: cfg)
        alt.theme.enable("mci")
    except Exception:
        try:  # altair <= 5.4
            alt.themes.register("mci", lambda: cfg)
            alt.themes.enable("mci")
        except Exception:
            pass


_CHARTS_READY = False


def inject_theme() -> None:
    """Apply the Dispatch Light stylesheet and chart templates.

    Call once, right after ``st.set_page_config``, on every page. Safe to call
    more than once per session; the chart templates are registered once.
    """
    global _CHARTS_READY
    st.html(_CSS)
    if not _CHARTS_READY:
        _register_plotly_theme()
        _register_altair_theme()
        _CHARTS_READY = True


def page_header(title: str, subtitle: str | None = None,
                fields: dict[str, str] | None = None, live: str | None = None) -> None:
    """Render the console title block.

    ``fields`` renders as a ``KEY value`` status strip under the rule — pass
    the two or three identifiers that say which run is on screen. ``subtitle``
    is free markup for the same strip (``<b>`` picks a value out) and is kept
    for call sites that build their own line. ``live`` adds a green pip with a
    label, for anything actually running.
    """
    st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)
    parts: list[str] = []
    if fields:
        parts += [f'<span><span class="dc-k">{k}</span><b>{v}</b></span>'
                  for k, v in fields.items()]
    if subtitle:
        parts.append(f"<span>{subtitle}</span>")
    if live:
        parts.append(f'<span class="dc-live">{live}</span>')
    if parts:
        st.markdown(f'<div class="dc-sub">{"".join(parts)}</div>', unsafe_allow_html=True)


def section(title: str, help: str | None = None) -> None:
    """A section rule with optional inline help, so pages stop mixing
    ``st.subheader`` + ``st.caption`` + ``st.markdown("###")`` for the same job."""
    st.subheader(title, help=help)


def kpi_row(items: list[tuple[str, object]] | dict[str, object],
            *, deltas: dict[str, object] | None = None,
            help: dict[str, str] | None = None) -> None:
    """Render a bordered metric strip from ``(label, value)`` pairs.

    Uses ``st.metric(border=True)`` and a horizontal container (Streamlit
    1.46+) instead of the old ``st.columns`` + negative-margin CSS: the cells
    stay equal width, wrap on narrow screens and need no selector hacks.
    """
    pairs = list(items.items()) if isinstance(items, dict) else list(items)
    if not pairs:
        return
    deltas = deltas or {}
    help = help or {}
    with st.container(horizontal=True, gap="small"):
        for label, value in pairs:
            st.metric(label, value, delta=deltas.get(label),
                      help=help.get(label), border=True)


def triage_badges(counts: dict[str, object]) -> None:
    """START triage counts as native badges — ``{"red": 8, "yellow": 14, ...}``.

    Colours come from the semantic families in config.toml, so the badges match
    the alerts and the chart palette instead of drifting from them.
    """
    if not counts:
        return
    with st.container(horizontal=True, gap="small"):
        for key, n in counts.items():
            colour = _TRIAGE_BADGE_COLOR.get(str(key).lower(), "gray")
            st.badge(f"{str(key).upper()} {n}", color=colour)
