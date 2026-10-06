"""
PK Outlier & Group Comparison Tool
-----------------------------------
A Streamlit app for:
  1) Outlier detection on Group / Subject / Time / Concentration PK data
     using IQR, Z-score, Modified Z-score, or Grubbs' test.
  2) T-test comparison between groups, on either the raw time-concentration
     data (per time point) or on PK parameters (Cmax, AUC, custom partial
     AUCs, etc.), with the ability to include/exclude specific groups/subjects.
  3) Publication-style PK profile plots (linear + semi-log) with custom
     group labels/colors and an optional MEC reference line.

Data entry is done via editable, spreadsheet-like grids: type directly, or
copy cells from Excel and paste them in (click the top-left cell of the grid,
then Ctrl+V / Cmd+V). File upload is also available.

Run with:
    streamlit run pk_outlier_app.py
"""

import io
import itertools
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from scipy import stats
import streamlit as st
import streamlit.components.v1 as components
# Plot presentation helpers are bundled here to avoid stale module imports.
"""Presentation-only helpers. Never changes analytical input or results."""
import math
import colorsys
import numpy as np
import plotly.graph_objects as go

PLOT_WIDTH, PLOT_HEIGHT = 640, 460
FONT_FAMILIES = {
    "맑은고딕": "Malgun Gothic, 맑은 고딕, Arial, sans-serif",
    "Times New Roman": "Times New Roman, Times, serif",
    "Arial": "Arial, Helvetica, sans-serif",
}
GROUP_COLORS = ["#1A1A1A", "#2196D2", "#F07832", "#20B486", "#D98ABB", "#EEAE22", "#66BCE8", "#AA78CD"]


def style_figure(fig, font, x_title, y_title):
    fig.update_layout(
        template="simple_white", width=PLOT_WIDTH, height=PLOT_HEIGHT,
        font=dict(family=font, size=13, color="#1F2937"),
        margin=dict(t=55, r=25, l=75, b=65),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    axis = dict(showline=True, linewidth=1.2, linecolor="black",
                showgrid=False, zeroline=False, ticks="outside", mirror=False,
                title_font=dict(family=font, size=14, color="black"))
    fig.update_xaxes(title_text=x_title, **axis)
    fig.update_yaxes(title_text=y_title, **axis)
    return fig


def profile_ranges(summary, show_sd, log_y, mec=None):
    """Zero-based linear axes; valid positive log limits including SD bounds."""
    means = summary["mean"].to_numpy(dtype=float)
    sd = summary["sd"].fillna(0).to_numpy(dtype=float) if show_sd else np.zeros(len(means))
    xmax = float(summary["Time"].max())
    upper = means + sd
    positive = np.concatenate([means[means > 0], (means - sd)[means - sd > 0]])
    if mec is not None and mec > 0:
        upper = np.append(upper, mec)
        positive = np.append(positive, mec)
    ymax = max(float(np.max(upper)), 0)
    if log_y:
        if not len(positive):
            return [0, max(xmax * 1.04, 1)], None
        low = math.floor(math.log10(float(np.min(positive))))
        high = max(math.ceil(math.log10(max(ymax, float(np.max(positive))))), low + 1)
        yrange = [low, high]
    else:
        yrange = [0, ymax * 1.12 if ymax > 0 else 1]
    return [0, xmax * 1.04 if xmax > 0 else 1], yrange


def distribution_figure(df, groups, times, grouped, colors, font, x_title, y_title):
    """Categorical time positions avoid overlapping boxes at irregular times."""
    fig = go.Figure()
    labels = [f"{float(t):g}" for t in times]
    subset = df[df["Time"].isin(times)]
    traces = [(g, subset[subset["Group"] == g]) for g in groups] if grouped else [("All groups", subset)]
    for i, (group, rows) in enumerate(traces):
        if rows.empty:
            continue
        fig.add_trace(go.Box(
            x=[f"{float(t):g}" for t in rows["Time"]], y=rows["Concentration"],
            name=str(group), offsetgroup=str(group), legendgroup=str(group),
            boxpoints="all", jitter=0.25, pointpos=0, quartilemethod="linear",
            marker=dict(color=colors[i % len(colors)] if grouped else colors[1], size=4, opacity=0.7),
            line=dict(width=1.2),
            customdata=rows[["Group", "Subject", "Time"]].to_numpy(),
            hovertemplate="Group: %{customdata[0]}<br>Subject: %{customdata[1]}<br>Time: %{customdata[2]}<br>Concentration: %{y}<extra>%{fullData.name}</extra>",
        ))
    style_figure(fig, font, x_title, y_title)
    fig.update_layout(boxmode="group", boxgap=0.3, boxgroupgap=0.12, showlegend=grouped)
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=labels,
                     tickangle=-45 if len(labels) > 8 else 0)
    return fig



def profile_style(fig, font, x_title, y_title, title):
    """Shared presentation for mean and individual PK profiles."""
    style_figure(fig, font, x_title, y_title)
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>" if title else "", x=0.5, xanchor="center",
                   font=dict(size=18, family=font, color="#1B4965")),
        legend=dict(orientation="v", bordercolor="lightgray", borderwidth=1,
                    bgcolor="rgba(255,255,255,0.75)", x=0.99, y=0.99,
                    xanchor="right", yanchor="top"),
        margin=dict(t=60, r=25, l=65, b=55),
    )
    fig.update_xaxes(title_text=f"<b>{x_title}</b>")
    fig.update_yaxes(title_text=f"<b>{y_title}</b>")
    return fig


def subject_shades(base_color, subjects):
    """Stable lightness variants of a group's selected mean color."""
    rgb = tuple(int(base_color[i:i+2], 16) / 255 for i in (1,3,5))
    hue, lightness, saturation = colorsys.rgb_to_hls(*rgb)
    if len(subjects) <= 1:
        return {s: base_color for s in subjects}
    low = lightness if saturation < 0.08 else max(0.16, min(lightness * 0.6, 0.45))
    high = min(0.78, max(lightness + 0.23, 0.70))
    shades = {}
    for subject, level in zip(subjects, np.linspace(low, high, len(subjects))):
        color = colorsys.hls_to_rgb(hue, level, saturation)
        shades[subject] = "#" + "".join(f"{round(c*255):02x}" for c in color)
    return shades


def individual_profile_figure(df, group, subjects, font, x_title, y_title, title,
                              base_color=GROUP_COLORS[0], color_subjects=None):
    """One unchanged original series per subject, styled like the mean plot."""
    fig = go.Figure()
    group_data = df[(df["Group"] == group) & df["Subject"].isin(subjects)]
    all_subjects = list(color_subjects) if color_subjects is not None else sorted(df.loc[df["Group"] == group, "Subject"].unique())
    shades = subject_shades(base_color, all_subjects)
    for subject in subjects:
        rows = group_data[group_data["Subject"] == subject].sort_values("Time", kind="stable")
        if rows.empty:
            continue
        subject_index = all_subjects.index(subject)
        fig.add_trace(go.Scatter(
            x=rows["Time"], y=rows["Concentration"], mode="lines+markers", name=str(subject),
            line=dict(color=shades[subject], width=2.5,
                      dash=["solid", "dash", "dot"][subject_index // 8 % 3]),
            marker=dict(size=7, color=shades[subject]),
            hovertemplate=f"{x_title}: %{{x}}<br>{y_title}: %{{y:.4g}}<extra>%{{fullData.name}}</extra>",
        ))
    profile_style(fig, font, x_title, y_title, title)
    xmax = max(float(group_data["Time"].max()), 0) if len(group_data) else 0
    ymax = max(float(group_data["Concentration"].max()), 0) if len(group_data) else 0
    fig.update_xaxes(range=[0, xmax * 1.04 if xmax else 1], autorange=False)
    fig.update_yaxes(range=[0, ymax * 1.12 if ymax else 1], autorange=False)
    return fig


st.set_page_config(page_title="PK Outlier & Comparison Tool", layout="wide")

# ========================================================================
# Restrained, professional styling
# ========================================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
}
h1 { color:#1B4965; font-size:1.8rem; border-bottom:2px solid #D6E4EE; padding-bottom:8px; margin-bottom:4px; }
h2 { color:#1B4965; font-size:1.3rem; border-bottom:1px solid #E3E8EC; padding-bottom:4px; margin-top:30px; }
h3 { color:#2C6E91; font-size:1.05rem; margin-top:14px; }
[data-testid="stMetricValue"] { color:#1B4965; }
[data-testid="stMetricLabel"] { color:#4A5A6A; }
.stTabs [data-baseweb="tab"] { color:#4A5A6A; }
.stTabs [aria-selected="true"] { color:#1B4965 !important; border-bottom:3px solid #1B4965 !important; }
.stButton button, .stDownloadButton button {
    background-color:#1B4965; color:#FFFFFF; border-radius:4px; border:none;
}
.stButton button:hover, .stDownloadButton button:hover { background-color:#143A50; color:#FFFFFF; }
[data-testid="stExpander"] { border:1px solid #E3E8EC; border-radius:6px; }
[data-testid="stCaptionContainer"] { color:#5A6B7B; }
[data-testid="stAlert"] { border-radius:6px; }
</style>
""", unsafe_allow_html=True)

# Red is reserved for flags/significance; group colors are shared by all plots.
FLAG_COLOR = "#C1121F"
FLAG_TINT = "#FBEAEA"

# PROFILE_PALETTE: default colors for the PK Profile Plot tab. These need to
# be clearly distinguishable from each other at a glance (e.g. Reference vs.
# Test groups), so this uses a brighter, distinct sequence. Still fully
# user-adjustable per group via the color pickers.
PROFILE_PALETTE = GROUP_COLORS
PALETTE = GROUP_COLORS

TC_COLUMNS = ["Group", "Subject", "Time", "Concentration"]
PK_DEFAULT_COLUMNS = ["Group", "Subject", "Cmax", "AUC", "AUC_partial"]

# ========================================================================
# Calculation functions — unchanged from the prior version.
# These are the ONLY functions that determine outlier flags / t-test
# results. Nothing below this block affects numerical output.
# ========================================================================

def iqr_flag(sub_vals, k=1.5):
    q1, q3 = sub_vals.quantile(0.25), sub_vals.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    mask = (sub_vals < lower) | (sub_vals > upper)
    detail = f"Q1={q1:.3g}, Q3={q3:.3g}, bounds=[{lower:.3g}, {upper:.3g}]"
    return mask, detail


def zscore_flag(sub_vals, threshold=3.0):
    mean, sd = sub_vals.mean(), sub_vals.std(ddof=1)
    if not sd or pd.isna(sd):
        z = pd.Series(0.0, index=sub_vals.index)
    else:
        z = (sub_vals - mean) / sd
    mask = z.abs() > threshold
    detail = f"mean={mean:.3g}, sd={sd:.3g}"
    return mask, detail, z


def modified_zscore_flag(sub_vals, threshold=3.5):
    median = sub_vals.median()
    mad = (sub_vals - median).abs().median()
    if not mad or pd.isna(mad):
        modz = pd.Series(0.0, index=sub_vals.index)
    else:
        modz = 0.6745 * (sub_vals - median) / mad
    mask = modz.abs() > threshold
    detail = f"median={median:.3g}, MAD={mad:.3g}"
    return mask, detail, modz


def grubbs_flag(sub_vals, alpha=0.05, iterative=True):
    """Iterative two-sided Grubbs' test. Returns a boolean mask aligned to sub_vals.index."""
    s = sub_vals.dropna()
    idx_list = list(s.index)
    vals = s.values.astype(float)
    flagged = []
    while True:
        n = len(vals)
        if n < 3:
            break
        mean, sd = vals.mean(), vals.std(ddof=1)
        if sd == 0:
            break
        abs_dev = np.abs(vals - mean)
        i_max = int(np.argmax(abs_dev))
        G = abs_dev[i_max] / sd
        t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit ** 2 / (n - 2 + t_crit ** 2))
        if G > G_crit:
            flagged.append(idx_list[i_max])
            vals = np.delete(vals, i_max)
            idx_list.pop(i_max)
            if not iterative:
                break
        else:
            break
    mask = sub_vals.index.isin(flagged)
    detail = f"n={len(s)}, alpha={alpha}"
    return pd.Series(mask, index=sub_vals.index), detail


MIN_N = {"IQR": 4, "Z-score": 3, "Modified Z-score": 3, "Grubbs' test": 3}


def run_outlier_detection(df, group_cols, value_col, method, params):
    out_rows = []
    grouping = df.groupby(group_cols) if group_cols else [(None, df)]
    for _, sub in grouping:
        vals = sub[value_col]
        n = vals.notna().sum()
        g = sub.copy()

        if n < MIN_N[method]:
            g["Is_Outlier"] = False
            g["Score"] = np.nan
            g["Detail"] = f"n={n} (< {MIN_N[method]} required), skipped"
        elif method == "IQR":
            mask, detail = iqr_flag(vals, k=params["k"])
            g["Is_Outlier"], g["Score"], g["Detail"] = mask, np.nan, detail
        elif method == "Z-score":
            mask, detail, z = zscore_flag(vals, threshold=params["threshold"])
            g["Is_Outlier"], g["Score"], g["Detail"] = mask, z, detail
        elif method == "Modified Z-score":
            mask, detail, modz = modified_zscore_flag(vals, threshold=params["threshold"])
            g["Is_Outlier"], g["Score"], g["Detail"] = mask, modz, detail
        elif method == "Grubbs' test":
            mask, detail = grubbs_flag(vals, alpha=params["alpha"], iterative=params["iterative"])
            g["Is_Outlier"], g["Score"], g["Detail"] = mask, np.nan, detail

        g["N_in_subset"] = n
        out_rows.append(g)
    return pd.concat(out_rows, ignore_index=True)


def linear_trapz_auc(time_vals, conc_vals):
    """Manual linear trapezoidal AUC (avoids relying on np.trapz/np.trapezoid,
    whose availability differs across numpy versions)."""
    t = np.asarray(time_vals, dtype=float)
    c = np.asarray(conc_vals, dtype=float)
    if len(t) < 2:
        return np.nan
    return float(np.sum((t[1:] - t[:-1]) * (c[1:] + c[:-1]) / 2.0))


def two_sample_ttest(a, b, equal_var=False):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return {"n1": len(a), "n2": len(b), "mean1": a.mean() if len(a) else np.nan,
                "mean2": b.mean() if len(b) else np.nan, "t_stat": np.nan, "p_value": np.nan,
                "significant": None}
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=equal_var)
    return {"n1": len(a), "n2": len(b), "mean1": a.mean(), "mean2": b.mean(),
            "t_stat": t_stat, "p_value": p_val, "significant": p_val < 0.05}

# ========================================================================
# Demo / template data (used only when the user explicitly requests it)
# ========================================================================

def empty_tc_df():
    return pd.DataFrame(columns=TC_COLUMNS)


def empty_pk_df():
    return pd.DataFrame(columns=PK_DEFAULT_COLUMNS)


def demo_tc_df():
    rng = np.random.default_rng(42)
    groups = {"Reference": 90, "Test1": 100, "Test2": 110, "Test3": 95}
    timepoints = [0, 0.5, 1, 2, 4, 8, 12, 24]
    rows = []
    for grp, base_dose in groups.items():
        for i in range(1, 5):
            subj = f"{grp}_S{i:02d}"
            base = rng.uniform(0.85, 1.15) * base_dose
            for t in timepoints:
                if t == 0:
                    conc = 0.0  # pre-dose baseline sample
                else:
                    conc = max(base * np.exp(-0.15 * t) + rng.normal(0, 2), 0)
                rows.append({"Group": grp, "Subject": subj, "Time": t, "Concentration": round(conc, 2)})

    # Inject one realistic outlier: pick a random existing (non-zero) sample
    # and replace its value, rather than adding a fabricated extra subject.
    demo_df = pd.DataFrame(rows)
    non_baseline_idx = demo_df.index[demo_df["Time"] != 0]
    outlier_idx = rng.choice(non_baseline_idx)
    demo_df.loc[outlier_idx, "Concentration"] = round(demo_df.loc[outlier_idx, "Concentration"] * 6 + 50, 2)
    return demo_df


def demo_pk_df():
    return pd.DataFrame([
        {"Group": "Reference", "Subject": "Reference_S01", "Cmax": 88.2, "AUC": 610.4, "AUC_partial": 210.1},
        {"Group": "Reference", "Subject": "Reference_S02", "Cmax": 91.5, "AUC": 630.2, "AUC_partial": 215.6},
        {"Group": "Test1", "Subject": "Test1_S01", "Cmax": 99.1, "AUC": 700.8, "AUC_partial": 240.3},
        {"Group": "Test1", "Subject": "Test1_S02", "Cmax": 102.4, "AUC": 715.0, "AUC_partial": 245.9},
    ])


def example_tc_snippet():
    return pd.DataFrame({
        "Group": ["Reference", "Reference", "Test1"],
        "Subject": ["Ref_01", "Ref_01", "Test1_01"],
        "Time": [0, 1, 0],
        "Concentration": [0, 45.2, 0],
    })


# ========================================================================
# Session state
# ========================================================================

if "tc_data" not in st.session_state:
    st.session_state.tc_data = empty_tc_df()
if "pk_data" not in st.session_state:
    st.session_state.pk_data = empty_pk_df()
for flag in ("confirm_clear_tc", "confirm_clear_pk", "pending_upload_tc", "pending_upload_pk"):
    if flag not in st.session_state:
        st.session_state[flag] = False

group_col, subj_col, time_col, conc_col = "Group", "Subject", "Time", "Concentration"

# ========================================================================
# Header
# ========================================================================

st.title("PK Outlier & Group Comparison Tool")
st.caption("Outlier detection, group comparisons, and profile plots for PK study data. · Interface v2026.10.06.4")

# ========================================================================
# Helper: handle a pending file upload with explicit confirmation
# ========================================================================

def handle_upload(uploader_key, session_key, pending_key, columns_hint):
    """Stores an uploaded file as 'pending' and only applies it once the
    user confirms (if the table already has data) so edits are never lost
    silently."""
    up = st.file_uploader(
        f"Upload CSV/Excel to replace this table (columns: {', '.join(columns_hint)})",
        type=["csv", "xlsx", "xls"], key=uploader_key,
    )
    if up is not None:
        sig = (up.name, up.size)
        if st.session_state.get(f"{uploader_key}_sig") != sig:
            st.session_state[f"{uploader_key}_bytes"] = up.getvalue()
            st.session_state[f"{uploader_key}_name"] = up.name
            st.session_state[f"{uploader_key}_sig"] = sig
            st.session_state[pending_key] = True

    if st.session_state.get(pending_key):
        current_n = len(st.session_state[session_key])

        def apply_upload():
            name = st.session_state[f"{uploader_key}_name"]
            raw = st.session_state[f"{uploader_key}_bytes"]
            new_df = pd.read_csv(io.BytesIO(raw)) if name.endswith(".csv") else pd.read_excel(io.BytesIO(raw))
            st.session_state[session_key] = new_df
            st.session_state[pending_key] = False

        if current_n == 0:
            apply_upload()
            st.rerun()
        else:
            st.warning(f"Uploading will replace the {current_n} row(s) currently in this table. This cannot be undone.")
            uc1, uc2 = st.columns(2)
            if uc1.button("Replace data with uploaded file", key=f"{uploader_key}_confirm"):
                apply_upload()
                st.rerun()
            if uc2.button("Cancel upload", key=f"{uploader_key}_cancel"):
                st.session_state[pending_key] = False
                st.rerun()

# ========================================================================
# STAGE 1: Data
# ========================================================================

def render_copyable_plot(fig, height):
    """Renders with the installed Plotly JavaScript bundle and an added
    'Copy image to clipboard' button, so the plot can be pasted into slides
    or reports without a separate download step. Falls back gracefully if
    the browser doesn't support the Clipboard image API."""
    fig_json = fig.to_json().replace("<", "\\u003c")
    plotly_js = get_plotlyjs()
    html = f"""
    <div style="max-width:{PLOT_WIDTH}px;width:100%;" id="plotdiv_{id(fig)}"></div>
    <div style="margin-top:6px;">
      <button id="copybtn_{id(fig)}" style="padding:6px 14px;background:#1B4965;color:#fff;
        border:none;border-radius:4px;cursor:pointer;font-family:Arial,sans-serif;font-size:13px;">
        Copy image to clipboard
      </button>
      <span id="copystatus_{id(fig)}" style="margin-left:10px;font-family:Arial,sans-serif;
        font-size:13px;color:#5A6B7B;"></span>
    </div>
    <script>{plotly_js}</script>
    <script>
    (function() {{
        var figData = {fig_json};
        var gd = document.getElementById('plotdiv_{id(fig)}');
        var ready = Plotly.newPlot(gd, figData.data, figData.layout, {{displayModeBar: true, responsive: true, toImageButtonOptions: {{format: 'png', width: figData.layout.width, height: figData.layout.height, scale: 2}}}});
        document.getElementById('copybtn_{id(fig)}').addEventListener('click', function() {{
            var statusEl = document.getElementById('copystatus_{id(fig)}');
            ready.then(function() {{ return Plotly.toImage(gd, {{format: 'png', width: figData.layout.width, height: figData.layout.height, scale: 2}}); }})
                .then(function(url) {{
                    return fetch(url).then(function(res) {{ return res.blob(); }});
                }})
                .then(function(blob) {{
                    if (navigator.clipboard && window.ClipboardItem) {{
                        navigator.clipboard.write([new ClipboardItem({{'image/png': blob}})]).then(function() {{
                            statusEl.innerText = 'Copied!';
                            setTimeout(function() {{ statusEl.innerText = ''; }}, 2000);
                        }}).catch(function() {{
                            statusEl.innerText = 'Copy not supported in this browser — use the camera icon above to download instead.';
                        }});
                    }} else {{
                        statusEl.innerText = 'Copy not supported in this browser — use the camera icon above to download instead.';
                    }}
                }}).catch(function() {{
                    statusEl.innerText = "Could not copy image — use the camera icon to download a PNG.";
                }});
        }});
    }})();
    </script>
    """
    if hasattr(st, "iframe"):
        st.iframe(html, height=height + 70)
    else:
        components.html(html, height=height + 70, scrolling=False)


st.header("Data")
entry_tab_tc, entry_tab_pk = st.tabs(["Time–Concentration Data", "PK Parameter Data"])

with entry_tab_tc:
    u1, u2 = st.columns(2)
    with u1:
        time_unit = st.text_input("Time unit", value="hours", key="tc_time_unit").strip() or "hours"
    with u2:
        conc_unit = st.text_input("Concentration unit", value="ng/mL", key="tc_conc_unit").strip() or "ng/mL"
    st.caption("Units label your existing values; editing a unit does not convert the data.")
    handle_upload("up_tc", "tc_data", "pending_upload_tc", TC_COLUMNS)

    with st.expander("Need help? Data format & instructions", expanded=False):
        st.markdown("Required columns: **Group**, **Subject**, **Time**, **Concentration**. Example:")
        st.dataframe(example_tc_snippet(), width="stretch", hide_index=True)
        st.markdown(
            "- Click the top-left cell of the editable grid below and paste (Ctrl+V / Cmd+V) a block copied from Excel.\n"
            "- Add or remove rows directly in the grid (use the `+`/row menu).\n"
            "- Or upload a CSV/Excel file above — you'll be asked to confirm before it replaces this table."
        )
        if st.button("Load example dataset", key="load_demo_tc"):
            st.session_state.tc_data = demo_tc_df()
            st.rerun()

    n_tc = len(st.session_state.tc_data)
    if n_tc == 0:
        st.info("No data yet. Paste or type your Time–Concentration data below, or upload a file above.")
    else:
        n_groups_tc = st.session_state.tc_data[group_col].nunique() if group_col in st.session_state.tc_data.columns else 0
        n_subj_tc = st.session_state.tc_data[subj_col].nunique() if subj_col in st.session_state.tc_data.columns else 0
        st.caption(f"{n_tc} row(s) · {n_groups_tc} group(s) · {n_subj_tc} subject(s) — preview below, expand 'Edit data' to see or change all rows.")
        st.dataframe(st.session_state.tc_data.head(5), width="stretch", hide_index=True, height=200,
                     column_config={"Time": f"Time ({time_unit})", "Concentration": f"Concentration ({conc_unit})"})

    with st.expander(f"Edit data ({n_tc} rows)", expanded=(n_tc == 0)):
        st.session_state.tc_data = st.data_editor(
            st.session_state.tc_data,
            num_rows="dynamic",
            width="stretch",
            key="tc_editor",
            column_config={
                "Group": st.column_config.TextColumn(required=True),
                "Subject": st.column_config.TextColumn(required=True),
                "Time": st.column_config.NumberColumn(label=f"Time ({time_unit})", required=True),
                "Concentration": st.column_config.NumberColumn(label=f"Concentration ({conc_unit})", required=True),
            },
        )
        cc1, _ = st.columns([1, 5])
        with cc1:
            if st.button("Clear data", key="clear_tc_btn"):
                st.session_state.confirm_clear_tc = True
        if st.session_state.confirm_clear_tc:
            st.warning("This will remove all rows from the Time–Concentration table. This cannot be undone.")
            yc1, yc2 = st.columns(2)
            if yc1.button("Yes, clear table", key="yes_clear_tc"):
                st.session_state.tc_data = empty_tc_df()
                st.session_state.confirm_clear_tc = False
                st.rerun()
            if yc2.button("Cancel", key="cancel_clear_tc"):
                st.session_state.confirm_clear_tc = False
                st.rerun()

with entry_tab_pk:
    with st.expander("Parameter units", expanded=False):
        st.caption("Set a label for each parameter. Values are not converted. Use matching units when comparing groups.")
        pk_units = {}
        for column in st.session_state.pk_data.columns:
            if column in ("Group", "Subject"):
                continue
            normalized = column.lower()
            default_unit = (f"{time_unit}·{conc_unit}" if normalized.startswith("auc") else
                            time_unit if normalized in ("tmax", "t_half", "t1/2") else
                            conc_unit if normalized.startswith("cmax") else "")
            pk_units[column] = st.text_input(f"{column} unit", value=default_unit, key=f"pk_unit_{column}").strip()
    handle_upload("up_pk", "pk_data", "pending_upload_pk", PK_DEFAULT_COLUMNS)

    with st.expander("Need help? Data format & instructions", expanded=False):
        st.markdown(
            "Default columns: **Group**, **Subject**, **Cmax**, **AUC**, **AUC_partial**. "
            "Add your own parameter columns below (e.g. AUC_0-12, Tmax, t1/2, CL/F) — the sheet is fully adjustable."
        )
        if st.button("Load example dataset", key="load_demo_pk"):
            st.session_state.pk_data = demo_pk_df()
            st.rerun()

    n_pk = len(st.session_state.pk_data)
    if n_pk == 0:
        st.info("No PK parameter data yet. This sheet is optional unless you want to run group comparisons on PK parameters.")
    else:
        n_groups_pk = st.session_state.pk_data[group_col].nunique() if group_col in st.session_state.pk_data.columns else 0
        st.caption(f"{n_pk} row(s) · {n_groups_pk} group(s) — preview below, expand 'Edit data' to see or change all rows.")
        st.dataframe(st.session_state.pk_data.head(5), width="stretch", hide_index=True, height=200,
                     column_config={c: f"{c} ({u})" for c, u in pk_units.items() if u})

    with st.expander(f"Edit data ({n_pk} rows)", expanded=(n_pk == 0)):
        col_a, col_b = st.columns([3, 1])
        with col_a:
            new_col_name = st.text_input("New parameter column name (e.g. 'AUC_0-12', 'Tmax', 't_half')", key="new_pk_col")
        with col_b:
            st.write("")
            st.write("")
            if st.button("Add column", key="add_pk_col_btn") and new_col_name.strip():
                if new_col_name.strip() not in st.session_state.pk_data.columns:
                    st.session_state.pk_data[new_col_name.strip()] = np.nan
                st.rerun()

        numeric_cols_pk = [c for c in st.session_state.pk_data.columns if c not in ("Group", "Subject")]
        cols_to_drop = st.multiselect("Remove parameter column(s)", numeric_cols_pk, key="pk_cols_to_drop")
        if cols_to_drop and st.button("Remove selected column(s)", key="remove_pk_cols_btn"):
            st.session_state.pk_data = st.session_state.pk_data.drop(columns=cols_to_drop)
            st.rerun()

        column_config_pk = {
            "Group": st.column_config.TextColumn(required=True),
            "Subject": st.column_config.TextColumn(required=True),
        }
        for c in st.session_state.pk_data.columns:
            if c not in ("Group", "Subject"):
                column_config_pk[c] = st.column_config.NumberColumn(label=f"{c} ({pk_units[c]})" if pk_units.get(c) else c)

        st.session_state.pk_data = st.data_editor(
            st.session_state.pk_data,
            num_rows="dynamic",
            width="stretch",
            key="pk_editor",
            column_config=column_config_pk,
        )
        pc1, _ = st.columns([1, 5])
        with pc1:
            if st.button("Clear data", key="clear_pk_btn"):
                st.session_state.confirm_clear_pk = True
        if st.session_state.confirm_clear_pk:
            st.warning("This will remove all rows from the PK Parameter table. This cannot be undone.")
            ypc1, ypc2 = st.columns(2)
            if ypc1.button("Yes, clear table", key="yes_clear_pk"):
                st.session_state.pk_data = empty_pk_df()
                st.session_state.confirm_clear_pk = False
                st.rerun()
            if ypc2.button("Cancel", key="cancel_clear_pk"):
                st.session_state.confirm_clear_pk = False
                st.rerun()

# ========================================================================
# Clean time-concentration data for analysis
# ========================================================================

with st.expander("Plot appearance", expanded=False):
    font_choice = st.selectbox("Plot font", list(FONT_FAMILIES), key="plot_font")
    st.caption("The selected font is used in all plots and exported images. If unavailable on your computer, the browser uses a fallback.")
plot_font = FONT_FAMILIES[font_choice]

df_raw = st.session_state.tc_data.copy()
required_tc_cols = {group_col, subj_col, time_col, conc_col}
if not required_tc_cols.issubset(df_raw.columns):
    df = pd.DataFrame(columns=TC_COLUMNS)
else:
    df = df_raw.copy()
    df[conc_col] = pd.to_numeric(df[conc_col], errors="coerce")
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[group_col, subj_col, time_col, conc_col])

all_groups = sorted(df[group_col].dropna().unique().tolist()) if not df.empty else []

# ========================================================================
# STAGE 2 & 3: Analysis & Results
# ========================================================================

if df.empty:
    st.header("Analysis")
    st.info("Add Time–Concentration data above (paste, type, or upload) to run outlier detection, group comparisons, and profile plots.")
else:
    st.header("Analysis")
    tab_outlier, tab_ttest, tab_profile = st.tabs(["Outlier Detection", "Group Comparison", "Profile Plot"])

    # --------------------------------------------------------------
    # TAB 1: Outlier detection
    # --------------------------------------------------------------
    with tab_outlier:
        with st.container(border=True):
            st.markdown("**Settings**")
            s1, s2, s3 = st.columns([1.1, 1.5, 1.2])
            with s1:
                method = st.selectbox(
                    "Method", ["IQR", "Z-score", "Modified Z-score", "Grubbs' test"], key="method_outlier",
                    help="Statistical method used to flag unusual concentration values.",
                )
            with s2:
                grouping_choice = st.radio(
                    "Comparison scope",
                    ["Group + Time point (recommended)", "Time point only", "Overall (whole dataset)"],
                    key="scope_outlier",
                    help=(
                        "Group + Time point: compares each subject to others in the SAME group AND SAME time point. "
                        "Best when groups are expected to have different concentration levels.\n\n"
                        "Time point only: compares across all groups at the same time point.\n\n"
                        "Overall: ignores both group and time — rarely appropriate for PK profiles."
                    ),
                )
            with s3:
                params = {}
                if method == "IQR":
                    params["k"] = st.slider("IQR multiplier (k)", 1.0, 3.0, 1.5, 0.1, key="k_outlier",
                                             help="Standard Tukey fence = 1.5")
                elif method == "Z-score":
                    params["threshold"] = st.slider("Z-score threshold", 1.5, 5.0, 3.0, 0.1, key="z_outlier")
                elif method == "Modified Z-score":
                    params["threshold"] = st.slider("Modified Z-score threshold", 1.5, 5.0, 3.5, 0.1, key="modz_outlier")
                elif method == "Grubbs' test":
                    params["alpha"] = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01, key="alpha_outlier")
                    params["iterative"] = st.checkbox("Remove outliers iteratively", value=True, key="iter_outlier")

        with st.expander("How does this method flag outliers?", expanded=False):
            method_help = {
                "IQR": "Flags values much higher or lower than the typical middle range of the comparison group. It is less influenced by extreme values than methods based on the mean.",
                "Z-score": "Flags values far from the group average relative to its usual variation. A higher threshold means a value must be further from the average to be flagged.",
                "Modified Z-score": "Flags values far from the group median (middle value), using a measure of typical variation that is less affected by extreme values.",
                "Grubbs' test": "Checks whether the most extreme value is unusually far from the group average for roughly normal data. Iterative mode repeats the check after each flag.",
            }
            st.markdown(f"**{method}:** {method_help[method]}")
            st.caption("Flags highlight unusual observations; they do not remove your data.")

        if grouping_choice.startswith("Group + Time"):
            group_cols = [group_col, time_col]
            scope_label = "Group + Time point"
        elif grouping_choice.startswith("Time point"):
            group_cols = [time_col]
            scope_label = "Time point only"
        else:
            group_cols = []
            scope_label = "Overall"

        result = run_outlier_detection(df, group_cols, conc_col, method, params)
        n_outliers = int(result["Is_Outlier"].sum())
        n_total = len(result)

        if method == "IQR":
            param_label = f"k={params['k']}"
        elif method == "Grubbs' test":
            param_label = f"alpha={params['alpha']}"
        else:
            param_label = f"threshold={params['threshold']}"
        st.caption(f"Showing results for **{method}** ({param_label}) · Comparison scope: **{scope_label}** · Time: {time_unit} · Concentration: {conc_unit}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Data points analyzed", n_total)
        c2.metric("Flagged as outlier", n_outliers)
        c3.metric("Flagged rate", f"{(n_outliers / n_total * 100) if n_total else 0:.1f}% of {n_total}")

        def comparison_group_label(row):
            if group_cols == [group_col, time_col]:
                return f"{row[group_col]} @ t={row[time_col]} {time_unit}"
            elif group_cols == [time_col]:
                return f"All groups @ t={row[time_col]} {time_unit}"
            return "All data"

        st.subheader("Flagged observations")
        outliers = result[result["Is_Outlier"]].copy()
        if n_outliers == 0:
            st.success("No observations flagged as outliers with the current method and settings.")
        else:
            outliers["Comparison group"] = outliers.apply(comparison_group_label, axis=1)
            f1, f2, f3 = st.columns(3)
            with f1:
                filt_groups = st.multiselect("Filter by group", sorted(outliers[group_col].unique()),
                                              default=sorted(outliers[group_col].unique()), key="flag_filter_group")
            with f2:
                filt_subjects = st.multiselect("Filter by subject", sorted(outliers[subj_col].unique()),
                                                default=sorted(outliers[subj_col].unique()), key="flag_filter_subj")
            with f3:
                filt_times = st.multiselect("Filter by time", sorted(outliers[time_col].unique()),
                                             default=sorted(outliers[time_col].unique()), key="flag_filter_time")
            flagged_view = outliers[
                outliers[group_col].isin(filt_groups)
                & outliers[subj_col].isin(filt_subjects)
                & outliers[time_col].isin(filt_times)
            ]
            show_cols = [group_col, subj_col, time_col, conc_col, "Comparison group", "Score", "Detail", "N_in_subset"]
            st.dataframe(
                flagged_view[show_cols].sort_values(by=[group_col, time_col]).rename(columns={"N_in_subset": "n in comparison group"}),
                width="stretch", hide_index=True,
            )
            dl1, _ = st.columns([1, 4])
            with dl1:
                st.download_button(
                    "Download flagged observations (CSV)",
                    data=flagged_view[show_cols].to_csv(index=False),
                    file_name="flagged_outliers.csv", mime="text/csv", key="download_flagged",
                )

        with st.expander("Full results & statistical details (all data points)"):
            result_display = result.copy()
            result_display["Flagged as outlier"] = result_display.pop("Is_Outlier").map({True: "Yes", False: "No"})
            st.dataframe(result_display, width="stretch", hide_index=True)
            st.download_button(
                "Download full results (CSV)",
                data=result_display.to_csv(index=False),
                file_name="pk_outlier_results.csv", mime="text/csv", key="download_outlier_full",
            )

        st.subheader("Visualization")
        viz_tab1, viz_tab2 = st.tabs(["Time vs. Concentration", "Distribution (box plot)"])

        with viz_tab1:
            fig = go.Figure()
            normal = result[~result["Is_Outlier"]]
            flagged_pts = result[result["Is_Outlier"]]
            color_map = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(all_groups)}

            for g in all_groups:
                sub_n = normal[normal[group_col] == g]
                fig.add_trace(go.Scatter(
                    x=sub_n[time_col], y=sub_n[conc_col], mode="markers",
                    marker=dict(color=color_map.get(g, "gray"), size=7, opacity=0.65),
                    name=f"{g}",
                    text=[f"Group: {g}<br>Subject: {s}<br>Time: {t}<br>Conc: {c:.3g}<br>Flagged: No"
                          for s, t, c in zip(sub_n[subj_col], sub_n[time_col], sub_n[conc_col])],
                    hoverinfo="text",
                ))

            fig.add_trace(go.Scatter(
                x=flagged_pts[time_col], y=flagged_pts[conc_col], mode="markers",
                marker=dict(color=FLAG_COLOR, size=13, symbol="x", line=dict(width=2)),
                name="Flagged as outlier",
                text=[f"Group: {g}<br>Subject: {s}<br>Time: {t}<br>Conc: {c:.3g}<br>Flagged: Yes"
                      for g, s, t, c in zip(flagged_pts[group_col], flagged_pts[subj_col], flagged_pts[time_col], flagged_pts[conc_col])],
                hoverinfo="text",
            ))
            style_figure(fig, plot_font, f"Time ({time_unit})", f"Concentration ({conc_unit})")
            fig.update_xaxes(rangemode="tozero")
            fig.update_yaxes(rangemode="tozero")
            render_copyable_plot(fig, PLOT_HEIGHT)

        with viz_tab2:
            box_group_by = st.radio("Group by", ["Time only", "Group + Time"], horizontal=True, key="box_group_by")
            available_times = sorted(df[time_col].unique().tolist())
            box_times = st.multiselect("Time points to display", available_times, default=available_times,
                                      key="box_times", help="Select fewer time points if the plot is crowded; analytical results are unchanged.")
            if box_times:
                fig_box = distribution_figure(df, all_groups, sorted(box_times), box_group_by == "Group + Time",
                                              PALETTE, plot_font, f"Time ({time_unit})", f"Concentration ({conc_unit})")
                render_copyable_plot(fig_box, PLOT_HEIGHT)
                st.caption("Each box shows Q1, median and Q3. Whiskers extend to the most extreme values within 1.5×IQR; all observations are overlaid. Time categories are evenly spaced, not a continuous time axis. Box whiskers are independent of the selected outlier method.")
                if len(box_times) * (len(all_groups) if box_group_by == "Group + Time" else 1) > 30:
                    st.info("Many boxes are displayed. Select fewer time points above to inspect the distributions more clearly.")
            else:
                st.info("Select at least one time point to display distributions.")

    # --------------------------------------------------------------
    # TAB 2: Group Comparison (t-test)
    # --------------------------------------------------------------
    with tab_ttest:
        with st.container(border=True):
            st.markdown("**Settings**")
            g1c, g2c, g3c = st.columns([1.6, 1.6, 1.2])
            with g1c:
                selected_groups = st.multiselect("Groups to include", all_groups, default=all_groups, key="ttest_groups")
            with g2c:
                pk_all = st.session_state.pk_data.copy()
                available_subjects_all = sorted(set(df[subj_col].unique().tolist()) | set(
                    pk_all[subj_col].unique().tolist() if subj_col in pk_all.columns else []
                ))
                excluded_subjects = st.multiselect(
                    "Exclude subjects", available_subjects_all, default=[], key="ttest_exclude",
                    help="E.g. subjects flagged as outliers in the Outlier Detection tab.",
                )
            with g3c:
                equal_var = st.checkbox("Assume equal variances", value=False, key="ttest_equal_var",
                                         help="Unchecked = Welch's t-test (does not assume equal variances) — the safer default.")
            comparison_type = st.radio(
                "Dataset to compare",
                ["Time–Concentration (per time point)", "PK Parameters"],
                key="ttest_dataset", horizontal=True,
            )

        if len(selected_groups) < 2:
            st.info("Select at least 2 groups above to run a comparison.")
        else:
            tc_subset = df[df[group_col].isin(selected_groups)]
            pk_subset_groups = pk_all[pk_all[group_col].isin(selected_groups)] if group_col in pk_all.columns else pk_all.iloc[0:0]

            tc_subset = tc_subset[~tc_subset[subj_col].isin(excluded_subjects)]
            pk_subset = pk_subset_groups[~pk_subset_groups[subj_col].isin(excluded_subjects)] if not pk_subset_groups.empty else pk_subset_groups

            if len(selected_groups) == 2:
                pairs = [tuple(selected_groups)]
            else:
                pairs = list(itertools.combinations(selected_groups, 2))
                st.caption(f"More than 2 groups selected — running all {len(pairs)} pairwise comparisons.")

            def style_significant(row):
                if row.get("significant") == "Yes (p<0.05)":
                    return [f"background-color:{FLAG_TINT}"] * len(row)
                return [""] * len(row)

            if comparison_type.startswith("Time"):
                st.caption(f"Comparing: **Time–Concentration data** · Groups: {', '.join(selected_groups)} · Excluded subjects: {len(excluded_subjects)}")
                st.subheader("Results: t-test at each time point")
                for ga, gb in pairs:
                    st.markdown(f"**{ga} vs {gb}**")
                    rows = []
                    for t in sorted(tc_subset[time_col].unique()):
                        a = tc_subset[(tc_subset[group_col] == ga) & (tc_subset[time_col] == t)][conc_col]
                        b = tc_subset[(tc_subset[group_col] == gb) & (tc_subset[time_col] == t)][conc_col]
                        res = two_sample_ttest(a, b, equal_var=equal_var)
                        rows.append({"Time": t, **res})
                    res_df = pd.DataFrame(rows)
                    res_df["significant"] = res_df["significant"].map({True: "Yes (p<0.05)", False: "No", None: "n<2, skipped"})
                    styled = res_df.style.format(
                        {"mean1": "{:.3g}", "mean2": "{:.3g}", "t_stat": "{:.3g}", "p_value": "{:.4f}"}
                    ).apply(style_significant, axis=1)
                    st.dataframe(styled, width="stretch", hide_index=True)
            else:
                if pk_subset.empty:
                    st.warning("No matching rows found in the PK Parameter sheet for the selected groups/subjects. "
                               "Fill in the PK Parameter sheet in the Data section above first.")
                else:
                    st.caption(f"Comparing: **PK Parameter data** · Groups: {', '.join(selected_groups)} · Excluded subjects: {len(excluded_subjects)}")
                    param_cols = [c for c in pk_subset.columns if c not in (group_col, subj_col)]
                    chosen_params = st.multiselect("Parameters to test", param_cols, default=param_cols, key="ttest_params")
                    with st.expander(f"PK parameter data used ({len(pk_subset)} rows)"):
                        st.dataframe(pk_subset, width="stretch", hide_index=True)

                    st.subheader("Results: t-test on PK parameters")
                    for ga, gb in pairs:
                        st.markdown(f"**{ga} vs {gb}**")
                        rows = []
                        for param in chosen_params:
                            a = pd.to_numeric(pk_subset[pk_subset[group_col] == ga][param], errors="coerce")
                            b = pd.to_numeric(pk_subset[pk_subset[group_col] == gb][param], errors="coerce")
                            res = two_sample_ttest(a, b, equal_var=equal_var)
                            rows.append({"Parameter": param, "Unit": pk_units.get(param, ""), **res})
                        res_df = pd.DataFrame(rows)
                        res_df["significant"] = res_df["significant"].map({True: "Yes (p<0.05)", False: "No", None: "n<2, skipped"})
                        styled = res_df.style.format(
                            {"mean1": "{:.3g}", "mean2": "{:.3g}", "t_stat": "{:.3g}", "p_value": "{:.4f}"}
                        ).apply(style_significant, axis=1)
                        st.dataframe(styled, width="stretch", hide_index=True)

    # --------------------------------------------------------------
    # TAB 3: PK Profile Plot
    # --------------------------------------------------------------
    with tab_profile:
        with st.container(border=True):
            st.markdown("**Groups & subjects**")
            p1, p2 = st.columns(2)
            with p1:
                profile_groups = st.multiselect("Groups to include", all_groups, default=all_groups, key="profile_groups")
            with p2:
                profile_subjects_all = sorted(df[df[group_col].isin(profile_groups)][subj_col].unique().tolist()) if profile_groups else []
                excluded_profile_subjects = st.multiselect(
                    "Subjects to exclude from this plot only", profile_subjects_all, default=[], key="profile_exclude_subj"
                )

            profile_flags = result[result["Is_Outlier"]].copy()
            st.caption(f"Outlier reference: {method} ({param_label}) · {scope_label}. Flags use all analysis data, before plot exclusions.")
            with st.expander(f"Flagged groups & subjects ({len(profile_flags)} observations)", expanded=False):
                if profile_flags.empty:
                    st.success("No observations flagged with the current outlier settings.")
                else:
                    flag_rows = []
                    for (flag_group, flag_subject), flagged_samples in profile_flags.groupby([group_col, subj_col]):
                        if flag_group not in profile_groups:
                            plot_status = "Group not included"
                        elif flag_subject in excluded_profile_subjects:
                            plot_status = "Subject excluded"
                        else:
                            plot_status = "Included in plot"
                        flag_rows.append({"Group": flag_group, "Subject": flag_subject,
                                          "Flagged points": len(flagged_samples),
                                          f"Flagged times ({time_unit})": ", ".join(f"{t:g}" for t in sorted(flagged_samples[time_col].unique())),
                                          "Plot status": plot_status})
                    st.dataframe(pd.DataFrame(flag_rows), hide_index=True, width="stretch", height=180)
                    st.caption("A flagged point does not mean the whole subject is invalid. Excluding a subject removes that subject's entire curve from the plots only.")

        if not profile_groups:
            st.info("Select at least one group above to build a plot.")
        else:
            profile_df = df[df[group_col].isin(profile_groups)]
            profile_df = profile_df[~profile_df[subj_col].isin(excluded_profile_subjects)]

            if profile_df.empty:
                st.warning("No data left after filtering — adjust the group/subject selections above.")
            else:
                with st.container(border=True):
                    st.markdown("**Labels & colors**")
                    st.caption("Default colors are chosen to be clearly distinguishable between groups. Adjust any of them below if needed.")
                    default_palette = PROFILE_PALETTE
                    rename_map, color_map_profile = {}, {}
                    label_cols = st.columns(min(len(profile_groups), 4)) if profile_groups else []
                    for i, g in enumerate(profile_groups):
                        with label_cols[i % len(label_cols)]:
                            with st.expander(f"'{g}'", expanded=False):
                                rename_map[g] = st.text_input("Display label", value=g, key=f"label_{g}")
                                color_map_profile[g] = st.color_picker("Color", value=default_palette[i % len(default_palette)], key=f"color_{g}")
                        rename_map.setdefault(g, g)
                        color_map_profile.setdefault(g, default_palette[i % len(default_palette)])

                with st.container(border=True):
                    st.markdown("**Titles & axes**")
                    for title_key, default_title in (("profile_xtitle", f"Time ({time_unit})"),
                                                     ("profile_ytitle", f"{conc_col} ({conc_unit})")):
                        previous = st.session_state.get(title_key + "_unit_default")
                        if title_key not in st.session_state or st.session_state[title_key] == previous:
                            st.session_state[title_key] = default_title
                        st.session_state[title_key + "_unit_default"] = default_title
                    tc1, tc2, tc3 = st.columns(3)
                    with tc1:
                        plot_title = st.text_input("Plot title (shown on the plot)", value="PK Profile", key="profile_title")
                    with tc2:
                        x_axis_title = st.text_input("X-axis title", key="profile_xtitle",
                                                      help="Edit the unit to match your data, e.g. 'Time (day)'.")
                    with tc3:
                        y_axis_title = st.text_input("Y-axis title", key="profile_ytitle")

                    ac1, ac2, ac3 = st.columns(3)
                    with ac1:
                        x_ticks_input = st.text_input("X-axis tick values (comma-separated, optional)", value="",
                                                       placeholder="e.g. 0,7,14,21,28,35,42", key="profile_xticks")
                    with ac2:
                        y_interval_linear = st.number_input("Y-axis interval — linear plot only (0 = auto)", min_value=0.0, value=0.0, key="profile_yint")
                    with ac3:
                        error_metric = st.radio("Error bars", ["SD", "None"], horizontal=True, key="profile_error")

                with st.container(border=True):
                    st.markdown("**MEC line (optional)**")
                    include_mec = st.checkbox("Include a minimum effective concentration (MEC) reference line", key="profile_mec_toggle")
                    mec_value, mec_label = None, "MEC"
                    if include_mec:
                        st.caption("Enter your own MEC value below — it is not calculated from the data.")
                        mv1, mv2 = st.columns(2)
                        with mv1:
                            mec_value = st.number_input(f"MEC value ({conc_unit})", min_value=0.0, value=0.0, step=1.0, key="profile_mec_value")
                        with mv2:
                            mec_label = st.text_input("MEC line label", value="MEC", key="profile_mec_label")
                        if mec_value == 0.0:
                            st.info("Enter your MEC value above to draw the reference line.")

                summary = (
                    profile_df.groupby([group_col, time_col])[conc_col]
                    .agg(mean="mean", sd="std", n="count")
                    .reset_index()
                )

                x_ticks = None
                if x_ticks_input.strip():
                    try:
                        x_ticks = [float(x.strip()) for x in x_ticks_input.split(",") if x.strip() != ""]
                    except ValueError:
                        st.warning("Couldn't parse X-axis tick values — using automatic ticks instead.")
                        x_ticks = None

                def build_profile_figure(log_y: bool):
                    xrange, yrange = profile_ranges(summary, error_metric == "SD", log_y,
                                                    mec_value if include_mec else None)
                    fig = go.Figure()
                    for g in profile_groups:
                        g_data = summary[summary[group_col] == g].sort_values(time_col)
                        if log_y:
                            g_data = g_data.copy()
                            g_data.loc[g_data["mean"] <= 0, "mean"] = np.nan
                        err = g_data["sd"].fillna(0) if error_metric == "SD" else None
                        error_config = None
                        if err is not None:
                            error_config = dict(type="data", array=err, visible=True, thickness=1, width=3)
                            if log_y and yrange is not None:
                                error_config.update(symmetric=False,
                                    arrayminus=np.minimum(err, np.maximum(g_data["mean"] - 10 ** yrange[0], 0)))
                        fig.add_trace(go.Scatter(
                            x=g_data[time_col], y=g_data["mean"],
                            mode="lines+markers",
                            name=rename_map.get(g, g),
                            line=dict(color=color_map_profile.get(g), width=2.5),
                            marker=dict(size=7, color=color_map_profile.get(g)),
                            error_y=error_config,
                            customdata=g_data[["sd", "n"]].to_numpy(),
                            hovertemplate=f"Time ({time_unit}): %{{x}}<br>Mean ({conc_unit}): %{{y:.4g}}<br>SD: %{{customdata[0]:.4g}}<br>n: %{{customdata[1]}}<extra>%{{fullData.name}}</extra>",
                        ))

                    if include_mec and mec_value:
                        fig.add_hline(y=mec_value, line_dash="dash", line_color="black",
                                      annotation_text=mec_label, annotation_position="top left")

                    profile_style(fig, plot_font, x_axis_title, y_axis_title, plot_title)
                    axis_title_font = dict(size=14, color="#000000", family=plot_font)
                    fig.update_xaxes(
                        title=dict(text=f"<b>{x_axis_title}</b>", font=axis_title_font),
                        showline=True, linewidth=1.2, linecolor="black", mirror=False,
                        showgrid=False, zeroline=False,
                    )
                    fig.update_xaxes(range=xrange, autorange=False)
                    if x_ticks:
                        fig.update_xaxes(tickmode="array", tickvals=x_ticks)
                    if log_y:
                        fig.update_yaxes(
                            title=dict(text=f"<b>{y_axis_title}</b>", font=axis_title_font),
                            type="log", range=yrange, autorange=False, dtick=1, exponentformat="none",
                            showline=True, linewidth=1.2, linecolor="black", mirror=False,
                            showgrid=False, zeroline=False,
                            minor=dict(showgrid=False, ticks=""),
                        )
                    else:
                        fig.update_yaxes(
                            range=yrange, autorange=False,
                            title=dict(text=f"<b>{y_axis_title}</b>", font=axis_title_font),
                            showline=True, linewidth=1.2, linecolor="black", mirror=False,
                            showgrid=False, zeroline=False,
                        )
                        if y_interval_linear > 0:
                            fig.update_yaxes(dtick=y_interval_linear)
                    return fig

                st.subheader("Plot")
                plot_tab_linear, plot_tab_log, plot_tab_individual = st.tabs(["Mean — linear", "Mean — semi-log", "Individual profiles"])
                with plot_tab_linear:
                    render_copyable_plot(build_profile_figure(log_y=False), PLOT_HEIGHT)
                with plot_tab_log:
                    if (summary["mean"] > 0).any():
                        render_copyable_plot(build_profile_figure(log_y=True), PLOT_HEIGHT)
                        st.caption("True semi-log: zero cannot be shown on Y. Nonpositive means are omitted; lower SD bars are clipped at the positive display minimum.")
                    else:
                        st.info("The true semi-log plot needs at least one positive mean concentration.")

                with plot_tab_individual:
                    indiv_group_options = [g for g in profile_groups if (profile_df[group_col] == g).any()]
                    individual_group = st.selectbox("Group for individual profiles", indiv_group_options,
                        format_func=lambda g: str(rename_map.get(g, g)), key="individual_group")
                    subject_options = sorted(profile_df.loc[profile_df[group_col] == individual_group, subj_col].unique().tolist())
                    individual_subjects = st.multiselect("Subjects to display", subject_options, default=subject_options,
                        key=f"individual_subjects_{individual_group}")
                    individual_title = st.text_input("Individual plot title", value="PK Profile", key="individual_title",
                        help="Edit the complete title, or leave it blank to hide the title. No group or individual suffix is added.")
                    if individual_subjects:
                        individual_fig = individual_profile_figure(profile_df, individual_group, individual_subjects,
                            plot_font, x_axis_title, y_axis_title,
                            individual_title, base_color=color_map_profile[individual_group],
                            color_subjects=sorted(df.loc[df[group_col] == individual_group, subj_col].unique().tolist()))
                        if x_ticks:
                            individual_fig.update_xaxes(tickmode="array", tickvals=x_ticks)
                        if y_interval_linear > 0:
                            individual_fig.update_yaxes(dtick=y_interval_linear)
                        render_copyable_plot(individual_fig, PLOT_HEIGHT)
                        st.caption("One line per subject in the selected group; shades follow the group's mean color. Original observations, no averaging or SD bars. Click a legend entry to hide/show a subject.")
                    else:
                        st.info("Select at least one subject to display individual profiles.")

                st.caption(
                    "Click \"Copy image to clipboard\" to paste the plot directly into slides or documents, "
                    "or use the camera icon in the plot's toolbar to download it as a PNG."
                )
