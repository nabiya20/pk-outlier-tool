"""
PK Outlier Assessment Tool
--------------------------
A Streamlit app for detecting outliers in time vs. concentration
pharmacokinetic (PK) data using the IQR (Interquartile Range) method.

Run with:
    streamlit run pk_outlier_app.py
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="PK Outlier Assessment Tool", layout="wide")

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def iqr_bounds(series: pd.Series, k: float = 1.5):
    """Return (lower_bound, upper_bound, Q1, Q3, IQR) for a numeric series."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper, q1, q3, iqr


def flag_outliers_per_timepoint(df, time_col, conc_col, k=1.5, min_n=4):
    """
    Apply IQR method WITHIN each time point (recommended for PK data,
    since concentration ranges naturally shift with time).

    min_n: minimum number of samples required at a time point before
           IQR is considered statistically meaningful. Time points with
           fewer samples are still flagged as "insufficient data".
    """
    results = []
    for t, group in df.groupby(time_col):
        n = group[conc_col].notna().sum()
        if n < min_n:
            lower, upper, q1, q3, iqr = (np.nan,) * 5
            note = f"n={n} (< {min_n}), IQR not computed"
        else:
            lower, upper, q1, q3, iqr = iqr_bounds(group[conc_col], k)
            note = f"n={n}"

        g = group.copy()
        g["Q1"] = q1
        g["Q3"] = q3
        g["IQR"] = iqr
        g["Lower_Bound"] = lower
        g["Upper_Bound"] = upper
        if n >= min_n:
            g["Is_Outlier"] = (g[conc_col] < lower) | (g[conc_col] > upper)
        else:
            g["Is_Outlier"] = False
        g["Note"] = note
        results.append(g)
    return pd.concat(results, ignore_index=True)


def flag_outliers_overall(df, conc_col, k=1.5):
    """Apply IQR method across the WHOLE dataset (ignores time grouping)."""
    lower, upper, q1, q3, iqr = iqr_bounds(df[conc_col], k)
    g = df.copy()
    g["Q1"], g["Q3"], g["IQR"] = q1, q3, iqr
    g["Lower_Bound"], g["Upper_Bound"] = lower, upper
    g["Is_Outlier"] = (g[conc_col] < lower) | (g[conc_col] > upper)
    return g


# ----------------------------------------------------------------------
# Sidebar - data input
# ----------------------------------------------------------------------

st.title("🧪 PK Outlier Assessment Tool")
st.caption("IQR-based outlier detection for time vs. concentration profiles")

st.sidebar.header("1. Load your data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

use_demo = st.sidebar.checkbox("Use demo data instead", value=uploaded_file is None)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)
elif use_demo:
    rng = np.random.default_rng(42)
    demo_rows = []
    subjects = [f"S{i:02d}" for i in range(1, 13)]
    timepoints = [0, 0.5, 1, 2, 4, 8, 12, 24]
    for subj in subjects:
        base = rng.uniform(80, 120)
        for t in timepoints:
            conc = base * np.exp(-0.15 * t) + rng.normal(0, 2)
            demo_rows.append({"Subject": subj, "Time": t, "Concentration": max(conc, 0)})
    # inject a couple of obvious outliers
    demo_rows.append({"Subject": "S13_outlier", "Time": 4, "Concentration": 250})
    demo_rows.append({"Subject": "S14_outlier", "Time": 12, "Concentration": 1})
    raw_df = pd.DataFrame(demo_rows)
else:
    st.info("Upload a file or check 'Use demo data' in the sidebar to get started.")
    st.stop()

st.sidebar.header("2. Map your columns")
columns = list(raw_df.columns)
time_col = st.sidebar.selectbox("Time column", columns, index=columns.index("Time") if "Time" in columns else 0)
conc_col = st.sidebar.selectbox(
    "Concentration column", columns, index=columns.index("Concentration") if "Concentration" in columns else 0
)
subj_col_options = ["(none)"] + columns
subj_col = st.sidebar.selectbox(
    "Subject/ID column (optional)",
    subj_col_options,
    index=subj_col_options.index("Subject") if "Subject" in subj_col_options else 0,
)

st.sidebar.header("3. Outlier settings")
method = st.sidebar.radio(
    "Analysis method",
    ["Per time point (recommended)", "Overall (whole dataset)"],
    help=(
        "Per time point: flags a value as an outlier relative to other subjects "
        "AT THE SAME time point. This is usually what you want for PK data, since "
        "concentration naturally rises and falls over time.\n\n"
        "Overall: flags a value as an outlier relative to ALL concentration values "
        "in the dataset, ignoring time. Rarely appropriate for PK profiles."
    ),
)
k_value = st.sidebar.slider(
    "IQR multiplier (k)", min_value=1.0, max_value=3.0, value=1.5, step=0.1,
    help="1.5 = standard Tukey outlier fence. 3.0 = 'extreme' outlier fence (more lenient)."
)
min_n = st.sidebar.number_input(
    "Minimum samples per time point to compute IQR", min_value=3, max_value=20, value=4,
    help="Time points with fewer samples than this won't have IQR computed (too few points for a meaningful quartile)."
)

# ----------------------------------------------------------------------
# Clean data
# ----------------------------------------------------------------------

df = raw_df.copy()
df[conc_col] = pd.to_numeric(df[conc_col], errors="coerce")
df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
n_dropped = df[conc_col].isna().sum() + df[time_col].isna().sum()
df = df.dropna(subset=[conc_col, time_col])

if n_dropped > 0:
    st.warning(f"Dropped {n_dropped} row(s) with non-numeric or missing Time/Concentration values.")

# ----------------------------------------------------------------------
# Run analysis
# ----------------------------------------------------------------------

if method.startswith("Per time point"):
    result = flag_outliers_per_timepoint(df, time_col, conc_col, k=k_value, min_n=min_n)
else:
    result = flag_outliers_overall(df, conc_col, k=k_value)

n_outliers = int(result["Is_Outlier"].sum())
n_total = len(result)

# ----------------------------------------------------------------------
# Display: summary metrics
# ----------------------------------------------------------------------

col1, col2, col3 = st.columns(3)
col1.metric("Total data points", n_total)
col2.metric("Outliers flagged", n_outliers)
col3.metric("Outlier rate", f"{(n_outliers / n_total * 100) if n_total else 0:.1f}%")

# ----------------------------------------------------------------------
# Display: plot
# ----------------------------------------------------------------------

st.subheader("Time vs. Concentration Profile")

fig = go.Figure()

normal = result[~result["Is_Outlier"]]
outliers = result[result["Is_Outlier"]]

hover_cols = [time_col, conc_col] + ([subj_col] if subj_col != "(none)" else [])

def make_hover(rows):
    if subj_col != "(none)":
        return [f"Subject: {s}<br>Time: {t}<br>Conc: {c:.3g}" for s, t, c in zip(rows[subj_col], rows[time_col], rows[conc_col])]
    return [f"Time: {t}<br>Conc: {c:.3g}" for t, c in zip(rows[time_col], rows[conc_col])]

fig.add_trace(go.Scatter(
    x=normal[time_col], y=normal[conc_col], mode="markers",
    marker=dict(color="steelblue", size=8),
    name="Normal",
    text=make_hover(normal), hoverinfo="text",
))

fig.add_trace(go.Scatter(
    x=outliers[time_col], y=outliers[conc_col], mode="markers",
    marker=dict(color="crimson", size=12, symbol="x", line=dict(width=2)),
    name="Outlier",
    text=make_hover(outliers), hoverinfo="text",
))

# overlay median line per time point for reference
median_line = df.groupby(time_col)[conc_col].median().reset_index()
fig.add_trace(go.Scatter(
    x=median_line[time_col], y=median_line[conc_col], mode="lines",
    line=dict(color="gray", dash="dash"), name="Median trend"
))

fig.update_layout(
    xaxis_title="Time", yaxis_title="Concentration",
    height=550, legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# Display: boxplots per time point
# ----------------------------------------------------------------------

if method.startswith("Per time point"):
    st.subheader("Distribution by Time Point")
    fig_box = go.Figure()
    for t, group in df.groupby(time_col):
        fig_box.add_trace(go.Box(y=group[conc_col], name=str(t), boxpoints="all", jitter=0.4))
    fig_box.update_layout(xaxis_title="Time", yaxis_title="Concentration", height=450)
    st.plotly_chart(fig_box, use_container_width=True)

# ----------------------------------------------------------------------
# Display: outlier table
# ----------------------------------------------------------------------

st.subheader("Flagged Outliers")
if n_outliers == 0:
    st.success("No outliers detected with the current settings.")
else:
    display_cols = [c for c in [subj_col, time_col, conc_col, "Q1", "Q3", "Lower_Bound", "Upper_Bound"] if c in result.columns and c != "(none)"]
    st.dataframe(
        outliers[display_cols].sort_values(by=time_col).style.format(precision=3),
        use_container_width=True,
    )

# ----------------------------------------------------------------------
# Display: full results + download
# ----------------------------------------------------------------------

with st.expander("View full results table"):
    st.dataframe(result, use_container_width=True)

csv_buffer = io.StringIO()
result.to_csv(csv_buffer, index=False)
st.download_button(
    "Download full results as CSV",
    data=csv_buffer.getvalue(),
    file_name="pk_outlier_results.csv",
    mime="text/csv",
)
