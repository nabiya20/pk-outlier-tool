"""
PK Outlier & Group Comparison Tool
-----------------------------------
A Streamlit app for:
  1) Outlier detection on Group / Subject / Time / Concentration PK data
     using IQR, Z-score, Modified Z-score, or Grubbs' test.
  2) T-test comparison between groups, on either the raw time-concentration
     data (per time point) or on derived PK parameters (Cmax, Tmax, AUClast),
     with the ability to include/exclude specific groups or subjects.

Run with:
    streamlit run pk_outlier_app.py
"""

import io
import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

st.set_page_config(page_title="PK Outlier & Comparison Tool", layout="wide")

# ========================================================================
# Outlier detection helpers
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
    """
    group_cols: list of columns defining the comparison subset
                (e.g. [] for overall, [time_col], or [group_col, time_col]).
    Returns df with added columns: Is_Outlier, Score, Detail, N_in_subset
    """
    out_rows = []
    grouping = df.groupby(group_cols) if group_cols else [(None, df)]
    for _, sub in grouping:
        vals = sub[value_col]
        n = vals.notna().sum()
        g = sub.copy()
        score = pd.Series(np.nan, index=sub.index)

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


# ========================================================================
# PK parameter helpers (for the PK-parameter t-test)
# ========================================================================

def linear_trapz_auc(time_vals, conc_vals):
    """Manual linear trapezoidal AUC (avoids relying on np.trapz/np.trapezoid,
    whose availability differs across numpy versions)."""
    t = np.asarray(time_vals, dtype=float)
    c = np.asarray(conc_vals, dtype=float)
    if len(t) < 2:
        return np.nan
    return float(np.sum((t[1:] - t[:-1]) * (c[1:] + c[:-1]) / 2.0))


def compute_pk_params(df, group_col, subj_col, time_col, conc_col):
    """Per-subject Cmax, Tmax, AUClast (linear trapezoidal) from time-concentration data."""
    rows = []
    for (grp, subj), g in df.groupby([group_col, subj_col]):
        g = g.sort_values(time_col).dropna(subset=[time_col, conc_col])
        if g.empty:
            continue
        cmax = g[conc_col].max()
        tmax = g.loc[g[conc_col].idxmax(), time_col]
        auc_last = linear_trapz_auc(g[time_col].values, g[conc_col].values)
        rows.append({group_col: grp, subj_col: subj, "Cmax": cmax, "Tmax": tmax, "AUClast": auc_last})
    return pd.DataFrame(rows)


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
# Sidebar: data loading & column mapping
# ========================================================================

st.title("🧪 PK Outlier & Group Comparison Tool")
st.caption("Outlier detection (IQR / Z-score / Modified Z-score / Grubbs) and group t-test comparisons for PK study data")

st.sidebar.header("1. Load your data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
use_demo = st.sidebar.checkbox("Use demo data instead", value=uploaded_file is None)

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
elif use_demo:
    rng = np.random.default_rng(42)
    groups = {"Reference": 90, "Test1": 100, "Test2": 110, "Test3": 95}
    timepoints = [0, 0.5, 1, 2, 4, 8, 12, 24]
    rows = []
    for grp, base_dose in groups.items():
        for i in range(1, 7):
            subj = f"{grp}_S{i:02d}"
            base = rng.uniform(0.85, 1.15) * base_dose
            for t in timepoints:
                conc = base * np.exp(-0.15 * t) + rng.normal(0, 2)
                rows.append({"Group": grp, "Subject": subj, "Time": t, "Concentration": max(conc, 0)})
    rows.append({"Group": "Test1", "Subject": "Test1_S99_outlier", "Time": 4, "Concentration": 300})
    rows.append({"Group": "Reference", "Subject": "Reference_S98_outlier", "Time": 12, "Concentration": 1})
    raw_df = pd.DataFrame(rows)
else:
    st.info("Upload a file or check 'Use demo data' in the sidebar to get started.")
    st.stop()

st.sidebar.header("2. Map your columns")
columns = list(raw_df.columns)


def guess(name, default_idx=0):
    return columns.index(name) if name in columns else default_idx


group_col = st.sidebar.selectbox("Group column", columns, index=guess("Group"))
subj_col = st.sidebar.selectbox("Subject column", columns, index=guess("Subject"))
time_col = st.sidebar.selectbox("Time column", columns, index=guess("Time"))
conc_col = st.sidebar.selectbox("Concentration column", columns, index=guess("Concentration"))

# Clean numeric columns
df = raw_df.copy()
df[conc_col] = pd.to_numeric(df[conc_col], errors="coerce")
df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
n_dropped = df[[conc_col, time_col]].isna().any(axis=1).sum()
df = df.dropna(subset=[conc_col, time_col])
if n_dropped > 0:
    st.warning(f"Dropped {n_dropped} row(s) with non-numeric or missing Time/Concentration values.")

all_groups = sorted(df[group_col].dropna().unique().tolist())

# ========================================================================
# Tabs
# ========================================================================

tab_outlier, tab_ttest = st.tabs(["🔍 Outlier Detection", "📊 T-Test Comparison"])

# ------------------------------------------------------------------------
# TAB 1: Outlier detection
# ------------------------------------------------------------------------
with tab_outlier:
    st.sidebar.header("3. Outlier detection settings")
    method = st.sidebar.selectbox(
        "Method", ["IQR", "Z-score", "Modified Z-score", "Grubbs' test"]
    )
    grouping_choice = st.sidebar.radio(
        "Compare within...",
        ["Group + Time point (recommended)", "Time point only", "Overall (whole dataset)"],
        help=(
            "Group + Time point: compares each subject to others in the SAME group AND SAME time point. "
            "Best when groups (e.g. Reference vs Test1) are expected to have different concentration levels.\n\n"
            "Time point only: compares across all groups at the same time point.\n\n"
            "Overall: ignores both group and time — rarely appropriate for PK profiles."
        ),
    )
    if grouping_choice.startswith("Group + Time"):
        group_cols = [group_col, time_col]
    elif grouping_choice.startswith("Time point"):
        group_cols = [time_col]
    else:
        group_cols = []

    params = {}
    if method == "IQR":
        params["k"] = st.sidebar.slider("IQR multiplier (k)", 1.0, 3.0, 1.5, 0.1)
    elif method == "Z-score":
        params["threshold"] = st.sidebar.slider("Z-score threshold", 1.5, 5.0, 3.0, 0.1)
    elif method == "Modified Z-score":
        params["threshold"] = st.sidebar.slider("Modified Z-score threshold", 1.5, 5.0, 3.5, 0.1)
    elif method == "Grubbs' test":
        params["alpha"] = st.sidebar.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
        params["iterative"] = st.sidebar.checkbox("Remove outliers iteratively (test again after each removal)", value=True)

    result = run_outlier_detection(df, group_cols, conc_col, method, params)
    n_outliers = int(result["Is_Outlier"].sum())
    n_total = len(result)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total data points", n_total)
    c2.metric("Outliers flagged", n_outliers)
    c3.metric("Outlier rate", f"{(n_outliers / n_total * 100) if n_total else 0:.1f}%")

    st.subheader("Time vs. Concentration Profile")
    fig = go.Figure()
    normal = result[~result["Is_Outlier"]]
    outliers = result[result["Is_Outlier"]]
    color_map = {g: c for g, c in zip(all_groups, ["steelblue", "seagreen", "darkorange", "purple", "teal", "brown"])}

    for g in all_groups:
        sub_n = normal[normal[group_col] == g]
        fig.add_trace(go.Scatter(
            x=sub_n[time_col], y=sub_n[conc_col], mode="markers",
            marker=dict(color=color_map.get(g, "gray"), size=7),
            name=f"{g} (normal)",
            text=[f"Group: {g}<br>Subject: {s}<br>Time: {t}<br>Conc: {c:.3g}"
                  for s, t, c in zip(sub_n[subj_col], sub_n[time_col], sub_n[conc_col])],
            hoverinfo="text",
        ))

    fig.add_trace(go.Scatter(
        x=outliers[time_col], y=outliers[conc_col], mode="markers",
        marker=dict(color="crimson", size=13, symbol="x", line=dict(width=2)),
        name="Outlier",
        text=[f"Group: {g}<br>Subject: {s}<br>Time: {t}<br>Conc: {c:.3g}"
              for g, s, t, c in zip(outliers[group_col], outliers[subj_col], outliers[time_col], outliers[conc_col])],
        hoverinfo="text",
    ))
    fig.update_layout(xaxis_title="Time", yaxis_title="Concentration", height=550,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribution by Group & Time Point")
    box_group_by = st.radio("Box plot grouping", ["By Time (all groups combined)", "By Group + Time"], horizontal=True)
    fig_box = go.Figure()
    if box_group_by.startswith("By Time"):
        for t, g in df.groupby(time_col):
            fig_box.add_trace(go.Box(y=g[conc_col], name=str(t), boxpoints="all", jitter=0.4))
        fig_box.update_layout(xaxis_title="Time", yaxis_title="Concentration", height=450)
    else:
        for grp in all_groups:
            g = df[df[group_col] == grp]
            fig_box.add_trace(go.Box(x=g[time_col], y=g[conc_col], name=grp, boxpoints="all", jitter=0.4))
        fig_box.update_layout(xaxis_title="Time", yaxis_title="Concentration", height=450, boxmode="group")
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Flagged Outliers")
    if n_outliers == 0:
        st.success("No outliers detected with the current settings.")
    else:
        show_cols = [group_col, subj_col, time_col, conc_col, "Score", "Detail", "N_in_subset"]
        st.dataframe(outliers[show_cols].sort_values(by=[group_col, time_col]), use_container_width=True)

    with st.expander("View full results table"):
        st.dataframe(result, use_container_width=True)

    csv_buffer = io.StringIO()
    result.to_csv(csv_buffer, index=False)
    st.download_button("Download full results as CSV", data=csv_buffer.getvalue(),
                        file_name="pk_outlier_results.csv", mime="text/csv", key="download_outlier")

# ------------------------------------------------------------------------
# TAB 2: T-test comparison
# ------------------------------------------------------------------------
with tab_ttest:
    st.subheader("Compare groups with a t-test")

    st.markdown("**Step 1 — choose which groups to include**")
    selected_groups = st.multiselect("Groups to include in comparison", all_groups, default=all_groups)

    if len(selected_groups) < 2:
        st.warning("Select at least 2 groups to run a comparison.")
        st.stop()

    subset = df[df[group_col].isin(selected_groups)]
    available_subjects = sorted(subset[subj_col].unique().tolist())

    st.markdown("**Step 2 — include/exclude specific subjects**")
    excluded_subjects = st.multiselect(
        "Subjects to EXCLUDE from this analysis (e.g. subjects you've flagged as outliers above)",
        available_subjects, default=[],
    )
    subset = subset[~subset[subj_col].isin(excluded_subjects)]

    equal_var = st.checkbox(
        "Assume equal variances (standard t-test)", value=False,
        help="Leave unchecked to use Welch's t-test (does not assume equal variances) — usually the safer default."
    )

    if len(selected_groups) == 2:
        pairs = [tuple(selected_groups)]
    else:
        pairs = list(itertools.combinations(selected_groups, 2))
        st.caption(f"More than 2 groups selected — running all {len(pairs)} pairwise comparisons.")

    comparison_type = st.radio(
        "What do you want to compare?",
        ["Time-concentration data (per time point)", "PK parameters (Cmax, Tmax, AUClast)"],
    )

    if comparison_type.startswith("Time-concentration"):
        st.markdown("### Results: t-test at each time point")
        for g1, g2 in pairs:
            st.markdown(f"**{g1} vs {g2}**")
            rows = []
            for t in sorted(subset[time_col].unique()):
                a = subset[(subset[group_col] == g1) & (subset[time_col] == t)][conc_col]
                b = subset[(subset[group_col] == g2) & (subset[time_col] == t)][conc_col]
                res = two_sample_ttest(a, b, equal_var=equal_var)
                rows.append({"Time": t, **res})
            res_df = pd.DataFrame(rows)
            res_df["significant"] = res_df["significant"].map({True: "Yes (p<0.05)", False: "No", None: "n<2, skipped"})
            st.dataframe(res_df.style.format({"mean1": "{:.3g}", "mean2": "{:.3g}", "t_stat": "{:.3g}", "p_value": "{:.4f}"}),
                         use_container_width=True)

    else:
        st.markdown("### PK parameters derived from time-concentration data")
        st.caption("Cmax = max observed concentration; Tmax = time of Cmax; AUClast = linear trapezoidal AUC up to the last time point, computed per subject.")
        pk_df = compute_pk_params(subset, group_col, subj_col, time_col, conc_col)
        st.dataframe(pk_df, use_container_width=True)

        st.markdown("### Results: t-test on PK parameters")
        for g1, g2 in pairs:
            st.markdown(f"**{g1} vs {g2}**")
            rows = []
            for param in ["Cmax", "Tmax", "AUClast"]:
                a = pk_df[pk_df[group_col] == g1][param]
                b = pk_df[pk_df[group_col] == g2][param]
                res = two_sample_ttest(a, b, equal_var=equal_var)
                rows.append({"Parameter": param, **res})
            res_df = pd.DataFrame(rows)
            res_df["significant"] = res_df["significant"].map({True: "Yes (p<0.05)", False: "No", None: "n<2, skipped"})
            st.dataframe(res_df.style.format({"mean1": "{:.3g}", "mean2": "{:.3g}", "t_stat": "{:.3g}", "p_value": "{:.4f}"}),
                         use_container_width=True)

        csv_buffer2 = io.StringIO()
        pk_df.to_csv(csv_buffer2, index=False)
        st.download_button("Download derived PK parameters as CSV", data=csv_buffer2.getvalue(),
                            file_name="pk_parameters.csv", mime="text/csv", key="download_pk")
