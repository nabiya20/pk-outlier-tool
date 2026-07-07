"""
PK Outlier & Group Comparison Tool
-----------------------------------
A Streamlit app for:
  1) Outlier detection on Group / Subject / Time / Concentration PK data
     using IQR, Z-score, Modified Z-score, or Grubbs' test.
  2) T-test comparison between groups, on either the raw time-concentration
     data (per time point) or on PK parameters (Cmax, AUC, custom partial
     AUCs, etc.), with the ability to include/exclude specific groups/subjects.

Data entry is done via editable, spreadsheet-like grids: you can type directly,
or copy cells from Excel and paste them in (click the top-left cell of the grid,
then Ctrl+V / Cmd+V). File upload is also available if you'd rather import a
whole file at once.

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
# Demo / template data
# ========================================================================

def make_demo_tc_data():
    rng = np.random.default_rng(42)
    groups = {"Reference": 90, "Test1": 100, "Test2": 110, "Test3": 95}
    timepoints = [0, 0.5, 1, 2, 4, 8, 12, 24]
    rows = []
    for grp, base_dose in groups.items():
        for i in range(1, 5):
            subj = f"{grp}_S{i:02d}"
            base = rng.uniform(0.85, 1.15) * base_dose
            for t in timepoints:
                conc = base * np.exp(-0.15 * t) + rng.normal(0, 2)
                rows.append({"Group": grp, "Subject": subj, "Time": t, "Concentration": round(max(conc, 0), 2)})
    rows.append({"Group": "Test1", "Subject": "Test1_S99_outlier", "Time": 4, "Concentration": 300})
    return pd.DataFrame(rows)


def make_demo_pk_data():
    return pd.DataFrame([
        {"Group": "Reference", "Subject": "Reference_S01", "Cmax": 88.2, "AUC": 610.4, "AUC_partial": 210.1},
        {"Group": "Reference", "Subject": "Reference_S02", "Cmax": 91.5, "AUC": 630.2, "AUC_partial": 215.6},
        {"Group": "Test1", "Subject": "Test1_S01", "Cmax": 99.1, "AUC": 700.8, "AUC_partial": 240.3},
        {"Group": "Test1", "Subject": "Test1_S02", "Cmax": 102.4, "AUC": 715.0, "AUC_partial": 245.9},
    ])


# ========================================================================
# Session state init
# ========================================================================

if "tc_data" not in st.session_state:
    st.session_state.tc_data = make_demo_tc_data()
if "pk_data" not in st.session_state:
    st.session_state.pk_data = make_demo_pk_data()

st.title("🧪 PK Outlier & Group Comparison Tool")
st.caption("Paste your data directly into the grids below (like Excel), or upload a file if you prefer.")

# ========================================================================
# Data entry
# ========================================================================

st.header("1. Enter your data")
entry_tab_tc, entry_tab_pk = st.tabs(["📈 Time–Concentration data", "🧮 PK Parameter data"])

with entry_tab_tc:
    st.markdown(
        "Columns: **Group, Subject, Time, Concentration**. "
        "Click the top-left cell and paste (Ctrl+V / Cmd+V) to bring in data copied from Excel. "
        "Right-click a row for more options, or use the `+` at the bottom to add rows."
    )
    up_tc = st.file_uploader("...or upload a CSV/Excel file to replace this table", type=["csv", "xlsx", "xls"], key="up_tc")
    if up_tc is not None:
        st.session_state.tc_data = pd.read_csv(up_tc) if up_tc.name.endswith(".csv") else pd.read_excel(up_tc)

    st.session_state.tc_data = st.data_editor(
        st.session_state.tc_data,
        num_rows="dynamic",
        use_container_width=True,
        key="tc_editor",
        column_config={
            "Group": st.column_config.TextColumn(required=True),
            "Subject": st.column_config.TextColumn(required=True),
            "Time": st.column_config.NumberColumn(required=True),
            "Concentration": st.column_config.NumberColumn(required=True),
        },
    )
    if st.button("Reset to demo data", key="reset_tc"):
        st.session_state.tc_data = make_demo_tc_data()
        st.rerun()

with entry_tab_pk:
    st.markdown(
        "Default columns: **Group, Subject, Cmax, AUC, AUC_partial**. "
        "Add your own parameter columns below (e.g. AUC_0-12, Tmax, t1/2, CL/F) — "
        "the sheet is fully adjustable."
    )
    up_pk = st.file_uploader("...or upload a CSV/Excel file to replace this table", type=["csv", "xlsx", "xls"], key="up_pk")
    if up_pk is not None:
        st.session_state.pk_data = pd.read_csv(up_pk) if up_pk.name.endswith(".csv") else pd.read_excel(up_pk)

    col_a, col_b = st.columns([3, 1])
    with col_a:
        new_col_name = st.text_input("New parameter column name (e.g. 'AUC_0-12', 'Tmax', 't_half')", key="new_pk_col")
    with col_b:
        st.write("")
        st.write("")
        if st.button("➕ Add column") and new_col_name.strip():
            if new_col_name.strip() not in st.session_state.pk_data.columns:
                st.session_state.pk_data[new_col_name.strip()] = np.nan
            st.rerun()

    numeric_cols = [c for c in st.session_state.pk_data.columns if c not in ("Group", "Subject")]
    cols_to_drop = st.multiselect("Remove parameter column(s)", numeric_cols)
    if cols_to_drop and st.button("🗑️ Remove selected column(s)"):
        st.session_state.pk_data = st.session_state.pk_data.drop(columns=cols_to_drop)
        st.rerun()

    column_config_pk = {
        "Group": st.column_config.TextColumn(required=True),
        "Subject": st.column_config.TextColumn(required=True),
    }
    for c in st.session_state.pk_data.columns:
        if c not in ("Group", "Subject"):
            column_config_pk[c] = st.column_config.NumberColumn()

    st.session_state.pk_data = st.data_editor(
        st.session_state.pk_data,
        num_rows="dynamic",
        use_container_width=True,
        key="pk_editor",
        column_config=column_config_pk,
    )
    if st.button("Reset to demo data", key="reset_pk"):
        st.session_state.pk_data = make_demo_pk_data()
        st.rerun()

# ========================================================================
# Clean time-concentration data for analysis
# ========================================================================

group_col, subj_col, time_col, conc_col = "Group", "Subject", "Time", "Concentration"

df = st.session_state.tc_data.copy()
required_tc_cols = {group_col, subj_col, time_col, conc_col}
if not required_tc_cols.issubset(df.columns):
    st.error(f"Time–Concentration sheet must have columns: {', '.join(required_tc_cols)}")
    st.stop()

df[conc_col] = pd.to_numeric(df[conc_col], errors="coerce")
df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
df = df.dropna(subset=[group_col, subj_col, time_col, conc_col])
all_groups = sorted(df[group_col].dropna().unique().tolist())

if df.empty:
    st.warning("No usable rows in the Time–Concentration sheet yet. Add data above to continue.")
    st.stop()

# ========================================================================
# Analysis tabs
# ========================================================================

st.header("2. Analysis")
tab_outlier, tab_ttest = st.tabs(["🔍 Outlier Detection", "📊 T-Test Comparison"])

# ------------------------------------------------------------------------
# TAB 1: Outlier detection
# ------------------------------------------------------------------------
with tab_outlier:
    st.sidebar.header("Outlier detection settings")
    method = st.sidebar.selectbox("Method", ["IQR", "Z-score", "Modified Z-score", "Grubbs' test"])
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
        params["iterative"] = st.sidebar.checkbox("Remove outliers iteratively", value=True)

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
    palette = ["steelblue", "seagreen", "darkorange", "purple", "teal", "brown"]
    color_map = {g: palette[i % len(palette)] for i, g in enumerate(all_groups)}

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

    tc_subset = df[df[group_col].isin(selected_groups)]
    pk_all = st.session_state.pk_data.copy()
    pk_subset_groups = pk_all[pk_all[group_col].isin(selected_groups)] if group_col in pk_all.columns else pk_all.iloc[0:0]

    available_subjects = sorted(set(tc_subset[subj_col].unique().tolist()) | set(
        pk_subset_groups[subj_col].unique().tolist() if subj_col in pk_subset_groups.columns else []
    ))

    st.markdown("**Step 2 — include/exclude specific subjects**")
    excluded_subjects = st.multiselect(
        "Subjects to EXCLUDE from this analysis (e.g. subjects you've flagged as outliers above)",
        available_subjects, default=[],
    )
    tc_subset = tc_subset[~tc_subset[subj_col].isin(excluded_subjects)]
    pk_subset = pk_subset_groups[~pk_subset_groups[subj_col].isin(excluded_subjects)] if not pk_subset_groups.empty else pk_subset_groups

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
        ["Time-concentration data (per time point)", "PK parameters (from the PK Parameter sheet)"],
    )

    if comparison_type.startswith("Time-concentration"):
        st.markdown("### Results: t-test at each time point")
        for g1, g2 in pairs:
            st.markdown(f"**{g1} vs {g2}**")
            rows = []
            for t in sorted(tc_subset[time_col].unique()):
                a = tc_subset[(tc_subset[group_col] == g1) & (tc_subset[time_col] == t)][conc_col]
                b = tc_subset[(tc_subset[group_col] == g2) & (tc_subset[time_col] == t)][conc_col]
                res = two_sample_ttest(a, b, equal_var=equal_var)
                rows.append({"Time": t, **res})
            res_df = pd.DataFrame(rows)
            res_df["significant"] = res_df["significant"].map({True: "Yes (p<0.05)", False: "No", None: "n<2, skipped"})
            st.dataframe(res_df.style.format({"mean1": "{:.3g}", "mean2": "{:.3g}", "t_stat": "{:.3g}", "p_value": "{:.4f}"}),
                         use_container_width=True)

    else:
        if pk_subset.empty:
            st.warning("No matching rows found in the PK Parameter sheet for the selected groups/subjects. "
                       "Fill in the PK Parameter sheet above (Step 1) first.")
        else:
            param_cols = [c for c in pk_subset.columns if c not in (group_col, subj_col)]
            chosen_params = st.multiselect("Parameters to test", param_cols, default=param_cols)
            st.dataframe(pk_subset, use_container_width=True)

            st.markdown("### Results: t-test on PK parameters")
            for g1, g2 in pairs:
                st.markdown(f"**{g1} vs {g2}**")
                rows = []
                for param in chosen_params:
                    a = pd.to_numeric(pk_subset[pk_subset[group_col] == g1][param], errors="coerce")
                    b = pd.to_numeric(pk_subset[pk_subset[group_col] == g2][param], errors="coerce")
                    res = two_sample_ttest(a, b, equal_var=equal_var)
                    rows.append({"Parameter": param, **res})
                res_df = pd.DataFrame(rows)
                res_df["significant"] = res_df["significant"].map({True: "Yes (p<0.05)", False: "No", None: "n<2, skipped"})
                st.dataframe(res_df.style.format({"mean1": "{:.3g}", "mean2": "{:.3g}", "t_stat": "{:.3g}", "p_value": "{:.4f}"}),
                             use_container_width=True)
