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
from scipy import stats
import streamlit as st
import streamlit.components.v1 as components

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

# Colors reserved: PALETTE for groups in the Outlier Detection view (muted,
# navy/teal/gray family — kept subdued on purpose so flagged points stand
# out), FLAG_COLOR exclusively for flagged/significant markers.
PALETTE = ["#1B4965", "#2C6E91", "#5FA8D3", "#7D8597", "#3C6E71", "#849324", "#8D5B4C", "#5C5C8A"]
FLAG_COLOR = "#C1121F"
FLAG_TINT = "#FBEAEA"

# PROFILE_PALETTE: default colors for the PK Profile Plot tab. These need to
# be clearly distinguishable from each other at a glance (e.g. Reference vs.
# Test groups), so this uses a high-contrast, colorblind-safe (Okabe-Ito
# based) sequence rather than the muted PALETTE above. Still fully
# user-adjustable per group via the color pickers.
PROFILE_PALETTE = ["#1A1A1A", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#8D5B4C"]

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
st.caption("Outlier detection, group comparisons, and profile plots for PK study data.")

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

st.header("Data")
entry_tab_tc, entry_tab_pk = st.tabs(["Time–Concentration Data", "PK Parameter Data"])

with entry_tab_tc:
    handle_upload("up_tc", "tc_data", "pending_upload_tc", TC_COLUMNS)

    with st.expander("Need help? Data format & instructions", expanded=False):
        st.markdown("Required columns: **Group**, **Subject**, **Time**, **Concentration**. Example:")
        st.dataframe(example_tc_snippet(), use_container_width=True, hide_index=True)
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
        st.dataframe(st.session_state.tc_data.head(5), use_container_width=True, hide_index=True, height=200)

    with st.expander(f"Edit data ({n_tc} rows)", expanded=(n_tc == 0)):
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
        st.dataframe(st.session_state.pk_data.head(5), use_container_width=True, hide_index=True, height=200)

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
                column_config_pk[c] = st.column_config.NumberColumn()

        st.session_state.pk_data = st.data_editor(
            st.session_state.pk_data,
            num_rows="dynamic",
            use_container_width=True,
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
        st.caption(f"Showing results for **{method}** ({param_label}) · Comparison scope: **{scope_label}**")

        c1, c2, c3 = st.columns(3)
        c1.metric("Data points analyzed", n_total)
        c2.metric("Flagged as outlier", n_outliers)
        c3.metric("Flagged rate", f"{(n_outliers / n_total * 100) if n_total else 0:.1f}% of {n_total}")

        def comparison_group_label(row):
            if group_cols == [group_col, time_col]:
                return f"{row[group_col]} @ t={row[time_col]}"
            elif group_cols == [time_col]:
                return f"All groups @ t={row[time_col]}"
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
                use_container_width=True, hide_index=True,
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
            st.dataframe(result_display, use_container_width=True, hide_index=True)
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
            fig.update_layout(xaxis_title="Time", yaxis_title="Concentration", height=500,
                               template="simple_white",
                               legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

        with viz_tab2:
            box_group_by = st.radio("Group by", ["Time only", "Group + Time"], horizontal=True, key="box_group_by")
            fig_box = go.Figure()
            if box_group_by == "Time only":
                for t, g in df.groupby(time_col):
                    fig_box.add_trace(go.Box(y=g[conc_col], name=str(t), boxpoints="all", jitter=0.4,
                                              marker_color=PALETTE[0]))
                fig_box.update_layout(xaxis_title="Time", yaxis_title="Concentration", height=450, template="simple_white")
            else:
                for i, grp in enumerate(all_groups):
                    g = df[df[group_col] == grp]
                    fig_box.add_trace(go.Box(x=g[time_col], y=g[conc_col], name=grp, boxpoints="all", jitter=0.4,
                                              marker_color=PALETTE[i % len(PALETTE)]))
                fig_box.update_layout(xaxis_title="Time", yaxis_title="Concentration", height=450,
                                       boxmode="group", template="simple_white")
            st.plotly_chart(fig_box, use_container_width=True)

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
                    st.dataframe(styled, use_container_width=True, hide_index=True)
            else:
                if pk_subset.empty:
                    st.warning("No matching rows found in the PK Parameter sheet for the selected groups/subjects. "
                               "Fill in the PK Parameter sheet in the Data section above first.")
                else:
                    st.caption(f"Comparing: **PK Parameter data** · Groups: {', '.join(selected_groups)} · Excluded subjects: {len(excluded_subjects)}")
                    param_cols = [c for c in pk_subset.columns if c not in (group_col, subj_col)]
                    chosen_params = st.multiselect("Parameters to test", param_cols, default=param_cols, key="ttest_params")
                    with st.expander(f"PK parameter data used ({len(pk_subset)} rows)"):
                        st.dataframe(pk_subset, use_container_width=True, hide_index=True)

                    st.subheader("Results: t-test on PK parameters")
                    for ga, gb in pairs:
                        st.markdown(f"**{ga} vs {gb}**")
                        rows = []
                        for param in chosen_params:
                            a = pd.to_numeric(pk_subset[pk_subset[group_col] == ga][param], errors="coerce")
                            b = pd.to_numeric(pk_subset[pk_subset[group_col] == gb][param], errors="coerce")
                            res = two_sample_ttest(a, b, equal_var=equal_var)
                            rows.append({"Parameter": param, **res})
                        res_df = pd.DataFrame(rows)
                        res_df["significant"] = res_df["significant"].map({True: "Yes (p<0.05)", False: "No", None: "n<2, skipped"})
                        styled = res_df.style.format(
                            {"mean1": "{:.3g}", "mean2": "{:.3g}", "t_stat": "{:.3g}", "p_value": "{:.4f}"}
                        ).apply(style_significant, axis=1)
                        st.dataframe(styled, use_container_width=True, hide_index=True)

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
                    tc1, tc2, tc3 = st.columns(3)
                    with tc1:
                        plot_title = st.text_input("Plot title (shown on the plot)", value="PK Profile", key="profile_title")
                    with tc2:
                        x_axis_title = st.text_input("X-axis title", value="Time (h)", key="profile_xtitle",
                                                      help="Edit the unit to match your data, e.g. 'Time (day)'.")
                    with tc3:
                        y_axis_title = st.text_input("Y-axis title", value=f"{conc_col} (ng/mL)", key="profile_ytitle")

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
                            mec_value = st.number_input("MEC value (ng/mL)", min_value=0.0, value=0.0, step=1.0, key="profile_mec_value")
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

                PLOT_WIDTH, PLOT_HEIGHT = 640, 460

                def build_profile_figure(log_y: bool):
                    fig = go.Figure()
                    for g in profile_groups:
                        g_data = summary[summary[group_col] == g].sort_values(time_col)
                        err = g_data["sd"].fillna(0) if error_metric == "SD" else None
                        fig.add_trace(go.Scatter(
                            x=g_data[time_col], y=g_data["mean"],
                            mode="lines+markers",
                            name=rename_map.get(g, g),
                            line=dict(color=color_map_profile.get(g), width=2.5),
                            marker=dict(size=7, color=color_map_profile.get(g)),
                            error_y=dict(type="data", array=err, visible=True) if err is not None else None,
                        ))

                    if include_mec and mec_value:
                        fig.add_hline(y=mec_value, line_dash="dash", line_color="black",
                                      annotation_text=mec_label, annotation_position="top left")

                    fig.update_layout(
                        title=dict(text=f"<b>{plot_title}</b>", x=0.5, xanchor="center", font=dict(size=18, family="Arial", color="#1B4965")),
                        template="simple_white",
                        font=dict(family="Arial", size=13, color="#1F2937"),
                        legend=dict(
                            bordercolor="lightgray", borderwidth=1, bgcolor="rgba(255,255,255,0.75)",
                            x=0.99, y=0.99, xanchor="right", yanchor="top",
                        ),
                        width=PLOT_WIDTH, height=PLOT_HEIGHT,
                        margin=dict(t=60, r=25, l=65, b=55),
                        plot_bgcolor="white", paper_bgcolor="white",
                    )
                    axis_title_font = dict(size=14, color="#000000", family="Arial Black, Arial, sans-serif")
                    fig.update_xaxes(
                        title=dict(text=f"<b>{x_axis_title}</b>", font=axis_title_font),
                        showline=True, linewidth=1.2, linecolor="black", mirror=False,
                        showgrid=True, gridcolor="#E3E8EC", zeroline=False,
                    )
                    if x_ticks:
                        fig.update_xaxes(tickmode="array", tickvals=x_ticks)
                    if log_y:
                        fig.update_yaxes(
                            title=dict(text=f"<b>{y_axis_title}</b>", font=axis_title_font),
                            type="log", dtick=1, exponentformat="none",
                            showline=True, linewidth=1.2, linecolor="black", mirror=False,
                            showgrid=True, gridcolor="#E3E8EC", zeroline=False,
                            minor=dict(showgrid=False, ticks=""),
                        )
                    else:
                        fig.update_yaxes(
                            title=dict(text=f"<b>{y_axis_title}</b>", font=axis_title_font),
                            showline=True, linewidth=1.2, linecolor="black", mirror=False,
                            showgrid=True, gridcolor="#E3E8EC", zeroline=False,
                        )
                        if y_interval_linear > 0:
                            fig.update_yaxes(dtick=y_interval_linear)
                    return fig

                def render_copyable_plot(fig, height):
                    """Renders a Plotly figure via plotly.js directly (CDN) with an added
                    'Copy image to clipboard' button, so the plot can be pasted into slides
                    or reports without a separate download step. Falls back gracefully if
                    the browser doesn't support the Clipboard image API."""
                    fig_json = fig.to_json()
                    html = f"""
                    <div id="plotdiv_{id(fig)}"></div>
                    <div style="margin-top:6px;">
                      <button id="copybtn_{id(fig)}" style="padding:6px 14px;background:#1B4965;color:#fff;
                        border:none;border-radius:4px;cursor:pointer;font-family:Arial,sans-serif;font-size:13px;">
                        Copy image to clipboard
                      </button>
                      <span id="copystatus_{id(fig)}" style="margin-left:10px;font-family:Arial,sans-serif;
                        font-size:13px;color:#5A6B7B;"></span>
                    </div>
                    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
                    <script>
                    (function() {{
                        var figData = {fig_json};
                        var gd = document.getElementById('plotdiv_{id(fig)}');
                        Plotly.newPlot(gd, figData.data, figData.layout, {{displayModeBar: true, responsive: false}});
                        document.getElementById('copybtn_{id(fig)}').addEventListener('click', function() {{
                            var statusEl = document.getElementById('copystatus_{id(fig)}');
                            Plotly.toImage(gd, {{format: 'png', width: figData.layout.width, height: figData.layout.height, scale: 2}})
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
                                }});
                        }});
                    }})();
                    </script>
                    """
                    components.html(html, height=height + 70, scrolling=False)

                st.subheader("Plot")
                plot_tab_linear, plot_tab_log = st.tabs(["Linear scale", "Semi-log scale"])
                with plot_tab_linear:
                    render_copyable_plot(build_profile_figure(log_y=False), PLOT_HEIGHT)
                with plot_tab_log:
                    render_copyable_plot(build_profile_figure(log_y=True), PLOT_HEIGHT)

                st.caption(
                    "Click \"Copy image to clipboard\" to paste the plot directly into slides or documents, "
                    "or use the camera icon in the plot's toolbar to download it as a PNG."
                )
