"""Presentation-only helpers. Never changes analytical input or results."""
import math
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


def individual_profile_figure(df, group, subjects, font, x_title, y_title, title):
    """One original time/concentration series per subject within a group."""
    fig = go.Figure()
    group_data = df[(df["Group"] == group) & df["Subject"].isin(subjects)]
    for i, subject in enumerate(subjects):
        rows = group_data[group_data["Subject"] == subject].sort_values("Time", kind="stable")
        if rows.empty:
            continue
        fig.add_trace(go.Scatter(
            x=rows["Time"], y=rows["Concentration"], mode="lines+markers", name=str(subject),
            line=dict(color=GROUP_COLORS[i % len(GROUP_COLORS)], width=1.5,
                      dash=["solid", "dash", "dot"][i // len(GROUP_COLORS) % 3]),
            marker=dict(size=5),
            hovertemplate=f"{x_title}: %{{x}}<br>{y_title}: %{{y:.4g}}<extra>%{{fullData.name}}</extra>",
        ))
    style_figure(fig, font, x_title, y_title)
    fig.update_layout(title=dict(text=title, x=0.5, xanchor="center"),
                      legend=dict(orientation="v", x=1.02, y=1, xanchor="left", yanchor="top"),
                      margin=dict(t=65, r=140, l=75, b=65))
    xmax = max(float(group_data["Time"].max()), 0) if len(group_data) else 0
    ymax = max(float(group_data["Concentration"].max()), 0) if len(group_data) else 0
    fig.update_xaxes(range=[0, xmax * 1.04 if xmax else 1], autorange=False)
    fig.update_yaxes(range=[0, ymax * 1.12 if ymax else 1], autorange=False)
    return fig


def true_log_individual_figure(source):
    fig = go.Figure(source)
    positive = []
    for trace in fig.data:
        values = np.asarray(trace.y, dtype=float)
        positive.extend(values[np.isfinite(values) & (values > 0)].tolist())
        trace.y = np.where(values > 0, values, np.nan)
    if not positive:
        return None
    low = math.floor(math.log10(min(positive)))
    high = max(math.ceil(math.log10(max(positive))), low + 1)
    fig.update_yaxes(type="log", range=[low, high], tickmode="auto", dtick=1, exponentformat="none")
    return fig
