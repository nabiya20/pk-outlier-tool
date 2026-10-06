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
