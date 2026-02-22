"""
Shared Plotly chart builders for the Options Tool.

All charts use a consistent dark theme with clean styling.
"""

import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Layout defaults
# ---------------------------------------------------------------------------

LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    margin=dict(l=50, r=20, t=40, b=40),
    height=350,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)

IV_COLORS = [
    "rgba(255,100,100,0.8)",   # low IV  - red
    "rgba(255,180,100,0.8)",   # med-low - orange
    "rgba(100,200,255,0.8)",   # current - blue
    "rgba(100,255,150,0.8)",   # med-high - green
    "rgba(200,150,255,0.8)",   # high IV - purple
]


def _base_fig(**kwargs):
    layout = {**LAYOUT_DEFAULTS, **kwargs}
    fig = go.Figure()
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# P&L at Expiry
# ---------------------------------------------------------------------------

def pnl_chart(spot_range, pnl, spot, strike):
    """P&L at expiry with zero line and strike/spot markers."""
    fig = _base_fig(title="P&L at Expiry", xaxis_title="Spot at Expiry",
                    yaxis_title="P&L ($)")

    # Color fill: green above zero, red below
    fig.add_trace(go.Scatter(
        x=spot_range, y=pnl, mode="lines",
        line=dict(color="rgba(100,200,255,0.9)", width=2),
        name="P&L", fill="tozeroy",
        fillcolor="rgba(100,200,255,0.1)",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.add_vline(x=spot, line_dash="dash", line_color="gray",
                  annotation_text=f"Spot {spot:,.0f}")
    fig.add_vline(x=strike, line_dash="dot", line_color="yellow",
                  annotation_text=f"Strike {strike:,.0f}")

    return fig


# ---------------------------------------------------------------------------
# Multi-IV curves (Price, Delta, Theta vs Spot)
# ---------------------------------------------------------------------------

def multi_iv_curve(spot_range, curves_data, title, yaxis_title,
                   spot=None, strike=None, zero_line=False):
    """
    Plot multiple curves (one per IV level) on the same chart.

    curves_data: list of (label, y_values) tuples.
    """
    fig = _base_fig(title=title, xaxis_title="Spot", yaxis_title=yaxis_title)

    for i, (label, y_vals) in enumerate(curves_data):
        color = IV_COLORS[i % len(IV_COLORS)]
        is_current = "current" in label.lower() or i == len(curves_data) // 2
        fig.add_trace(go.Scatter(
            x=spot_range, y=y_vals, mode="lines",
            line=dict(color=color, width=2.5 if is_current else 1.2),
            name=label,
        ))

    if zero_line:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8)

    if spot is not None:
        fig.add_vline(x=spot, line_dash="dash", line_color="gray", line_width=0.8)

    if strike is not None:
        fig.add_vline(x=strike, line_dash="dot", line_color="yellow",
                      line_width=0.8)

    return fig


# ---------------------------------------------------------------------------
# Time Decay chart
# ---------------------------------------------------------------------------

def time_decay_chart(dte_arr, price_arr, title="Price Decay over Time",
                     spot=None, strike=None):
    """Price vs DTE (time on x-axis, counting down)."""
    fig = _base_fig(title=title, xaxis_title="DTE", yaxis_title="Price ($)")

    fig.add_trace(go.Scatter(
        x=dte_arr, y=price_arr, mode="lines",
        line=dict(color="rgba(100,200,255,0.9)", width=2),
        name="Price",
    ))

    # X-axis reversed (high DTE on left)
    fig.update_xaxes(autorange="reversed")

    return fig


def time_decay_multi(dte_arr, data_dict, title="Greeks over Time"):
    """Multiple Greek lines vs DTE."""
    fig = _base_fig(title=title, xaxis_title="DTE")

    colors = ["rgba(100,200,255,0.9)", "rgba(255,150,100,0.9)",
              "rgba(100,255,150,0.9)"]
    for i, (label, vals) in enumerate(data_dict.items()):
        fig.add_trace(go.Scatter(
            x=dte_arr, y=vals, mode="lines",
            line=dict(color=colors[i % len(colors)], width=2),
            name=label,
        ))

    fig.update_xaxes(autorange="reversed")
    return fig


# ---------------------------------------------------------------------------
# IV Smile chart
# ---------------------------------------------------------------------------

def smile_chart(smile_df, title="IV Smile", spot=None):
    """IV smile with raw and smoothed curves."""
    fig = _base_fig(title=title, xaxis_title="Strike", yaxis_title="IV",
                    height=300)

    if "iv_raw" in smile_df.columns:
        fig.add_trace(go.Scatter(
            x=smile_df["strike"], y=smile_df["iv_raw"], mode="lines",
            line=dict(color="rgba(150,150,150,0.4)", width=1),
            name="Raw",
        ))

    fig.add_trace(go.Scatter(
        x=smile_df["strike"], y=smile_df["iv"], mode="lines",
        line=dict(color="rgba(100,200,255,0.9)", width=2),
        name="Smoothed",
    ))

    if spot is not None:
        fig.add_vline(x=spot, line_dash="dash", line_color="gray",
                      line_width=0.8, annotation_text="Spot")

    return fig
