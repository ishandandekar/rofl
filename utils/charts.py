import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def create_reserve_gap_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df["reserve_gap"],
            nbinsx=50,
            name="Reserve Gap",
            marker_color="lightblue",
            opacity=0.7,
        )
    )

    fig.update_layout(
        title="Reserve Gap Distribution",
        xaxis_title="Reserve Gap ($)",
        yaxis_title="Frequency",
        showlegend=False,
        height=400,
    )

    return fig


def create_action_distribution_chart(df: pd.DataFrame) -> go.Figure:
    action_counts = df["recommended_action_idx"].value_counts().sort_index()

    fig = go.Figure(
        data=[
            go.Bar(
                x=action_counts.index, y=action_counts.values, marker_color="lightgreen"
            )
        ]
    )

    fig.update_layout(
        title="RL Action Distribution",
        xaxis_title="Action Index",
        yaxis_title="Count",
        height=400,
    )

    return fig


def create_confidence_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df["policy_confidence"],
            nbinsx=30,
            name="Confidence",
            marker_color="lightcoral",
            opacity=0.7,
        )
    )

    fig.update_layout(
        title="Policy Confidence Distribution",
        xaxis_title="Confidence Score",
        yaxis_title="Frequency",
        showlegend=False,
        height=400,
    )

    return fig


def create_claim_development_chart(claim_data: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Reserve Development", "Cumulative Payments"),
        vertical_spacing=0.1,
    )

    fig.add_trace(
        go.Scatter(
            x=claim_data["as_at_date"],
            y=claim_data["outstanding"],
            mode="lines+markers",
            name="Outstanding Reserve",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=claim_data["as_at_date"],
            y=claim_data["paid"],
            mode="lines+markers",
            name="Cumulative Paid",
            line=dict(color="green"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"Claim Development - {claim_data.iloc[0]['claim_id']}",
        height=600,
        showlegend=True,
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Amount ($)", row=1, col=1)
    fig.update_yaxes(title_text="Amount ($)", row=2, col=1)

    return fig


def create_training_progress_chart(metrics: dict, model_type: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Episode Rewards", "Episode Lengths"),
        vertical_spacing=0.1,
    )

    episodes = list(range(len(metrics["episode_rewards"])))

    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=metrics["episode_rewards"],
            mode="lines",
            name=f"{model_type.upper()} Reward",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=metrics["episode_lengths"],
            mode="lines",
            name=f"{model_type.upper()} Length",
            line=dict(color="green"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{model_type.upper()} Training Progress", height=600, showlegend=True
    )

    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Steps", row=2, col=1)

    return fig


def create_model_comparison_chart(dqn_metrics: dict, ppo_metrics: dict) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Average Rewards", "Average Episode Lengths"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]],
    )

    dqn_episodes = list(range(len(dqn_metrics["episode_rewards"])))
    ppo_episodes = list(range(len(ppo_metrics["episode_rewards"])))

    fig.add_trace(
        go.Scatter(
            x=dqn_episodes,
            y=dqn_metrics["episode_rewards"],
            mode="lines",
            name="DQN Reward",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ppo_episodes,
            y=ppo_metrics["episode_rewards"],
            mode="lines",
            name="PPO Reward",
            line=dict(color="red"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dqn_episodes,
            y=dqn_metrics["episode_lengths"],
            mode="lines",
            name="DQN Length",
            line=dict(color="blue"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=ppo_episodes,
            y=ppo_metrics["episode_lengths"],
            mode="lines",
            name="PPO Length",
            line=dict(color="red"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(title="Model Comparison: DQN vs PPO", height=400, showlegend=True)

    fig.update_xaxes(title_text="Episode", row=1, col=1)
    fig.update_xaxes(title_text="Episode", row=1, col=2)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Steps", row=1, col=2)

    return fig


def create_industry_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot_df = (
        df.groupby(["industry", "loss_category"]).size().reset_index(name="count")
    )
    heatmap_data = pivot_df.pivot(
        index="industry", columns="loss_category", values="count"
    ).fillna(0)

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale="Blues",
            showscale=True,
        )
    )

    fig.update_layout(
        title="Claims by Industry and Loss Category",
        xaxis_title="Loss Category",
        yaxis_title="Industry",
        height=500,
    )

    return fig


def create_portfolio_summary_charts(df: pd.DataFrame) -> dict:
    charts = {}

    charts["reserve_gap"] = create_reserve_gap_chart(df)
    charts["industry_distribution"] = px.pie(
        df.groupby("industry").size().reset_index(name="count"),
        values="count",
        names="industry",
        title="Claims by Industry",
    )
    charts["loss_category_distribution"] = px.pie(
        df.groupby("loss_category").size().reset_index(name="count"),
        values="count",
        names="loss_category",
        title="Claims by Loss Category",
    )
    charts["industry_heatmap"] = create_industry_heatmap(df)

    return charts
