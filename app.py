import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.db import ROFLDatabase
from utils.rl_inference import ROFLInference
from utils.charts import create_reserve_gap_chart, create_action_distribution_chart

st.set_page_config(
    page_title="ROFL - Reinforcement-Optimized Financial Loss Reserves",
    page_icon="📊",
    layout="wide",
)

st.title("🤖 ROFL - Reinforcement-Optimized Financial Loss Reserves")
st.markdown("---")


@st.cache_resource
def load_inference_engine():
    return ROFLInference()


@st.cache_data
def load_portfolio_stats():
    db = ROFLDatabase()
    return db.get_portfolio_stats()


def main():
    inference_engine = load_inference_engine()

    try:
        inference_engine.load_models("dqn")
        st.success("✅ DQN model loaded successfully")
    except:
        st.warning("⚠️ No trained model found. Please train the model first.")
        st.info("Run `python -m rl.train` to train the models.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Portfolio Overview", "🔍 Claim Explorer", "🧠 RL Analysis", "⚙️ Settings"]
    )

    with tab1:
        portfolio_overview_tab(inference_engine)

    with tab2:
        claim_explorer_tab(inference_engine)

    with tab3:
        rl_analysis_tab(inference_engine)

    with tab4:
        settings_tab(inference_engine)


def portfolio_overview_tab(inference_engine):
    st.header("Portfolio Overview")

    col1, col2, col3, col4 = st.columns(4)

    try:
        stats = load_portfolio_stats()

        with col1:
            st.metric("Total Claims", f"{stats['total_claims']:,}")
        with col2:
            st.metric("Total Snapshots", f"{stats['total_snapshots']:,}")
        with col3:
            st.metric("Industries", stats["industries"])
        with col4:
            st.metric("Loss Categories", stats["loss_categories"])
    except:
        st.error("Unable to load portfolio statistics")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Reserve Gap Distribution")
        try:
            df = inference_engine.db.conn.execute(
                "SELECT reserve_gap FROM claims_long LIMIT 10000"
            ).df()
            if len(df) > 0:
                fig = px.histogram(
                    df, x="reserve_gap", nbins=50, title="Reserve Gap Distribution"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating reserve gap chart: {e}")

    with col2:
        st.subheader("Industry Breakdown")
        try:
            df = inference_engine.db.conn.execute("""
                SELECT industry, COUNT(*) as count 
                FROM claims_long 
                GROUP BY industry 
                ORDER BY count DESC
            """).df()
            if len(df) > 0:
                fig = px.pie(
                    df, values="count", names="industry", title="Claims by Industry"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating industry chart: {e}")

    st.markdown("---")
    st.subheader("Recent RL Recommendations")

    try:
        recent_inferences = inference_engine.get_inference_history(limit=100)
        if len(recent_inferences) > 0:
            st.dataframe(recent_inferences.head(10), use_container_width=True)
        else:
            st.info("No inference history available")
    except Exception as e:
        st.error(f"Error loading inference history: {e}")


def claim_explorer_tab(inference_engine):
    st.header("Claim Explorer")

    col1, col2 = st.columns(2)

    with col1:
        claim_id = st.text_input("Enter Claim ID:", placeholder="e.g., CLAIM_12345")

    with col2:
        as_at_date = st.date_input("As At Date:", datetime.now().date())

    if st.button("🔍 Analyze Claim"):
        if claim_id:
            try:
                claim_data = inference_engine.db.conn.execute(f"""
                    SELECT * FROM claims_long 
                    WHERE claim_id = '{claim_id}' 
                    ORDER BY as_at_date
                """).df()

                if len(claim_data) == 0:
                    st.error(f"No data found for claim ID: {claim_id}")
                    return

                st.success(f"Found {len(claim_data)} snapshots for claim {claim_id}")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Reserve Development")
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=claim_data["as_at_date"],
                            y=claim_data["outstanding"],
                            mode="lines+markers",
                            name="Outstanding Reserve",
                            line=dict(color="blue"),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=claim_data["as_at_date"],
                            y=claim_data["paid"],
                            mode="lines+markers",
                            name="Cumulative Paid",
                            line=dict(color="green"),
                        )
                    )
                    fig.update_layout(title="Reserve Development Over Time", height=400)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Latest RL Recommendation")
                    latest_snapshot = claim_data.iloc[-1]

                    try:
                        recommendation = inference_engine.predict_single(
                            latest_snapshot
                        )

                        st.metric(
                            "Recommended Action",
                            f"{recommendation['recommended_pct']:+.1%}",
                        )
                        st.metric(
                            "Confidence", f"{recommendation['policy_confidence']:.1%}"
                        )
                        st.metric(
                            "New Reserve",
                            f"${recommendation['recommended_new_reserve']:,.0f}",
                        )

                        explanations = inference_engine.generate_explanations(
                            latest_snapshot
                        )
                        if explanations:
                            st.subheader("Top Explanations")
                            exp_df = pd.DataFrame(explanations)
                            st.dataframe(
                                exp_df[
                                    ["feature_name", "shap_value", "explanation_rank"]
                                ],
                                use_container_width=True,
                            )
                    except Exception as e:
                        st.error(f"Error generating recommendation: {e}")

                st.markdown("---")
                st.subheader("Claim History")
                st.dataframe(claim_data, use_container_width=True)

            except Exception as e:
                st.error(f"Error analyzing claim: {e}")
        else:
            st.warning("Please enter a claim ID")


def rl_analysis_tab(inference_engine):
    st.header("RL Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Performance Metrics")
        try:
            metrics = inference_engine.get_model_performance_metrics()

            if "error" not in metrics:
                st.metric("Total Inferences", metrics["total_inferences"])
                st.metric("Average Confidence", f"{metrics['avg_confidence']:.1%}")
                st.metric("Confidence Std Dev", f"{metrics['confidence_std']:.2f}")

                st.subheader("Action Distribution")
                action_df = pd.DataFrame(
                    list(metrics["action_distribution"].items()),
                    columns=["Action", "Count"],
                )
                fig = px.bar(
                    action_df, x="Action", y="Count", title="RL Action Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(metrics["error"])
        except Exception as e:
            st.error(f"Error loading metrics: {e}")

    with col2:
        st.subheader("Model Comparison")

        model_type = st.selectbox("Select Model:", ["dqn", "ppo"])

        if st.button("🔄 Switch Model"):
            try:
                inference_engine.load_models(model_type)
                st.success(f"✅ {model_type.upper()} model loaded")
            except Exception as e:
                st.error(f"Error loading {model_type} model: {e}")

        st.info(f"Currently using: {inference_engine.current_model_type.upper()} model")

    st.markdown("---")
    st.subheader("Inference History")

    col1, col2, col3 = st.columns(3)

    with col1:
        start_date = st.date_input("Start Date:", datetime.now() - timedelta(days=30))

    with col2:
        end_date = st.date_input("End Date:", datetime.now())

    with col3:
        limit = st.number_input("Limit:", min_value=10, max_value=1000, value=100)

    if st.button("📊 Load History"):
        try:
            history_df = inference_engine.get_inference_history(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )

            if len(history_df) > 0:
                st.dataframe(history_df.head(limit), use_container_width=True)

                col1, col2 = st.columns(2)

                with col1:
                    fig = px.histogram(
                        history_df,
                        x="policy_confidence",
                        title="Confidence Distribution",
                        nbins=20,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = px.scatter(
                        history_df.head(500),
                        x="recommended_pct",
                        y="policy_confidence",
                        title="Action vs Confidence",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No inference history found for the selected period")
        except Exception as e:
            st.error(f"Error loading history: {e}")


def settings_tab(inference_engine):
    st.header("Settings")

    st.subheader("Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Current Model",
            inference_engine.current_model_type.upper()
            if inference_engine.current_model_type
            else "None",
        )
        st.metric("State Dimension", inference_engine.state_dim)
        st.metric("Action Space", inference_engine.action_dim)

    with col2:
        st.metric(
            "Actions Available", len(inference_engine.config["environment"]["actions"])
        )
        st.metric("Learning Rate", inference_engine.config["training"]["learning_rate"])
        st.metric("Gamma (Discount)", inference_engine.config["training"]["gamma"])

    st.markdown("---")
    st.subheader("Available Actions")

    actions_df = pd.DataFrame(
        {
            "Action Index": range(
                len(inference_engine.config["environment"]["actions"])
            ),
            "Reserve Adjustment": inference_engine.config["environment"]["actions"],
        }
    )
    st.dataframe(actions_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Database Information")

    try:
        db_stats = inference_engine.db.get_portfolio_stats()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Database Path", inference_engine.db.db_path)

        with col2:
            st.metric("Total Claims", db_stats["total_claims"])

        with col3:
            st.metric(
                "Avg Reserve Gap",
                f"${db_stats['avg_reserve_gap']:,.0f}"
                if db_stats["avg_reserve_gap"]
                else "N/A",
            )
    except Exception as e:
        st.error(f"Error loading database information: {e}")


if __name__ == "__main__":
    main()
