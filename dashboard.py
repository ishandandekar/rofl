import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.db import ROFLDatabase
from utils.rl_inference import ROFLInference
from utils.charts import (
    create_confidence_chart,
    create_claim_development_chart,
)

st.set_page_config(
    page_title="ROFL Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00a67e;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_inference_engine():
    return ROFLInference()


@st.cache_data
def load_portfolio_data():
    db = ROFLDatabase()
    return db.get_training_data(limit=50000)


@st.cache_data
def get_portfolio_insights(_db):
    """Generate meaningful insights from portfolio data"""
    try:
        # Get key metrics
        stats = _db.get_portfolio_stats()
        # Reserve gap analysis
        gap_data = _db.conn.execute("""
            SELECT 
                reserve_gap,
                industry,
                loss_category,
                CASE 
                    WHEN reserve_gap > 0 THEN 'Under-reserved'
                    WHEN reserve_gap < 0 THEN 'Over-reserved'
                    ELSE 'Adequate'
                END as reserve_status
            FROM claims_long 
            WHERE reserve_gap IS NOT NULL
        """).df()

        # Recent inference trends
        recent_inferences = _db.conn.execute("""
            SELECT 
                DATE(inference_ts) as inference_date,
                AVG(policy_confidence) as avg_confidence,
                COUNT(*) as inference_count,
                AVG(recommended_pct) as avg_action
            FROM rofl_inference 
            WHERE inference_ts >= DATE('now', '-30 days')
            GROUP BY DATE(inference_ts)
            ORDER BY inference_date DESC
        """).df()

        return {
            "stats": stats,
            "gap_analysis": gap_data,
            "inference_trends": recent_inferences,
        }
    except Exception as e:
        st.error(f"Error generating insights: {e}")
        return None


def render_sidebar():
    """Render improved sidebar with navigation and controls"""
    st.sidebar.markdown("## 🎛️ Dashboard Controls")

    # Model selection
    model_type = st.sidebar.selectbox(
        "Active Model",
        ["dqn", "ppo"],
        help="Choose which RL model to use for recommendations",
    )

    # Date range filter
    st.sidebar.markdown("### 📅 Date Range")
    end_date = st.sidebar.date_input("End Date", datetime.now().date())
    start_date = st.sidebar.date_input("Start Date", end_date - timedelta(days=90))

    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Min Confidence Threshold",
        0.0,
        1.0,
        0.5,
        0.05,
        help="Filter recommendations by confidence level",
    )

    # Industry filter
    try:
        db = ROFLDatabase()
        industries = (
            db.conn.execute("""
            SELECT DISTINCT industry FROM claims_long ORDER BY industry
        """)
            .df()["industry"]
            .tolist()
        )

        selected_industries = st.sidebar.multiselect(
            "Filter by Industry",
            industries,
            default=industries[:5] if len(industries) > 5 else industries,
        )
    except:
        selected_industries = []

    return {
        "model_type": model_type,
        "start_date": start_date,
        "end_date": end_date,
        "confidence_threshold": confidence_threshold,
        "selected_industries": selected_industries,
    }


def render_kpi_metrics(stats, insights):
    """Render key performance indicators with insights"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_color = "normal" if stats["total_claims"] > 0 else "off"
        st.metric(
            "Total Claims",
            f"{stats['total_claims']:,}",
            delta="Active" if stats["total_claims"] > 0 else "No data",
            delta_color=delta_color,
        )

    with col2:
        avg_gap = stats.get("avg_reserve_gap", 0)
        gap_status = (
            "⚠️ Under"
            if avg_gap > 1000
            else "✅ Balanced"
            if avg_gap > -1000
            else "⚠️ Over"
        )
        st.metric(
            "Avg Reserve Gap",
            f"${avg_gap:,.0f}",
            delta=gap_status,
            delta_color="inverse" if avg_gap < -1000 else "normal",
        )

    with col3:
        st.metric(
            "Industries",
            stats["industries"],
            delta="Diverse" if stats["industries"] > 3 else "Focused",
        )

    with col4:
        st.metric(
            "Loss Categories",
            stats["loss_categories"],
            delta="Complex" if stats["loss_categories"] > 5 else "Simple",
        )

    # Insights section
    if insights and "gap_analysis" in insights:
        gap_data = insights["gap_analysis"]
        if len(gap_data) > 0:
            under_reserved = len(
                gap_data[gap_data["reserve_status"] == "Under-reserved"]
            )
            total_claims = len(gap_data)
            under_reserve_pct = (under_reserved / total_claims) * 100

            if under_reserve_pct > 30:
                st.markdown(
                    """
                <div class="warning-box">
                    <strong>⚠️ Portfolio Risk Alert:</strong> {:.1f}% of claims are under-reserved. 
                    Consider reviewing reserve policies for high-risk segments.
                </div>
                """.format(under_reserve_pct),
                    unsafe_allow_html=True,
                )
            elif under_reserve_pct < 10:
                st.markdown(
                    """
                <div class="insight-box">
                    <strong>✅ Portfolio Health:</strong> Reserve adequacy looks good with only {:.1f}% under-reserved claims.
                </div>
                """.format(under_reserve_pct),
                    unsafe_allow_html=True,
                )


def render_portfolio_overview(insights, filters):
    """Render enhanced portfolio overview with actionable insights"""
    st.markdown(
        '<h2 class="main-header">📊 Portfolio Intelligence</h2>', unsafe_allow_html=True
    )

    if not insights:
        st.error("Unable to load portfolio insights")
        return

    gap_data = insights["gap_analysis"]

    # Top row - Key charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Reserve Gap Analysis")
        if len(gap_data) > 0:
            fig = px.histogram(
                gap_data,
                x="reserve_gap",
                color="reserve_status",
                nbins=50,
                title="Reserve Gap Distribution by Status",
                color_discrete_map={
                    "Under-reserved": "#ff6b6b",
                    "Over-reserved": "#4ecdc4",
                    "Adequate": "#45b7d1",
                },
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Key insight
            median_gap = gap_data["reserve_gap"].median()
            st.info(f"📈 Median reserve gap: ${median_gap:,.0f}")

    with col2:
        st.subheader("🏭 Industry Performance")
        if len(gap_data) > 0:
            industry_summary = (
                gap_data.groupby("industry")
                .agg({"reserve_gap": ["mean", "median", "count"]})
                .round(0)
            )
            industry_summary.columns = ["Avg Gap", "Median Gap", "Claim Count"]
            industry_summary = industry_summary.sort_values("Avg Gap", ascending=False)

            fig = px.bar(
                industry_summary.reset_index(),
                x="industry",
                y="Avg Gap",
                title="Average Reserve Gap by Industry",
                color="Claim Count",
                color_continuous_scale="Blues",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Bottom row - Heatmap and trends
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔥 Risk Heatmap")
        try:
            db = ROFLDatabase()
            heatmap_data = db.conn.execute("""
                SELECT industry, loss_category, COUNT(*) as claim_count,
                       AVG(reserve_gap) as avg_gap
                FROM claims_long 
                WHERE reserve_gap IS NOT NULL
                GROUP BY industry, loss_category
                HAVING claim_count > 5
            """).df()

            if len(heatmap_data) > 0:
                pivot_data = heatmap_data.pivot(
                    index="industry", columns="loss_category", values="avg_gap"
                ).fillna(0)

                fig = px.imshow(
                    pivot_data,
                    title="Average Reserve Gap by Industry & Loss Category",
                    color_continuous_scale="RdYlBu_r",
                    labels=dict(x="Loss Category", y="Industry", color="Avg Gap ($)"),
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating heatmap: {e}")

    with col2:
        st.subheader("📈 Recent Trends")
        if "inference_trends" in insights and len(insights["inference_trends"]) > 0:
            trends = insights["inference_trends"]

            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=("Average Confidence", "Daily Inference Volume"),
                vertical_spacing=0.1,
            )

            fig.add_trace(
                go.Scatter(
                    x=trends["inference_date"],
                    y=trends["avg_confidence"],
                    mode="lines+markers",
                    name="Avg Confidence",
                    line=dict(color="#1f77b4"),
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=trends["inference_date"],
                    y=trends["inference_count"],
                    mode="lines+markers",
                    name="Inference Count",
                    line=dict(color="#ff7f0e"),
                ),
                row=2,
                col=1,
            )

            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recent inference data available")


def render_claim_analyzer(inference_engine, filters):
    """Render enhanced claim analysis tool"""
    st.markdown(
        '<h2 class="main-header">🔍 Claim Intelligence Analyzer</h2>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        claim_id = st.text_input(
            "Enter Claim ID",
            placeholder="e.g., CLAIM_12345",
            help="Search for a specific claim to analyze",
        )

    with col2:
        # Get unique claim IDs for autocomplete
        try:
            claim_ids = (
                inference_engine.db.conn.execute("""
                SELECT DISTINCT claim_id FROM claims_long 
                ORDER BY claim_id LIMIT 1000
            """)
                .df()["claim_id"]
                .tolist()
            )

            selected_claim = st.selectbox(
                "Or select from recent claims",
                options=[""] + claim_ids,
                help="Choose from available claim IDs",
            )

            if selected_claim and not claim_id:
                claim_id = selected_claim
        except:
            pass

    with col3:
        as_at_date = st.date_input(
            "As At Date", datetime.now().date(), help="Analysis date for the claim"
        )

    if claim_id:
        try:
            # Get claim data
            claim_data = inference_engine.db.conn.execute(f"""
                SELECT * FROM claims_long 
                WHERE claim_id = '{claim_id}' 
                ORDER BY as_at_date
            """).df()

            if len(claim_data) == 0:
                st.error(f"No data found for claim ID: {claim_id}")
                return

            # Claim summary card
            latest = claim_data.iloc[-1]
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Industry", latest["industry"])
            with col2:
                st.metric("Loss Category", latest["loss_category"])
            with col3:
                st.metric("Development Age", f"{latest['t']} months")
            with col4:
                st.metric("Status", latest["claim_status"])

            # Development chart
            st.subheader("📈 Claim Development Pattern")
            fig = create_claim_development_chart(claim_data)
            st.plotly_chart(fig, use_container_width=True)

            # RL Analysis
            st.subheader("🤖 RL Recommendation Analysis")

            try:
                recommendation = inference_engine.predict_single(latest)

                col1, col2, col3 = st.columns(3)

                with col1:
                    action_color = (
                        "🟢"
                        if recommendation["recommended_pct"] > 0
                        else "🔴"
                        if recommendation["recommended_pct"] < 0
                        else "🟡"
                    )
                    st.metric(
                        f"{action_color} Recommended Action",
                        f"{recommendation['recommended_pct']:+.1%}",
                    )

                with col2:
                    confidence_color = (
                        "High"
                        if recommendation["policy_confidence"] > 0.7
                        else "Medium"
                        if recommendation["policy_confidence"] > 0.5
                        else "Low"
                    )
                    st.metric(
                        "Confidence",
                        f"{recommendation['policy_confidence']:.1%}",
                        delta=confidence_color,
                    )

                with col3:
                    current_reserve = latest["outstanding"]
                    new_reserve = recommendation["recommended_new_reserve"]
                    change = new_reserve - current_reserve
                    st.metric(
                        "New Reserve", f"${new_reserve:,.0f}", delta=f"${change:,.0f}"
                    )

                # Explanations
                explanations = inference_engine.generate_explanations(latest)
                if explanations:
                    st.subheader("🧠 Key Factors Driving Recommendation")
                    exp_df = pd.DataFrame(explanations)

                    # Create a more visual explanation
                    for i, exp in exp_df.head(5).iterrows():
                        impact_icon = "📈" if exp["shap_value"] > 0 else "📉"
                        st.markdown(f"""
                        **{impact_icon} {exp["feature_name"]}**: 
                        {abs(exp["shap_value"]):.3f} impact score
                        """)

            except Exception as e:
                st.error(f"Error generating RL analysis: {e}")

            # Historical recommendations
            st.subheader("📋 Historical Recommendations")
            try:
                history = inference_engine.db.conn.execute(f"""
                    SELECT * FROM rofl_inference 
                    WHERE claim_id = '{claim_id}'
                    ORDER BY inference_ts DESC
                    LIMIT 10
                """).df()

                if len(history) > 0:
                    st.dataframe(
                        history[
                            [
                                "as_at_date",
                                "recommended_pct",
                                "policy_confidence",
                                "model_type",
                            ]
                        ],
                        use_container_width=True,
                    )
                else:
                    st.info("No previous recommendations found")
            except Exception as e:
                st.warning(f"Could not load recommendation history: {e}")

        except Exception as e:
            st.error(f"Error analyzing claim: {e}")


def render_model_insights(inference_engine, filters):
    """Render model performance and insights"""
    st.markdown(
        '<h2 class="main-header">🧠 Model Intelligence</h2>', unsafe_allow_html=True
    )

    # Model performance metrics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Model Performance")
        try:
            metrics = inference_engine.get_model_performance_metrics()

            if "error" not in metrics:
                # Performance metrics
                perf_col1, perf_col2, perf_col3 = st.columns(3)

                with perf_col1:
                    st.metric("Total Inferences", f"{metrics['total_inferences']:,}")

                with perf_col2:
                    avg_conf = metrics["avg_confidence"]
                    conf_trend = (
                        "↗️ High"
                        if avg_conf > 0.7
                        else "➡️ Medium"
                        if avg_conf > 0.5
                        else "↘️ Low"
                    )
                    st.metric("Avg Confidence", f"{avg_conf:.1%}", delta=conf_trend)

                with perf_col3:
                    st.metric("Confidence Std", f"{metrics['confidence_std']:.3f}")

                # Action distribution
                if "action_distribution" in metrics:
                    action_df = pd.DataFrame(
                        list(metrics["action_distribution"].items()),
                        columns=["Action", "Count"],
                    )

                    # Add action labels
                    action_labels = {
                        -20: "-20%",
                        -10: "-10%",
                        -5: "-5%",
                        0: "Hold",
                        5: "+5%",
                        10: "+10%",
                        20: "+20%",
                    }
                    action_df["Action_Label"] = action_df["Action"].map(action_labels)

                    fig = px.bar(
                        action_df,
                        x="Action_Label",
                        y="Count",
                        title="RL Action Distribution",
                        color="Count",
                        color_continuous_scale="Blues",
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(metrics["error"])
        except Exception as e:
            st.error(f"Error loading model metrics: {e}")

    with col2:
        st.subheader("⚙️ Model Configuration")

        config_info = {
            "Model Type": inference_engine.current_model_type.upper()
            if inference_engine.current_model_type
            else "None",
            "State Dimension": inference_engine.state_dim,
            "Action Space": inference_engine.action_dim,
            "Available Actions": len(inference_engine.config["environment"]["actions"]),
        }

        for key, value in config_info.items():
            st.metric(key, value)

        # Model switcher
        if st.button(f"🔄 Switch to {filters['model_type'].upper()} Model"):
            try:
                inference_engine.load_models(filters["model_type"])
                st.success(
                    f"✅ {filters['model_type'].upper()} model loaded successfully"
                )
                st.rerun()
            except Exception as e:
                st.error(f"Error switching model: {e}")

    # Recent inference analysis
    st.markdown("---")
    st.subheader("🔍 Recent Inference Analysis")

    try:
        history_df = inference_engine.get_inference_history(
            start_date=filters["start_date"].strftime("%Y-%m-%d"),
            end_date=filters["end_date"].strftime("%Y-%m-%d"),
        )

        if len(history_df) > 0:
            # Filter by confidence threshold
            filtered_df = history_df[
                history_df["policy_confidence"] >= filters["confidence_threshold"]
            ]

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Total Inferences",
                    len(history_df),
                    delta=f"{len(filtered_df)} above threshold",
                )

                # Confidence distribution
                fig = create_confidence_chart(history_df)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Action vs Confidence scatter
                fig = px.scatter(
                    history_df.head(1000),  # Limit for performance
                    x="recommended_pct",
                    y="policy_confidence",
                    color="model_type",
                    title="Action vs Confidence Analysis",
                    opacity=0.6,
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            # Recent inferences table
            with st.expander("📋 View Recent Inferences"):
                display_df = filtered_df.head(50)[
                    [
                        "claim_id",
                        "as_at_date",
                        "recommended_pct",
                        "policy_confidence",
                        "model_type",
                        "inference_ts",
                    ]
                ]
                st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No inference history found for the selected period")

    except Exception as e:
        st.error(f"Error loading inference analysis: {e}")


def main():
    # Initialize inference engine
    try:
        inference_engine = load_inference_engine()
        inference_engine.load_models("dqn")
    except Exception as e:
        st.error(f"❌ Failed to initialize models: {e}")
        st.info("💡 Please train the models first using `python -m rl.train`")
        return

    # Render sidebar
    filters = render_sidebar()

    # Main header
    st.markdown(
        """
    <div style="text-align: center; padding: 2rem 0;">
        <h1 class="main-header">🤖 ROFL Dashboard</h1>
        <p style="font-size: 1.2rem; color: #666;">Reinforcement-Optimized Financial Loss Reserves</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Load insights
    with st.spinner("Loading portfolio insights..."):
        insights = get_portfolio_insights(inference_engine.db)

    # Main navigation tabs
    tab1, tab2, tab3 = st.tabs(
        ["📊 Portfolio Intelligence", "🔍 Claim Analyzer", "🧠 Model Insights"]
    )

    with tab1:
        if insights:
            render_kpi_metrics(insights["stats"], insights)
            render_portfolio_overview(insights, filters)

    with tab2:
        render_claim_analyzer(inference_engine, filters)

    with tab3:
        render_model_insights(inference_engine, filters)

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        ROFL Dashboard v1.0 | Real-time RL-powered reserve optimization
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
