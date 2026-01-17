
"""rofl_streamlit_app.py

Streamlit app to visualize policy actions and offline evaluation metrics for ROFL contextual bandit.

Usage:
    streamlit run rofl_streamlit_app.py

Note: edit DUCKDB_PATH and FEATURE_COLS to match your environment.
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import duckdb
from rofl_bandit import generate_synthetic_data, LinUCB, LinearThompson, NeuralBootstrapBandit, train_reward_model, OffPolicyEvaluator, ips_weighted_regression, predict_from_models

st.set_page_config(page_title='ROFL Bandit Explorer', layout='wide')

st.title('ROFL — Contextual Bandit Explorer')

# Sidebar: data source
st.sidebar.header('Data')
DATA_SOURCE = st.sidebar.radio('Data source', ['Synthetic', 'DuckDB'])

if DATA_SOURCE == 'Synthetic':
    n = st.sidebar.slider('n samples (synthetic)', 1000, 20000, 4000, step=500)
    data = generate_synthetic_data(n=n, n_arms=5, d=8, seed=42)
    X = data['X']
    rewards = data['rewards']
    actions = data['actions']
    propensities = data['propensities']
    n_arms = data['true_thetas'].shape[0]
else:
    st.sidebar.text('Edit DUCKDB_PATH and FEATURE_COLS in the app file to match your DB')
    DUCKDB_PATH = st.sidebar.text_input('DuckDB path', '/path/to/your.duckdb')
    QUERY = st.sidebar.text_area('DuckDB SQL', 'SELECT * FROM claims_snapshot LIMIT 1000')
    if st.sidebar.button('Load from DuckDB'):
        try:
            con = duckdb.connect(DUCKDB_PATH)
            df = con.execute(QUERY).fetchdf()
            st.sidebar.success('Loaded df shape: {}'.format(df.shape))
            # user must edit FEATURE_COLS to match their table
            FEATURE_COLS = [c for c in df.columns if c not in ('action','reward','propensity')]
            X = df[FEATURE_COLS].values
            actions = df['action'].astype(int).values
            rewards = df['reward'].values
            propensities = df['propensity'].values
            n_arms = int(actions.max() + 1)
        except Exception as e:
            st.sidebar.error(f'Failed to load DuckDB: {e}')
            st.stop()
    else:
        st.info('Press "Load from DuckDB" after editing path and query, or choose Synthetic data.')

# Train or load policies
st.sidebar.header('Policies')
train_linucb = st.sidebar.checkbox('Train LinUCB (offline imitation)', True)
train_lts = st.sidebar.checkbox('Train Linear Thompson (offline imitation)', True)
train_neural = st.sidebar.checkbox('Train Neural ensemble', True)

if 'X' not in globals():
    st.stop()

d = X.shape[1]
if train_linucb:
    linucb = LinUCB(n_arms=n_arms, n_features=d, alpha=0.8)
    for i in range(X.shape[0]):
        linucb.update(actions[i], X[i], rewards[i])
else:
    linucb = None

if train_lts:
    lts = LinearThompson(n_arms=n_arms, n_features=d, v2=1.0, lambda_reg=1.0)
    for i in range(X.shape[0]):
        lts.update(actions[i], X[i], rewards[i])
else:
    lts = None

if train_neural:
    neural = NeuralBootstrapBandit(n_arms=n_arms, n_features=d, n_models=6)
    neural.fit(X, actions, rewards)
else:
    neural = None

# Compute target actions
policy_choices = []
if linucb is not None:
    target_linucb = np.array([linucb.select_arm(X[i]) for i in range(X.shape[0])])
    policy_choices.append(('linucb', target_linucb))
if lts is not None:
    target_lts = np.array([lts.select_arm(X[i]) for i in range(X.shape[0])])
    policy_choices.append(('linear_thompson', target_lts))
if neural is not None:
    target_neural = np.array([neural.select_arm(X[i]) for i in range(X.shape[0])])
    policy_choices.append(('neural', target_neural))

# IPS-weighted regression policy
models = ips_weighted_regression(X, actions, rewards, propensities, n_arms)
preds = predict_from_models(models, X)
greedy_from_ips = np.argmax(preds, axis=1)
policy_choices.append(('ips_weighted_regression', greedy_from_ips))

# Display policy metrics
st.header('Offline evaluation')
rows = []
q_hat = train_reward_model(X, actions, rewards, n_arms)
for name, targ in policy_choices:
    ips = OffPolicyEvaluator.ips(rewards, actions, targ, propensities, clip=100.0)
    dr = OffPolicyEvaluator.dr(rewards, actions, targ, propensities, q_hat)
    rows.append({'policy': name, 'IPS': ips, 'DR': dr, 'action_entropy': float(pd.Series(targ).value_counts(normalize=True).entropy())})
df_metrics = pd.DataFrame(rows)
st.dataframe(df_metrics)

# Action distribution plot for selected policy
st.header('Action distributions')
selected = st.selectbox('Select policy to inspect', df_metrics['policy'].tolist())
selected_actions = dict(policy_choices)[selected]
counts = pd.Series(selected_actions).value_counts().sort_index()
plot_df = pd.DataFrame({'action': counts.index.astype(int), 'count': counts.values})
chart = alt.Chart(plot_df).mark_bar().encode(
    x=alt.X('action:O', title='Arm / Action'),
    y=alt.Y('count:Q', title='Count'),
    tooltip=['action', 'count']
).properties(width=600, height=300)
st.altair_chart(chart)

st.write('---')
st.write('To run a live test, export the recommended actions to a table and deploy LinUCB in online mode with conservative exploration.')
