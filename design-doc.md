# ROFL — Reinforcement-Optimized Financial Loss Reserves
Technical Design Document (TDD)

Version: 0.1
Owner: Ishan Dandekar
Date: 05th December 2025


## Executive Summary
ROFL is a reinforcement learning–driven system for dynamic claim reserve optimization.
It ingests historical claim development data, learns optimal reserve adjustment strategies using DQN/PPO, and produces:
- Reserve recommendations (increase / decrease / hold)
- Confidence levels
- Explanations (feature attribution: SHAP or contribution-based)
-  Portfolio analytics
-  Audit-quality inference logs

ROFL aims to provide:
-  Better reserve adequacy
-  Lower IBNR volatility
-  Faster reserve reaction time to emerging patterns
-  Evidence-backed reserve decisioning for clients

## System architecture overview

                +-----------------------------+
                |      Raw Client Data        |
                +-----------------------------+
                          |
                          ▼
          +-----------------------------------------+
          |     ETL + Feature Engineering Layer     |
          +-----------------------------------------+
            | snapshot tables  | feature tensors
            ▼                  ▼
   +----------------+     +---------------------+
   | DuckDB (OLAP)  |<--->| RL Training System  |
   | claims_long     |     | (DQN / PPO)         |
   | claims_snapshot |     +---------------------+
   | rofl_inference |             |
   +----------------+              ▼
                               model.pt
                                 |
                                 ▼
                      +--------------------------+
                      | Streamlit RL Dashboard   |
                      |  - RL recommendations    |
                      |  - reserve analytics     |
                      |  - explanations          |
                      +--------------------------+

## Algorithms used
ROFL will integrate 3 categories of ML algorithms:

1. Reinforcement Learning Algorithms
  A. Deep Q-Network (DQN)
    - Type: Off-policy, value-based RL
    - Usage: Baseline RL algorithm for discrete reserve actions
    - Advantages: Simple, robust, stable with experience replay
    - Actions: -20%, -10%, -5%, 0%, +5%, +10%, +20%
  B. Proximal Policy Optimization (PPO)
    - Type: On-policy, actor-critic
    - Usage: Advanced exploration + more consistent convergence
    - Advantages:
      - Handles stochasticity
      - More appropriate for reserve development sequence modeling
      - Regularized policy updates prevent instability
  C. Environment Simulation Model ("Claim Development Micro-simulator")
    - A simplified claim development engine defined by:
    - Paid velocity curve
    - Outstanding reserve adjustment drift
    - Industry × Loss category priors
    - Shock loss probability
    - Used for:
      - RL training
      - Scenario rollouts during inference to compute p10/p90
2. Auxiliary ML Models
SHAP Explainers
- Used for feature attribution:
- Which variables influenced the RL policy?
- Why did it recommend +10% instead of -5%?

## Data Architecture
1. DuckDB Data Model
ROFL uses three primary tables + optional views.
1.1 `claims_long`

| Column         | Type    | Description                      |
| -------------- | ------- | -------------------------------- |
| claim_id       | TEXT    | Unique claim                     |
| client_id      | TEXT    | Client identifier                |
| industry       | TEXT    | Industry category                |
| loss_category  | TEXT    | Motor / Property / Liability etc |
| cause_of_loss  | TEXT    | Text                             |
| cause_code     | TEXT    | Encoded category                 |
| date_of_loss   | DATE    | Claim accident date              |
| date_notified  | DATE    | FNOL                             |
| as_at_date     | DATE    | Reporting snapshot               |
| paid           | DOUBLE  | Cumulative paid                  |
| outstanding    | DOUBLE  | Case reserve                     |
| total_incurred | DOUBLE  | paid + outstanding               |
| claim_status   | TEXT    | open/closed                      |
| t              | INTEGER | Development age in months        |
| paid_velocity  | DOUBLE  | Derived                          |
| reserve_gap    | DOUBLE  | Derived                          |

1.2 `claims_snapshot`

| Column         | Type    |
| -------------- | ------- |
| claim_id       | TEXT    |
| as_at_date     | DATE    |
| paid           | DOUBLE  |
| outstanding    | DOUBLE  |
| total_incurred | DOUBLE  |
| t              | INTEGER |
| industry       | TEXT    |
| loss_category  | TEXT    |
| reserve_gap    | DOUBLE  |


1.3 `rofl_inference`

| Column                  | Type      |
| ----------------------- | --------- |
| claim_id                | TEXT      |
| as_at_date              | DATE      |
| recommended_action_idx  | INT       |
| recommended_pct         | DOUBLE    |
| recommended_new_reserve | DOUBLE    |
| policy_confidence       | DOUBLE    |
| model_type              | TEXT      |
| model_path              | TEXT      |
| inference_ts            | TIMESTAMP |


## Feature Engineering
1. Base Features
  - Paid
  - Outstanding
  - Total incurred
  - Development age (t)
  - Reserve gap
  - Paid acceleration
  - Industry encoding
  - Loss category encoding

2. Derived Features
  - Paid velocity: (paid_t - paid_{t-1}) / outstanding
  - Outstanding drift: outstanding_t - outstanding_{t-1}
  - Shock detection: Boolean flag based on jump thresholds
  - Age bucket: early / mid / tail

3. State vector structure
```bash

state = [
    normalized_paid,
    normalized_outstanding,
    normalized_total_incurred,
    t,
    industry_onehot/embedding,
    loss_category_onehot/embedding,
    reserve_gap,
    paid_velocity,
]
```

## Reinforcement learning Environment
1. State Representation
One claim snapshot.

2. Actions
Discrete reserve adjustment percentages:

[-0.20, -0.10, -0.05, 0.0, +0.05, +0.10, +0.20]

3. Reward Function
Reward balances:
  - Claim adequacy
  - Smooth adjustments
  - Avoiding excessive reserve volatility

```markdown

reward = 
    - under_reserve_penalty
    - volatility_penalty
    + accuracy_bonus
    + stability_bonus

```

4. Transition Function
Uses a stochastic claim development simulator:
  - Paid development curve (industry x category)
  - Reserve drift model
  - Shock-loss probability

## Model training steps
  - Extract claims_long → convert to training episodes
  - Compute features
  - Build state vectors
  - Train DQN and PPO agents
  - Evaluate against baselines
  - Export model (model.pt)
  - Register model metadata into DuckDB

## Streamlit dashboard architecture
Tabs
1. Portfolio Overview
  - Reserve gap distribution
  - Heatmaps
  - RL recommendation summary

2. Claim Explorer
  - Reserve development path
  - RL decisions per as-at
  - Confidence scores
3. RL Output Layer
  - Model inference logs
  - SHAP explanations (top drivers)
  - Accept/Reject workflow (optional)
4. Scenario Simulation
  - RL vs deterministic reserve path comparison

## Project structure
```bash

rofl/
│
├── app.py                        # Streamlit dashboard
├── config/
│   └── settings.yaml             # env configs
│
├── data/
│   ├── claims_raw.csv
│   └── duckdb/
│
├── rl/
│   ├── env.py                    # environment logic
│   ├── dqn_agent.py              # DQN training
│   ├── ppo_agent.py              # PPO training
│   ├── networks.py               # neural nets
│   ├── replay_buffer.py
│   └── train.py                  # training orchestrator
│
├── utils/
│   ├── db.py                     # duckdb connectors
│   ├── transforms.py             # feature engineering
│   ├── charts.py                 # altair charts
│   ├── rl_inference.py           # inference + persistence
│   └── logging.py
│
├── outputs/
│   ├── models/
│   └── logs/
│
├── schemas/
│   ├── claims_schema.json
│   ├── inference_schema.json
│   └── explanation_schema.json
│
└── README.md
```

## Technical Requirements
1. Runtime
  - Python 3.10+
  - PyTorch (with CUDA if available)
  - Streamlit
  - DuckDB
  - Altair charts
  - Pydantic for CLI validation

2. Performance Requirements
  - Able to train DQN on 50–100k claims in < 10 hours
  - Inference on 10k claim snapshots in < 3 seconds
  - Dashboard refresh time < 0.5 seconds
  - DuckDB query latency < 50 ms

3. Reliability
- RL inference must be logged deterministically
- Every recommended reserve must have:
  - timestamp
  - model version
  - deterministic seed

4. Explainability
For each claim inference:
- store SHAP top-5 features
- store q_values or policy_logits
- produce human-readable reasons

5. Governance & Controls
- Model versioning
- Audit trails
- Fallback rule-based reserves
- Stress tests (shock-loss scenarios)
- Bias checks (industry, category)

## Scalability Constraints
- DuckDB is OLAP-optimized and handles ~100M rows easily
- RL state tensors stored off-database for speed
- Streaming inference possible in next iteration

## Risks & Mitigations

| Risk                        | Mitigation                                             |
| --------------------------- | ------------------------------------------------------ |
| RL may overfit              | Use PPO entropy bonus + dropout                        |
| State scaling mismatch      | Use feature store manifest versioning                  |
| Claim development too noisy | Add simulation priors + smoothing                      |
| Clients distrust RL         | Provide transparent explanations + override capability |
| Model drift                 | Weekly retraining with drift detection                 |
