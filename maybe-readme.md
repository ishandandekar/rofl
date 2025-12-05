# ROFL - Reinforcement-Optimized Financial Loss Reserves

ROFL is a reinforcement learning system for dynamic claim reserve optimization in insurance. It uses Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) algorithms to learn optimal reserve adjustment strategies.

## Features

- **Reinforcement Learning**: DQN and PPO agents for reserve optimization
- **Feature Engineering**: Automated feature extraction from claim data
- **Explainability**: SHAP-like explanations for RL decisions
- **Dashboard**: Interactive Streamlit dashboard for analysis
- **Database**: DuckDB for efficient data storage and querying
- **Real-time Inference**: Fast reserve recommendations with confidence scores

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd rofl

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Train both DQN and PPO models
python -m rl.train

# Or train with your own data
python -m rl.train --data_path data/your_claims.csv
```

### 3. Launch Dashboard

```bash
# Start the Streamlit dashboard
streamlit run app.py
```

## Project Structure

```
rofl/
├── app.py                        # Streamlit dashboard
├── config/
│   └── settings.yaml             # Configuration settings
├── data/
│   ├── claims_raw.csv            # Sample claims data
│   └── duckdb/                   # DuckDB database files
├── rl/
│   ├── env.py                    # RL environment
│   ├── dqn_agent.py              # DQN agent implementation
│   ├── ppo_agent.py              # PPO agent implementation
│   ├── networks.py               # Neural network architectures
│   └── train.py                  # Training orchestrator
├── utils/
│   ├── db.py                     # Database utilities
│   ├── transforms.py             # Feature engineering
│   ├── charts.py                 # Visualization utilities
│   └── rl_inference.py           # Inference engine
├── outputs/
│   ├── models/                   # Trained models
│   └── logs/                     # Training logs
└── schemas/                      # Data schemas
```

## Data Format

ROFL expects claims data in the following format:

| Column | Type | Description |
|--------|------|-------------|
| claim_id | TEXT | Unique claim identifier |
| client_id | TEXT | Client identifier |
| industry | TEXT | Industry category |
| loss_category | TEXT | Loss type (Motor, Property, etc.) |
| cause_of_loss | TEXT | Description of loss cause |
| date_of_loss | DATE | Claim accident date |
| date_notified | DATE | FNOL date |
| as_at_date | DATE | Reporting snapshot date |
| paid | DOUBLE | Cumulative paid amount |
| outstanding | DOUBLE | Case reserve |
| total_incurred | DOUBLE | paid + outstanding |
| claim_status | TEXT | open/closed |
| t | INTEGER | Development age in months |

## Configuration

Edit `config/settings.yaml` to customize:

- Training parameters (episodes, learning rate, batch size)
- Model architectures (hidden layers, activation functions)
- Environment settings (actions, reward weights)
- Database paths and inference settings

## RL Algorithms

### Deep Q-Network (DQN)
- Off-policy, value-based RL
- Experience replay for stable training
- Epsilon-greedy exploration
- Target network for stability

### Proximal Policy Optimization (PPO)
- On-policy, actor-critic method
- Clipped surrogate objective
- Entropy regularization for exploration
- More stable convergence

## Environment

The RL environment models claim development with:

- **State**: Normalized claim features (paid, outstanding, development age, etc.)
- **Actions**: Reserve adjustments [-20%, -10%, -5%, 0%, +5%, +10%, +20%]
- **Reward**: Balances reserve adequacy, volatility, and accuracy

## Dashboard Features

### Portfolio Overview
- Reserve gap distribution
- Industry and loss category breakdowns
- Recent RL recommendations

### Claim Explorer
- Individual claim analysis
- Reserve development paths
- RL recommendations with explanations

### RL Analysis
- Model performance metrics
- Action distribution analysis
- Inference history and trends

## API Usage

```python
from utils.rl_inference import ROFLInference

# Initialize inference engine
inference = ROFLInference()

# Load trained models
inference.load_models("dqn")  # or "ppo"

# Make prediction on a claim snapshot
result = inference.predict_single(claim_data)

# Generate explanations
explanations = inference.generate_explanations(claim_data)
```

## Training

```python
from rl.train import ROFLTrainer

# Initialize trainer
trainer = ROFLTrainer()

# Train both models
metrics = trainer.train_both("data/claims_raw.csv")

# Compare models
comparison = trainer.compare_models()
```

## Performance Requirements

- Training: 50-100k claims in < 10 hours
- Inference: 10k claim snapshots in < 3 seconds
- Dashboard: Refresh time < 0.5 seconds
- Database: Query latency < 50ms

## Model Explainability

Each RL recommendation includes:
- Confidence score
- Top 5 feature attributions
- Human-readable explanations
- Model version and timestamp

## Governance & Controls

- Model versioning and audit trails
- Fallback rule-based reserves
- Stress testing for shock-loss scenarios
- Bias checks across industries and categories

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Check the documentation
- Review the example notebooks
- Open an issue on GitHub

## Roadmap

- [ ] Streaming inference for real-time processing
- [ ] Advanced ensemble methods
- [ ] Multi-objective optimization
- [ ] Industry-specific pre-trained models
- [ ] Advanced visualization and reporting