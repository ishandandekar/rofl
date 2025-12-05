import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


class ROFLDatabase:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.db_path = Path(self.config["database"]["path"])
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._initialize_tables()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _initialize_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS claims_long (
                claim_id TEXT,
                client_id TEXT,
                industry TEXT,
                loss_category TEXT,
                cause_of_loss TEXT,
                cause_code TEXT,
                date_of_loss DATE,
                date_notified DATE,
                as_at_date DATE,
                paid DOUBLE,
                outstanding DOUBLE,
                total_incurred DOUBLE,
                claim_status TEXT,
                t INTEGER,
                paid_velocity DOUBLE,
                reserve_gap DOUBLE
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS claims_snapshot (
                claim_id TEXT,
                as_at_date DATE,
                paid DOUBLE,
                outstanding DOUBLE,
                total_incurred DOUBLE,
                t INTEGER,
                industry TEXT,
                loss_category TEXT,
                reserve_gap DOUBLE
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS rofl_inference (
                claim_id TEXT,
                as_at_date DATE,
                recommended_action_idx INTEGER,
                recommended_pct DOUBLE,
                recommended_new_reserve DOUBLE,
                policy_confidence DOUBLE,
                model_type TEXT,
                model_path TEXT,
                inference_ts TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS shap_explanations (
                claim_id TEXT,
                as_at_date DATE,
                feature_name TEXT,
                feature_value DOUBLE,
                shap_value DOUBLE,
                explanation_rank INTEGER
            )
        """)

    def load_claims_data(self, csv_path: str) -> int:
        df = pd.read_csv(csv_path)
        self.conn.execute("INSERT OR REPLACE INTO claims_long SELECT * FROM df")
        return len(df)

    def get_training_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        query = "SELECT * FROM claims_long ORDER BY claim_id, as_at_date"
        if limit:
            query += f" LIMIT {limit}"
        return self.conn.execute(query).df()

    def get_claims_snapshot(self, as_at_date: str) -> pd.DataFrame:
        query = f"""
            SELECT * FROM claims_snapshot 
            WHERE as_at_date = '{as_at_date}'
            ORDER BY claim_id
        """
        return self.conn.execute(query).df()

    def save_inference(self, inference_data: pd.DataFrame):
        self.conn.execute("INSERT INTO rofl_inference SELECT * FROM inference_data")

    def save_explanations(self, explanations: pd.DataFrame):
        self.conn.execute("INSERT INTO shap_explanations SELECT * FROM explanations")

    def get_portfolio_stats(self) -> Dict[str, Any]:
        result = self.conn.execute("""
                SELECT 
                    COUNT(DISTINCT claim_id) as total_claims,
                    COUNT(*) as total_snapshots,
                    AVG(reserve_gap) as avg_reserve_gap,
                    STDDEV(reserve_gap) as std_reserve_gap,
                    COUNT(DISTINCT industry) as industries,
                    COUNT(DISTINCT loss_category) as loss_categories
                FROM claims_long
            """).fetchone()

        if result is None:
            return {
                "total_claims": 0,
                "total_snapshots": 0,
                "avg_reserve_gap": 0.0,
                "std_reserve_gap": 0.0,
                "industries": 0,
                "loss_categories": 0,
            }

        return {
            "total_claims": result[0],
            "total_snapshots": result[1],
            "avg_reserve_gap": result[2],
            "std_reserve_gap": result[3],
            "industries": result[4],
            "loss_categories": result[5],
        }


def close(self):
    self.conn.close()
