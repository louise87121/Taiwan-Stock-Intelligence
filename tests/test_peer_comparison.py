import pandas as pd
import yaml

from src.config import CONFIG_DIR
from src.transform.peer_comparison import build_semiconductor_peer_comparison


def test_semiconductor_peer_ranking_and_winbond_config():
    snapshot = pd.DataFrame(
        {
            "month": ["2024-01", "2024-01", "2024-01"],
            "stock_id": ["2344", "2330", "2881"],
            "stock_name": ["華邦電", "台積電", "富邦金"],
            "industry_group": ["Semiconductor", "24", "Financials"],
            "revenue": [100, 200, 300],
            "revenue_yoy": [0.2, 0.1, 0.5],
            "monthly_return": [0.03, 0.01, 0.2],
            "volatility_20d": [0.04, 0.02, 0.01],
            "pe_ratio": [12, 20, 10],
            "financial_health_score": [70, 80, 90],
            "risk_level": ["Moderate Risk", "Low Risk", "Low Risk"],
        }
    )
    peers = build_semiconductor_peer_comparison(snapshot)
    assert set(peers["stock_id"]) == {"2344", "2330"}
    assert peers.loc[peers["stock_id"].eq("2344"), "revenue_yoy_rank"].iloc[0] == 1
    assert peers.loc[peers["stock_id"].eq("2330"), "volatility_rank"].iloc[0] == 1

    with (CONFIG_DIR / "stocks.yml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    winbond = [stock for stock in config["stocks"] if stock["stock_id"] == "2344"][0]
    assert winbond["stock_name"] == "華邦電"
