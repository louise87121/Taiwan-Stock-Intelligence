from src.api.finmind_client import FinMindClient


def test_build_params_includes_dataset_and_query_params_without_token():
    client = FinMindClient(token="")
    params = client.build_params("TaiwanStockPrice", data_id="2344", start_date="2021-01-01")
    assert params["dataset"] == "TaiwanStockPrice"
    assert params["data_id"] == "2344"
    assert params["start_date"] == "2021-01-01"
    assert "token" not in params


def test_build_params_includes_token_when_available():
    client = FinMindClient(token="secret-token")
    params = client.build_params("TaiwanStockPER", data_id="2344")
    assert params["token"] == "secret-token"
