from backend.backtest import BacktestAnalyzer
from backend.data_collector import DataLoader
from backend.data_analysis import PairTradingAnalyzer

import pandas as pd

tickers = ["ABEV3.SA", "B3SA3.SA"]

data_loader = DataLoader(tickers=tickers,
                         start_date="2023-06-01",
                         end_date="2024-09-01")
data = data_loader.fetch_data()
data = data_loader.preprocess_data()

trading_analyzer = PairTradingAnalyzer(data)

backtest = BacktestAnalyzer(data, trading_analyzer)
backtest.run_analysis(start_date='2024-01-01', end_date='2024-08-30')