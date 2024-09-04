from backend.data_collector import DataLoader
from backend.data_analysis import PairTradingAnalyzer, DataAnalyzer
from datetime import datetime, timedelta
import unittest

class TestPairTradingAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tickers = [
            "VALE3.SA", "PETR4.SA", "ITUB4.SA", "PETR3.SA", "BBAS3.SA", "ELET3.SA", "BBDC4.SA", "WEGE3.SA", 
            "B3SA3.SA", "ITSA4.SA", "ABEV3.SA", "BPAC11.SA", "RENT3.SA", "JBSS3.SA", "EQTL3.SA", "PRIO3.SA", 
            "RADL3.SA", "SUZB3.SA", "RDOR3.SA", "EMBR3.SA", "SBSP3.SA", "RAIL3.SA", "UGPA3.SA", "VBBR3.SA", 
            "BBSE3.SA", "GGBR4.SA", "ENEV3.SA", "CMIG4.SA", "VIVT3.SA", "BRFS3.SA"
        ]
        data_loader = DataLoader(tickers=cls.tickers, 
                                 start_date=(datetime.today() - timedelta(days = 120)).strftime("%Y-%m-%d"), 
                                 end_date="2024-08-24")
        data = data_loader.fetch_data()
        data = data_loader.preprocess_data()
        cls.data = data  # Aqui está a correção
        cls.data_analyzer = DataAnalyzer(cls.data)
        cls.pair_analyzer = PairTradingAnalyzer(cls.data)

    def test_coint_pairs(self):
        pairs = self.data_analyzer.test_cointegrated_pairs(self.tickers)
        print(f"Cointegrated pairs: {pairs}")
        for ticker1, ticker2, _, _ in pairs:
            self.assertFalse(self.data_analyzer.data[ticker1].isnull().all(), f"{ticker1} data is empty or NaN")
            self.assertFalse(self.data_analyzer.data[ticker2].isnull().all(), f"{ticker2} data is empty or NaN")
        self.assertIsInstance(pairs, list)
        self.assertTrue(len(pairs) > 0) 

    def test_stationary_pairs(self):
        pairs = self.data_analyzer.test_cointegrated_pairs(['ITUB4.SA', 'SBSP3.SA'])
        stationary_pairs = self.pair_analyzer.find_stationary_pairs(pairs)
        print(f"Stationary pairs: {stationary_pairs}")
        self.assertIsInstance(stationary_pairs, list)

    def test_pair_metrics_storage(self):
        pairs = self.data_analyzer.test_cointegrated_pairs(['ITUB4.SA', 'SBSP3.SA'])
        stationary_pairs = self.pair_analyzer.find_stationary_pairs(pairs)
        
        # Debugging: Print the stationary pairs identified
        print(f"Stationary pairs found: {stationary_pairs}")
        
        self.pair_analyzer.store_pair_metrics(stationary_pairs)
        
        # Debugging: Print the stored pair metrics
        print(f"Stored pair metrics: {self.pair_analyzer.pair_metrics}")
        
        self.assertIn(('ITUB4.SA', 'SBSP3.SA'), self.pair_analyzer.pair_metrics)
        print(f"Pair metrics: {self.pair_analyzer.pair_metrics}")

if __name__ == '__main__':
    unittest.main()