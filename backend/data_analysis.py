from statsmodels.tsa.stattools import adfuller, coint, acf, pacf
import statsmodels.api as sm
import numpy as np

class DataAnalyzer:
    def __init__(self, data):
        self.data = data

    def perform_adfuller(self, series):
        result = adfuller(series)
        return result

    def calculate_coint(self, series1, series2):
        score, p_value, _ = coint(series1, series2)
        return score, p_value

    def calculate_acf_pacf(self, series, lags=40):
        lag_acf = acf(series, nlags=lags)
        lag_pacf = pacf(series, nlags=lags, method='ols')
        return lag_acf, lag_pacf

    def test_cointegrated_pairs(self, tickers):
        pairs = []
        n = len(tickers)
        for i in range(n):
            for j in range(i + 1, n):
                series1 = self.data[tickers[i]]
                series2 = self.data[tickers[j]]
                score, p_value = self.calculate_coint(series1, series2)
                if p_value < 0.01:  # Usando um nível de significância de 1%
                    pairs.append((tickers[i], tickers[j], score, p_value))
        return pairs

class PairTradingAnalyzer(DataAnalyzer):
    def __init__(self, data):
        super().__init__(data)
        self.regression_results = {}
        self.pair_metrics = {}

    def check_stationarity(self, series):
        adf_test = self.perform_adfuller(series)
        return adf_test[1] < 0.01 

    def analyze_pair(self, ticker1, ticker2):
        series1 = self.data[ticker1]
        series2 = self.data[ticker2]
        ols_result = sm.OLS(series1, sm.add_constant(series2)).fit()
        self.regression_results[(ticker1, ticker2)] = ols_result
        residuals = ols_result.resid
        return self.check_stationarity(residuals)

    def find_stationary_pairs(self, pairs):
        stationary_pairs = []
        for ticker1, ticker2, score, p_value in pairs:
            if self.analyze_pair(ticker1, ticker2):
                stationary_pairs.append((ticker1, ticker2, score, p_value))
        return stationary_pairs

    def normaliza_zscore(self, series):
        return (series - series.mean()) / np.std(series)

    def analyze_selected_pairs(self, pairs):
        selected_pairs = []
        for pair in pairs:
            ticker1, ticker2, score, p_value = pair
            ols_result = self.regression_results[(ticker1, ticker2)]
            beta0 = ols_result.params[0]
            beta = ols_result.params[1]
            residuo = ols_result.resid
            residuo_norm = self.normaliza_zscore(residuo)

            # Meia-vida
            half_life = np.round(-np.log(2) / np.log(acf(residuo_norm.dropna(), alpha=0.05, nlags=1)[0][1]), 2)

            # Estacionariedade
            adf_diff1 = self.perform_adfuller(residuo_norm)

            # Condições para seleção
            if beta > 0 and adf_diff1[4]['5%'] < -2.8 and adf_diff1[1] < 0.01 and half_life < 2.0:
                selected_pairs.append((ticker1, ticker2, score, p_value))

        return selected_pairs
    
    # def store_pair_metrics(self, pairs):
    #     self.pair_metrics = {}
    #     for ticker1, ticker2, score, p_value in pairs:
    #         if self.analyze_pair(ticker1, ticker2):
    #             ols_result = self.regression_results[(ticker1, ticker2)]
    #             residuals = ols_result.resid
    #             adf_test = self.perform_adfuller(residuals)
    #             beta = ols_result.params[1]
    #             half_life = np.round(-np.log(2) / np.log(acf(self.normaliza_zscore(residuals).dropna(), alpha=0.05, nlags=1)[0][1]), 2)
    #             correlation = self.data[ticker1].corr(self.data[ticker2])
    #             self.pair_metrics[(ticker1, ticker2)] = {
    #                 "score": score,
    #                 "p_value": p_value,
    #                 "adf_p_value": adf_test[1],
    #                 "beta": beta,
    #                 "half_life": half_life,
    #                 "correlation": correlation
    #             } 

    def calculate_and_store_pair_metrics(self, pairs):
        self.pair_metrics = {}
        for ticker1, ticker2, score, p_value in pairs:
            y = self.data[ticker1]
            X = sm.add_constant(self.data[ticker2])
            model = sm.OLS(y, X).fit()
            residuals = model.resid
            
            adf_test = adfuller(residuals)
            half_life = np.round(-np.log(2) / np.log(acf(residuals, nlags=1)[1]), 2)
            
            self.pair_metrics[(ticker1, ticker2)] = {
                "score": score,
                "p_value": p_value,
                "adf_p_value": adf_test[1],
                "beta": model.params[1],
                "alpha": model.params[0],
                "residuals_mean": residuals.mean(),
                "residuals_std": residuals.std(),
                "half_life": half_life
            }
        
        return self.pair_metrics