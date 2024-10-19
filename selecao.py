# Lista de tickers
tickers_list = [
    "VALE3.SA", "PETR4.SA", "ITUB4.SA", "PETR3.SA", "BBAS3.SA", "ELET3.SA", "BBDC4.SA", "WEGE3.SA",
    "B3SA3.SA", "ITSA4.SA", "ABEV3.SA", "BPAC11.SA", "RENT3.SA", "JBSS3.SA", "EQTL3.SA", "PRIO3.SA",
    "RADL3.SA", "SUZB3.SA", "RDOR3.SA", "EMBR3.SA", "SBSP3.SA", "RAIL3.SA", "UGPA3.SA", "VBBR3.SA",
    "BBSE3.SA", "GGBR4.SA", "ENEV3.SA", "CMIG4.SA", "VIVT3.SA", "BRFS3.SA"
]

import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# Função para calcular a correlação dos pares e filtrar com base no limiar
def filter_by_correlation(data, pairs, threshold=0.6):
    corr_matrix = data.corr()
    filtered_pairs = [(x, y) for x, y in pairs if abs(corr_matrix.loc[x, y]) >= threshold]
    return filtered_pairs

# Função para realizar o teste de Engle-Granger para validar a cointegração
def engle_granger_test(series1, series2):
    X = sm.add_constant(series2)
    ols_result = sm.OLS(series1, X).fit()
    residuals = ols_result.resid
    adf_test = adfuller(residuals)
    return adf_test[1]  # Retorna o p-valor

# Período de 5 anos
start_date = "2019-01-01"
end_date = "2024-09-30"

# Coletar dados de fechamento ajustado
data = yf.download(tickers_list, start=start_date, end=end_date)['Adj Close'].dropna()

# Gerar todas as combinações possíveis de pares
pairs = list(combinations(data.columns, 2))

# Filtrar pares por correlação
pairs = filter_by_correlation(data, pairs, threshold=0.6)

# Lista para armazenar os resultados
johansen_results = []

# Parâmetros do teste
significance_level = 0.05
min_obs = 250

# Loop pelos pares
for pair in pairs:
    ticker1, ticker2 = pair
    df_pair = data[[ticker1, ticker2]].dropna()

    if len(df_pair) >= min_obs:
        result = coint_johansen(df_pair, det_order=0, k_ar_diff=1)
        trace_stat = result.lr1[0]
        crit_value = result.cvt[0, 1]

        if trace_stat > crit_value:
            p_value_eg = engle_granger_test(df_pair[ticker1], df_pair[ticker2])
            if p_value_eg < significance_level:
                johansen_results.append({
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'trace_stat': trace_stat,
                    'crit_value': crit_value,
                    'engle_granger_p_value': p_value_eg
                })

# Resultados finais
johansen_df = pd.DataFrame(johansen_results)
print(johansen_df)
