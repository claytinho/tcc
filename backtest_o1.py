import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint, acf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

# Função para coletar os dados dos ativos
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Garantir que todas as datas úteis estejam presentes
    data = data.reindex(all_dates).interpolate()  # Reindexar para incluir todas as datas e interpolar valores faltantes
    return data

def test_stationarity(series):
    adf_result = adfuller(series)
    return adf_result  # P-value

# Função para calcular as métricas estatísticas para um par de ativos
def calculate_statistics(series1, series2):
    # Teste de cointegração
    coint_result = coint(series1, series2)
    p_value_coint = coint_result[1]
    coint_score = coint_result[0] < coint_result[2][1] and p_value_coint < 0.01 

    # Regressão OLS
    X = sm.add_constant(series2)
    ols_result = sm.OLS(series1, X).fit()
    alpha, beta = ols_result.params
    residuals = ols_result.resid

    # Teste de Estacionariedade
    adf_result = test_stationarity(residuals)
    p_value_adf = adf_result[1]
    adf_score = adf_result[0] <= adf_result[4]['5%'] and p_value_adf < 0.01
    
    # Teste de Estacionariedade nas Séries Individuais
    adf_series1 = test_stationarity(series1)
    adf_series2 = test_stationarity(series2)

    # Teste de Heterocedasticidade (Breusch-Pagan)
    lm_stat, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(residuals, X)

    # Teste de Autocorrelação dos Resíduos (Durbin-Watson)
    dw_stat = durbin_watson(residuals)

    # Cálculos dos resíduos
    residuals_normalized = (residuals - np.mean(residuals)) / np.std(residuals)
    last_residual_normalized = residuals_normalized.iloc[-1]

    # Half-life dos resíduos
    try:
        half_life = np.round(-np.log(2) / np.log(acf(last_residual_normalized, alpha=0.05, nlags=1)[0][1]), 2)
    except:
        half_life = np.nan

    return {
        'score_coint': coint_score,
        'p_value': p_value_coint,
        'score_adf': adf_score,
        'P_value_adf': p_value_adf,
        'residual': last_residual_normalized,
        'beta': beta,
        'half_life': half_life,
        'breusch': lm_pvalue
    }

# Função para verificar as condições de trading
def check_trading_conditions(row):
    # Definir as condições para o trading
    coint_condition = row['score_coint']  # Cointegração dos ativos
    adf_condition = row['score_adf']  # Estacionariedade dos resíduos 
    residual_condition = abs(row['residual']) > 1.5  # Resíduo normalizado fora da faixa [-1.5, 1.5]
    autocorrelation_condition = row['breusch']  < 0.05

    # Se todas as condições forem atendidas, marcar como OK para trading
    if coint_condition and adf_condition and residual_condition and autocorrelation_condition:
        return 1  # Sinal de trading "OK"
    else:
        return 0  # Não é um bom dia para trading

# Função para calcular a quantidade de ações compradas e vendidas
def calculate_trade_volumes(price_long, price_short, beta, total_investment):
    investment_per_asset = total_investment / 2
    qty_long = investment_per_asset / price_long
    qty_short = (investment_per_asset * beta) / price_short
    return qty_long, qty_short

# Função para calcular o retorno financeiro
def calculate_return(entry_price_long, exit_price_long, qty_long, entry_price_short, exit_price_short, qty_short):
    # Retorno no ativo long
    long_return = (exit_price_long - entry_price_long) * qty_long
    # Retorno no ativo short (note que você ganha quando o preço cai)
    short_return = (entry_price_short - exit_price_short) * qty_short
    # Retorno total
    return long_return + short_return

# Função principal para rodar os cálculos diariamente
def run_daily_backtest(tickers, start_date, end_date, window_size, start_calc_day):
    data = fetch_data(tickers, start_date, end_date)
    results = []

    # Calcular a data de início após 90 dias úteis
    calc_start_date = pd.to_datetime(start_date) + pd.offsets.BDay(start_calc_day)

    for current_date in data.index:
        # Apenas começar após 90 dias úteis
        if current_date < calc_start_date:
            continue

        print(f"Processando data: {current_date}")

        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                ticker1, ticker2 = tickers[i], tickers[j]

                # Selecionar os últimos 60 dias úteis de dados para cada ativo
                start_window = current_date - pd.offsets.BDay(window_size)
                series1 = data[ticker1].loc[start_window:current_date].dropna()
                series2 = data[ticker2].loc[start_window:current_date].dropna()

                # Realizar os cálculos apenas se houver dados suficientes
                if len(series1) >= window_size and len(series2) >= window_size:
                    stats = calculate_statistics(series1, series2)

                    result = {
                        'date': current_date,
                        'ticker1': ticker1,
                        'ticker2': ticker2,
                        'price_ticker1': data[ticker1].loc[current_date],
                        'price_ticker2': data[ticker2].loc[current_date],
                        **stats
                    }

                    results.append(result)
                    print(f"Estatísticas calculadas para {ticker1} e {ticker2} na data {current_date}")
                else:
                    print(f"Dados insuficientes para {ticker1} e {ticker2} na data {current_date}")

    df_results = pd.DataFrame(results)
    return df_results

# Código principal
tickers_list = [
    'ABEV3.SA', 'BBSE3.SA', 'BPAC11.SA', 'BRFS3.SA', 'BBDC4.SA',
    'ELET3.SA', 'EQTL3.SA', 'GGBR4.SA', 'PETR3.SA', 'PETR4.SA',
    'PRIO3.SA', 'RAIL3.SA', 'RADL3.SA', 'RENT3.SA', 'SUZB3.SA',
    'UGPA3.SA', 'VALE3.SA', 'VIVT3.SA'
]


start_date = "2021-01-01"
end_date = "2024-09-30"

# Executar a função de backtest
df_daily_statistics = run_daily_backtest(tickers_list, start_date, end_date, window_size=60, start_calc_day=120)

# Aplicar a função ao DataFrame
df_daily_statistics['trade_signal'] = df_daily_statistics.apply(check_trading_conditions, axis=1)

# Definir o total a ser investido em cada operação
total_investment = 10000  # Exemplo de R$10.000 para cada par

# Inicializar colunas para quantidade de ações e retorno
df_daily_statistics['qty_ticker1'] = np.nan  # Quantidade de ações compradas (long)
df_daily_statistics['qty_ticker2'] = np.nan  # Quantidade de ações vendidas (short)
df_daily_statistics['trade_return'] = np.nan  # Retorno financeiro

# Iterar sobre cada par separadamente
pairs = df_daily_statistics[['ticker1', 'ticker2']].drop_duplicates()

for _, pair in pairs.iterrows():
    ticker1 = pair['ticker1']
    ticker2 = pair['ticker2']
    pair_data = df_daily_statistics[(df_daily_statistics['ticker1'] == ticker1) & (df_daily_statistics['ticker2'] == ticker2)].copy()
    pair_data = pair_data.sort_values('date').reset_index(drop=True)

    trade_open = False
    entry_row = None  # Linha de entrada na operação
    entry_residual = None  # Resíduo no momento da entrada

    for i in range(len(pair_data)):
        row = pair_data.iloc[i]

        # Verificar se o sinal de trading está ativo, o beta é positivo e não há operação aberta
        if row['trade_signal'] == 1 and row['beta'] > 0 and not trade_open:
            # Calcular a quantidade de ações com base no preço e beta
            qty_long, qty_short = calculate_trade_volumes(row['price_ticker1'], row['price_ticker2'], row['beta'], total_investment)

            # Registrar as quantidades no DataFrame principal
            idx = df_daily_statistics[(df_daily_statistics['date'] == row['date']) &
                                      (df_daily_statistics['ticker1'] == ticker1) &
                                      (df_daily_statistics['ticker2'] == ticker2)].index
            if len(idx) == 1:
                df_daily_statistics.loc[idx, 'qty_ticker1'] = qty_long
                df_daily_statistics.loc[idx, 'qty_ticker2'] = qty_short
            else:
                print(f"Alerta: Mais de um índice encontrado para {ticker1} e {ticker2} na data {row['date']}")

            # Marcar que a operação foi aberta
            trade_open = True
            entry_row = row  # Armazenar a linha de entrada
            entry_residual = row['residual']  # Registrar o resíduo de entrada

            # Armazenar as quantidades e preços de entrada
            entry_qty_long = qty_long
            entry_qty_short = qty_short
            entry_price_long = row['price_ticker1']
            entry_price_short = row['price_ticker2']

            print(f"Operação aberta para {ticker1}/{ticker2} na data {row['date']}")
            print(f"Resíduo de entrada: {entry_residual}")

        # Se uma operação estiver aberta, verificar o cruzamento do zero em relação ao resíduo de entrada
        elif trade_open:
            current_residual = row['residual']

            # Se o resíduo cruzar o zero em relação ao resíduo de entrada, fechar a operação
            if entry_residual * current_residual < 0:
                # Recuperar os preços de saída
                exit_price_long = row['price_ticker1']
                exit_price_short = row['price_ticker2']

                # Calcular o retorno financeiro com base nos preços de entrada e saída
                trade_return = calculate_return(
                    entry_price_long, exit_price_long, entry_qty_long,
                    entry_price_short, exit_price_short, entry_qty_short
                )

                # Registrar o retorno no dia de fechamento da operação
                idx = df_daily_statistics[(df_daily_statistics['date'] == row['date']) &
                                          (df_daily_statistics['ticker1'] == ticker1) &
                                          (df_daily_statistics['ticker2'] == ticker2)].index
                df_daily_statistics.loc[idx, 'trade_return'] = trade_return

                # Marcar que a operação foi fechada
                trade_open = False
                entry_row = None  # Resetar a linha de entrada
                entry_residual = None  # Resetar o resíduo de entrada

                print(f"Operação fechada para {ticker1}/{ticker2} na data {row['date']}")
                print(f"Quantidades: Long - {entry_qty_long}, Short - {entry_qty_short}")
                print(f"Preços de entrada: Long - {entry_price_long}, Short - {entry_price_short}")
                print(f"Preços de saída: Long - {exit_price_long}, Short - {exit_price_short}")
                print(f"Retorno da operação: {trade_return}")

            # Não atualizar o entry_residual; mantemos o resíduo de entrada para comparação

# Exibir os resultados
print(df_daily_statistics[['date', 'ticker1', 'ticker2', 'price_ticker1', 'price_ticker2', 'qty_ticker1', 'qty_ticker2', 'trade_return']])

# Salvar os resultados em CSV se houver resultados
if not df_daily_statistics.empty:
    df_daily_statistics.to_csv('daily_statistics.csv', index=False, decimal=',', sep=';')
    print("Resultados salvos em daily_statistics.csv")
else:
    print("Nenhum resultado para salvar!")