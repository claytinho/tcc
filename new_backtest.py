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
    return adf_result[1]  # P-value


# Função para calcular as métricas estatísticas para um par de ativos
def calculate_statistics(series1, series2):
    # Teste de cointegração
    score, p_value, _ = coint(series1, series2)

    # Regressão OLS
    X = sm.add_constant(series2)
    ols_result = sm.OLS(series1, X).fit()
    alpha, beta = ols_result.params
    residuals = ols_result.resid

    # ADF para os resíduos
    adf_test = test_stationarity(residuals)
    adf = adf_test  # P-value do ADF


    # Teste de Estacionariedade nas Séries Individuais
    adf_series1 = test_stationarity(series1)
    adf_series2 = test_stationarity(series2)

    # Half-life dos resíduos
    lag_acf = acf(residuals, nlags=40)
    half_life = np.round(-np.log(2) / np.log(lag_acf[1]), 2)

    # Teste de Heterocedasticidade (Breusch-Pagan)
    lm_stat, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(residuals, X)

    # Teste de Autocorrelação dos Resíduos (Durbin-Watson)
    dw_stat = durbin_watson(residuals)

    # Cálculos dos resíduos
    residuals_normalized = (residuals - np.mean(residuals)) / np.std(residuals)
    last_residual_normalized = residuals_normalized[-1]

    return {
        'score': score,
        'p_value': p_value,
        'residual': last_residual_normalized,
        'adf_residual': adf,
        'beta': beta,
        'alpha': alpha,
        'half_life': half_life,
        'heteroscedasticity_pvalue': lm_pvalue,
        'durbin_watson_stat': dw_stat,
        'adf_series1': adf_series1,
        'adf_series2': adf_series2
    }

# Função principal para rodar os cálculos diariamente
def run_daily_backtest(tickers, start_date, end_date, window_size=60, start_calc_day=90):
    data = fetch_data(tickers, start_date, end_date)
    results = []
    
    # Calcular a data de início após 90 dias úteis
    calc_start_date = pd.to_datetime(start_date) + pd.offsets.BDay(start_calc_day)

    for current_date in data.index:
        # Apenas começar após 90 dias úteis
        if current_date < calc_start_date:
            continue

        print(f"Processando data: {current_date}")
        day_data = {'date': current_date}

        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                ticker1, ticker2 = tickers[i], tickers[j]

                # Selecionar os últimos 60 dias úteis de dados para cada ativo
                start_window = current_date - pd.offsets.BDay(window_size)
                series1 = data[ticker1].loc[start_window:current_date].dropna()
                series2 = data[ticker2].loc[start_window:current_date].dropna()

                print(f"Verificando {ticker1} e {ticker2} com dados de {len(series1)} dias e {len(series2)} dias")

                # Realizar os cálculos apenas se houver dados suficientes
                if len(series1) >= window_size and len(series2) >= window_size:
                    stats = calculate_statistics(series1, series2)
                    day_data.update({
                        'ticker1': ticker1,
                        'ticker2': ticker2,
                        'price_ticker1': data[ticker1].loc[current_date],
                        'price_ticker2': data[ticker2].loc[current_date],
                        **stats
                    })
                    print(f"Estatísticas calculadas para {ticker1} e {ticker2} na data {current_date}")
                else:
                    print(f"Dados insuficientes para {ticker1} e {ticker2} na data {current_date}")

        if len(day_data) > 1:
            results.append(day_data)

    df_results = pd.DataFrame(results)
    return df_results

# Parâmetros de exemplo
tickers_list = ["VALE3.SA", "ABEV3.SA"]
start_date = "2022-01-01"
end_date = "2024-09-30"

def check_trading_conditions(row):
    # Definir as condições para o trading
    coint_condition = row['p_value'] < 0.10  # Cointegração significativa
    adf_condition = row['adf_residual'] < 0.10  # Estacionariedade dos resíduos
    residual_condition = abs(row['residual']) > 1.5  # Resíduo normalizado fora da faixa [-2, 2]

    # Se todas as condições forem atendidas, marcar como OK para trading
    if coint_condition and adf_condition and residual_condition:
        return 1  # Sinal de trading "OK"
    else:
        return 0  # Não é um bom dia para trading


# Executar a função de backtest
df_daily_statistics = run_daily_backtest(tickers_list, start_date, end_date, window_size=90, start_calc_day=120)

# Aplicar a função ao DataFrame
df_daily_statistics['trade_signal'] = df_daily_statistics.apply(check_trading_conditions, axis=1)


# Definir o total a ser investido em cada operação
total_investment = 10000  # Exemplo de $10.000 para cada par

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

# Inicializar colunas para quantidade de ações e retorno
df_daily_statistics['qty_ticker1'] = np.nan  # Quantidade de ações compradas (long)
df_daily_statistics['qty_ticker2'] = np.nan  # Quantidade de ações vendidas (short)
df_daily_statistics['trade_return'] = np.nan  # Retorno financeiro

# Flag para controlar se uma operação está aberta
trade_open = False
entry_day = None  # Dia de entrada na operação
last_residual = None  # Armazenar o último resíduo para monitorar o cruzamento do zero

# Iterar sobre o DataFrame para identificar os dias de trading e calcular as quantidades e retorno
for i in range(len(df_daily_statistics)):
    row = df_daily_statistics.iloc[i]

    # Verificar se o sinal de trading está ativo, o beta é positivo e não há operação aberta
    if row['trade_signal'] == 1 and row['beta'] > 0 and not trade_open:
        # Calcular a quantidade de ações com base no preço e beta
        qty_long, qty_short = calculate_trade_volumes(row['price_ticker1'], row['price_ticker2'], row['beta'], total_investment)
        
        # Registrar as quantidades no DataFrame
        df_daily_statistics.at[i, 'qty_ticker1'] = qty_long
        df_daily_statistics.at[i, 'qty_ticker2'] = qty_short
        
        # Marcar que a operação foi aberta
        trade_open = True
        entry_day = i  # Armazenar o índice do dia de entrada
        last_residual = row['residual']  # Registrar o resíduo de entrada

    # Se uma operação estiver aberta, verificar o cruzamento do zero
    elif trade_open:
        current_residual = row['residual']

        # Se o resíduo cruzar o zero, fechar a operação
        if (last_residual > 0 and current_residual < 0) or (last_residual < 0 and current_residual > 0):
            # Recuperar os preços e quantidades da entrada da operação
            entry_price_long = df_daily_statistics.at[entry_day, 'price_ticker1']
            entry_price_short = df_daily_statistics.at[entry_day, 'price_ticker2']
            qty_long = df_daily_statistics.at[entry_day, 'qty_ticker1']
            qty_short = df_daily_statistics.at[entry_day, 'qty_ticker2']

            # Calcular o retorno financeiro com base nos preços de entrada e saída
            trade_return = calculate_return(
                entry_price_long, row['price_ticker1'], qty_long,  # Preços e quantidades do long
                entry_price_short, row['price_ticker2'], qty_short  # Preços e quantidades do short
            )

            # Registrar o retorno no dia de fechamento da operação
            df_daily_statistics.at[i, 'trade_return'] = trade_return

            # Marcar que a operação foi fechada
            trade_open = False
            entry_day = None  # Resetar o dia de entrada

        # Atualizar o último resíduo
        last_residual = current_residual

# Exibir os resultados
print(df_daily_statistics[['date', 'ticker1', 'ticker2', 'price_ticker1', 'price_ticker2', 'qty_ticker1', 'qty_ticker2', 'trade_return']])



# Salvar os resultados em CSV se houver resultados
if not df_daily_statistics.empty:
    df_daily_statistics.to_csv('daily_statistics.csv', index=False, decimal=',', sep=';')
    print("Resultados salvos em daily_statistics.csv")
else:
    print("Nenhum resultado para salvar!")
