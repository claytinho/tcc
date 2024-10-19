import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from joblib import Parallel, delayed

# Função para coletar os dados dos ativos
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    data = data.reindex(all_dates).interpolate()
    return data

def rolling_ols_stats(df, window_size=120):
    df['beta'] = np.nan
    df['residual'] = np.nan

    for _, pair in df[['ticker1', 'ticker2']].drop_duplicates().iterrows():
        ticker1 = pair['ticker1']
        ticker2 = pair['ticker2']

        # Filtrar os dados para o par específico
        pair_data = df[(df['ticker1'] == ticker1) & (df['ticker2'] == ticker2)].copy()
        pair_data = pair_data.sort_values('date')

        # Regressão OLS móvel (rolling)
        exog = sm.add_constant(pair_data['price_ticker2'])
        endog = pair_data['price_ticker1']
        rolling_model = RollingOLS(endog=endog, exog=exog, window=window_size)
        rolling_results = rolling_model.fit()

        # Obter os parâmetros estimados (intercepto e beta)
        params = rolling_results.params

        # Calculando manualmente os valores previstos: intercepto + beta * exog
        predicted_values = params['const'] + params['price_ticker2'] * pair_data['price_ticker2']

        # Calculando os resíduos (residuals)
        residuals = endog - predicted_values

        # Atualizando o DataFrame com os betas e os resíduos
        df.loc[(df['ticker1'] == ticker1) & (df['ticker2'] == ticker2), 'beta'] = params['price_ticker2']
        df.loc[(df['ticker1'] == ticker1) & (df['ticker2'] == ticker2), 'residual'] = residuals

    return df

# Função para verificar as condições de entrada dinâmicas com base na volatilidade
def dynamic_entry_exit_conditions(row, residual_volatility, k_factor=1.5):
    residual_condition = abs(row['residual']) > k_factor * residual_volatility
    return residual_condition

# Função para aplicar as condições dinâmicas de trading
def check_dynamic_trading_conditions(df, window_size=60, k_factor=1.5):
    for _, pair in df[['ticker1', 'ticker2']].drop_duplicates().iterrows():
        ticker1 = pair['ticker1']
        ticker2 = pair['ticker2']

        pair_data = df[(df['ticker1'] == ticker1) & (df['ticker2'] == ticker2)].copy()
        pair_data = pair_data.sort_values('date')

        pair_data['residual_volatility'] = pair_data['residual'].rolling(window=window_size).std()

        for i, row in pair_data.iterrows():
            if not np.isnan(row['residual_volatility']):
                df.loc[i, 'trade_signal'] = dynamic_entry_exit_conditions(row, row['residual_volatility'], k_factor=k_factor)

    return df

# Função principal para rodar os cálculos diariamente
def run_daily_backtest_parallel(tickers, start_date, end_date, window_size, pairs=None):
    data = fetch_data(tickers, start_date, end_date)

    if pairs is None:
        raise ValueError("Nenhuma lista de pares fornecida.")
    else:
        tickers_in_pairs = set([t for pair in pairs for t in pair])
        missing_tickers = tickers_in_pairs - set(data.columns)
        if missing_tickers:
            raise ValueError(f"Tickers não encontrados nos dados: {missing_tickers}")

    results = []

    # calc_start_date = pd.to_datetime(start_date) + pd.offsets.BDay(start_calc_day)
    dates = data.index[data.index >= pd.to_datetime(start_date)]

    def process_date_pair(current_date, ticker_pair):
        ticker1, ticker2 = ticker_pair
        start_window = current_date - pd.offsets.BDay(window_size)
        series1 = data[ticker1].loc[start_window:current_date].dropna()
        series2 = data[ticker2].loc[start_window:current_date].dropna()

        if len(series1) >= window_size and len(series2) >= window_size:
            result = {
                'date': current_date,
                'ticker1': ticker1,
                'ticker2': ticker2,
                'price_ticker1': data[ticker1].loc[current_date],
                'price_ticker2': data[ticker2].loc[current_date]
            }
            return result
        else:
            return None

    for current_date in dates:
        date_results = Parallel(n_jobs=-1)(
            delayed(process_date_pair)(current_date, pair) for pair in pairs
        )
        date_results = [res for res in date_results if res is not None]
        results.extend(date_results)

    df_results = pd.DataFrame(results)
    return df_results

# Função para calcular a quantidade de ações compradas e vendidas com ajuste no beta
def calculate_trade_volumes(price_long, price_short, beta, total_investment):
    beta_adjusted = max(-2.0, min(beta, 2.0))

    investment_per_asset = total_investment / 2
    qty_long = investment_per_asset / price_long
    qty_short = (investment_per_asset * beta_adjusted) / price_short
    return qty_long, qty_short

# Função para calcular o retorno financeiro
def calculate_return(entry_price_long, current_price_long, qty_long, entry_price_short, current_price_short, qty_short):
    long_return = (current_price_long - entry_price_long) * qty_long
    short_return = (entry_price_short - current_price_short) * qty_short
    return long_return + short_return

# Função para aplicar as operações e calcular os retornos financeiros
def apply_trading(df, total_investment=10000):
    open_trades = {}

    df['qty_ticker1'] = np.nan
    df['qty_ticker2'] = np.nan
    df['trade_return'] = np.nan
    df['daily_trade_return'] = 0.0

    pairs_df = df[['ticker1', 'ticker2']].drop_duplicates()

    for _, pair in pairs_df.iterrows():
        ticker1 = pair['ticker1']
        ticker2 = pair['ticker2']
        pair_data = df[(df['ticker1'] == ticker1) & (df['ticker2'] == ticker2)].copy().sort_values('date').reset_index(drop=True)

        for i in range(len(pair_data)):
            row = pair_data.iloc[i]
            current_date = row['date']
            idx = df[(df['date'] == current_date) & (df['ticker1'] == ticker1) & (df['ticker2'] == ticker2)].index

            trade_info = open_trades.get((ticker1, ticker2), None)

            if trade_info is None and row['trade_signal'] == 1 and row['beta'] > 0:
                qty_long, qty_short = calculate_trade_volumes(row['price_ticker1'], row['price_ticker2'], row['beta'], total_investment)
                df.loc[idx, 'qty_ticker1'] = qty_long
                df.loc[idx, 'qty_ticker2'] = qty_short

                open_trades[(ticker1, ticker2)] = {
                    'entry_price_long': row['price_ticker1'],
                    'entry_price_short': row['price_ticker2'],
                    'qty_long': qty_long,
                    'qty_short': qty_short,
                    'entry_residual': row['residual'],
                    'entry_date': current_date
                }
            elif trade_info is not None:
                current_price_long = row['price_ticker1']
                current_price_short = row['price_ticker2']
                unrealized_return = calculate_return(
                    trade_info['entry_price_long'], current_price_long, trade_info['qty_long'],
                    trade_info['entry_price_short'], current_price_short, trade_info['qty_short']
                )
                df.loc[idx, 'daily_trade_return'] = unrealized_return

                if trade_info['entry_residual'] * row['residual'] < 0:
                    df.loc[idx, 'trade_return'] = unrealized_return
                    del open_trades[(ticker1, ticker2)]
            else:
                df.loc[idx, 'daily_trade_return'] = 0.0

    return df

# Código principal
tickers_list = [
    'ABEV3.SA', 'B3SA3.SA', 'BBAS3.SA', 'BBDC4.SA', 'BBSE3.SA', 'BPAC11.SA', 'CMIG4.SA', 'ELET3.SA',
    'EMBR3.SA', 'ENEV3.SA', 'EQTL3.SA', 'ITUB4.SA', 'PRIO3.SA', 'RADL3.SA', 'RAIL3.SA', 'RDOR3.SA',
    'SUZB3.SA', 'BRFS3.SA', 'GGBR4.SA', 'RENT3.SA', 'VALE3.SA', 'PETR3.SA', 'WEGE3.SA', 'SBSP3.SA',
    'JBSS3.SA', 'VBBR3.SA', 'PETR4.SA'
]

pairs = [
    ('B3SA3.SA', 'BBDC4.SA'), ('B3SA3.SA', 'RENT3.SA'), ('BBAS3.SA', 'EQTL3.SA'),
    ('BBAS3.SA', 'PETR3.SA'), ('BBAS3.SA', 'PETR4.SA'), ('BBDC4.SA', 'ENEV3.SA'),
    ('BBSE3.SA', 'PRIO3.SA'), ('BPAC11.SA', 'RADL3.SA'), ('BPAC11.SA', 'RAIL3.SA'),
    ('ENEV3.SA', 'PRIO3.SA'), ('ENEV3.SA', 'RDOR3.SA'), ('EQTL3.SA', 'PRIO3.SA'),
    ('RADL3.SA', 'RAIL3.SA')
]

start_date = "2019-01-01"
end_date = "2024-09-30"

# Passo 1: Executar a função de backtest
df_daily_statistics = run_daily_backtest_parallel(
    tickers_list, start_date, end_date, window_size=120, pairs=pairs
)

# Passo 2: Calcular estatísticas usando Rolling OLS
df_daily_statistics = rolling_ols_stats(df_daily_statistics, window_size=120)

# Passo 3: Definir condições dinâmicas de entrada/saída com volatilidade ajustada
df_daily_statistics = check_dynamic_trading_conditions(df_daily_statistics, window_size=60, k_factor=1.5)

# Passo 4: Aplicar operações de trading e calcular os retornos financeiros
df_daily_statistics = apply_trading(df_daily_statistics, total_investment=10000)

# Salvar os resultados em CSV se houver resultados
if not df_daily_statistics.empty:
    df_daily_statistics.to_csv('daily_statistics.csv', index=False, decimal=',', sep=';')
    print("Resultados salvos em daily_statistics.csv")
else:
    print("Nenhum resultado para salvar!")
