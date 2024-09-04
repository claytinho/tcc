import pandas as pd
import numpy as np
from scipy import stats

class BacktestAnalyzer:
    def __init__(self, data, trading_analyzer):
        self.data = data
        self.trading_analyzer = trading_analyzer
        self.bimonthly_analyses = {}
        self.position_status = {}
        self.active_pairs = set()  # Conjunto para armazenar pares ativos
        self.backtest_results = []

    def run_analysis(self, start_date, end_date):
        current_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        while current_date <= end_date:
            if self._is_bimonthly_update(current_date):
                self._perform_bimonthly_analysis(current_date)

            self._check_signals(current_date)
            current_date += pd.offsets.BDay(1)

        self._print_results()

    def _is_bimonthly_update(self, date):
        # Verifica se é o primeiro dia útil de um mês ímpar (janeiro, março, maio, etc.)
        return date.day == 1 and date.month % 2 == 1

    def _perform_bimonthly_analysis(self, current_date):
        print(f"Performing bimonthly analysis for date: {current_date}")
        end_date = current_date
        start_date = end_date - pd.Timedelta(days=120)  # Usando 120 dias (aproximadamente 4 meses) de dados
        data_slice = self.data.loc[start_date:end_date]
        
        self.trading_analyzer.data = data_slice
        selected_pairs = self.trading_analyzer.test_cointegrated_pairs(data_slice.columns)
        stationary_pairs = self.trading_analyzer.find_stationary_pairs(selected_pairs)
        
        self.bimonthly_analyses[current_date.strftime('%Y-%m')] = stationary_pairs

    def _check_signals(self, current_date):
        bimonthly_key = (current_date.replace(day=1) - pd.offsets.MonthBegin(1)).strftime('%Y-%m')
        current_pairs = set(tuple(pair[:2]) for pair in self.bimonthly_analyses.get(bimonthly_key, []))
        
        # Adicionar pares atuais ao conjunto de pares ativos
        self.active_pairs.update(current_pairs)
        
        # Verificar sinais para todos os pares ativos
        for ticker1, ticker2 in self.active_pairs:
            self._check_pair_signals(current_date, ticker1, ticker2, (ticker1, ticker2) in current_pairs)

    def _check_pair_signals(self, current_date, ticker1, ticker2, is_current):
        # Calcular o z-score para o dia atual
        lookback = 30  # Período para calcular o z-score
        end_date = current_date
        start_date = end_date - pd.Timedelta(days=lookback)
        
        data_slice = self.data.loc[start_date:end_date, [ticker1, ticker2]]
        spread = data_slice[ticker1] - data_slice[ticker2]
        zscore = (spread.iloc[-1] - spread.mean()) / spread.std()

        position_key = (ticker1, ticker2)
        position = self.position_status.get(position_key)

        entry = False
        exit = False
        maintain = False

        if position:
            if (position == "Long" and zscore > 0) or (position == "Short" and zscore < 0):
                exit = True
                del self.position_status[position_key]
            else:
                maintain = True
        elif is_current:  # Só permite novas entradas se o par passou nos testes estatísticos no período atual
            if zscore > 2.0:
                entry = True
                self.position_status[position_key] = "Short"
            elif zscore < -2.0:
                entry = True
                self.position_status[position_key] = "Long"

        self.backtest_results.append({
            'date': current_date,
            'pair': f"{ticker1}/{ticker2}",
            'zscore': zscore,
            'entry': entry,
            'maintain': maintain,
            'exit': exit
        })

    def _print_results(self):
        df = pd.DataFrame(self.backtest_results)
        df = df.pivot(index='date', columns='pair', values=['zscore', 'entry', 'maintain', 'exit'])
        df.columns = [f"{col[1]}_{col[0]}" for col in df.columns]
        print(df)
        
        # Salvar o DataFrame em um arquivo CSV
        csv_filename = 'backtest_results.csv'
        df.to_csv(csv_filename)
        print(f"\nResultados salvos em {csv_filename}")
