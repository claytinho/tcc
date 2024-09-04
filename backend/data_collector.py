import yfinance as yf
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self, tickers, start_date=None, end_date=datetime.today()):
        self.tickers = tickers
        self.start_date = start_date if start_date else datetime.today() - timedelta(days=3*365)
        self.end_date = end_date
        self.data = None

    def fetch_data(self):
        self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']
        return self.data

    def preprocess_data(self):
        # Drop NaN values and remove any columns (tickers) that are empty after this
        self.data = self.data.dropna(axis=1, how='all')  # Remove colunas com todos os valores NaN
        return self.data

# # Lista de tickers para testar
# tickers = ['AAPL', 'MSFT']

# # Criando uma instância da classe DataLoader sem fornecer o start_date
# data_loader = DataLoader(tickers=tickers)

# # Imprimindo a data inicial para verificar se está correta
# print(f"Start Date: {data_loader.start_date}")

# # Buscando e processando os dados
# data = data_loader.fetch_data()
# processed_data = data_loader.preprocess_data()

# # Imprimindo os primeiros registros para verificar
# print(processed_data.head())