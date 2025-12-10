import yfinance as yf

ticker = "AAPL"
df = yf.Ticker(ticker).history(period = "max")

print(df.columns)
