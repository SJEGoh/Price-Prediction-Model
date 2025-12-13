from helper import load_data, test
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def main():
    # BAC 0.95, 10 years
    ticker = "AMD"
    split = 0.975
    t = yf.Ticker(ticker)
    df = t.history(period = "20y", auto_adjust=True)
    # walk forward validation 
    l = len(df)
    count = 0
    model = pd.DataFrame(columns = ["JS", "Wass", "RMSE"])
    min_JS = float('inf')
    best_paths = None
    best_price = None
    gbm = pd.DataFrame(columns = ["JS", "Wass", "RMSE"])
    garch = pd.DataFrame(columns = ["JS", "Wass", "RMSE"])
    for i in range(0, l//2, int(l * (1 - split))):
        count += 1
        train_df = df.iloc[i:i + l//2]
        train_price, test_price, log_train_ser, log_test_ser, df_train = load_data(train_df, split=split)
        model_metrics, gbm_metrics, garch_metrics, paths, test_price = test(train_price, test_price, log_train_ser, log_test_ser, df_train, t)
        model.loc[len(model)] = model_metrics
        gbm.loc[len(gbm)] = gbm_metrics
        garch.loc[len(garch)] = garch_metrics
        if model_metrics[0] < min_JS:
            min_JS = model_metrics[0]
            best_paths = paths
            best_price = test_price
    
    print(model)
if __name__ == "__main__":
    main()
