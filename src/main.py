from helper import load_data, test, metrics_summary
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
    min_JS = float('inf')
    best_paths = None
    best_price = None
    model = pd.DataFrame(columns = ["JS", "Wass", "RMSE"])
    gbm = pd.DataFrame(columns = ["JS", "Wass", "RMSE"])
    garch = pd.DataFrame(columns = ["JS", "Wass", "RMSE"])
    for i in range(0, l//2, int(l * (1 - split))):
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
    print("Model Metrics:")
    metrics_summary(model)
    print("GBM Metrics:")
    metrics_summary(gbm)
    print("GARCH Metrics:")
    metrics_summary(garch)
    plt.figure(figsize = (12, 6))
    plt.plot(best_price.index, best_price.values, color = "black", label = "Actual")
    plt.plot(best_price.index, best_paths[0], color = "blue", label = "Model")
    plt.plot(best_price.index, best_paths[1], color = "red", linestyle = "--", label = "GBM")
    plt.plot(best_price.index, best_paths[2], color = "green", linestyle = ":", label = "GARCH")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
