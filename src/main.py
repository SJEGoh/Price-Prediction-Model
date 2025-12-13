from helper import load_data, test, metrics_summary
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def main():
    # BAC 0.95, 10 years
    ticker = "GME"
    split = 0.99
    t = yf.Ticker(ticker)
    df = t.history(period = "20y", auto_adjust=True)
    # walk forward validation 
    l = len(df)
    max_JS = float('-inf')
    worst_paths = None
    worst_price = None
    min_JS = float('inf')
    best_paths = None
    best_price = None
    model = pd.DataFrame(columns = ["JS_med", "JS_dist", "Wass", "RMSE"])
    gbm = pd.DataFrame(columns = ["JS_med", "JS_dist", "Wass", "RMSE"])
    garch = pd.DataFrame(columns = ["JS_med", "JS_dist", "Wass", "RMSE"])
    for i in range(0, l//2, int(l * (1 - split))):
        window_df = df.iloc[i:i + l//2]
        print(f"Train period: {window_df.index[0]} - {window_df.index[-int((l * (1 - split)))]}")
        print(f"Test period: {window_df.index[1-int((l * (1 - split)))]} - {window_df.index[-1]}\n")

        train_price, test_price, log_train_ser, log_test_ser, df_train = load_data(window_df, split=split)
        model_metrics, gbm_metrics, garch_metrics, paths, test_price = test(train_price, test_price, log_train_ser, log_test_ser, df_train, t)
        model.loc[len(model)] = model_metrics
        gbm.loc[len(gbm)] = gbm_metrics
        garch.loc[len(garch)] = garch_metrics
        if model_metrics[0] < min_JS:
            min_JS = model_metrics[0]
            best_paths = paths
            best_price = test_price
        if model_metrics[0] > max_JS:
            max_JS = model_metrics[0]
            worst_paths = paths
            worst_price = test_price
    print("Model Metrics:")
    metrics_summary(model)
    print("GBM Metrics:")
    metrics_summary(gbm)
    print("GARCH Metrics:")
    metrics_summary(garch)
    plt.figure(figsize = (12, 6))
    plt.plot(best_price.index, best_price.values, color = "black", label = "Actual")
    plt.plot(best_price.index, best_paths[0], color = "blue", label = "Model")
    plt.plot(best_price.index, best_paths[1], color = "red", linestyle = "--", label = "GBM", alpha = 0.5)
    plt.plot(best_price.index, best_paths[2], color = "green", linestyle = ":", label = "GARCH", alpha = 0.5)
    plt.legend()
    plt.show()

    plt.figure(figsize = (12, 6))
    plt.plot(worst_price.index, worst_price.values, color = "black", label = "Actual")
    plt.plot(worst_price.index, worst_paths[0], color = "blue", label = "Model")
    plt.plot(worst_price.index, worst_paths[1], color = "red", linestyle = "--", label = "GBM", alpha = 0.5)
    plt.plot(worst_price.index, worst_paths[2], color = "green", linestyle = ":", label = "GARCH", alpha = 0.5)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
