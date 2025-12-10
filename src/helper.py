import pandas as pd

def earnings(earn_df, feat):
    feat.index = feat.index.normalize()
    earn_df["Earnings Date"] = pd.to_datetime(earn_df["Earnings Date"]).dt.normalize()
    earn_df = earn_df.sort_values("Earnings Date")

# Compute EPS Surprise

# Forward-fill EPS numbers only *after* earnings, then freeze until next release
# Create a daily index version of earnings
    earn_daily = pd.DataFrame(index=feat.index)
    earn_daily = earn_daily.join(earn_df.set_index("Earnings Date"), how="left")

# Forward-fill ONLY past EPS data (safe, no leakage)
    earn_daily["Reported EPS"]      = earn_daily["Reported EPS"].ffill()
    earn_daily["EPS Estimate"]         = earn_daily["EPS Estimate"].ffill()
    earn_daily["Surprise(%)"]    = earn_daily["Surprise(%)"].ffill()

# Key ML-ready features:
    feat["eps_actual"]      = earn_daily["Reported EPS"]
    feat["eps_est"]         = earn_daily["EPS Estimate"]
    feat["eps_surprise"] = earn_daily["Surprise(%)"]

# Indicator for positive / negative surprise
    feat["eps_positive"] = (feat["eps_surprise"] > 0).astype(int)
    feat["eps_negative"] = (feat["eps_surprise"] < 0).astype(int)

    feat.fillna({"eps_actual":-999}, inplace=True)
    feat.fillna({"eps_est":-999}, inplace=True)
    feat.fillna({"eps_surprise":-999}, inplace=True)

    feat.fillna({"eps_positive":0}, inplace=True)
    feat.fillna({"eps_negative":0}, inplace=True)
    return feat

