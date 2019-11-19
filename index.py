# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# %%
ep = pd.read_csv("data/ele_power_data.csv")
wd = pd.read_csv("data/wether_data.csv")
df = pd.concat([wd, ep], axis=1)
df = df.drop(columns="DATE")

# '日', '休日', '現地気圧', '海面気圧', '降水量', '平均気温', '最高気温', '最低気温', '平均湿度',
# '最小湿度', '平均風速', '最大風速', '最大瞬間風速', '日照時間', 'DATE', 'ピーク時供給力(万kW)',
# '予想最大電力(万kW)', '予想使用率(%)', '実績最大電力(万kW)', '実績最大電力発生時間帯'

col = [
    "Date", "Holiday", "Local pressure",
    "Sea level pressure", "Precipitation amount",
    "Average temperature", "Highest temperature",
    "Lowest Temperature", "Average humidity",
    "Minimum humidity", "Average wind speed",
    "Maximum wind speed", "Maximum instantaneous wind speed",
    "Sunshine hours", "Peak supply capacity[10^5kW]",
    "Expected maximum power[10^5kW]", "Expected usage[%]",
    "Actual maximum power[10^5kW]", "Actual maximum power generation time zone"
]
df = pd.DataFrame(df.values, columns=col)
df = df.replace({'Precipitation amount': {"--": None}})
df = df.replace("3.8 )", "3.8")
df = df.replace("2.3 )", "2.3")
df = df.replace("1.7 )", "1.7")
df = df.replace("4.6 )", "4.6")
df = df.replace("6.2 )", "6.2")
df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)
print(df.info())
df.head()

# %%
X = df.iloc[:, 2:14].values  # 説明変数
y_col_name = "Actual maximum power[10^5kW]"
y = df.loc[:, [y_col_name]].values  # 目的変数

# 説明変数、目的変数を標準化
ss = StandardScaler()
X_std = ss.fit_transform(X)
y_std = ss.fit_transform(y)

# %%
