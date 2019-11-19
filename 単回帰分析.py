# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

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
# 単回帰分析
def regression(x_name, y_name="Actual maximum power[10^5kW]"):
    x_col_name = x_name
    X = df.loc[:, [x_col_name]].values  # 説明変数
    y_col_name = y_name
    y = df.loc[:, [y_col_name]].values  # 目的変数

    # 変数変換
    quad = PolynomialFeatures(degree=2)
    X_quad = quad.fit_transform(X)
    # cubic = PolynomialFeatures(degree=3)
    # X_cubic = cubic.fit_transform(X)

    # 学習
    lr = LinearRegression()  # linear 次元1
    lr.fit(X, y)
    lr_quad = LinearRegression()  # linear 次元2
    lr_quad.fit(X_quad, y)
    # lr_cubic = LinearRegression()  # linear 次元3
    # lr_cubic.fit(X_cubic, y)

    # プロット
    plt.scatter(X, y, color="gray", label="data")
    plt.plot(X, lr.predict(X), color="red", label="linear")
    plt.plot(X, lr_quad.predict(X_quad), color="blue", label="quad")
    # plt.plot(X, lr_cubic.predict(X_cubic), color="green", label="cubic")
    plt.legend(loc="upper right")
    plt.xlabel(x_col_name)
    plt.ylabel(y_col_name)
    plt.show()


# %%
# 単回帰分析を実行
for x in ["Local pressure", "Sea level pressure",
          "Average temperature", "Highest temperature",
          "Lowest Temperature", "Average humidity",
          "Minimum humidity", "Average wind speed",
          "Maximum wind speed", "Maximum instantaneous wind speed",
          "Sunshine hours"]:
    regression(x_name=x)


# %%
