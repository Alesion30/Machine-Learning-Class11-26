import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from pandas.plotting import register_matplotlib_converters


def linear_regression(df, x_name, y_name):
    print("data: {}".format(x_name))
    print("target: {}\n".format(y_name))

    x_col_name = x_name
    X = df.loc[:, [x_col_name]].values  # 説明変数
    y_col_name = y_name
    y = df.loc[:, [y_col_name]].values  # 目的変数

    # 変数変換
    quad = PolynomialFeatures(degree=2)
    X_quad = quad.fit_transform(X)

    # 学習
    lr = LinearRegression()  # linear 次元1
    lr.fit(X, y)
    lr_quad = LinearRegression()  # quad 次元2
    lr_quad.fit(X_quad, y)

    # プロット
    plt.scatter(X, y, color="gray", label="data")
    plt.plot(X, lr.predict(X), color="red", label="linear")
    plt.plot(X, lr_quad.predict(X_quad), color="blue", label="quad")
    plt.legend(loc="upper right")
    plt.xlabel(x_col_name)
    plt.ylabel(y_col_name)
    plt.show()


if __name__ == "__main__":
    register_matplotlib_converters()

    # Load dataset
    ep = pd.read_csv("data/ele_power_data.csv")
    wd = pd.read_csv("data/wether_data.csv")
    df = pd.concat([wd, ep], axis=1)

    # Trim dataset
    df = df.drop(columns="DATE")
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
    df = df.replace({'Precipitation amount': {"--": None}}).replace("3.8 )", "3.8").replace(
        "2.3 )", "2.3").replace("1.7 )", "1.7").replace("4.6 )", "4.6").replace("6.2 )", "6.2")
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)

    # Run Linear Regression y="Actual maximum power[10^5kW]" 実績最大電力(万kW)
    feature_name = ["Average temperature",
                    "Highest temperature", "Lowest Temperature"]
    # feature_name = ["Local pressure", "Sea level pressure",
    #                 "Average temperature", "Highest temperature",
    #                 "Lowest Temperature", "Average humidity",
    #                 "Minimum humidity", "Average wind speed",
    #                 "Maximum wind speed", "Maximum instantaneous wind speed",
    #                 "Sunshine hours"]
    for x in feature_name:
        linear_regression(
            df=df, x_name=x, y_name="Actual maximum power[10^5kW]")
