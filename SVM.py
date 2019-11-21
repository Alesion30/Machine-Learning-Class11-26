# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# %%
ep = pd.read_csv("data/ele_power_data.csv")
wd = pd.read_csv("data/wether_data.csv")
df = pd.concat([wd, ep], axis=1)
df = df.drop(columns="DATE")

# '日', '休日', '現地気圧', '海面気圧', '降水量', '平均気温', '最高気温', '最低気温', '平均湿度',
# '最小湿度', '平均風速', '最大風速', '最大瞬間風速', '日照時間', 'ピーク時供給力(万kW)',
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
week = df[df['Holiday'] == 0.0].loc[:, [
    "Average temperature", "Actual maximum power[10^5kW]"]]
holi = df[df['Holiday'] == 1.0].loc[:, [
    "Average temperature", "Actual maximum power[10^5kW]"]]
X_0 = week.values[:, 0]
y_0 = week.values[:, 1]
X_1 = holi.values[:, 0]
y_1 = holi.values[:, 1]

plt.scatter(X_0, y_0, color="blue", label="weekday")
plt.scatter(X_1, y_1, color="red", label="holiday")
plt.xlabel("Average temperature")
plt.ylabel("Actual maximum power[10^5kW]")
plt.legend(loc="lower right")
plt.show()

# %%
data = df.loc[:, ["Average temperature",
                  "Actual maximum power[10^5kW]", "Holiday"]]
X = data.iloc[:, [0, 1]].values
y = data.iloc[:, [2]].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)


# %%
# ２次元データの決定曲線をプロットする関数


def plot_decision_regions(X, y, classifier, resolution=0.02):
    from matplotlib.colors import ListedColormap
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
                    c=cmap(idx), marker=markers[idx], label=cl)
    plt.show()


# %%
rbf_svm = SVC(kernel='rbf', gamma=0.005, C=1.0)
rbf_svm.fit(X_train, y_train)


# %%
print(rbf_svm.score(X_test, y_test))
print(rbf_svm.score(X_train, y_train))

# %%
plot_decision_regions(X, y, classifier=rbf_svm)


# %%
