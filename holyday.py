# %%
import numpy as np
import pandas as pd

# %%
ep = pd.read_csv("data/ele_power_data.csv")
wd = pd.read_csv("data/wether_data.csv")
df = pd.concat([wd, ep], axis=1)
df = df.drop(columns="DATE")

# %%
train = df.loc[:,['平均気温', '実績最大電力(万kW)', '休日']]

# %%
train = np.array(train)

# %%
train_x = train[:, 0:2]

# %%
train_y = train[:, 2]

# %%
import matplotlib.pyplot as plt

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
plt.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1], 'o')
plt.plot(train_x[train_y == 0, 0], train_x[train_y == 0, 1], 'x')
plt.show()

# %%
theta = np.random.rand(4)

# %%
mu = train_x.mean(axis=0)

# %%
sigma = train_x.std(axis=0)

# %%
def standardize(x):
    return (x - mu) / sigma

# %%
train_z = standardize(train_x)

# %%
def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])
    x3 = x[:, 0, np.newaxis] ** 2
    return np.hstack([x0, x, x3])

# %%
X = to_matrix(train_z)

# %%
def f(x):
    return 1 / (1 + np.exp(-np.dot(x, theta)))

# %%
ETA = 1e-3

# %%
epoch = 5000

# %%
for _ in range(epoch):
    theta = theta - ETA * np.dot(f(X) - train_y, X)

# %%
x1 = np.linspace(-2, 2, 100)

# %%
x2 = -(theta[0] + theta[1] * x1 + theta[3] * x1 ** 2) / theta[2]

# %%
plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x')
plt.plot(x1, x2, color = 'red', linestyle='-')
plt.show()

# %%
