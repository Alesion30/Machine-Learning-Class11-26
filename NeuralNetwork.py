from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pathlib

# 参考WEBサイト
# https://www.tensorflow.org/tutorials/keras/regression?hl=ja

# CSVを読み取りデータセットを作成する
def read_dataset():
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
    df = df.replace({'Precipitation amount': {"--": "0.0"}}).replace("3.8 )", "3.8").replace(
        "2.3 )", "2.3").replace("1.7 )", "1.7").replace("4.6 )", "4.6").replace("6.2 )", "6.2")
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)

    # Holidayはカテゴリーなのでワンホットエンコーディングを行う
    is_holiday = df.pop('Holiday')
    df['Weekdays'] = (is_holiday == 0)*1.0
    df['Holidays'] = (is_holiday == 1)*1.0

    return df

def build_model():
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
        optimizer=optimizer,
        metrics=['mae', 'mse'])
    return model

if __name__ == '__main__':
    dataset = read_dataset()
    date = dataset.pop('Date')

    # 訓練データとテストデータに分ける。訓練: 8割  テスト: 2割
    train_dataset = dataset.sample(frac=0.8,random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    '''
    sns.pairplot(train_dataset[["Average temperature", "Peak supply capacity[10^5kW]", "Local pressure"]], diag_kind="kde")
    plt.show()
    '''

    # ラベルと特徴量の分離
    train_labels = train_dataset.pop('Peak supply capacity[10^5kW]')
    test_labels = test_dataset.pop('Peak supply capacity[10^5kW]')

    model = build_model()

    example_batch = train_dataset[:10]
    example_result = model.predict(example_batch)
    print(example_result)

    print('finish')