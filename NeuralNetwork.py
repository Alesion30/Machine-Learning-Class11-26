import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    df = df.replace({'Precipitation amount': {"--": None}}).replace("3.8 )", "3.8").replace(
        "2.3 )", "2.3").replace("1.7 )", "1.7").replace("4.6 )", "4.6").replace("6.2 )", "6.2")
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)

    # Holidayはカテゴリーなのでワンホットエンコーディングを行う
    is_holiday = df.pop('Holiday')
    df['Weekdays'] = (is_holiday == 0)*1.0
    df['Holidays'] = (is_holiday == 1)*1.0

    return df

if __name__ == '__main__':
    dataset = read_dataset()

    # 訓練データとテストデータに分ける。訓練: 8割  テスト: 2割
    train_dataset = dataset.sample(frac=0.8,random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    sns.pairplot(train_dataset[["Average temperature", "Peak supply capacity[10^5kW]", "Local pressure"]], diag_kind="kde")
    plt.show()

    print('finish')