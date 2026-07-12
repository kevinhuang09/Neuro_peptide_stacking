from collections import Counter
import pandas as pd
import numpy as np
import random

def changeBinaryFeatureInDf(dataDf):
    """
    因為 LightGBM 無法接受 binary feature 以及 int64 type 的 feature, 所以人工處理.
    binary feature: 隨機挑 1% 來改變數值 (加減 0.01 or 乘上 0.99/1.01)
    int64 type feature: 手動轉換為 float64 type
    :param dataDf:
    :return:
    """
    pd.options.mode.chained_assignment = None
    threshold = len(dataDf.index.tolist()) * (95 / 100)
    for column in dataDf.columns.to_list():
        dfUniqueValue = dataDf[column].unique()
        count_class = Counter(dataDf[column])
        dfCounterValue = pd.DataFrame.from_dict(count_class, orient='index', columns=["Count"])
        dfCounterValueSum = dfCounterValue['Count'].nlargest(2).sum()
        if (dfCounterValueSum >= threshold) and (column != 'y'):
            value1 = dfCounterValue['Count'].nlargest(2).index.tolist()[0]
            value2 = dfCounterValue['Count'].nlargest(2).index.tolist()[1]
            if 0 in dfUniqueValue:
                convertMaxValue = value1 - 0.01
                convertMinValue = value2 + 0.01
                print('Value of ' + str(column) + ' Converted')
                print(str(value1) + '  --+0.01-->  ' + str(convertMaxValue))
                print(str(value2) + '  --+0.01-->  ' + str(convertMinValue))
            else:
                convertMaxValue = value1 * 0.99
                convertMinValue = value2 * 1.01
                print('Value of ' + str(column) + ' Converted')
                print(str(value1) + '  --*0.99-->  ' + str(convertMaxValue))
                print(str(value2) + '  --*1.01-->  ' + str(convertMinValue))
            value1Index = list(dataDf.loc[dataDf[column] == value1].index[:])
            value2Index = list(dataDf.loc[dataDf[column] == value2].index[:])
            countMax_int = int(np.ceil(len(value1Index) * 0.1))
            countMin_int = int(np.ceil(len(value2Index) * 0.1))
            randomMaxIndex = random.sample(value1Index, countMax_int)
            randomMinIndex = random.sample(value2Index, countMin_int)
            dataDf[column].loc[randomMaxIndex] = dataDf[column].loc[randomMaxIndex].replace(value1, convertMaxValue)
            dataDf[column].loc[randomMinIndex] = dataDf[column].loc[randomMinIndex].replace(value2, convertMinValue)
        if (dataDf[column].dtype.name == 'int64') and (column != 'y'):
            dataDf[column] = dataDf[column].astype('float64')
            print('Type of ' + str(column) + ' Converted')
            print('int64  ---->  float64')

    return dataDf
