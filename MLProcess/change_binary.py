from collections import Counter
import pandas as pd
import numpy as np
import random

def changeBinaryFeatureInDf(dataDf):
    pd.options.mode.chained_assignment = None
    threshold = len(dataDf.index.tolist()) * (95 / 100)
    for column in dataDf.columns.to_list():
        dfUniqueValue = dataDf[column].unique()
        count_class = Counter(dataDf[column])
        dfCounterValue = pd.DataFrame.from_dict(count_class, orient='index', columns=["Count"])
        topValues = dfCounterValue['Count'].nlargest(2)          # 先取出來，只算一次
        dfCounterValueSum = topValues.sum()
        # 加上 len(topValues) >= 2 的防呆，避免整欄只有 1 個值時 IndexError
        if (dfCounterValueSum >= threshold) and (column != 'y') and (len(topValues) >= 2):
            value1 = topValues.index.tolist()[0]
            value2 = topValues.index.tolist()[1]
            if 0 in dfUniqueValue:
                convertMaxValue = value1 - 0.01
                convertMinValue = value2 + 0.01
                print('Value of ' + str(column) + ' Converted')
                print(str(value1) + ' --+0.01--> ' + str(convertMaxValue))
                print(str(value2) + ' --+0.01--> ' + str(convertMinValue))
            else:
                convertMaxValue = value1 * 0.99
                convertMinValue = value2 * 1.01
                print('Value of ' + str(column) + ' Converted')
                print(str(value1) + ' --*0.99--> ' + str(convertMaxValue))
                print(str(value2) + ' --*1.01--> ' + str(convertMinValue))
            value1Index = list(dataDf.loc[dataDf[column] == value1].index[:])
            value2Index = list(dataDf.loc[dataDf[column] == value2].index[:])
            countMax_int = int(np.ceil(len(value1Index) * 0.1))
            countMin_int = int(np.ceil(len(value2Index) * 0.1))
            randomMaxIndex = random.sample(value1Index, countMax_int)
            randomMinIndex = random.sample(value2Index, countMin_int)
            dataDf[column].loc[randomMaxIndex] = dataDf[column].loc[randomMaxIndex].replace(value1, convertMaxValue)
            dataDf[column].loc[randomMinIndex] = dataDf[column].loc[randomMinIndex].replace(value2, convertMinValue)
        elif (dfCounterValueSum >= threshold) and (column != 'y') and (len(topValues) < 2):
            # 整欄只有 1 個值，沒有第二種數值可以互換，跳過 binary 調整，僅提示
            print(f'[Skip] Column "{column}" has only one unique value in this dataset, skip binary adjustment.')
        if (dataDf[column].dtype.name == 'int64') and (column != 'y'):
            dataDf[column] = dataDf[column].astype('float64')
            print('Type of ' + str(column) + ' Converted')
            print('int64 ----> float64')
    return dataDf
