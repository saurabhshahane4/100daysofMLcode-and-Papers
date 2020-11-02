import numpy as np
import pandas as pd
import random
import time
from collections import deque
from sklearn import preprocessing



df = pd.read_csv("LTC-USD.csv", names = ['time','low','high','open','close','volume'])

df.head()

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = 'LTC-USD'

def classify (current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df):
    df = df.drop('future', 1)
    
    for col in df.columns:
        if col != 'target':
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)
    
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
    random.shuffle(sequential_data)
    
    buys = []
    sells = []
    
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    random.shuffle(buys)
    random.shuffle(sells)
    
    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]
    
    sequential_data = buys + sells
    random.shuffle(sequential_data)
    
    X = []
    y = []
    
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
        
    return np.array(X), y


#All Cryptos in single dataframe
main_df = pd.DataFrame()
ratios = ['LTC-USD', 'BCH-USD', 'BTC-USD', 'ETH-USD'] 

for ratio in ratios:
    ratio = ratio.split('.csv')[0]
    dataset = f'../input/{ratio}.csv'
    df = pd.read_csv(dataset, names=['time','low','high','open','close','volume'])
    df.rename(columns={'close':f'{ratio}_close','volume':f'{ratio}_volume'}, inplace=True)
    df.set_index('time', inplace=True)
    df = df[[f'{ratio}_close',f'{ratio}_volume']]
    
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df.fillna(method='ffill', inplace=True)
main_df.dropna(inplace=True)
print(main_df.head())


main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'].main_df['future']))
main_df.dropna(inplace=True)
print(main_df.head())

times = sorted(main_df.index.values)
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]

print(time)
print(last_5pct)

validation_main_df = main_df[(main_df.index >= last_5pct)]
train_main_df = main_df[(main_df.index < last_5pct)]

print(validation_main_df.head())
print(train_main_df.head())

train_x, train_y = preprocess_df(train_main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

print(train_x.shape[1:])

