import csv

import tensorflow as tf
import numpy as np
from datetime import datetime
import pandas as pd
from fbprophet import Prophet

def rmse(a, b):
    sum = 0
    for i in range(len(a)):
        sum = sum + (a[i] - b[i]) ** 2
    return np.sqrt(sum / len(a))


def Bayseian(txs, forecastDay, weather, unit):
    global mockForecastDictionary
    global realForecastDictionary

    newyear = pd.DataFrame({
        'holiday': 'newyear',
        'ds': pd.to_datetime(['2010-02-14', '2011-02-03', '2012-01-23',
                              '2013-02-10', '2014-01-31', '2015-02-19',
                              '2016-02-09', '2017-02-28']),
        'lower_window': -1,
        'upper_window': 1,
    })
    thanksgiving = pd.DataFrame({
        'holiday': 'thanksgiving',
        'ds': pd.to_datetime(['2010-09-22', '2011-09-12', '2012-09-30',
                              '2013-09-19', '2014-09-09', '2015-09-27',
                              '2016-09-15', '2017-10-04']),
        'lower_window': -1,
        'upper_window': 1,
    })

    if unit is 'day':
        # print("here2")
        if (len(txs) < 366):
            model = Prophet()
            model.fit(txs)
            future = model.make_future_dataframe(periods=forecastDay)
            forecastProphetTable = model.predict(future)

        else:
            # 여기만 시범삼아 고치는
            holidaybeta = pd.concat((newyear, thanksgiving))

            model = Prophet(weekly_seasonality=False, yearly_seasonality=True, holidays= holidaybeta)
            model.add_seasonality(model, name='monthly', period=30.5, fourier_order=5)
            # 날씨는 어떻게 추가할수 있을까? API에는 존재하지 않는다.
            # 주간 시즌을 없애고 월별 주기 생성

            # 양대 명절만을 고려해놨다. 추가요구사항은 좀 더 알려줬으면 싶은데
            # 연휴와 휴가철 역시 마찬가지. 이것에 대한 데이터가 필요하다.
            # 여기까지는 seasonality and holiday로 처리가능
            # 계약종료, 신규업체는 우리가 무슨수로 알 수 있지? 뼈대는 무엇? -> changepoint강제처리가 1
            # 화물연대 파업은 무슨수로 알 수 있지? 뼈대는 무엇? ->이미 발생한 파업은 outlier처리를 해야하고
            # 앞으로 예고된 파업에 대해서는 파업요소를 홀리데이에 강제로 추가해줘야 할거같다.
            model.fit(txs)
            future = model.make_future_dataframe(periods=forecastDay)
            forecastProphetTable = model.predict(future)

    elif unit is 'week':
        # print("here2")
        if (len(txs) < 53):
            model = Prophet()
            model.fit(txs)
            future = model.make_future_dataframe(periods=forecastDay, freq='w')
            forecastProphetTable = model.predict(future)

        else:
            model = Prophet(yearly_seasonality=True)
            model.fit(txs)
            future = model.make_future_dataframe(periods=forecastDay, freq='w')
            forecastProphetTable = model.predict(future)

    elif unit is 'month':
        # print("here2")
        if (len(txs) < 12):
            model = Prophet()
            model.fit(txs)
            future = model.make_future_dataframe(periods=forecastDay, freq='m')
            forecastProphetTable = model.predict(future)

        else:
            # print("here")
            holidaybeta = pd.concat((newyear, thanksgiving))
            model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True, holidays=holidaybeta)
            weatherL = weather[:-forecastDay]
            txs['weather'] = weatherL
            model.add_regressor('weather')
            model.fit(txs)
            future = model.make_future_dataframe(periods=forecastDay, freq='m')
            future['weather'] = weather
            forecastProphetTable = model.predict(future)

    # date = [d.strftime('%Y-%m-%d') for d in forecastProphetTable['ds']]
    return [np.exp(y) for y in forecastProphetTable['yhat'][-forecastDay:]]

datadates = pd.read_csv('fordates.csv', header=0)
datavalues = pd.read_csv('forvalue.csv', header = 0)
dataweathers = pd.read_csv('weather2.csv', header = 0)
rawArrayDatas = pd.concat((datadates, datavalues), axis=1)

forecastDay = 8

# ds = rawArrayDatas[0][:-forecastDay]
ds = rawArrayDatas['date'][:-forecastDay]
# print(ds)
# # TODO bayseian에 대해서는 input값이 0인 상황처리 필요
y = list(np.log(rawArrayDatas['values'][:-forecastDay]))
# print(y)
sales = list(zip(ds, y))
# print(sales)
txsForRealForecastBayesian = pd.DataFrame(data=sales, columns=['ds', 'y'])
# print(txsForRealForecastBayesian)
realForecastDictionary = {}
realForecastDictionary['Bayseian'] = Bayseian(txsForRealForecastBayesian, forecastDay, dataweathers, 'month')

testY = {}
testY['Bayseian'] = list(rawArrayDatas['values'][-forecastDay:])
print(realForecastDictionary['Bayseian'])
print(testY['Bayseian'])
resultofrmse = rmse(testY['Bayseian'], realForecastDictionary['Bayseian'])
print(resultofrmse)