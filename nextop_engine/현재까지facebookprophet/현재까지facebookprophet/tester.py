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


def Bayseian(txs, forecastDay, amountOfRain, maxOfTemp, minOfTemp, forcastedRain, forcastedMax, forcastedMin, unit):
    global mockForecastDictionary
    global realForecastDictionary




    if unit is 'day':
        # print("here2")
        if (len(txs) < 366):
            model = Prophet()
            model.fit(txs)
            future = model.make_future_dataframe(periods=forecastDay)
            forecastProphetTable = model.predict(future)

        else:

            # print(holidaybeta)


            # model.add_seasonality(model, name='monthly', period=30.5, fourier_order=5)


            if(amountOfRain.any()) :
                print('here')
                txs['rain_amount'] = amountOfRain
                model.add_regressor('rain_amount')
                txs['max_temp'] = maxOfTemp
                model.add_regressor('max_temp')
                txs['min_temp'] = minOfTemp
                model.add_regressor('min_temp')
            else :
                print('none')

            model.fit(txs)
            future = model.make_future_dataframe(periods=forecastDay, freq= 'd')

            # print(future)
            # future['weather'] = weather
            if(forcastedRain.any()) :
                future['rain_amount'] = forcastedRain
                future['max_temp'] = forcastedMax
                future['min_temp'] = forcastedMin
            else :
                print('none')

            forecastProphetTable = model.predict(future)
            # print(forecastProphetTable)

            # usecaseofholiday = forecastProphetTable[(forecastProphetTable['newyear'] + forecastProphetTable['thanksgiving'] + forecastProphetTable['chocostick']).abs() > 0][['ds', 'newyear', 'thanksgiving', 'chocostick']][:]
            # print(usecaseofholiday)

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
            holidaybeta = pd.concat((newyear, thanksgiving, chocostick))
            model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True, holidays=holidaybeta)
            # weatherL = weather[:-forecastDay]
            # txs['weather'] = weatherL
            # model.add_regressor('weather')
            model.fit(txs)
            future = model.make_future_dataframe(periods=forecastDay, freq='m')
            # future['weather'] = weather
            forecastProphetTable = model.predict(future)
            # usecaseofholiday = forecastProphetTable[(forecastProphetTable['newyear'] + forecastProphetTable['thanksgiving'] + forecastProphetTable['chocostick']).abs() > 0][
            # ['ds', 'newyear', 'thanksgiving', 'chocostick']][:]
            # print(usecaseofholiday)

    # date = [d.strftime('%Y-%m-%d') for d in forecastProphetTable['ds']]
    return [np.exp(y) for y in forecastProphetTable['yhat'][-forecastDay:]]


# ------------------------------------------------------------원본 데이터들
dataByDates = pd.read_csv('KPPuc77cubcc4ud22cuc785(10_17)_restructured_restructured.csv', header = 0)
dataDates = pd.to_datetime(dataByDates['ds'])
dataValues = dataByDates['y']
dataRainAmount = dataByDates['rain_amount']
dataTempMax = dataByDates['temp_max']
dataTempMin = dataByDates['temp_min']
# print(dataDates)
# print(dataDates)

rawArrayDatas = pd.concat((dataDates, dataValues), axis=1)
# print(rawArrayDatas)
# ------------------------------------------------------------2016년까지의 데이터들. 2017예측을 위해서 사용한다.

numberOf2017 = 333
forecastDay = 365

# # TODO bayseian에 대해서는 input값이 0인 상황처리 필요
ds = rawArrayDatas['ds'][:-numberOf2017]
y = list(np.log(rawArrayDatas['y'][:-numberOf2017]))# 왜 로그를 씌워야하는지는 아직도 의문이다. 필요성이 있는가?
rainAmount = dataRainAmount[:-numberOf2017]
tempMax = dataTempMax[:-numberOf2017]
tempMin = dataTempMin[:-numberOf2017]
#-------------------------------------------------------------2017예측하고 rmse 구하기

salesFor2017 = list(zip(ds, y))
# print(sales)
txsFor2017 = pd.DataFrame(data=salesFor2017, columns=['ds', 'y'])
# print(txsFor2017)
testFor2017 = {}
testFor2017['Bayseian'] = Bayseian(txsFor2017, numberOf2017, rainAmount, tempMax, tempMin, dataRainAmount, dataTempMax, dataTempMin, 'day')
YFor2017 = {}
YFor2017['Bayseian'] = list(rawArrayDatas['y'][-numberOf2017:])
print(testFor2017['Bayseian'])
print(YFor2017['Bayseian'])
resultofrmse = rmse(YFor2017['Bayseian'], testFor2017['Bayseian'])
print(resultofrmse)
# ------------------------------------------------------------2018 예측하기
#TODO : 날씨와 같은것을 이벤트로 처리하기. 일단은 테스터로 올렸습니다. 알고리즘에 바로 올리면 문제가 생겨서요 ㅜㅜ
