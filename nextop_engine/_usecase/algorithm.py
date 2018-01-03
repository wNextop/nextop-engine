import os
import sys
path_name= os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(path_name)

from _element.feature_control import *
from _element.calculations import minMaxNormalizer, minMaxDeNormalizer

import tensorflow as tf
import numpy as np
from datetime import datetime
import pandas as pd
from fbprophet import Prophet
from collections import OrderedDict

def LSTM(txs, forecastDay, features):
    tf.reset_default_graph()
    tf.set_random_seed(77)
    # Add basic date related features to the table
    year = lambda x: datetime.strptime(x, "%Y-%m-%d").year
    dayOfWeek = lambda x: datetime.strptime(x, "%Y-%m-%d").weekday()
    month = lambda x: datetime.strptime(x, "%Y-%m-%d").month
    weekNumber = lambda x: datetime.strptime(x, "%Y-%m-%d").strftime('%V')
    txs['year'] = txs['ds'].map(year)
    txs['month'] = txs['ds'].map(month)
    txs['weekNumber'] = txs['ds'].map(weekNumber)
    txs['dayOfWeek'] = txs['ds'].map(dayOfWeek)

    # Add non-basic date related features to the table
    seasons = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]  # dec - feb is winter, then spring, summer, fall etc
    season = lambda x: seasons[(datetime.strptime(x, "%Y-%m-%d").month - 1)]
    day_of_week01s = [0, 0, 0, 0, 0, 1, 1]
    day_of_week01 = lambda x: day_of_week01s[(datetime.strptime(x, "%Y-%m-%d").weekday())]
    txs['season'] = txs['ds'].map(season)
    txs['dayOfWeek01'] = txs['ds'].map(day_of_week01)

    # Backup originalSales
    originalSales = list(txs['y'])
    sales = list(txs['y'])

    # week number는 경계부분에서 약간 오류가 있다.
    if features is 'DayOfWeek_WeekNumber_Month_Season':
        tempxy = [list(txs['dayOfWeek']), list(txs['weekNumber']), list(txs['month']), list(txs['season']), sales]
    elif features is 'DayOfWeek01_WeekNumber_Month_Season':
        tempxy = [list(txs['dayOfWeek01']), list(txs['weekNumber']), list(txs['month']), list(txs['season']), sales]

    elif features is 'WeekNumber_Month_Season_Year':
        tempxy = [list(txs['weekNumber']), list(txs['month']), list(txs['season']), list(txs['year']), sales]

    xy = np.array(tempxy).transpose().astype(np.float)

    # Backup originalXY for denormalize
    originalXY = np.array(tempxy).transpose().astype(np.float)
    xy = minMaxNormalizer(xy)

    # TRAIN PARAMETERS
    # data_dim은 y값 도출을 위한 feature 가지수+1(독립변수 가지수 +1(y포함))
    data_dim = 5
    # data_dim크기의 data 한 묶음이 seq_length만큼 input으로 들어가
    seq_length = 10
    # output_dim(=forecastDays)만큼의 다음날 y_data를 예측
    output_dim = forecastDay
    # hidden_dim은 정말 임의로 설정
    hidden_dim = 100
    # learning rate은 배우는 속도(너무 크지도, 작지도 않게 설정)
    learning_rate = 0.001
    iterations = 1000
    # Build a series dataset(seq_length에 해당하는 전날 X와 다음 forecastDays에 해당하는 Y)
    x = xy
    y = xy[:, [-1]]
    dataX = []
    dataY = []
    for i in range(0, len(y) - seq_length - forecastDay + 1):
        _x = x[i:i + seq_length]
        _y = y[i + seq_length:i + seq_length + forecastDay]
        _y = np.reshape(_y, (forecastDay))
        dataX.append(_x)
        dataY.append(_y)

    train_size = int(len(dataY) - forecastDay)
    # train_size = int(len(dataY) * 0.7)
    test_size = len(dataY) - train_size

    trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:])

    trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:])

    X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
    Y = tf.placeholder(tf.float32, [None, forecastDay])

    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)
    loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    denormalizedTestY = originalSales[train_size + seq_length:]
    #     denormalizedTestY_feed=np.array([[i] for i in denormalizedTestY])

    targets = tf.placeholder(tf.float32, [None, 1])
    predictions = tf.placeholder(tf.float32, [None, 1])

    count = 0
    with tf.Session() as sess:

        # 초기화
        init = tf.global_variables_initializer()
        sess.run(init)

        # Training step
        for i in range(iterations):
            _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
            print("[step: {}] loss: {}".format(i, step_loss))

        # Test step
        test_predict = minMaxDeNormalizer(sess.run(Y_pred, feed_dict={X: testX}), originalXY)
        realSale = minMaxDeNormalizer(testY[-1], originalXY)
    return np.square(test_predict[-1]).tolist()

# TODO : 엘에스티엠 개선


def Bayseian(txs, forecastDay, unit):
    global mockForecastDictionary
    global realForecastDictionary

    if unit is 'day':
        if (len(txs) < 366):
            model = Prophet()
            model.fit(txs)
            future = model.make_future_dataframe(periods=forecastDay)
            forecastProphetTable = model.predict(future)

        else:
            model = Prophet(yearly_seasonality=True)
            model.fit(txs)
            future = model.make_future_dataframe(periods=forecastDay)
            forecastProphetTable = model.predict(future)


    elif unit is 'week':
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
        if (len(txs) < 12):
            model = Prophet()
            model.fit(txs)
            future = model.make_future_dataframe(periods=forecastDay, freq='m')
            forecastProphetTable = model.predict(future)

        else:
            model = Prophet(yearly_seasonality=True)
            model.fit(txs)
            future = model.make_future_dataframe(periods=forecastDay, freq='m')
            forecastProphetTable = model.predict(future)

    # date = [d.strftime('%Y-%m-%d') for d in forecastProphetTable['ds']]
    return [np.exp(y) for y in forecastProphetTable['yhat'][-forecastDay:]]

def Bayseian2(txs, forecastDay, unit, weather=None):
    global mockForecastDictionary
    global realForecastDictionary

    if isinstance(txs, type(OrderedDict())):
        txs= feature_control.dict_to_df(txs)

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
    chocostick = pd.DataFrame({
        'holiday': 'chocostick',
        'ds': pd.to_datetime(['2010-11-11', '2011-11-11', '2012-11-11',
                              '2013-11-11', '2014-11-11', '2015-11-11',
                              '2016-11-11', '2017-11-11']),
        'lower_window': 0,
        'upper_window': 0,
    })

    if unit is 'day':
        # print("here2")
        if (len(txs) < 366):
            model = Prophet()
            model.fit(txs)
            future = model.make_future_dataframe(periods=forecastDay)
            forecastProphetTable = model.predict(future)

        else:

            holidaybeta = pd.concat((newyear, thanksgiving))

            model = Prophet(weekly_seasonality=False, yearly_seasonality=True, holidays= holidaybeta)
            model.add_seasonality(model, name='monthly', period=30.5, fourier_order=5)

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
            holidaybeta = pd.concat((newyear, thanksgiving, chocostick))
            model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True, holidays=holidaybeta)

            # weatherL = weather[:-forecastDay]
            # txs['weather'] = weatherL
            # model.add_regressor('weather')
            for feature in txs.columns.values.tolist():
                if not (feature == 'ds' or feature == 'y'):
                    model.add_regressor(feature)
            model.fit(txs)
            future = model.make_future_dataframe(periods=forecastDay)   # 일 별로 수정...!
            # future = model.make_future_dataframe(periods=forecastDay, freq='m')
            # future['weather'] = weather

            #TODO: 강수량, 날씨 데이터를 예측할 날짜에 넣어 줘야 predict 함수가 돌아갑니다.
            # 현재 future dataframe은 이렇게 생겼어요.
            #              ds
            # 0    2010-07-01
            # 1    2010-07-02
            # 2    2010-07-03
            # 3    2010-07-04
            # 4    2010-07-05
            # 5    2010-07-06
            # 6    2010-07-07
            # 7    2010-07-08
            # 8    2010-07-09
            # 9    2010-07-10
            # ...         ...
            # 2692 2017-11-21
            # 2693 2017-11-22
            # 2694 2017-11-23
            # 2695 2017-11-24
            # 2696 2017-11-25
            # 2697 2017-11-26
            # 2698 2017-11-27
            # 2699 2017-11-28
            # 2700 2017-11-29
            # 2701 2017-11-30
            # 2702 2017-12-01       << 여기서부터 예측할 날짜>>
            # 2703 2017-12-02
            # 2704 2017-12-03
            # 2705 2017-12-04
            # 2706 2017-12-05
            # 2707 2017-12-06
            # 2708 2017-12-07
            # print(future)
            forecastProphetTable = model.predict(future)

            usecaseofholiday = forecastProphetTable[10:70][
            ['ds', 'newyear', 'thanksgiving', 'chocostick']][:]
            print(usecaseofholiday)

    # date = [d.strftime('%Y-%m-%d') for d in forecastProphetTable['ds']]
    return [np.exp(y) for y in forecastProphetTable['yhat'][-forecastDay:]]

# TODO : 페이스북 라이브러리 조절 파라메터 도큐멘 공부해서 성능 높이기

# Main ########################################################################
if __name__ == '__main__':
    inputfilename= 'KPP일별투입(10_17)_restructured_restructured.xlsx'
    #이미 수정한 데이터로 진행
    df_dir= '\\_element\\data\\private\\'
    #해당 주소에 데이터를 넣어 주어야!
    temp_data_dir= '\\_element\\data\\temp_data'
    txs= pd.read_excel(path_name+df_dir+inputfilename, sheet_name=None)
    if isinstance(txs, type(OrderedDict())):
        txs= dict_to_df(txs)
    Bayseian2(txs, 7, 'month')

    #           ds  rain_amount   temp_max   temp_min      y
    # 0 2010-07-01          0.0  30.500000  24.900000  79590
    # 1 2010-07-02         70.0  25.799999  22.700001  79456
    # 2 2010-07-03          1.0  25.600000  22.600000  48469
    # 3 2010-07-04          0.0  29.700001  23.100000   1045
    # 4 2010-07-05          0.0  30.600000  21.799999  65049
    # 5 2010-07-06          0.0  31.000000  22.400000  84245
    # 6 2010-07-07          0.0  29.600000  21.900000  80493
    # 7 2010-07-08          0.0  29.000000  21.299999  73930
    # 8 2010-07-09          0.0  29.299999  21.799999  80166
    # 9 2010-07-10          0.0  29.100000  22.900000  49542
    # 
    # 323 2017-11-21          0.5       6.8       0.4  121680
    # 324 2017-11-22          0.0       9.5      -0.8  122640
    # 325 2017-11-23          0.8       4.3      -0.6  110220
    # 326 2017-11-24          0.1       2.9      -2.5  120647
    # 327 2017-11-25         15.7       6.4       0.8   67756
    # 328 2017-11-26          0.0       6.4      -2.4     100
    # 329 2017-11-27          0.0       8.1       1.8  117480
    # 330 2017-11-28          1.2       9.8       4.2  122192
    # 331 2017-11-29          0.0       5.7      -3.2  120685
    # 332 2017-11-30          0.0       1.8      -7.1  108386
    # Initial log joint probability = -104.505
    #     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
    #       77       2476.86    0.00233501       105.151   2.298e-05       0.001      123  LS failed, Hessian reset
    #       99       2477.04   0.000121835       89.7766      0.3404           1      156
    #     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
    #      199       2477.52   0.000532963       73.4935           1           1      282
    #     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
    #      236       2477.66   6.85527e-05       112.256   1.151e-06       0.001      388  LS failed, Hessian reset
    #      299       2477.74    1.7124e-06        82.655           1           1      462
    #     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
    #      335       2477.74   8.79426e-06       81.8554    1.15e-07       0.001      556  LS failed, Hessian reset
    #      354       2477.75   6.82779e-08       78.3674      0.3942      0.3942      584
