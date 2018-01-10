import os
import sys
path_name= os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(path_name)

import _element.feature_control as ftc
import _element.calculations as calc
import _element.varr as varr
inputfilename= varr.INPUT_FILENAME
#이미 수정한 데이터로 진행
df_dir= varr.DF_DIR
temp_data_dir= varr.TEMP_DATA_DIR

import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from fbprophet import Prophet
from collections import OrderedDict
from scipy.special import expit
import copy

def Bayseian2(txs, forecastDay, unit):
    global mockForecastDictionary
    global realForecastDictionary

    holidaybeta= varr.HOLYDAYBETA
    if ftc.is_dict(txs):
        txs_raw= ftc.dict_to_df(txs, varr.COLNAME_KPPDAILY)
    else: txs_raw= copy.deepcopy(txs)

    txs= ftc.cut_df(txs_raw, varr.START_DATE, (varr.LAST_DATE- timedelta(days= forecastDay-1)))
    # print(txs.head())
    # print(txs.tail())

    if unit is 'day':
        # print("here2")
        if (len(txs) < 366):    seasonality_option= (False, True, True, True, 'd')
        else:                   seasonality_option= (False, True, True, True, 'd')

    elif unit is 'week':
        # print("here2")
        if (len(txs) < 53):     seasonality_option= (False, False, True, True, 'w')
        else:                   seasonality_option= (False, False, True, True, 'w')

    elif unit is 'month':
        # print("here2")
        if (len(txs) < 12):     seasonality_option= (False, False, False, True, 'm')
        else:                   seasonality_option= (False, False, False, True, 'm')

    model = Prophet(weekly_seasonality= False, \
                    yearly_seasonality=seasonality_option[3], \
                    holidays= holidaybeta)
    if seasonality_option[2]:
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    if seasonality_option[1]:
        model.add_seasonality(name='weekly', period=7, fourier_order=21, prior_scale= 100)

    for feature in txs.columns.values.tolist():
        if not (feature == 'ds' or feature == 'y'):
            model.add_regressor(feature)

    model.fit(txs)
    future = model.make_future_dataframe(periods=forecastDay, freq= seasonality_option[4])
    # future['ds']= pd.to_datetime(future['ds'], format= "%Y-%m-%d")

    future= pd.merge(future, txs_raw, how='left', on='ds')
    print(future[future.isnull().any(axis=1)])
    future.dropna(axis=0, inplace=True)
    # print(future)
    forecastProphetTable= model.predict(future)
    return (model, future, forecastProphetTable)
    # date = [d.strftime('%Y-%m-%d') for d in forecastProphetTable['ds']]

def extract_info_from(future, forecastProphetTable, forecastDay):
    result_forecast= forecastProphetTable['yhat'][-forecastDay:]
    expit(result_forecast)
    result_df= pd.concat([future[-forecastDay:], result_forecast], axis=1)
    event_parameter_df= forecastProphetTable[\
                            (forecastProphetTable['newyear'] + forecastProphetTable['thanksgiving'] + forecastProphetTable['chocostick']).abs() > 0][\
                        ['ds', 'newyear', 'thanksgiving', 'chocostick']]
    return (result_forecast, result_df, event_parameter_df)

# TODO : 페이스북 라이브러리 조절 파라메터 도큐멘 공부해서 성능 높이기

# Main ########################################################################
if __name__ == '__main__':
    txs= pd.read_excel(path_name+df_dir+inputfilename, sheet_name=None)
    if ftc.is_dict(txs):
        txs= ftc.dict_to_df(txs, varr.COLNAME_KPPDAILY)

    (model, future, forecastProphetTable)= Bayseian2(txs, varr.FORECASTDAY, 'day')
    (result_forecast, result_df, event_parameter_df)= extract_info_from(future, forecastProphetTable, 31)

    print(result_df)
    print(forecastProphetTable.head(20))
    print(event_parameter_df)

    model.plot(forecastProphetTable)
    model.plot_components(forecastProphetTable)
    calc.rms_error(result_df['y'], result_df['yhat'])
    calc.map_error(result_df['y'], result_df['yhat'])
    # save_as_xlsx(result_df, 'KPP일별투입(10_17).xlsx', specialfilename= 'result_Prophet.xlsx',\
    #             dirpath= 'C:\\Studying\\myvenv\\Project_Nextop\\nextop-engine\\nextop_engine\\_element\\data\\private\\')
    # save_as_xlsx(usecaseofholiday, 'KPP일별투입(10_17).xlsx', specialfilename= 'result_Prophet_usecase.xlsx',\
    #             dirpath= 'C:\\Studying\\myvenv\\Project_Nextop\\nextop-engine\\nextop_engine\\_element\\data\\private\\')
    #
    # (usecaseofholiday, result_df, result_forecast)= Bayseian2(txs, 180, 'month')
    # print(result_df)
    # save_as_xlsx(result_df, 'KPP일별투입(10_17).xlsx', specialfilename= 'result_Prophet_180.xlsx',\
    #             dirpath= 'C:\\Studying\\myvenv\\Project_Nextop\\nextop-engine\\nextop_engine\\_element\\data\\private\\')
    # save_as_xlsx(usecaseofholiday, 'KPP일별투입(10_17).xlsx', specialfilename= 'result_Prophet_usecase_180.xlsx',\
    #             dirpath= 'C:\\Studying\\myvenv\\Project_Nextop\\nextop-engine\\nextop_engine\\_element\\data\\private\\')
