import pandas as pd
import numpy as np
import copy
from datetime import datetime
from collections import OrderedDict
import os, sys
if __name__=="__main__":
    path_name= os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    sys.path.append(path_name)
df_dir='C:\\Studying\\myvenv\\Project_Nextop\\nextop-engine\\nextop_engine\\_element\\data\\private\\'
temp_data_dir='C:\\Studying\\myvenv\\Project_Nextop\\nextop-engine\\nextop_engine\\_element\\data\\temp_data'

def opener(inputfilename):
    '''
    엑셀 파일을 엽니다.
    현재는 특정한 디렉토리가 inputfilename에 적혀져 있어야 열릴 수 있는 상태입니다.
    '''
    dict_of_dfs= pd.read_excel(inputfilename, sheet_name=None)
    return dict_of_dfs

def struct(inputfilename):
    """
    건수별로 이루어져 있는 df를 날짜를 index로, 건수별 유형 코드를 feature로 하는 df로 만듭니다.
    현재는 inputfilename이 input값으로 되어 있고, 이후 df를 인풋으로 하도록 수정할 예정입니다.
    또 현재는 '발송일', '유형', '수량'이라는 feature만을 가진 것으로 짜여져 있는 점,
    날짜 형식이 YYYMMDD 형식인 경우만을 고려하는 점도 수정해야 합니다.
    """
    dict_of_dfs= opener(inputfilename)
    date_change= lambda x: datetime(int(x[:4]),int(x[4:6]),int(x[6:])) if len(x)==8 \
                            else datetime.strptime(x, "%Y-%m-%d")
    result_dict={}
    for dfsheetname, df in dict_of_dfs.items():
        df['발송일']=df['발송일'].map(str).map(date_change)
        df= df.groupby([df['발송일'], df['유형']])['수량'].sum().unstack('유형')
        df.fillna(0, inplace= True)
        df['y_sum']= df.sum(axis=1)
        result_dict[dfsheetname]= df
    return result_dict

def save_as_xlsx(dict_of_dfs, inputfilename, specialfilename=None, dirpath= None):
    """
    여러 개의 dfsheet로 되어 있는 dictionary(또는 OrderedDict) 데이터를 엑셀에 저장합니다.
    아직 디렉토리를 설정할 수 없어 나중에 수정해야 합니다.
    """
    if specialfilename==None: specialfilename= df_dir + inputfilename[:-5] + '_restructured' + inputfilename[-5:]
    if dirpath: specialfilename= dirpath+specialfilename
    writer= pd.ExcelWriter(specialfilename, engine= 'xlsxwriter')
    if is_dict(dict_of_dfs):
        for (dfsheetname, df) in dict_of_dfs.items():
            df.to_excel(writer, sheet_name= dfsheetname)
    else: dict_of_dfs.to_excel(writer, sheet_name= 'data_merged')
    writer.save()
    return None

def add_temp_data(inputfilename, datainfo):
    """
    날씨 관련 정보를 불러와 df로 구성합니다.
    현재는 강수량의 경우 'rain_amount', 기온의 경우 'temp_max'와 'temp_min'으로
    저장되도록 짜 놓았습니다. 다른 feature가 추가되어야 되면 수정해야 합니다.
    """
    year= int(inputfilename[-13:-9])
    month= 1
    if year== 2010: month +=6
    drop_index=[]
    positive_value= lambda x: max(x, 0)

    df2= pd.read_csv(inputfilename)
    df2.set_axis(['ds', 'hour', 'amount'], axis='columns', inplace=True)

    for index, day, _, _ in df2.itertuples():
        if len(day)>4:
            month+= 1
            drop_index.append(index)
        else:
            try: df2.at[index, 'ds']=datetime(year, month, int(day))
            except: print(df2.loc[index, :])
    df2.drop(drop_index, inplace=True)

    if datainfo== 'rain':
        df2= df2.groupby(df2['ds'])['amount'].sum()
        df2.rename('rain_amount', inplace= True)
        df2= df2.map(positive_value)
    elif datainfo== 'temp':
        df2= df2[df2['amount'].map(int) != -1]
        agg_func= {'temp_max': np.max, 'temp_min': np.min}
        df2= df2.groupby(df2['ds'])['amount'].agg(agg_func)
    return df2

def dir_walk(data_path, dict_of_dfs):
    for root, dirs, files in os.walk(data_path):
        for file_ in files:
            print(os.path.join(root, file_))
            df2= add_temp_data(os.path.join(root, file_), file_[5:9])
            result_dict={}
            if is_dict(dict_of_dfs):
                for (dfsheetname, df) in zip(dict_of_dfs.keys(), dict_of_dfs.values()):
                    df_result= df.join(df2, how='inner', lsuffix= '_left', rsuffix= '_right')
                    if not df_result.empty:
                        result_dict[dfsheetname]= df_result
                    else: result_dict[dfsheetname]= df
                dict_of_dfs= copy.deepcopy(result_dict)
                for df in dict_of_dfs.values():
                    print(df)
            else:
                dict_of_dfs= dict_of_dfs.join(df2, how='left', on='ds', lsuffix='', rsuffix='_added')
                dict_of_dfs.fillna(0, inplace= True)
                col_list= list(dict_of_dfs.columns)
                for feature in col_list:
                    if feature[-6:]== '_added':
                        originalft= feature[:-6]
                        dict_of_dfs[originalft]= dict_of_dfs[originalft]+ dict_of_dfs[feature]
                        dict_of_dfs.drop(feature, axis= 1, inplace= True)
    return dict_of_dfs

def is_dict(dict_of_dfs):
    if isinstance(dict_of_dfs, type(OrderedDict())) or isinstance(dict_of_dfs, dict):
        return True
    else: return False

def dict_to_df(dict_of_dfs, column_list, y_column= 'y_sum'):
    df_txs= pd.DataFrame(columns= column_list)
    if is_dict(dict_of_dfs):
        for sheetname, df in dict_of_dfs.items():
            df.reset_index(drop= False, inplace= True)
            df.rename(columns= {'발송일': 'ds',\
                            'y_sum': 'y'}, inplace=True)
            df= df[column_list]
            df_txs= pd.concat([df_txs, df])
        df_txs.sort_values(by='ds')
    else:
        df_txs= dict_of_dfs
        df_txs.reset_index(drop= False, inplace= True)
        df_txs.rename(columns= {'발송일': 'ds',\
                        'y_sum': 'y'}, inplace=True)
    return df_txs

def cut_df(txs, start_date, end_date):
    txs= txs[(txs['ds']>= start_date) & (end_date > txs['ds'])]
    return txs

# Main #########################################################################

if __name__== '__main__':
    df= opener(df_dir+'KPP일별투입(10_17)_restructured.xlsx')
    df= dict_to_df(df, ['ds', 'rain_amount', 'temp_max', 'temp_min', 'y'])
    print(df)
    df= cut_df(df, datetime(2010, 7, 1), datetime(2017, 7, 1))
    print(df.head(10))
    print(df.tail(10))
