import pandas as pd
import numpy as np
import copy
from datetime import datetime
import os
df_dir='C:\\Studying\\myvenv\\Project_Nextop\\nextop-engine\\nextop_engine\\_element\\data\\df_kpp\\'
temp_data_dir='C:\\Studying\\myvenv\\Project_Nextop\\nextop-engine\\nextop_engine\\_element\\data\\temp_data'

def struct(inputfilename):
    dict_of_dfs= pd.read_excel(inputfilename, sheet_name=None)
    date_change= lambda x: datetime.strptime(x[:4]+'-'+x[4:6]+'-'+x[6:], "%Y-%m-%d") if len(x)==8 \
                            else datetime.strptime(x, "%Y-%m-%d")
    result_dict={}

    for (dfsheetname, df) in zip(dict_of_dfs.keys(), dict_of_dfs.values()):
        # print(pd.pivot_table(df, index=['발송일'], values=['수량'], columns=['유형'], \
        #                         aggfunc=[np.sum], fill_value=0))
        df['발송일']=df['발송일'].map(str).map(date_change)
        df= df.groupby([df['발송일'], df['유형']])['수량'].sum().unstack('유형')
        df.fillna(0, inplace= True)
        df['y_sum']= df.sum(axis=1)
        result_dict[dfsheetname]= df
        # print(df.columns)
        # print(df.head(30))
    return result_dict

def save_as_xlsx(dict_of_dfs, inputfilename, outputfilename=None):
    if outputfilename==None: outputfilename= df_dir + inputfilename[:-5] + '_restructured' + inputfilename[-5:]
    writer= pd.ExcelWriter(outputfilename, engine= 'xlsxwriter')
    for (dfsheetname, df) in zip(dict_of_dfs.keys(), dict_of_dfs.values()):
        df.to_excel(writer, sheet_name= dfsheetname)
    writer.save()
    return None

def add_temp_data(inputfilename, datainfo):
    year= int(inputfilename[-13:-9])
    month= 1
    if year== 2010: month +=6
    drop_index=[]
    positive_value= lambda x: max(x, 0)

    df2= pd.read_csv(inputfilename)
    df2.set_axis(['발송일', 'hour', 'amount'], axis='columns', inplace=True)

    for index, day, _, _ in df2.itertuples():
        if len(day)>4:
            month+= 1
            drop_index.append(index)
        else:
            try: df2.at[index, '발송일']=datetime(year, month, int(day)).strftime("%Y-%m-%d")
            except: print(df2.loc[index, :])
    df2.drop(drop_index, inplace=True)

    if datainfo== 'rain':
        df2= df2.groupby(df2['발송일'])['amount'].sum()
        df2.rename('rain_amount', inplace= True)
        df2= df2.map(positive_value)
    elif datainfo== 'temp':
        df2= df2[df2['amount'].map(int) != -1]
        agg_func= {'temp_max': np.max, 'temp_min': np.min}
        df2= df2.groupby(df2['발송일'])['amount'].agg(agg_func)
    # print(df2.head(10))
    return df2

def dir_walk(data_path, df_inputfilename):
    dict_of_dfs= struct(df_inputfilename)

    for root, dirs, files in os.walk(data_path):
        for file_ in files:
            print(os.path.join(root, file_))
            df2= add_temp_data(os.path.join(root, file_), file_[5:9])
            result_dict={}
            for (dfsheetname, df) in zip(dict_of_dfs.keys(), dict_of_dfs.values()):
                # df= pd.concat([df, df2], axis=1, join= 'inner')
                df_result= df.join(df2, how='inner', lsuffix= file_[5:9], rsuffix= '_right')
                # print(df_result)
                if not df_result.empty:
                    result_dict[dfsheetname]= df_result
                else: result_dict[dfsheetname]= df
            dict_of_dfs= copy.deepcopy(result_dict)
    for df in dict_of_dfs.values():
        print(df)
    return dict_of_dfs

# Main

if __name__== '__main__':
    inputfilename= 'KPP일별투입(10_17).xlsx'
    dict_of_dfs=dir_walk(temp_data_dir, df_dir+inputfilename)
    save_as_xlsx(dict_of_dfs, inputfilename)
    # struct(df_dir+'\\kpp_sampledata.xlsx')
    # add_temp_data('2017_rain.csv', 'rain')
    # add_temp_data('2017_temp.csv', 'temp')
