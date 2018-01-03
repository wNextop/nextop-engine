import algorithm
import pandas as pd
import datetime
#엑셀 파일에서 날짜-y_sum 으로 txs를 만들고,  weather를 만든다.

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


algorithm.BayseianNoCompare(txs, 7, weather, rain ,'day')

algorithm.BayseianNoCompare(txs, 7, weather,'day')

algorithm.BayseianNoCompare(txs, 7, weather, rain ,'month')

algorithm.BayseianNoCompare(txs, 7, weather,'month')