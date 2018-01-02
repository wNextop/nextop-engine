import pandas as pd
import numpy as np
from datetime import datetime
import os
os.chdir('C:\\Studying\\myvenv\\Project_Nextop\\nextop-engine\\nextop_engine\\_element')

def struct(inputfilename):
    dict_of_dfs= pd.read_excel(inputfilename, sheet_name=None)
    date_change= lambda x: datetime.strptime(x[:4]+'-'+x[4:6]+'-'+x[6:], "%Y-%m-%d") if len(x)==8 \
                            else datetime.strptime(x, "%Y-%m-%d")

    for df in dict_of_dfs.values():
        # print(pd.pivot_table(df, index=['발송일'], values=['수량'], columns=['유형'], \
        #                         aggfunc=[np.sum], fill_value=0))
        df['발송일']=df['발송일'].map(str).map(date_change)
        df= df.groupby([df['발송일'], df['유형']])['수량'].sum().unstack('유형')
        df.fillna(0, inplace= True)

        df['y_sum']= df.sum(axis=1)
        print(df.columns)
        print(df.head(30))
    return dict_of_dfs

def output(inputfilename, outputfilename):
    dict_of_dfs= struct(inputfilename)
    return None


# Main

if __name__== '__main__':
    struct('sample_data.xlsx')
