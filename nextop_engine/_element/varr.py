import pandas as pd
from datetime import datetime, timedelta

newyear = pd.DataFrame({
    'holiday': 'newyear',
    'ds': pd.to_datetime(['2011-02-03', '2012-01-23',
                          '2013-02-10', '2014-01-31', '2015-02-19',
                          '2016-02-09', '2017-02-28', '2018-02-16']),
    'lower_window': -2,
    'upper_window': 2,
})

thanksgiving = pd.DataFrame({
    'holiday': 'thanksgiving',
    'ds': pd.to_datetime(['2010-09-22', '2011-09-12', '2012-09-30',
                          '2013-09-19', '2014-09-09', '2015-09-27',
                          '2016-09-15', '2017-10-04', '2018-09-24']),
    'lower_window': -2,
    'upper_window': 2,
})

chocostick = pd.DataFrame({
    'holiday': 'chocostick',
    'ds': pd.to_datetime(['2010-11-11', '2011-11-11', '2012-11-11',
                          '2013-11-11', '2014-11-11', '2015-11-11',
                          '2016-11-11', '2017-11-11', '2018-11-11']),
    'lower_window': -2,
    'upper_window': 2,
})

christmas = pd.DataFrame({
    'holiday': 'christmas',
    'ds': pd.to_datetime(['2010-12-25', '2011-12-25', '2012-12-25',
                          '2013-12-25', '2014-12-25', '2015-12-25',
                          '2016-12-25', '2017-12-25', '2018-12-25']),
    'lower_window': -2,
    'upper_window': 2,
})

thanksgivingbefore = pd.DataFrame({
    'holiday': 'thanksgivingbefore',
    'ds': pd.to_datetime(['2010-09-14', '2011-09-04', '2012-09-22',
                          '2013-09-11', '2014-09-01', '2015-09-19',
                          '2016-09-07', '2017-09-26', '2018-09-16']),
    'lower_window': -4,
    'upper_window': 4,
})

newyearbefore = pd.DataFrame({
    'holiday': 'newyearbefore',
    'ds': pd.to_datetime(['2011-01-26', '2012-01-15',
                          '2013-02-02', '2014-01-23', '2015-02-11',
                          '2016-02-01', '2017-02-20', '2018-02-08']),
    'lower_window': -4,
    'upper_window': 4,
})

HOLYDAYBETA = pd.concat((newyear, thanksgiving, chocostick, christmas, newyearbefore, thanksgivingbefore))
TEMP_DATA_DIR= '\\_element\\data\\temp_data'
ALG_PRPPHET_DIR= '\\_usecase\\algorithm_prophet.py'
INPUT_FILENAME= 'KPP일별투입(10_17)_restructured_restructured.xlsx'
DF_DIR= '\\_element\\data\\private\\'
COLNAME_KPPDAILY= ['ds', 'rain_amount', 'temp_max', 'temp_min', 'y']
START_DATE= datetime(2010, 7, 1)
START_DATE_STR= START_DATE.strftime("%Y-%m-%d")
FORECASTDAY= 30
LAST_DATE= datetime(2017, 11, 30)
END_DATE= (LAST_DATE - timedelta(days=FORECASTDAY))
END_DATE_STR= END_DATE.strftime("%Y-%m-%d")
# print(holidaybeta)
#           ds       holiday  lower_window  upper_window
# 0 2011-02-03       newyear            -1             1
# 1 2012-01-23       newyear            -1             1
# 2 2013-02-10       newyear            -1             1
# 3 2014-01-31       newyear            -1             1
# 4 2015-02-19       newyear            -1             1
# 5 2016-02-09       newyear            -1             1
# 6 2017-02-28       newyear            -1             1
# 7 2018-02-16       newyear            -1             1
# 0 2010-09-22  thanksgiving            -1             1
# 1 2011-09-12  thanksgiving            -1             1
# 2 2012-09-30  thanksgiving            -1             1
# 3 2013-09-19  thanksgiving            -1             1
# 4 2014-09-09  thanksgiving            -1             1
# 5 2015-09-27  thanksgiving            -1             1
# 6 2016-09-15  thanksgiving            -1             1
# 7 2017-10-04  thanksgiving            -1             1
# 8 2018-09-24  thanksgiving            -1             1
# 0 2010-11-11    chocostick             0             0
# 1 2011-11-11    chocostick             0             0
# 2 2012-11-11    chocostick             0             0
# 3 2013-11-11    chocostick             0             0
# 4 2014-11-11    chocostick             0             0
# 5 2015-11-11    chocostick             0             0
# 6 2016-11-11    chocostick             0             0
# 7 2017-11-11    chocostick             0             0
# 8 2018-11-11    chocostick             0             0
