from fbprophet import Prophet
import pandas as pd
# help(Prophet.add_regressor)

def weatherbinary(ds):
    date = pd.to_datetime(ds)
    if date.weakday() == 6 and (date.month > 8 or date.month < 2):
        return 1
    else:
        return 0

date = {'state' : ['ohio', 'ohio', 'ohio', 'nevada', 'nevada'],
        'year' : [2000, 2001, 2002, 2001, 2002],
        'pop' : [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = pd.DataFrame(date, columns=['year', 'state', 'pop'])
frame['one'] = 1

print(frame)