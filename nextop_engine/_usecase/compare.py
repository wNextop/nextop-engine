from .._element.calculations import rmse


def AlgorithmCompare(testY, algorithm):
    global mockForecastDictionary
    nameOfBestAlgorithm = 'LSTM'
    minData = rmse(testY, mockForecastDictionary[nameOfBestAlgorithm])
    rms = 0
    for algorithm in mockForecastDictionary.keys():
        rms = rmse(testY, mockForecastDictionary[algorithm])
        if rms < minData:
            nameOfBestAlgorithm = algorithm
    print('testY is: ', testY)
    print('\n')
    print('LSTM forecast :', mockForecastDictionary['LSTM'], '\n@@@@@LSTM rmse: ',
          rmse(testY, mockForecastDictionary['LSTM']))
    print('Bayseian forecast :', mockForecastDictionary['Bayseian'], '\n@@@@@Bayseian rmse: ',
          rmse(testY, mockForecastDictionary['Bayseian']))
    print('\n')
    print(nameOfBestAlgorithm, 'WON!!!!!!')
    return nameOfBestAlgorithm
