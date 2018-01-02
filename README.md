# nextop-engine

Powerful python package which is for time-series-prediction & industrial-engineering-optimization

## for contributers (협업자들을 위한 안내)

`./nextop-engine` 이란 디렉터리가 주된 python package 입니다.

나머지 파일들은 package 의 설치를 돕는 파일들입니다

contributer들 께서는 패키지의 디버깅이나 성능 개선 작업을 `./nextop-engine` 안에 있는 파일들로 해주시면 감사하겠습니다.

패키지 내 구성에 대해 간단히 소개하겠습니다.

### 1. `_element` package

`_element` package 는 기본적인 계산이나 수식들을 포함합니다.

`_usecase` 에서 사용될 재료들이라고 생각할 수 있습니다.

`nextop-engine` 패키지 사용자에 의해 직접 사용될 일은 없습니다.

### 2. `_usecase` package

`_usecase` package 는 자주 사용되는 동작들을 정의합니다.

`_usecase` package 는 `ui` 패키지 내의 모듈들에서 사용될 재료들이라고 생각할 수 있습니다.

`nextop-engine` 패키지 사용자에 의해 직접 사용될 일은 없습니다.

#### `algorithm.py`

LSTM, Facebook prophet , ARIMA 등 예측 알고리즘들을 정의하고 있습니다.

#### `compare.py`

알고리즘 성능 비교 동작을 정의하고 있습니다.

### 3. `ui` module

`ui`는 패키지에서 직접 사용되는 부분으로 `_usecase` 들을 실행해 직접적인 작업을 처리합니다.

## change log (주요 수정 내역)

2018-01-02
    .gitignore를 생성했습니다. .gitingore에는 csv, xlsx 등 데이터 파일, env 파일 등이 포함됩니다. 데이터 등도 지금은 아무데나 넣어 놨는데 이후에 정리할 수 있으면 정리하겠습니다.
    feature 설정이나 추가 등을 위한 메서드를 모은 feature_control.py를 분리했습니다. 이후 데이터 전처리, 조정과 관련된 함수를 여기에 모으도록 하겠습니다.
    데이터를 날짜별/유형별로 정리하는 struct함수를 feature_control.py에 작성했습니다. 이후에는 컬럼 명을 넣으면 자동으로 정리될 수 있도록 하면 좋을 것 같지만 지금 당장은 저 정도만 정리해 두었어요.
