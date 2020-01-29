노래의 계이름과 음정길이를 코드화하여 LSTM 신경망으로 훈련시키는 코드입니다.
노래제목은 "학교종이 땡땡땡"이며 "c"="도", "d"="레" 등으로 순서대로 코드화 하였고 알파벳뒤의 숫자는 음정의 길이를 의미합니다.

500번의 훈련횟수가 넘어가면서부터는 전체곡을 100% 구현해내는데 성공하였습니다. 데이터에 따라 다르겠지만 LSTM의 경우 대게 정상패턴 학습효율이 높게 나타납니다.
코드는 처음 네음절만 듣고 나머지 전체곡을 예측해내는 신경망을 구성하였습니다.

예측 그래프의 급격한 피치들은 데이터의 특성상 하나만 틀려도 전체패턴이 달라지므로 나오는 현상입니다.
전체 학습횟수가 늘어날수록 피치가 줄어들고 완만해져갑니다. 실험이 필요하신 분들은 epoch 횟수를 늘려보시기 바랍니다.

자세한 내용은 LSTM.ipynb 파일과 lstm.py 파일을 참고하십시오.


<br>
<img width="459" alt="스크린샷 2020-01-29 오후 11 51 13" src="https://user-images.githubusercontent.com/45910733/73367170-7e7f1d00-42f2-11ea-9bef-acbeb448aa9e.png">
<img width="1202" alt="스크린샷 2020-01-29 오후 11 52 14" src="https://user-images.githubusercontent.com/45910733/73367174-8048e080-42f2-11ea-9f55-720a908c7d54.png">
