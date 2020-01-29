# LSTM1
LSTM is model 

This is a code to train the LSTM neural network by coding the system name and pitch length of the song.
The title of the song is "School Paper." Coded "c" = c, "d" = d in order, and the number after the alphabet means the length of the pitch.

Accuracy is close to 100%, and many LSTM cases have a high probability of doing this.
Chord constructed a neural network that only listens to the first four syllables and predicts the rest of the song.

See the LSTM.ipynb file and lstm.py file for more information.


노래의 계이름과 음정길이를 코드화하여 LSTM 신경망으로 훈련시키는 코드입니다.
노래제목은 "학교종이 땡땡땡"이며 "c"="도", "d"="레" 등으로 순서대로 코드화 하였고 알파벳뒤의 숫자는 음정의 길이를 의미합니다.

500번의 훈련횟수가 넘어가면서부터는 전체곡을 100% 구현해내는데 성공하였습니다. 데이터에 따라 다르겠지만 LSTM의 경우 대게 정상패턴 학습효율이 높게 나타납니다.
코드는 처음 네음절만 듣고 나머지 전체곡을 예측해내는 신경망을 구성하였습니다.

자세한 내용은 LSTM.ipynb 파일과 lstm.py 파일을 참고하십시오.



