# -*- coding: utf-8 -*-
'''
학교종이 땡땡땡 악보코드
g4, g4, a4, a4, g4, g4, e4, g4, g4, e4, e4, d4,  
g4, g4, a4, a4, g4, g4, e4, g4, e4, d4, e4, c4
'''
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import np_utils


np.random.seed(5)

# loss값의 이력을 저장합니다.
class LossHistory(keras.callbacks.Callback):
    def init(self):
        self.losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# 데이터를 학습에 맞게 조각내는 함수입니다.
def seq2dataset(seq, window_size):
    dataset = []
    for i in range(len(seq)-window_size):
        subset = seq[i:(i+window_size+1)]
        dataset.append([code2idx[item] for item in subset])
    return np.array(dataset)        



# 계이름과 음정의 코드화합니다.

code2idx = {'c4':0, 'd4':1, 'e4':2, 'f4':3, 'g4':4, 'a4':5, 'b4':6}

idx2code = {0:'c4', 1:'d4', 2:'e4', 3:'f4', 4:'g4', 5:'a4', 6:'b4'}

# 악보를 코드화하여 정의합니다.

seq = [g4, g4, a4, a4, g4, g4, e4, g4, g4, e4, e4, d4,  
g4, g4, a4, a4, g4, g4, e4, g4, e4, d4, e4, c4]

# 데이터를 신경망에 맞게 조각냅니다.

dataset = seq2dataset(seq, window_size = 4)

print(dataset.shape)

# 입력과 출력(정답)을 나눕니다.
x_train = dataset[:,0:4]
y_train = dataset[:,4]

max_idx_value = 13

# 입력값을 정규화합니다.
x_train = x_train / float(max_idx_value)

# 입력을 (샘플 수, 타임스텝, 특성 수)로 형태 변환합니다.
x_train = np.reshape(x_train, (24, 4, 1))

# 라벨값에 대한 one-hot 인코딩 수행
y_train = np_utils.to_categorical(y_train)

one_hot_vec_size = y_train.shape[1]

print("one hot encoding vector size is ", one_hot_vec_size)

# 모델을 설계합니다.
model = Sequential()
model.add(LSTM(128, batch_input_shape = (1, 4, 1), stateful=True))
model.add(Dense(one_hot_vec_size, activation='softmax'))
    
# 모델 학습과정 설정합니다.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습시키기
num_epochs = 250

history = LossHistory() #loss값을 저장할 객체를 생성합니다.

history.init()

for epoch_idx in range(num_epochs):
    print ('epochs : ' + str(epoch_idx) )
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2, shuffle=False, callbacks=[history]) # 50 is X.shape[0]
    model.reset_states()
    
# 학습과정을 모니터링합니다.
# %matplotlib inline
import matplotlib.pyplot as plt

plt.plot(history.losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# 모델의 정확도를 평가합니다.
scores = model.evaluate(x_train, y_train, batch_size=1)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
model.reset_states()

# 모델을 사용합니다.

pred_count = 24 # 최대 예측 개수 정의

# 한 스텝을 예측합니다.

seq_out = ['g8', 'e8', 'e4', 'f8']
pred_out = model.predict(x_train, batch_size=1)

for i in range(pred_count):
    idx = np.argmax(pred_out[i]) # one-hot 인코딩을 인덱스 값으로 변환합니다.
    seq_out.append(idx2code[idx]) # seq_out는 최종 악보이므로 인덱스 값을 코드로 변환하여 저장합니다.

model.reset_states()
    
print("one step prediction : ", seq_out)

# 곡 전체를 예측합니다

seq_in = ['g8', 'e8', 'e4', 'f8']
seq_out = seq_in
seq_in = [code2idx[it] / float(max_idx_value) for it in seq_in] # 코드를 인덱스값으로 변환

for i in range(pred_count):
    sample_in = np.array(seq_in)
    sample_in = np.reshape(sample_in, (1, 4, 1)) # 샘플 수, 타입스텝 수, 속성 수
    pred_out = model.predict(sample_in)
    idx = np.argmax(pred_out)
    seq_out.append(idx2code[idx])
    seq_in.append(idx / float(max_idx_value))
    seq_in.pop(0)

model.reset_states()
    
print("full song prediction : ", seq_out)
