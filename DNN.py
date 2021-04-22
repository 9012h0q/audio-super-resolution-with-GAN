import scipy
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding1D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.optimizers import Adam,RMSprop
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import keras.backend as K
import joblib
import numpy.random as random
import scipy.signal as signal
from load_data import LoadData
import soundfile

LRSHAPE = 129#65
HRSHAPE = 128#192
class DNN():
    def __init__(self,
                 n_input=LRSHAPE,  # n_frames (9) * narrow_band_window (256/2 + 1)
                 n_output=HRSHAPE,  # high_band_window (512/4 + 1)
                 n_hidden=256,
                 n_layers=4):
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.data = joblib.load('data/ld_train')
        self.data_val = joblib.load('data/ld_val')
        self.model = Sequential()
        self.model.add(Dense(self.n_hidden, input_shape=(self.n_input,)))
        self.model.add(Activation('relu'))

        for idx in range(self.n_layers):
            self.model.add(Dense(n_hidden))
            self.model.add(Activation('relu'))

        self.model.add(Dense(self.n_output))
        self.model.compile(optimizer='adam', loss='mse')  # , metrics=[])

if __name__ == '__main__':
    dnn = DNN()
    dnn.model.summary()
    history = dnn.model.fit(dnn.data.X_train,dnn.data.Y_train,epochs=1,batch_size=2048,validation_data=(dnn.data_val.X_train,dnn.data_val.Y_train))
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1,len(loss)-1)
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.legend()
    # plt.show()


    def reconstruct_low_high(X_low, X_high, X_low_phase):
        X_log_magnitude = np.hstack([X_low, X_high])
        flipped = X_log_magnitude[:, :][:, ::-1]  # 利用对称性质
        X_log_magnitude = np.hstack([X_log_magnitude, flipped])

        flipped = -1 * X_low_phase[:, :][:, ::-1]
        X_phase = np.hstack([X_low_phase, flipped])
        flipped = -1 * X_phase[:, :][:, ::-1]
        X_phase = np.hstack([X_phase, flipped])
        flipped = -1 * X_phase[:, :][:, ::-1]
        X_phase = np.hstack([X_phase, flipped])
        _, n = X_log_magnitude.shape
        X_phase = X_phase[:, 0:n]
        return X_log_magnitude, X_phase
    def reconstruct_low_high2(X_low, X_high, X_low_phase=None, X_high_phase=None):
        """ Reconstruct from X_low, Y_high and assume conjugate symmetry """

        # bug in preprocessing
        if X_high.shape[1] == 129:
            # Slice off first index
            X_high = X_high[:, 1:]

        # windows, N = X_log_magnitude.shape
        X_log_magnitude = np.hstack([X_low, X_high])

        # Conjugate symmetric only take non-redundant points
        # Slice last two indices and flip
        flipped = X_log_magnitude[:, 1:-1][:, ::-1]  # 利用对称性质
        X_log_magnitude = np.hstack([X_log_magnitude, flipped])

        if X_low_phase is not None and X_high_phase is not None:
            X_phase = np.hstack([X_low_phase, X_high_phase])
            # Multipl by -1 to take complex conjugate
            flipped_phase = -1 * X_phase[:, 1:-1][:, ::-1]  # 实信号，虚部相反，因此角度相反
            X_phase = np.hstack([X_phase, flipped_phase])
            return X_log_magnitude, X_phase
        else:
            return X_log_magnitude
    def stft(x, **params):
        '''
        :param x: 输入信号
        :param params: {fs:采样频率；
                        window:窗。默认为汉明窗；
                        nperseg： 每个段的长度，默认为256，
                        noverlap:重叠的点数。指定值时需要满足COLA约束。默认是窗长的一半，
                        nfft：fft长度，
                        detrend：（str、function或False）指定如何去趋势，默认为Flase，不去趋势。
                        return_onesided：默认为True，返回单边谱。
                        boundary：默认在时间序列两端添加0
                        padded：是否对时间序列进行填充0（当长度不够的时候），
                        axis：可以不必关心这个参数}
        :return: f:采样频率数组；t:段时间数组；Zxx:STFT结果
        '''
        f, t, zxx = signal.stft(x, **params)
        return f, t, zxx


    def stft_specgram(x, picname=None, **params):  # picname是给图像的名字，为了保存图像
        f, t, zxx = stft(x, **params)
        # plt.subplot(2,1,i)
        plt.pcolormesh(t, f, np.abs(zxx), cmap='gnuplot')
        plt.colorbar()
        plt.ylabel('Frequency(kHz)')
        plt.xlabel('Time(s)')
        plt.tight_layout()
        plt.legend()
        #plt.savefig(picname + '.svg', format="svg")
        plt.show()
        return t, f, zxx

    data_test = joblib.load('data/ld_test')
    idx = 0
    LSD, SNR = [], []
    for waveform in data_test.waveforms:
        # if(idx % 50 == 0):
        #     stft_specgram(waveform, 1)

        X = data_test.stft(waveform, 512, 256)
        X_log_magnitude, X_phase = data_test.decompose_stft(X)
        X_low, X_high, X_low_phase, X_high_phase = data_test.extract_low_high(X_log_magnitude, X_phase)
        m, n = X_low.shape
        #X_low = X_low[:, np.newaxis, :]  # 添加一维以符合网络的输入
        Y_hat = dnn.model.predict(X_low)

        #X_low = X_low.reshape(m, n)  # 还原

        # m, n = Y_hat.shape

       # m, p, n = Y_hat.shape
        #Y_hat = Y_hat.reshape(m, n)  # 还原
        n_samples = len(waveform)

        Xhat_log_magnitude, Xhat_phase = reconstruct_low_high2(X_low, Y_hat, X_low_phase, X_high_phase)
        #Xhat_log_magnitude, Xhat_phase = reconstruct_low_high(X_low, Y_hat, X_low_phase)
        Xhat_log_magnitude = Xhat_log_magnitude[:, 0:512]  # 有514段，取前512段
        Xhat_phase = Xhat_phase[:, 0:512]
        Xhat = data_test.compose_stft(Xhat_log_magnitude, Xhat_phase)
        xhat = data_test.istft(Xhat, n_samples)

        if (idx % 50 == 0):
            stft_specgram(xhat, "DNN")
        elif (idx % 50 == 9):
            stft_specgram(xhat, "DNN2")
        soundfile.write('data/test_output/' + str(idx) + '.wav', xhat, 16000)  # 48000
        idx += 1

        a, b = Y_hat.shape
        lsd, snr = [], []
        for i in range(a):
            x1 = np.sqrt(np.average((X_high[i] - Y_hat[i]) ** 2))
            x2 = np.sum((X_high[i] - Y_hat[i]) ** 2)
            x3 = np.sum(X_high[i] ** 2)
            lsd.append(x1)
            snr.append(10 * np.log10(x3 / x2))
        lsd = np.average(lsd)
        snr = np.average(snr)
        LSD.append(lsd)
        SNR.append(snr)
    print("SNR={0},LSD={1}".format(np.average(SNR), np.average(LSD)))
