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
from GAN.PixelShuffler import PixelShuffler
import soundfile
LRSHAPE = 129
HRSHAPE = 128
class CNN():
    def __init__(self,):
        self.generator = self.build_generator()
        self.generator.compile(loss='mse',optimizer=Adam(0.00001))
        self.data = joblib.load('data/ld_train')
        self.data_val = joblib.load('data/ld_val')

    def build_generator(self):
        audio_lr = Input(shape=(32, LRSHAPE))
        c1 = Conv1D(filters=256, kernel_size=7, strides=2, padding='same')(audio_lr)
        b1 = BatchNormalization()(c1)
        a1 = LeakyReLU(alpha=0.2)(b1)

        c2 = Conv1D(filters=512, kernel_size=5, strides=2, padding='same')(a1)
        b2 = BatchNormalization()(c2)
        a2 = LeakyReLU(alpha=0.2)(b2)

        # c3 = Conv1D(filters=512, kernel_size=3, strides=2, padding='same')(a2)
        # b3 = BatchNormalization()(c3)
        # a3 = LeakyReLU(alpha=0.2)(b3)
        #
        # c4 = Conv1D(filters=1024, kernel_size=3, strides=2, padding='same')(a3)
        # b4 = BatchNormalization()(c4)
        # a4 = LeakyReLU(alpha=0.2)(b4)
        #
        # c5 = Conv1D(filters=512, kernel_size=3, strides=1, padding='same')(a4)
        # u5 = UpSampling1D(size=2)(c5)
        # b5 = BatchNormalization()(u5)
        # a5 = LeakyReLU(alpha=0.2)(b5)
        # A5 = Add()([a5, a3])

        c6 = Conv1D(filters=512, kernel_size=5, strides=1, padding='same')(a2)
        # u6 = UpSampling1D(size=2)(c6)
        b6 = BatchNormalization()(c6)
        a6 = LeakyReLU(alpha=0.2)(b6)
        # A6 = Add()([a6, a2])

        c7 = Conv1D(filters=256, kernel_size=7, strides=1, padding='same')(a6)
        #u7 = PixelShuffler()(c7)
        u7 = UpSampling1D(size=2)(c7)
        b7 = BatchNormalization()(u7)
        a7 = LeakyReLU(alpha=0.2)(b7)
        # A7 = Add()([a7, a1])

        c8 = Conv1D(filters=HRSHAPE, kernel_size=7, strides=1, padding='same')(a7)
        #u8 = PixelShuffler()(c8)
        u8 = UpSampling1D(size=2)(c8)
        b8 = BatchNormalization()(u8)
        a8 = LeakyReLU(alpha=0.2)(b8)

        c9 = Conv1D(filters=HRSHAPE, kernel_size=9, strides=1, padding='same')(a8)
        # b9 = BatchNormalization()(c9)
        # a9 = LeakyReLU(alpha=0.2)(c9)

        return Model(audio_lr, c9)
        # audio_lr = Input(shape=(32,LRSHAPE))
        # c1 = Conv1D(filters=256, kernel_size=7, strides=2,padding='same')(audio_lr)
        # b1 = BatchNormalization()(c1)
        # a1 = LeakyReLU(alpha=0.2)(b1)
        #
        # c2 = Conv1D(filters=512, kernel_size=5, strides=2,padding='same')(a1)
        # b2 = BatchNormalization()(c2)
        # a2 = LeakyReLU(alpha=0.2)(b2)
        #
        # c3 = Conv1D(filters=512, kernel_size=3, strides=2,padding='same')(a2)
        # b3 = BatchNormalization()(c3)
        # a3 = LeakyReLU(alpha=0.2)(b3)
        #
        # c4 = Conv1D(filters=1024, kernel_size=3, strides=2,padding='same')(a3)
        # b4 = BatchNormalization()(c4)
        # a4 = LeakyReLU(alpha=0.2)(b4)
        #
        # c5 = Conv1D(filters=512, kernel_size=3, strides=1,padding='same')(a4)
        # u5 = UpSampling1D(size=2)(c5)
        # b5 = BatchNormalization()(u5)
        # a5 = LeakyReLU(alpha=0.2)(b5)
        # A5 = Add()([a5,a3])
        #
        # c6 = Conv1D(filters=512, kernel_size=5, strides=1,padding='same')(A5)
        # u6 = UpSampling1D(size=2)(c6)
        # b6 = BatchNormalization()(u6)
        # a6 = LeakyReLU(alpha=0.2)(b6)
        # A6 = Add()([a6,a2])
        #
        # c7 = Conv1D(filters=256, kernel_size=7, strides=1,padding='same')(A6)
        # u7 = UpSampling1D(size=2)(c7)
        # b7 = BatchNormalization()(u7)
        # a7 = LeakyReLU(alpha=0.2)(b7)
        # A7 = Add()([a7,a1])
        #
        # c8 = Conv1D(filters=HRSHAPE, kernel_size=7, strides=1,padding='same')(A7)
        # u8 = UpSampling1D(size=2)(c8)
        # b8 = BatchNormalization()(u8)
        # a8 = LeakyReLU(alpha=0.2)(b8)
        #
        # c9 = Conv1D(filters=HRSHAPE, kernel_size=9, strides=1,padding='same')(a8)
        # # b9 = BatchNormalization()(c9)
        # #a9 = LeakyReLU(alpha=0.2)(c9)
        #
        # return Model(audio_lr, c9)

if __name__ == '__main__':
    srgan = CNN()
    srgan.generator.summary()
    m, n = srgan.data.X_train.shape
    X_train = srgan.data.X_train[0:m // 32 * 32]
    X_train = X_train.reshape(m // 32, 32, n)

    m, n = srgan.data.Y_train.shape
    Y_train = srgan.data.Y_train[0:m // 32 * 32]
    Y_train = Y_train.reshape(m // 32, 32, n)

    m, n = srgan.data_val.X_train.shape
    X_train_val = srgan.data_val.X_train[0:m // 32 * 32]
    X_train_val = X_train_val.reshape(m // 32, 32, n)

    m, n = srgan.data_val.Y_train.shape
    Y_train_val = srgan.data_val.Y_train[0:m // 32 * 32]
    Y_train_val = Y_train_val.reshape(m // 32, 32, n)

    srgan.generator.load_weights("weights/cnn_gen.h5")
    srgan.generator.fit(X_train, Y_train, epochs=5, batch_size=32,validation_data=(X_train_val,Y_train_val))
    srgan.generator.save_weights("weights/cnn_gen.h5")
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
    def stft_specgram(x, i, picname=None, **params):  # picname是给图像的名字，为了保存图像
        f, t, zxx = stft(x, **params)
        plt.subplot(2, 1, i)
        plt.pcolormesh(t, f, np.abs(zxx))
        plt.colorbar()
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.tight_layout()
        plt.legend()
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
        # X_low = X_low[:,np.newaxis,:]#添加一维以符合网络的输入
        X_low = X_low[0:m // 32 * 32]
        X_high = X_high[0:m // 32 * 32]
        X_low_phase = X_low_phase[0:m // 32 * 32]
        X_high_phase = X_high_phase[0:m // 32 * 32]

        X_low = X_low.reshape(m // 32, 32, n)

        Y_hat = srgan.generator.predict(X_low)
        # Y_hat = X_high.reshape(m//32,32,n-1)

        X_low = X_low.reshape(m // 32 * 32, n)  # 还原

        # m, n = Y_hat.shape

        p, _, q = Y_hat.shape
        Y_hat = Y_hat.reshape(p * _, q)  # 还原
        # n_samples = len(waveform)
        n_samples = (m // 32 * 32 + 1) * 256

        Xhat_log_magnitude, Xhat_phase = reconstruct_low_high2(X_low, Y_hat, X_low_phase, X_high_phase)
        #Xhat_log_magnitude, Xhat_phase = reconstruct_low_high(X_low, Y_hat, X_low_phase)
        Xhat_log_magnitude = Xhat_log_magnitude[:, 0:512]  # 有514段，取前512段
        Xhat_phase = Xhat_phase[:, 0:512]
        Xhat = data_test.compose_stft(Xhat_log_magnitude, Xhat_phase)
        xhat = data_test.istft(Xhat, n_samples)


        #if (idx % 50 == 0):
            #stft_specgram(xhat, 2)
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


