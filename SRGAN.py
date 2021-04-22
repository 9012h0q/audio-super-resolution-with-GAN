import scipy
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding1D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.optimizers import Adam,RMSprop
from keras.callbacks import ModelCheckpoint
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import keras.backend as K
import joblib
import numpy.random as random
from load_data import LoadData
LRSHAPE = 129
HRSHAPE = 128
class SRGAN():
    def __init__(self,):
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=Adam(0.0000001,0.5),
                                   metrics=['accuracy'])
        self.data = joblib.load('data/ld_train')
        self.data_val = joblib.load('data/ld_val')

        self.discriminator.trainable = False
        audio_lr = Input(shape=(32,LRSHAPE))
        fake_hr = self.generator(audio_lr)
        validity = self.discriminator(fake_hr)
        self.combined = Model(audio_lr, [validity,fake_hr])
        self.combined.compile(loss=['binary_crossentropy','mse'],
                              loss_weights=[0.001,0.1],
                              optimizer=Adam(0.0001,0.5))
        # self.combined = Model(audio_lr,validity)
        # self.combined.compile(loss='binary_crossentropy',
        #                       optimizer=Adam(0.00005))

    #一次取时间上相邻的32帧，总共取batch_size次
    def load_data(self,batch_size=64,idx=0):
        n_examples, n_features = self.data.X_train.shape
        X1, Y1 = [],[]
        X2, Y2 = [],[]
        for i in range(batch_size):
            idx = random.randint(0, high=n_examples-32)
            X = self.data.X_train[idx:idx+32]
            Y = self.data.Y_train[idx:idx+32]
            X1.append(X)
            Y1.append(Y)
            X2 = np.array(X1)
            Y2 = np.array(Y1)
            idx += 32
        return X2,Y2
    def load_data_val(self,batch_size=64,idx=0):
        n_examples, n_features = self.data_val.X_train.shape
        X1, Y1 = [],[]
        X2, Y2 = [],[]
        for i in range(batch_size):
            idx = random.randint(0, high=n_examples-32)
            X = self.data.X_train[idx:idx+32]
            Y = self.data.Y_train[idx:idx+32]
            X1.append(X)
            Y1.append(Y)
            X2 = np.array(X1)
            Y2 = np.array(Y1)
            idx += 32
        return X2,Y2
    def scheduler(self,models,epoch):
        # 学习率下降
        if epoch % 100 == 0 and epoch != 0:
            for model in models:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, lr * 0.5)
    def train(self, epochs=100, batch_size=64):
        Dloss1,Dloss,Gloss,Gloss_val=[],[],[],[]
        self.generator.load_weights("weights/gen_epoch.h5")
        #self.discriminator.load_weights("weights/dis_epoch.h5")
        for epoch in range(epochs):
            #self.scheduler([self.discriminator], epoch)
            #训练判别网络
            lr,hr = self.load_data(batch_size,idx=epoch*batch_size*32)
            fake_hr = self.generator.predict(lr)
            valid = np.ones((batch_size,1))
            fake = np.zeros((batch_size,1))
            labels = np.concatenate([fake,valid])
            #labels += 0.05*np.random.random(labels.shape)#给标签中添加随机噪声
            #hr_fake_hr = np.concatenate([hr, fake_hr])#标签互换
            hr_fake_hr = np.concatenate([fake_hr, hr])#标签不互换
            d_loss = self.discriminator.train_on_batch(hr_fake_hr, labels)

            #训练生成网络
            lr, hr = self.load_data(batch_size,idx=epoch*batch_size*32)
            valid = np.ones((batch_size,1))
            self.discriminator.trainable = False
            g_loss = self.combined.train_on_batch(lr, [valid,hr])
            #g_loss = self.combined.train_on_batch(lr,valid)

            #验证集
            lr, hr = self.load_data_val(batch_size,idx=epoch*batch_size*32)
            g_loss_val = self.combined.evaluate(lr,[valid,hr])
            #g_loss_val = self.combined.evaluate(lr,valid)
            #每一轮都保存权重
            #self.generator.save_weights("weights/gen_epoch"+str(epoch)+".h5")

            Gloss_val.append(g_loss_val[1])
            Gloss.append(g_loss[1])
            # Gloss.append(g_loss)
            # Gloss_val.append(g_loss_val)
            Dloss.append(d_loss[0])
            Dloss1.append(d_loss[1])

        os.makedirs('weights' , exist_ok=True)

        plt.plot(range(1, epochs + 1), Dloss, label='Dloss')
        plt.plot(range(1, epochs + 1), Dloss1, label='Daccuracy')
        plt.plot(range(1,epochs+1),Gloss,label='Gloss')
        plt.plot(range(1,epochs+1),Gloss_val,label='Gval_loss')
        #plt.title('Drate:1e-6 Grate:1e-4 ')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()
        self.generator.save_weights("weights/gen_epoch.h5" )
        self.discriminator.save_weights("weights/dis_epoch.h5" )
    def build_generator(self):
        audio_lr = Input(shape=(32,LRSHAPE))
        c1 = Conv1D(filters=256, kernel_size=7, strides=2,padding='same')(audio_lr)
        b1 = BatchNormalization()(c1)
        a1 = LeakyReLU(alpha=0.2)(b1)

        c2 = Conv1D(filters=512, kernel_size=5, strides=2,padding='same')(a1)
        b2 = BatchNormalization()(c2)
        a2 = LeakyReLU(alpha=0.2)(b2)

        c3 = Conv1D(filters=512, kernel_size=3, strides=2,padding='same')(a2)
        b3 = BatchNormalization()(c3)
        a3 = LeakyReLU(alpha=0.2)(b3)

        c4 = Conv1D(filters=1024, kernel_size=3, strides=2,padding='same')(a3)
        b4 = BatchNormalization()(c4)
        a4 = LeakyReLU(alpha=0.2)(b4)

        c5 = Conv1D(filters=512, kernel_size=3, strides=1,padding='same')(a4)
        u5 = UpSampling1D(size=2)(c5)
        b5 = BatchNormalization()(u5)
        a5 = LeakyReLU(alpha=0.2)(b5)
        A5 = Add()([a5,a3])

        c6 = Conv1D(filters=512, kernel_size=5, strides=1,padding='same')(A5)
        u6 = UpSampling1D(size=2)(c6)
        b6 = BatchNormalization()(u6)
        a6 = LeakyReLU(alpha=0.2)(b6)
        A6 = Add()([a6,a2])

        c7 = Conv1D(filters=256, kernel_size=7, strides=1,padding='same')(A6)
        u7 = UpSampling1D(size=2)(c7)
        b7 = BatchNormalization()(u7)
        a7 = LeakyReLU(alpha=0.2)(b7)
        A7 = Add()([a7,a1])

        c8 = Conv1D(filters=HRSHAPE, kernel_size=7, strides=1,padding='same')(A7)
        u8 = UpSampling1D(size=2)(c8)
        b8 = BatchNormalization()(u8)
        a8 = LeakyReLU(alpha=0.2)(b8)

        c9 = Conv1D(filters=HRSHAPE, kernel_size=9, strides=1,padding='same')(a8)
        # b9 = BatchNormalization()(c9)
        # c9 = LeakyReLU(alpha=0.2)(c9)

        return Model(audio_lr, c9)
    def build_discriminator(self):
        audio_hr = Input(shape=(32,HRSHAPE))
        c1 = Conv1D(filters=1024, kernel_size=(7,), strides=(2,), padding='same')(audio_hr)
        a1 = LeakyReLU(alpha=0.2)(c1)

        c2 = Conv1D(filters=1024, kernel_size=(5,), strides=(2,), padding='same')(a1)
        a2 = LeakyReLU(alpha=0.2)(c2)

        c3 = Conv1D(filters=1024, kernel_size=(3,), strides=(2,), padding='same')(a2)
        a3 = LeakyReLU(alpha=0.2)(c3)

        f1 = Flatten()(a3)
        # d1 = Dense(4096)(f1)
        # a4 = LeakyReLU(alpha=0.2)(d1)
        d2 = Dense(2048)(f1)
        a5 = LeakyReLU(alpha=0.2)(d2)
        d3 = Dense(1,activation='sigmoid')(a5)
        return Model(audio_hr,d3)

if __name__ == '__main__':
    srgan = SRGAN()
    #srgan.generator.summary()
    srgan.discriminator.summary()
    srgan.train(100,64)


