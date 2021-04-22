import os
import scipy.io.wavfile
import librosa
import numpy as np
import scipy.signal as sps
import sys
import joblib
import scipy as sp
from moviepy.editor import*
#import musdb
class LoadData(object):
    def __init__(self, high_window_size=512,
                 high_window_shift=256,
                 low_window_size=256,
                 low_window_shift=128,
                 sampling_rate=16000,#48000
                 data_dir='data/musdb18/test/',
                 wave_dir='data/musdbwav/test/testdata/',
                 wave_dirs='data/musdbwav/test/',
                 upsample=2
                 ):
        #self.mus = musdb.DB(root=data_dir,subsets=['train'])

        self.high_window_size = high_window_size
        self.high_window_shift = high_window_shift
        self.low_window_size = low_window_size
        self.low_window_shift = low_window_shift
        self.data_dir = data_dir
        self.wave_dir = wave_dir
        self.wave_dirs = wave_dirs
        self.sampling_rate = sampling_rate
        self.upsample = upsample

        #self.creatdata(data_dir)
        self.waveforms=self.loaddata(wave_dirs)
        self.create_training_set(self.waveforms)

    def creatdata(self,data_dir):
        files=[]
        for i in os.listdir(data_dir):
            j=os.path.join(data_dir,i)
            files.append(j)
        waveforms=[]
        for file in files:
            subfile=[]
            m = 0
            for i in os.listdir(file):
                j = os.path.join(file, i)
                subfile.append(j)
            for x in subfile:
                m +=1
                audio = VideoFileClip(x).audio
                audio.write_audiofile(self.wave_dir+str(m)+'.wav')

    def loaddata(self,data_dir):
        files = []
        for i in os.listdir(data_dir):
            j = os.path.join(data_dir, i)
            files.append(j)
        waveforms = []
        for file in files:
            subfile = []
            for i in os.listdir(file):
                j = os.path.join(file, i)
                subfile.append(j)
            for x in subfile:
                waveform, rate = librosa.load(x,sr=self.sampling_rate)
                waveforms.append(waveform)
        return waveforms

    def create_training_set(self, train_waveforms):
        """
        Create training and validation set
        and compute mean and correllation matrix
        """
        print("Extracting features...")

        X_train, Y_train, X_train_phase, Y_train_phase = \
            self.pipeline(train_waveforms)
        # self.X_train = X_train
        # self.Y_train = Y_train
        # self.X_train_phase = X_train_phase
        # self.Y_train_phase = Y_train_phase
        self.X_train = np.vstack(X_train)
        self.Y_train = np.vstack(Y_train)
        self.X_train_phase = np.vstack(X_train_phase)
        self.Y_train_phase = np.vstack(Y_train_phase)

    def pipeline(self, waveforms):
        """ Takes generator of waveforms and returns generator of
            low-band and high-band features """
        X_lows, X_highs, X_lows_phase, X_highs_phase = [], [], [], []

        for waveform in waveforms:
                # First high band features
                X = self.stft(waveform, self.high_window_size, self.high_window_shift)
                X_log_magnitude, X_phase = self.decompose_stft(X)
                X_low, X_high, X_low_phase, X_high_phase = \
                    self.extract_low_high(X_log_magnitude, X_phase)
                X_lows.append(X_low)
                X_highs.append(X_high)
                X_lows_phase.append(X_low_phase)
                X_highs_phase.append(X_high_phase)
        return X_lows, X_highs, X_lows_phase, X_highs_phase

    def stft(self, x, window_size=512, window_shift=256):
        """ STFT with non-symmetric Hamming window """
        w = sps.hamming(window_size, sym=False)
        X = np.array([sp.fft.fft(w * x[i:i + window_size])
                      for i in range(0, len(x) - window_size, window_shift)])
        return X

    def istft(self, X, n_samples, window_shift=256):
        """ iSTFT with symmetric Hamming window """
        n_windows, window_size = X.shape
        # x_len = window_size + (n_windows-1)*window_shift

        x = sp.zeros(n_samples)

        for n, i in enumerate(range(0, len(x) - window_size, window_shift)):
            x[i:i + window_size] += sp.real(sp.ifft(X[n]))
        return x

    def decompose_stft(self, X):
        """ Takes windowed STFT and compute ln mag and phase """
        # Replace zeros with fudge
        X[X == 0] = 1e-8
        X_log_magnitude = 2 * np.log(np.absolute(X))
        X_phase = np.angle(X, deg=False)  # 弧度制表示复数的角度

        return X_log_magnitude, X_phase

    def extract_low_high(self, X_log_magnitude, X_phase, split=True):
        """ Extract high and low bands from X_log_magnitude """

        def split(X, n):
            """ Takes as input array X and returns a split column at X[:,n] """
            return X[:, :n], X[:, n:]

        windows, N = X_log_magnitude.shape

        # Conjugate symmetric only take non-redundant points
        X_log_magnitude = X_log_magnitude[:, :(N // 2) + 1]
        # Conjugate symmetric only take non-redundant points
        X_phase = X_phase[:, :(N // 2) + 1]

        # If we want to split into high and low components
        # I break out the cases manually because it's easier to follow than eqn
        if split:  # 进一步降采样
            if self.upsample == 2:
                X_low, X_high = split(X_log_magnitude, (N // 4) + 1)
                X_low_phase, X_high_phase = split(X_phase, (N // 4) + 1)
            elif self.upsample == 4:
                X_low, X_high = split(X_log_magnitude, (N // 8) + 1)
                X_low_phase, X_high_phase = split(X_phase, (N // 8) + 1)
            elif self.upsample == 6:
                X_low, X_high = split(X_log_magnitude, int(np.ceil((N // 12) + 1)))
                X_low_phase, X_high_phase = split(X_phase, int(np.ceil((N // 12) + 1)))
            elif self.upsample == 8:
                X_low, X_high = split(X_log_magnitude, (N // 16) + 1)
                X_low_phase, X_high_phase = split(X_phase, (N // 16) + 1)

            return X_low, X_high, X_low_phase, X_high_phase
        else:
            return X_log_magnitude

    def reconstruct_low_high(self, X_low, X_high, X_low_phase=None, X_high_phase=None):
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

    def compose_stft(self, X_log_magnitude, X_phase):
        """ Do reverse operation of decompose_stft """
        return np.exp(0.5 * X_log_magnitude + 1j * X_phase)

if __name__ == "__main__":
  ld = LoadData()
  joblib.dump(ld,'data/ld_test')
