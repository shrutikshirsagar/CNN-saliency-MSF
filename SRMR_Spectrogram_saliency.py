import os, sys
import scipy.io.wavfile as wav
from subprocess import call
import numpy as np
import pickle
import scipy.io
from srmrpy import srmr
from scipy import io, polyval, polyfit, sqrt, stats, signal
import tables
import librosa

def srmr_audio(path, file, FS=16000):
    """
    http://stft.readthedocs.io/en/latest/index.html
    Receive specific folder and file to extract Modulation Features
    All audio are resample to 16 kHz if FS is not specified
    """
    s, fs = librosa.load(file)
    #fs, s = wav.read('%s/%s' % (path, file))
    dim = len(s.shape)
    if (dim>1):
        s = s[:, 0]
    if (fs != FS):
        n_s = round(len(s) * (FS / fs))
        s = signal.resample(s, n_s)
    
    ratio, energy = srmr(s, FS)

    return energy

class MFStats(object):

    def __init__(self, mf_param):
        print("Modulation stats")
        self.mf_param = mf_param

    def get_mf_fea_1(self):
        """
        # Energy distribution of speech along the modulation frequency
        """
        mf_mean = np.mean(self.mf_param, axis=1)
        return mf_mean

    def get_mf_fea_2(self):
        """
        Spectral flatness
        """
        mf_flat = scipy.stats.gmean(self.mf_param, axis=1)/np.mean(self.mf_param, axis=1)
        return mf_flat

    def get_mf_fea_3(self):
        """
        Spectral centroid
        """
        multiplier = np.arange(1, 24)
        mf_num = np.einsum('i,kij->kj', multiplier, self.mf_param)
        mf_denom = np.einsum('kij->kj', self.mf_param)
        mf_cent = mf_num / mf_denom
        return mf_cent

    def get_mf_fea_4(self):
        """
        Modulation spectral centroid
        """
        mf_spect = np.empty(shape=[self.mf_param.shape[0], 0])
        multiplier = np.arange(1, 9)  # 8 Modulation Features
        idx = np.array([[0, 1, 2, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16, 17], [18, 19, 20, 21, 22]])
        for i in range(0, 5):
            aux = self.mf_param[:, idx[i], 0:8]
            mf_num = np.einsum('kij->kj', aux)
            mf_num = np.einsum('j,kj->k', multiplier, mf_num)
            mf_denom = np.einsum('kij->k', aux)
            mf_spect = np.column_stack((mf_spect, mf_num / mf_denom))
        return mf_spect

    def get_mf_fea_5(self):
        """
        Linear regression and square error
        """
        nObs = self.mf_param.shape[0]
        mf_slope = np.empty(shape=[nObs, 0])
        mf_err = np.empty(shape=[nObs, 0])
        xaxis = np.arange(1, 9)  # 8 Modulation Features
        x = np.reshape(xaxis, (1, 8))
        x = np.repeat(x, nObs, axis=0).T
        idx = np.array([[0, 1, 2, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16, 17], [18, 19, 20, 21, 22]])
        for i in range(0, 5):
            aux = self.mf_param[:, idx[i], 0:8]
            mf_vlr = np.einsum('kij->kj', aux).T
            (ar, br) = polyfit(xaxis, mf_vlr, 1)
            xr = polyval([ar, br], x)
            # compute the mean square error
            err = sqrt(sum((xr - mf_vlr) ** 2) / xaxis.shape[0])
            mf_slope = np.column_stack((mf_slope, ar))
            mf_err = np.column_stack((mf_err, err))
        return np.concatenate((mf_slope, mf_err), axis=1)

    def get_stats(self):
        mfstats = self.mf_param
        mfstats = np.reshape(mfstats, (mfstats.shape[0], mfstats.shape[1] * mfstats.shape[2]))
        mfstats = np.concatenate((mfstats, self.get_mf_fea_1()), axis=1)
        mfstats = np.concatenate((mfstats, self.get_mf_fea_2()), axis=1)
        mfstats = np.concatenate((mfstats, self.get_mf_fea_3()), axis=1)
        mfstats = np.concatenate((mfstats, self.get_mf_fea_4()), axis=1)
        mfstats = np.concatenate((mfstats, self.get_mf_fea_5()), axis=1)
        return mfstats

    def moving_stats(self,modfea, w_size):
        fea1 = np.empty(shape=[0, modfea.shape[1]])
        fea2 = np.empty(shape=[0, modfea.shape[1]])
        fea3 = np.empty(shape=[0, modfea.shape[1]])
        fea4 = np.empty(shape=[0, modfea.shape[1]])
        fea5 = np.empty(shape=[0, modfea.shape[1]])
        fea6 = np.empty(shape=[0, modfea.shape[1]])
        fea7 = np.empty(shape=[0, modfea.shape[1]])
        fea8 = np.empty(shape=[0, modfea.shape[1]])

        extraframe = int(w_size/2)

        modtmp = np.reshape(modfea[0], (1, len(modfea[0])))
        modtmp = np.repeat(modtmp, extraframe, axis=0)
        modfea = np.vstack((modtmp, modfea))

        modtmp = np.reshape(modfea[len(modfea)-1], (1, len(modfea[len(modfea)-1])))
        modtmp = np.repeat(modtmp, extraframe+1, axis=0)
        modfea = np.vstack((modfea, modtmp))

        print("Extracting mean...")
        for i in range(0, len(modfea)):
            mf_mean = np.mean(modfea[0+i:w_size+i], axis=0)
            fea1 = np.concatenate((fea1, np.reshape(mf_mean, (1, len(mf_mean)))), axis=0)
        print("Extracting std...")
        for i in range(0, len(modfea)):
            mf_std = np.std(modfea[0+i:w_size+i], axis=0)
            fea2 = np.concatenate((fea2, np.reshape(mf_std, (1, len(mf_std)))), axis=0)
        print("Extracting skewness...")
        for i in range(0, len(modfea)):
            mf_skewness = stats.skew(modfea[0+i:w_size+i], axis=0)
            fea3 = np.concatenate((fea3, np.reshape(mf_skewness, (1, len(mf_skewness)))), axis=0)
        print("Extracting kurtosis...")
        for i in range(0, len(modfea)):
            mf_kurtosis = stats.kurtosis(modfea[0+i:w_size+i], axis=0)
            fea4 = np.concatenate((fea4, np.reshape(mf_kurtosis, (1, len(mf_kurtosis)))), axis=0)
        print("Extracting range...")
        for i in range(0, len(modfea)):
            mf_ptp = np.ptp(modfea[0+i:w_size+i], axis=0)
            fea5 = np.concatenate((fea5, np.reshape(mf_ptp, (1, len(mf_ptp)))), axis=0)
        print("Extracting variance...")
        for i in range(0, len(modfea)):
            mf_variance = np.var(modfea[0+i:w_size+i], axis=0)
            fea6 = np.concatenate((fea6, np.reshape(mf_variance, (1, len(mf_variance)))), axis=0)
        print("Extracting min...")
        for i in range(0, len(modfea)):
            mf_min = np.amin(modfea[0+i:w_size+i], axis=0)
            fea7 = np.concatenate((fea7, np.reshape(mf_min, (1, len(mf_min)))), axis=0)
        print("Extracting max...")
        for i in range(0, len(modfea)):
            mf_max = np.amax(modfea[0+i:w_size+i], axis=0)
            fea8 = np.concatenate((fea8, np.reshape(mf_max, (1, len(mf_max)))), axis=0)

        mfstats = np.empty(shape=[fea1.shape[0], 0])
        mfstats = np.concatenate((mfstats, fea1), axis=1)
        mfstats = np.concatenate((mfstats, fea2), axis=1)
        mfstats = np.concatenate((mfstats, fea3), axis=1)
        mfstats = np.concatenate((mfstats, fea4), axis=1)
        mfstats = np.concatenate((mfstats, fea5), axis=1)
        mfstats = np.concatenate((mfstats, fea6), axis=1)
        mfstats = np.concatenate((mfstats, fea7), axis=1)
        mfstats = np.concatenate((mfstats, fea8), axis=1)

        return mfstats
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import glob






   
data_msf=np.empty((0,23,8))

pathin = "/media/amrgaballah/Backup_Plus/stress/IEMOCAP/Audio/*.wav"
file_label = '/media/amrgaballah/Backup_Plus/IEMPCAP_final_label.csv'
df = pd.read_csv(file_label)
label_a= np.empty((0,1))
label_v = np.empty((0,1))
for file in glob.glob(pathin):
    print('file', file)

   

    mf = srmr_audio("%s/" % (pathin), file)
    mf = np.einsum('ijk->kij', mf)

    stats = MFStats(mf)

 
    mrs = np.reshape(mf, (mf.shape[0], mf.shape[1] * mf.shape[2]))
    data1 = np.mean( mf, axis=0)
    print(data1.shape)
   

    data_msf=np.vstack((data_msf,data1[None]))
    print('final', data_msf.shape)
    filename1 = os.path.basename(file).split('.wa')[0]
    print(filename1)
    label_arousal1  = df.loc[df['filename']==filename1]['Arousal']
    label_valence1  = df.loc[df['filename']==filename1]['valence']
    label_a = np.vstack((label_a, label_arousal1))
    print(label_a.shape)
    label_v = np.vstack((label_v, label_valence1))
    print(label_v.shape)
data_file = h5py.File('/media/amrgaballah/Backup_Plus/stress/mod_feat11.hdf', 'w')
data_file.create_dataset('x', data=data_msf)
data_file.create_dataset('y', data=label_a)
data_file.create_dataset('z', data=label_v)
data_file.close()




