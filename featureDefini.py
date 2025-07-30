import numpy as np
import librosa

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path)
        if len(y) < 2048:  # çok kısa sesleri atla
            print(f"Uyarı: {file_path} çok kısa, atlandı.")
            return None

        features = []

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))

        # Chroma
        stft = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        features.extend(np.mean(chroma, axis=1))

        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        features.extend(np.mean(contrast, axis=1))

        # Zero-Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(y)
        features.extend(np.mean(zcr, axis=1))  # ZCR'nin ortalamasını ekliyoruz

        # Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        features.extend(np.mean(mel_spectrogram, axis=1))  # Mel Spectrogram'ın ortalamasını ekliyoruz

        # Root Mean Square Energy (RMSE)
        rmse = librosa.feature.rms(y=y)
        features.extend(np.mean(rmse, axis=1))  # RMSE'nin ortalamasını ekliyoruz

        # Spectral Roll-off
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)  # %85'lik enerji seviyesini kullan
        features.extend(np.mean(rolloff, axis=1))

        return np.array(features)
    except Exception as e:
        print(f"Hata: {file_path} işlenemedi. ({e})")
        return None
