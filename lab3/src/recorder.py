import os
import sounddevice as sd
from time import time
from scipy.io.wavfile import write
from api import send
import base64
import numpy as np
from scipy.io.wavfile import read
from scipy.signal import resample_poly


store_audio = True

def preprocess_audio(audio):
    sample_rate = 48000
    audio_mono = audio.reshape(-1)
    audio_float32 = audio_mono.astype(np.float32) / (2**31)
    audio_16k = resample_poly(audio_float32, up=16000, down=sample_rate)
    encoded = base64.b64encode(audio_16k.tobytes()).decode('ascii')
    sampling_rate = 16000
    return (encoded, sampling_rate)

def callback(indata, frames, callback_time, status):
    global store_audio

    if status:
        print(f'Audio status: {status}')

    if store_audio:
        audio, sr = preprocess_audio(indata)
        response = send(audio, sr)
        prediction = response.get('output', None)
        print(f'Prediction: {prediction}')
        # timestamp = time()
        # filename = f'output/{timestamp}.wav'
        # write(filename, 48000, indata)
        # filesize_in_kb = os.path.getsize(filename) / 1024
        # print(f'Size: {filesize_in_kb:.2f} KB')

def main():
    global store_audio
    with sd.InputStream(
        device=1,  # replace with the USB microphone's device index
        channels=1,
        dtype='int32',
        samplerate=48000,
        blocksize=48000,
        callback=callback,
    ):
        while True:
            key = input('Press Q to stop or P to pause/resume saving: ')
            if key in ('q', 'Q'):
                print('Stop recording.')
                break
            if key in ('p', 'P'):
                store_audio = not store_audio

if __name__ == '__main__':
    main()