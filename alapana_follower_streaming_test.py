import time

import librosa
import matplotlib.pyplot as plt
import numpy as np
from alapana_nn.synth import simple_synth
import soundfile as sf
import wave
import sys
import pyaudio
from alapana_nn.utils import Util
from alapana_nn.pitchTrack import PitchTrack
import threading
import queue


threshold = 0.03
dev = "cpu"
pitch_tracker = PitchTrack('rmvpe', rmvpe_threshold=threshold, device=dev)
CHUNK = 2048
num_chunks = 1000


class Worker(threading.Thread):
    def __init__(self, q: queue.Queue, sample_rate=16000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q = q
        self.fs = sample_rate
        self.pitch_tracker = PitchTrack('rmvpe', rmvpe_threshold=0.03, device='cpu')
        self.midi = []
        self.running_mean = []

    def run(self):
        prev_x = np.zeros((CHUNK,), dtype=float)

        # dummy run to initialize
        cat_x = np.hstack([prev_x, prev_x])
        m = self.pitch_tracker.track(audio=cat_x, fs=self.fs, return_cents=True)
        residual = [m[len(m) // 2:3 * len(m) // 4]]
        self.midi = []
        i = 0
        while True:
            try:
                buffer = self.q.get(timeout=0.5)  # 1s timeout
            except queue.Empty:
                return

            t = time.time()
            temp = np.hstack([prev_x, buffer])
            prev_x = buffer
            m = self.pitch_tracker.track(audio=temp, fs=self.fs, return_cents=True)
            m[:len(m) // 4] = residual[-1]
            self.midi.extend(m[: len(m) // 2])
            self.running_mean = Util.lpf(np.array(self.midi), 0.9, restore_zeros=True, ignore_zeros=True)

            residual.append(m[len(m) // 2:3 * len(m) // 4])
            print(f"processed {i}th frame")
            i += 1
            # print(time.time() - t)
            self.q.task_done()


with wave.open("vocal_16k.wav", 'rb') as wf:
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    q = queue.Queue()
    worker = Worker(q, sample_rate=wf.getframerate())

    worker.start()
    i = 0
    t = time.time()
    while len(data := wf.readframes(CHUNK)):
        # print(time.time() - t)
        # t = time.time()

        # max time allowed for computation here is 20ms
        _x = np.frombuffer(data, dtype=np.int16)
        _x = _x.astype(float) / (2 ** 15)
        q.put_nowait(_x)
        print(f"put {i}th frame")
        stream.write(data)
        # x = (x * (2**15)).astype(np.int16)
        # data = x.tobytes()

        i += 1

        if i == num_chunks:
            break

    print("Done...")
    stream.close()
    p.terminate()

q.join()
worker.join()
