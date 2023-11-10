import sounddevice as sd
import numpy as np
import queue
import webrtcvad
import io
import wave
from whisper_jax import FlaxWhisperPipline

# VAD 인스턴스 생성
vad = webrtcvad.Vad()
vad.set_mode(3)  # VAD 모드 설정

# Whisper 모델 인스턴스화
pipeline = FlaxWhisperPipline("openai/whisper-tiny")

# 오디오 스트림 콜백 함수
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

samplerate = 16000  # 샘플 레이트 설정
channels = 1
q = queue.Queue()

# 오디오 프레임 길이 설정 (예: 10ms)
frame_length = int(samplerate * 0.01)  # 16000 * 0.01 = 160 샘플

with sd.InputStream(device=34, callback=audio_callback, samplerate=samplerate, channels=channels, blocksize=frame_length):
    while True:
        audio_block = q.get()
        # 16비트 PCM 형식으로 변환
        audio_frame = (audio_block * 32767).astype(np.int16).tobytes()

        is_speech = vad.is_speech(audio_frame, samplerate)
        if is_speech:
            with io.BytesIO() as f:
                wf = wave.open(f, 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(2)  # 16비트 오디오
                wf.setframerate(samplerate)
                wf.writeframes(audio_frame)
                f.seek(0)
                audio_data = f.read()

                # Whisper 모델로 오디오 데이터 텍스트로 변환
                text = pipeline(audio_data)
                print("Transcribed text:", text)
