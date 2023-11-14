from whisper_jax import FlaxWhisperPipline
import time

# instantiate pipeline
pipeline = FlaxWhisperPipline("openai/whisper-tiny")

# JIT compile the forward call - slow, but we only do once
start_time = time.time()
text = pipeline("audio.mp3")
end_time = time.time()
execution_time = end_time - start_time

print("Transcribed text:", text)
print("Execution time:", execution_time, "seconds")

# used cached function thereafter - super fast!!
start_time2 = time.time()
text2 = pipeline("audio.mp3")
end_time2 = time.time()
execution_time2 = end_time2 - start_time2

print("Transcribed text:", text2)
print("Execution time:", execution_time2, "seconds")
