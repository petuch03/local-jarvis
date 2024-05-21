import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import speech_recognition as sr
import os

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.float16,
    device="mps",
    model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {
        "attn_implementation": "sdpa"},
)

recognizer = sr.Recognizer()


def recognize_speech():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source, timeout=5)
        try:
            with open("test_audio.wav", "wb") as f:
                f.write(audio.get_wav_data())

            outputs = pipe("test_audio.wav", chunk_length_s=30, batch_size=24, return_timestamps=True)
            os.remove("test_audio.wav")
            recognized_text = outputs["text"]
            print(f"Recognized: {recognized_text}")
            return recognized_text
        except Exception as e:
            print(f"Error: {e}")
            return None


if __name__ == "__main__":
    recognize_speech()
