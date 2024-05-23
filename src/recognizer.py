from colorama import Fore
from colorama import init as colorama_init
import speech_recognition as sr
from transformers import pipeline
import torch
import os
import logging

colorama_init(autoreset=True)

recognizer = sr.Recognizer()
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    torch_dtype=torch.float16,
    device="cpu",
    model_kwargs={"attn_implementation": "sdpa"}
)


def recognize_speech():
    with sr.Microphone() as source:
        print(Fore.GREEN + "Listening...")
        audio = recognizer.listen(source, phrase_time_limit=10)
        try:
            # Save the audio to a file
            with open("audio.wav", "wb") as f:
                f.write(audio.get_wav_data())
            print(Fore.RED + "Recognizing...")
            # Call Whisper
            outputs = pipe("audio.wav", chunk_length_s=30, batch_size=24,
                           return_timestamps=True,
                           generate_kwargs={"language": "en"})
            os.remove("audio.wav")
            recognized_text = outputs["text"]
            print(Fore.RED + f"Recognized: {recognized_text}")
            return recognized_text
        except Exception as e:
            print(f"Error: {e}")
            return None
