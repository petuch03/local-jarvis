from TTS.api import TTS
import subprocess
import torch
import os

tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=torch.cuda.is_available())


def speak_response(response):
    tts_model.tts_to_file(text=response, file_path="response.wav")
    subprocess.run(["afplay", "response.wav"])
    os.remove("response.wav")


if __name__ == "__main__":
    response = "Who are you and what can you do?"
    speak_response(response)
