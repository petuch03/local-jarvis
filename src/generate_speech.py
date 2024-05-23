import os
import subprocess
from contextlib import contextmanager

import torch
from TTS.api import TTS
import sys


@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


with suppress_stdout_stderr():
    tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False,
                    gpu=torch.cuda.is_available())


def speak_response(response):
    tts_model.tts_to_file(text=response, file_path="resp.wav")
    subprocess.run(["afplay", "resp.wav"])
    os.remove("resp.wav")
