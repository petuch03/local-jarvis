import os
import subprocess
import speech_recognition as sr
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from TTS.api import TTS

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Initialize Whisper model pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.float16,
    device="mps",
    model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {
        "attn_implementation": "sdpa"},
)

# Initialize TTS model
tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=torch.cuda.is_available())


# Function to recognize speech using Whisper
def recognize_speech():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source, timeout=5)
        try:
            # Save the audio to a file
            with open("audio.wav", "wb") as f:
                f.write(audio.get_wav_data())

            # Call Whisper
            outputs = pipe("audio.wav", chunk_length_s=30, batch_size=24, return_timestamps=True)
            os.remove("audio.wav")
            recognized_text = outputs["text"]
            print(f"Recognized: {recognized_text}")
            return recognized_text
        except Exception as e:
            print(f"Error: {e}")
            return None


# Function to generate a response using llama.cpp with gemma 2b-it
def generate_response(prompt):
    try:
        # Call llama.cpp with the prompt
        process = subprocess.Popen(["/Users/es/Desktop/ml-playground/llama.cpp/build/bin/main",
                                    "--model",
                                    "/Users/es/Desktop/ml-playground/jarvis/gemma-2b-it-fp16.gguf",
                                    "--prompt", prompt],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error in llama.cpp: {stderr.decode('utf-8')}")
            return "Sorry, I couldn't generate a response."

        response = stdout.decode('utf-8')
        return response
    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, I couldn't generate a response."


# Function to speak the response using TTS model
def speak_response(response):
    tts_model.tts_to_file(text=response, file_path="resp.wav")
    # Play the generated speech
    subprocess.run(["afplay", "resp.wav"])
    os.remove("resp.wav")


# Main function to handle dialogue
def main():
    wake_word = "Jarvis"
    print(f"Say '{wake_word}' to wake me up.")

    while True:
        print("Waiting for wake word...")
        recognized_text = recognize_speech()
        if recognized_text and wake_word.lower() in recognized_text.lower():
            print("How can I help you?")
            user_input = recognize_speech()
            if user_input:
                response = generate_response(user_input)
                print(f"Jarvis: {response}")
                speak_response(response)


if __name__ == "__main__":
    main()