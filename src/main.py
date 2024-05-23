import time

from colorama import Fore
from colorama import init as colorama_init

colorama_init(autoreset=True)
print(Fore.RED + "Setting everything up...")

t = time.time()
import recognizer as rec

print(Fore.RED + f"Setting Recognizer up took {time.time() - t} seconds")

t = time.time()
import generate_speech as speech

print(Fore.RED + f"Setting Speech Generation Module up took {time.time() - t} seconds")

t = time.time()
import generate_text as text

print(Fore.RED + f"Setting Text Generation Module up took {time.time() - t} seconds")


def main():
    print(Fore.GREEN + "Jarvis is here.")
    with speech.suppress_stdout_stderr():
        speech.speak_response("Jarvis is here.")
    print(Fore.GREEN + "What do you want to say? You have 10 seconds for each phrase")
    while True:
        t = time.time()
        user_input = rec.recognize_speech()
        if user_input:
            response = text.generate_response(user_input)
            print(Fore.GREEN + f"Jarvis: {response}")
            with speech.suppress_stdout_stderr():
                speech.speak_response(response)
            print(Fore.RED + f"Processing took {time.time() - t} seconds")


if __name__ == "__main__":
    main()
