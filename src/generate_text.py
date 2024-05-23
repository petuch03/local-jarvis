import subprocess
from colorama import Fore
from colorama import init as colorama_init

colorama_init(autoreset=True)


def generate_response(prompt):
    prompt = (f"Your are artificial virtual assistant Jarvis, act as my common interlocutor. "
              f"We already had several messages before that. "
              f"Briefly answer given message below:\n {prompt}\n### Response:\n")
    try:
        process = subprocess.Popen(["/Users/es/Desktop/ml-playground/llama.cpp/build/bin/main",
                                    "--model",
                                    "/Users/es/Desktop/ml-playground/jarvis/gemma-2b-it-fp16.gguf",
                                    "--prompt", prompt],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(Fore.RED + f"Error in llama.cpp: {stderr.decode('utf-8')}")
            return "Sorry, I couldn't generate a response."

        response = stdout.decode('utf-8')
        response = response.split("### Response:", 1)[-1][:-5].strip()
        return response
    except Exception as e:
        print(Fore.RED + f"Error: {e}")
        return "Sorry, I couldn't generate a response."
