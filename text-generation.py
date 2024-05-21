import subprocess


def generate_response(prompt):
    try:
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


if __name__ == "__main__":
    prompt = "Hello, how are you?"
    response = generate_response(prompt)
    print(f"Response: {response}")