### Local Jarvis

Basically it is a simple conversation "something" utilizing speech-to-text + gen-text + text-to-speech models.
Everything runs locally.

## Overview
Used technologies:
- Whisper-small (CPU, because MPS is not fully supported)
- Gemma-2b-it (llama.cpp)
- TTS library

## Demo
[Link to demo](https://drive.google.com/file/d/1-Bh8HNeQvUSscbXIppwndH6kXztdeAr3/view) \
As you can see on the demo, whisper struggles with unknown words (example #3). However, on common speech it works fine. \
Demo is trimmed to enhance your watching experience. 