import time

import src.generate_text as g
import src.recognizer as rec

# g.generate_response("i am about to commit a suicide, what are reasons for me to live?")
t = time.time()
print("Starting")
outputs = rec.pipe("activation.wav", chunk_length_s=30, batch_size=24,
                   return_timestamps=True, generate_kwargs={"language": "en"})
print(f"Ended in {time.time() - t} seconds")

print(outputs)
