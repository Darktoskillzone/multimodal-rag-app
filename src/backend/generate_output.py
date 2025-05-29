import torch
import threading
# initialise distilqwen 1.5B

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, TextIteratorStreamer

from ..backend.embedder import embed_text
from ..backend.retrieve_document import get_top_k_similar_docs

# Load the tokenizer
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    trust_remote_code=True,
    torch_dtype=torch.float16  # Use float16 for efficiency on GPU
)

# Constants
MAX_TOKENS = 2048
RESERVED_FOR_OUTPUT = 200
MAX_INPUT_TOKENS = MAX_TOKENS - RESERVED_FOR_OUTPUT

# check user input token size

def count_tokens(text):
    return len(tokenizer.tokenize(text))

def fit_chunks(user_query, chunks):
    # Format user section
    base_prompt = f"""Context:
    

    Question: {user_query}

    Response:
    """
    base_tokens = count_tokens(base_prompt)
    user_tokens = count_tokens(user_query)

    available_tokens = MAX_INPUT_TOKENS - base_tokens - user_tokens

    final_chunks = []
    used_tokens = 0

    for chunk in chunks:
        chunk_tokens = count_tokens(chunk[1])
        if used_tokens + chunk_tokens <= available_tokens:
            final_chunks.append(chunk[1])
            used_tokens += chunk_tokens
        else:
            break

    return final_chunks

# if user input is too long, return default error message


# else, calculate the number of retrieved chunks to include in prompt

def build_prompt(user_query, top_k_chunks):
    # Fit as many as possible
    fitted_chunks = fit_chunks(user_query, top_k_chunks)

    # Build prompt
    retrieved_context = "\n\n".join(fitted_chunks)
    prompt = f"""Context:
    {retrieved_context}

    Question: {user_query}

    Respond concisely.
    """

    return prompt

def generate_output(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Streamer (optional for real-time printing)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Generate output
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.6,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer  # remove this line if you just want final text
    )

    # Decode (if not using streamer)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

def stream_output(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Streamer (optional for real-time printing)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Launch generation in a background thread
    generation_thread = threading.Thread(target=model.generate, kwargs={
        **inputs,
        **{
        "max_new_tokens": 200,
        "temperature": 0.6,
        "top_p": 0.9,
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
        }
    })
    generation_thread.start()

    # Yield tokens as they become available
    for token in streamer:
        yield token

# account for additional buffer tokens as context in the prompt

if __name__ == "__main__":
    prompt = build_prompt("What is the capital of France?", get_top_k_similar_docs(embed_text("What is the capital of France?").tolist(), k=5))
    print(generate_output(prompt))  