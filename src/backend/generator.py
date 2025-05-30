import torch
import threading
# initialise distilqwen 1.5B

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, TextIteratorStreamer

from backend.retriever import Retriever

class Generator:
    def __init__(self, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", embedding_model_name: str = "BAAI/bge-base-en-v1.5"):
    
        # Constants
        self.MAX_TOKENS = 2048
        self.RESERVED_FOR_OUTPUT = 200
        self.MAX_INPUT_TOKENS = self.MAX_TOKENS - self.RESERVED_FOR_OUTPUT
    
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto", 
            trust_remote_code=True,
            torch_dtype=torch.float16  # Use float16 for efficiency on GPU
        )

        self.retriever = Retriever("BAAI/bge-base-en-v1.5")
        
        self.base_prompt = (
            "You are an expert assistant.\n"
            "Context:\n{context}\n\n"
            "Query:\n{query}\n"
            "Please provide a helpful response."
        )
    
    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def count_tokens(self, text):
        return len(self.tokenize(text))
    
    def retrieve(self, query):
        return self.retriever.retrieve(query)
    
    def fit_chunks(self, query):

        retrieved_chunks = self.retrieve(query)

        base_tokens = self.count_tokens(self.base_prompt)

        user_tokens = self.count_tokens(query)

        available_tokens = self.MAX_INPUT_TOKENS - base_tokens - user_tokens

        output_chunks = []
        used_tokens = 0

        for chunk in retrieved_chunks:
            chunk_tokens = self.count_tokens(chunk[1])
            if used_tokens + chunk_tokens <= available_tokens:
                output_chunks.append(chunk[1])
                used_tokens += chunk_tokens
            else:
                break

        return output_chunks
    
    def build_prompt(self, query):
        # Fit as many as possible
        fitted_chunks = self.fit_chunks(query)

        # Build prompt
        context = "\n\n".join(fitted_chunks)
        prompt = f"{self.base_prompt}".format(context=context, query=query)

        return prompt
    
    def generate_output(self, query):

        # Build the prompt
        prompt = self.build_prompt(query)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Streamer (optional for real-time printing)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Generate output
        output = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer  # remove this line if you just want final text
        )

        # Decode (if not using streamer)
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded_output
    
    def stream_output(self, query):

        # Build the prompt
        prompt = self.build_prompt(query)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Streamer (optional for real-time printing)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Launch generation in a background thread
        generation_thread = threading.Thread(target=self.model.generate, kwargs={
            **inputs,
            **{
            "max_new_tokens": 200,
            "temperature": 0.6,
            "top_p": 0.9,
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
            }
        })
        generation_thread.start()

        # Yield tokens as they become available
        for token in streamer:
            yield token

if __name__ == "__main__":
    generator = Generator("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "BAAI/bge-base-en-v1.5")
    query = "Tell me about the formatting."
    print(generator.generate_output(query))
