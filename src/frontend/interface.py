import gradio as gr

from backend.retriever import Retriever
from backend.generator import Generator

def most_similar_text(input_text):
    query_embedding = retriever.embed(input_text).tolist()
    return retriever.retrieve(query_embedding, k=5)

def generate_response(input_text, history):
    generator = Generator("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "BAAI/bge-base-en-v1.5")
    
    partial_message = ""
    # stream_output yields tokens or small chunks
    for new_token in generator.stream_output(input_text):
        partial_message += new_token
        yield partial_message

if __name__ == "__main__":
    demo = gr.ChatInterface(fn=generate_response)
    demo.launch()