import gradio as gr

from ..backend.embedder import embed_text
from ..backend.retrieve_document import get_top_k_similar_docs
from ..backend.generate_output import build_prompt, generate_output, stream_output

def most_similar_text(input_text):
    return get_top_k_similar_docs(embed_text(input_text).tolist(), k=5)

def generate_response(input_text, history):
    prompt = build_prompt(input_text, get_top_k_similar_docs(embed_text(input_text).tolist(), k=5))
    
    partial_message = ""
    # stream_output yields tokens or small chunks
    for new_token in stream_output(prompt):
        partial_message += new_token
        yield partial_message
    
demo = gr.ChatInterface(fn=generate_response)
demo.launch()