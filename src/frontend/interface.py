import gradio as gr

from ..backend.embedder import embed_text
from ..backend.retrieve_document import get_top_k_similar_docs

def most_similar_text(input_text):
    return get_top_k_similar_docs(embed_text(input_text).tolist(), k=5)

demo = gr.Interface(fn=most_similar_text, inputs="text", outputs="text")
demo.launch()   