from transformers import AutoTokenizer

def tokenize_documents(text: str, return_offset_mappings: bool = False) -> list:
    """
    Tokenizes the input text using the DeepSeek-R1-Distill-Qwen-1.5B tokenizer.
    Args:
        text (str): The input text to tokenize.
        return_offset_mappings (bool, optional): If True, returns a list of tuples containing token IDs and their corresponding text spans. If False, returns a list of token IDs only. Defaults to False.
    Returns:
        list: 
            - If return_offset_mappings is False: A list of token IDs (int).
            - If return_offset_mappings is True: A list of tuples, each containing a token ID (int) and the corresponding text span (str).
    Example:
        >>> tokenize_documents("Hello world", return_offset_mappings=True)
        [(123, 'Hello'), (456, ' world')]
    """

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    if return_offset_mappings:
        encoding = tokenizer(text, return_offsets_mapping=True)
        token_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]

        # Map tokens back to their text spans
        tokens_with_text = []
        for token_id, (start, end) in zip(token_ids, offsets):
            token_text = text[start:end]
            tokens_with_text.append((token_id, token_text))
            
        return tokens_with_text
    else:
        return tokenizer(text)["input_ids"]
    

if __name__ == "__main__":
    sample_text = "This is a sample text for tokenization."
    tokens = tokenize_documents(sample_text, return_offset_mappings=True)
    print(f"Tokenized text: {tokens}")
    print(f"Number of tokens: {len(tokens)}")