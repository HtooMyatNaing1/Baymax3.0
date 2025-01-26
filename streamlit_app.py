import streamlit as st
from transformers import TextStreamer
import torch
from unsloth import FastLanguageModel

# Initialize model
max_seq_length = 2048  # Choose any!
dtype = None  # Auto detection, use specific dtype if needed (Float16, Bfloat16, etc.)
load_in_4bit = True  # Use 4bit quantization to save memory

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="KhantKyaw/Brain2",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)  # Enable faster inference

# Set up Streamlit UI components
st.title("Chatbot using Brain Tumor Model")

# Create a user input box
user_input = st.text_input("Ask something about Brain Tumors:")

# Display output when the user asks a question
if user_input:
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Input:
    {}

    ### Response:
    {}"""

    # Tokenize input
    inputs = tokenizer(
        [alpaca_prompt.format(user_input, "")], return_tensors="pt"
    ).to("cuda")

    # Generate response
    text_streamer = TextStreamer(tokenizer)
    output_ids = model.generate(**inputs, streamer=text_streamer, max_new_tokens=500)

    # Decode and extract the response
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response_start = full_output.find("### Response:") + len("### Response:")
    response = full_output[response_start:].strip()

    # Display the response
    st.write("Response:", response)
