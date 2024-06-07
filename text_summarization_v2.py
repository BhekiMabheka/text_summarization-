import streamlit as st
import fitz  # PyMuPDF
import re
from transformers import pipeline
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

def read_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def count_words(text):
    words = re.findall(r'\b\w+\b', text)
    return len(words)

@st.cache_resource
def load_summarization_model():
    model_name  = "facebook/bart-large-cnn"
    #summarization
    summarizer = pipeline(model_name, model="t5-small")
    return summarizer

st.title("PDF Reader and Summarizer")

uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")

def bart(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    min_len = 100
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, length_penalty=2.0,
                                 min_length=min_len, num_beams=4,
                                 early_stopping=True)  #  min_length=min_len, max_length=1024, max_length=2*min_len, min_length=min_len,
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if uploaded_files:
    combined_text = ""
    for uploaded_file in uploaded_files:
        st.write(f"**Filename: {uploaded_file.name}**")
        pdf_text = read_pdf(uploaded_file)
        combined_text += pdf_text + "\n\n"  # Adding newline characters to separate texts

        if st.button("Summarize Text"):
            with st.spinner("Summarizing..."):
                summarized_text = bart(text=combined_text)
                st.subheader("Summarized Text")
                st.write(summarized_text)

    # total_word_count = count_words(combined_text)
    # st.write(f"**Total Word Count: {total_word_count}**")
    # st.text_area("Combined Content", combined_text, height=500)

    # summarizer = load_summarization_model()
    # if st.button("Summarize Text"):
    #     with st.spinner("Summarizing..."):
    #         summarized_text = summarizer(combined_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    #         st.subheader("Summarized Text")
    #         st.write(summarized_text)




 

