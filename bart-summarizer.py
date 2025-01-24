import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer

# Load the BART model and tokenizer
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit UI
st.title("Text Summarizer")
st.write("Enter your text below, and I'll summarize it for you!")

# Text input
input_text = st.text_area("Input Text", height=200)

# Summarization parameters
max_length = st.slider("Max Summary Length", min_value=50, max_value=200, value=100)
min_length = st.slider("Min Summary Length", min_value=10, max_value=100, value=30)
num_beams = st.slider("Number of Beams", min_value=1, max_value=10, value=4)

# Summarize button
if st.button("Summarize"):
    if input_text.strip() == "":
        st.warning("Please enter some text to summarize!")
    else:
        with st.spinner("Generating summary..."):
            # Tokenize the input text
            inputs = tokenizer([input_text], max_length=1024, truncation=True, return_tensors="pt")

            # Generate the summary
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True,
            )

            # Decode the summary
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Display the summary
            st.subheader("Summary")
            st.write(summary)