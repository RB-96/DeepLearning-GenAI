from Text_Summarizer.settings import ENV
import streamlit as st
from transformers import pipeline
from streamlit_extras.add_vertical_space import add_vertical_space

env = ENV()

# Set page configuration
st.set_page_config(
    page_title="Text Summarization",
    layout="centered",
    initial_sidebar_state="collapsed",
)

@st.cache_resource
def load_model():
    tokenizer, model = env.loading_variables_and_fucntions()
    return pipeline("summarization", model=model, tokenizer=tokenizer)


# Load summarization pipeline
summarizer = load_model()

# App Header
st.title("Text Summarization App 📄")

# Text Input Section
placeholder_text = "Paste your text here for summarization..."  # Placeholder/sample text
input_text = st.text_area("Enter the text to summarize:", value=placeholder_text, height=150)

# Add vertical space
add_vertical_space(3)

# Submit Button
if st.button("Summarize!"):
    if input_text and input_text != placeholder_text:
        with st.spinner("Generating summary... ⏳"):
            # Summarization Parameters
            gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

            # Perform summarization
            summary_result = summarizer(input_text, **gen_kwargs)[0]["summary_text"]
            cleaned_summary = summary_result.replace("<n>", "\n") 
            
        # Trigger Balloons
        st.balloons()

        # Display the summary
        st.success("Highlights:")
        st.write(cleaned_summary)
    else:
        st.error("Please enter some text to summarize!")