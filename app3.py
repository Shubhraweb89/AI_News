import streamlit as st
import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load fine-tuned model
model_path = "E:/CIS_Project/bart_summarizer_with_rl"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cpu")
model.eval()

# File to log feedback - using absolute path
FEEDBACK_LOG = os.path.abspath("feedback_log.csv")
print(f"Feedback will be saved to: {FEEDBACK_LOG}")

def generate_summary(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    output_ids = model.generate(
        inputs["input_ids"],
        max_length=128,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9
    )
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary, output_ids

def log_feedback(article, summary, feedback):
    # Create feedback data
    feedback_data = {
        "article": [article],
        "summary": [summary],
        "feedback": [feedback]
    }
    df = pd.DataFrame(feedback_data)
    
    try:
        # Check if file exists
        file_exists = os.path.exists(FEEDBACK_LOG)
        
        # Write to CSV
        df.to_csv(
            FEEDBACK_LOG,
            mode='a' if file_exists else 'w',
            index=False,
            header=not file_exists
        )
        return True
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")
        return False

# UI
st.set_page_config(page_title="Personalized News Summarizer", layout="centered")
st.title("🧠 Personalized AI News Summarizer")
st.markdown("Enter a news article below. Get a summary, then provide feedback to improve future summaries.")

article_input = st.text_area("✍️ Paste your news article here:", height=250)

if st.button("🔍 Generate Summary"):
    if article_input.strip() == "":
        st.warning("Please enter a valid article.")
    else:
        with st.spinner("Generating summary..."):
            summary_text, output_ids = generate_summary(article_input)
        st.subheader("📝 Summary:")
        st.write(summary_text)

        feedback = st.radio("Was this summary helpful?", ("👍 Like", "👎 Dislike"))
        feedback_value = 1 if feedback == "👍 Like" else 0

        if st.button("📩 Submit Feedback"):
            if log_feedback(article_input, summary_text, feedback_value):
                st.success("Thank you! Your feedback has been recorded.")
            else:
                st.error("Failed to save feedback. Please try again.")