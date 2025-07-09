import streamlit as st
import pandas as pd
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model (unchanged)
model_path = "E:/CIS_Project/bart_summarizer_with_rl"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cpu")
model.eval()

# 1. Use a reliable path for feedback log
FEEDBACK_DIR = Path("feedback_data")
FEEDBACK_DIR.mkdir(exist_ok=True)  # Create directory if it doesn't exist
FEEDBACK_LOG = FEEDBACK_DIR / "feedback_log.csv"

# Print debug info
st.write(f"Feedback will be saved to: {FEEDBACK_LOG.absolute()}")

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
    try:
        # Create new feedback entry
        new_entry = pd.DataFrame({
            "article": [article],
            "summary": [summary],
            "feedback": [feedback]
        })
        
        # Check if file exists
        if FEEDBACK_LOG.exists():
            # Append without header
            new_entry.to_csv(FEEDBACK_LOG, mode='a', header=False, index=False)
        else:
            # Create new file with header
            new_entry.to_csv(FEEDBACK_LOG, mode='w', header=True, index=False)
        
        st.success("Feedback saved successfully!")
        return True
    except Exception as e:
        st.error(f"Failed to save feedback: {str(e)}")
        return False

# UI
st.set_page_config(page_title="Personalized News Summarizer", layout="centered")
st.title("🧠 Personalized AI News Summarizer")
st.markdown("Enter a news article below. Get a summary, then provide feedback to improve future summaries.")

article_input = st.text_area("✍️ Paste your news article here:", height=250)

if st.button("🔍 Generate Summary"):
    if not article_input.strip():
        st.warning("Please enter a valid article.")
    else:
        with st.spinner("Generating summary..."):
            summary_text, _ = generate_summary(article_input)
        
        st.subheader("📝 Summary:")
        st.write(summary_text)
        
        # Store in session state to use after feedback
        st.session_state.article = article_input
        st.session_state.summary = summary_text

# Feedback section (only show if summary exists)
if 'summary' in st.session_state:
    feedback = st.radio("Was this summary helpful?", ("👍 Like", "👎 Dislike"))
    feedback_value = 1 if feedback == "👍 Like" else 0
    
    if st.button("📩 Submit Feedback"):
        success = log_feedback(
            st.session_state.article,
            st.session_state.summary,
            feedback_value
        )
        
        if success:
            # Clear after successful submission
            del st.session_state.article
            del st.session_state.summary
            st.rerun()  # Refresh to clear the feedback UI