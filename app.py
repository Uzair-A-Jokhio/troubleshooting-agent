import gradio as gr
import spacy 
from textblob import TextBlob
import subprocess

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

knowledge_base = {
    "slow computer": "Try closing background applications or upgrading RAM.",
    "internet issue": "Check your router or reconnect to Wi-Fi.",
    "no power": "Ensure the power cable is connected or battery is charged.",
    "not responding": "Force close the application and restart.",
    "overheating": "Clean the vents and ensure proper airflow."
}

# Sentiment analysis
def analyze_sentiment(query):
    blob = TextBlob(query)
    return blob.sentiment.polarity

def troubleshoot_agent(message, history):
    query = message
    query_lower = query.lower()
    sentiment = analyze_sentiment(query)

    # Initialize response
    response = "❓ I couldn't identify the issue. Can you provide more details or rephrase your problem?"

    # Exact match
    for problem, solution in knowledge_base.items():
        if problem in query_lower:
            priority = "High" if sentiment < -0.3 else "Normal"
            response = f"🔍 Issue: **{problem.capitalize()}**\n💡 Solution: {solution}\n⚠️ Priority: {priority}"
            break
    else:
        # Fallback: keyword matching
        doc = nlp(query)
        keywords = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"]]

        for keyword in keywords:
            for problem, solution in knowledge_base.items():
                if keyword in problem:
                    priority = "High" if sentiment < -0.3 else "Normal"
                    response = f"🤔 Issue (guessed): **{problem.capitalize()}**\n💡 Possible Solution: {solution}\n⚠️ Priority: {priority}"
                    break
            else:
                continue
            break

    # Update chat history
    history.append((message, response))  # Format: (user_message, bot_response)
    return history, history

# ChatInterface UI
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(elem_id="chatbot")
    msg = gr.Textbox(placeholder="Ask your question...")
    state = gr.State([])  # To store chat history as list of messages
    
    msg.submit(troubleshoot_agent, inputs=[msg, state], outputs=[chatbot, state])
    
demo.launch()