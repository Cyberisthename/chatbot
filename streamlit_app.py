
import streamlit as st
import requests

# --- Custom JARVIS UI Styling ---
st.set_page_config(page_title="J.A.R.V.I.S. AI", page_icon="🤖", layout="wide")
st.markdown(
        """
        <style>
        body, .stApp { background-color: #0a192f; color: #00eaff; }
        .css-18e3th9 { background: #0a192f; }
        .css-1d391kg { background: #112240; }
        .stChatMessage, .stTextInput, .stButton, .stMarkdown, .stAlert {
                font-family: 'Orbitron', 'Consolas', 'Arial', sans-serif;
                color: #00eaff;
        }
        .stChatMessage { border-left: 4px solid #00eaff; margin-bottom: 1em; }
        .stTextInput > div > input { background: #112240; color: #00eaff; border: 1px solid #00eaff; }
        .stButton > button { background: #00eaff; color: #0a192f; border-radius: 8px; }
        .stMarkdown a { color: #64ffda; }
        .stAlert { background: #112240; border-left: 4px solid #64ffda; }
        </style>
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap" rel="stylesheet">
        """,
        unsafe_allow_html=True
)

# --- JARVIS Header ---
st.markdown("""
<div style='display:flex;align-items:center;gap:1em;'>
    <img src='https://upload.wikimedia.org/wikipedia/commons/6/6a/Iron_Man_Mark_III_helmet.png' width='60'>
    <h1 style='font-family:Orbitron,sans-serif;color:#00eaff;margin-bottom:0;'>J.A.R.V.I.S. AI</h1>
</div>
<div style='font-family:Orbitron,sans-serif;color:#64ffda;font-size:1.1em;margin-bottom:1em;'>
    <b>Just A Rather Very Intelligent System</b> &mdash; Your Personal AI Assistant
</div>
<div style='color:#8892b0;font-size:1em;'>
    <b>Tip:</b> To search the web, start your message with <code>search:</code>, <code>google</code>, or <code>websearch:</code>.<br>
    <b>Memory:</b> Use <code>remember:</code> to teach J.A.R.V.I.S. facts. Use <code>recall facts</code> to see what he knows.
</div>
---
""", unsafe_allow_html=True)

# --- Chat State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Chat Display ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
prompt = st.chat_input("Type your command or question for J.A.R.V.I.S....")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={"messages": st.session_state.messages}
        )
        response.raise_for_status()
        data = response.json()
        answer = data["message"]["content"]
    except Exception as e:
        answer = f"[Error communicating with local J.A.R.V.I.S. backend: {e}]"
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
