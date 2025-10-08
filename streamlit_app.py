
import streamlit as st
import requests

# --- Custom JARVIS UI Styling ---
st.set_page_config(page_title="J.A.R.V.I.S. AI", page_icon="🤖", layout="wide")
st.markdown(
        """
        <style>
        @keyframes glow {
            0% { box-shadow: 0 0 5px #00eaff; }
            50% { box-shadow: 0 0 20px #00eaff; }
            100% { box-shadow: 0 0 5px #00eaff; }
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        body, .stApp { 
            background-color: #0a192f; 
            color: #00eaff;
            background-image: radial-gradient(circle at 50% 50%, #112240 0%, #0a192f 100%);
        }
        .css-18e3th9 { background: transparent; }
        .css-1d391kg { 
            background: rgba(17, 34, 64, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            margin: 10px;
            animation: fadeIn 0.5s ease-out;
        }
        .stChatMessage, .stTextInput, .stButton, .stMarkdown, .stAlert {
            font-family: 'Orbitron', 'Consolas', 'Arial', sans-serif;
            color: #00eaff;
        }
        .stChatMessage { 
            border-left: 4px solid #00eaff; 
            margin-bottom: 1em;
            padding: 15px;
            background: rgba(17, 34, 64, 0.6);
            border-radius: 10px;
            animation: fadeIn 0.3s ease-out;
            transition: all 0.3s ease;
        }
        .stChatMessage:hover {
            transform: translateX(5px);
            box-shadow: 0 0 15px rgba(0, 234, 255, 0.2);
        }
        .stTextInput > div > input { 
            background: rgba(17, 34, 64, 0.8);
            color: #00eaff; 
            border: 2px solid #00eaff;
            border-radius: 10px;
            padding: 10px 15px;
            transition: all 0.3s ease;
            animation: glow 2s infinite;
        }
        .stTextInput > div > input:focus {
            border-color: #64ffda;
            box-shadow: 0 0 20px rgba(100, 255, 218, 0.3);
        }
        .stButton > button { 
            background: #00eaff;
            color: #0a192f; 
            border-radius: 8px;
            border: none;
            padding: 8px 20px;
            font-weight: bold;
            transition: all 0.3s ease;
            animation: pulse 2s infinite;
        }
        .stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 20px rgba(0, 234, 255, 0.4);
        }
        .stMarkdown a { 
            color: #64ffda;
            text-decoration: none;
            border-bottom: 1px solid transparent;
            transition: all 0.3s ease;
        }
        .stMarkdown a:hover {
            border-bottom-color: #64ffda;
            text-shadow: 0 0 10px rgba(100, 255, 218, 0.5);
        }
        .stAlert { 
            background: rgba(17, 34, 64, 0.8);
            border-left: 4px solid #64ffda;
            border-radius: 10px;
            animation: fadeIn 0.5s ease-out;
        }
        </style>
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
        <script>
            function typingEffect(element, text, speed = 50) {
                let i = 0;
                element.innerHTML = '';
                function type() {
                    if (i < text.length) {
                        element.innerHTML += text.charAt(i);
                        i++;
                        setTimeout(type, speed);
                    }
                }
                type();
            }
            
            document.addEventListener('DOMContentLoaded', (event) => {
                // Add hover effects to chat messages
                document.querySelectorAll('.stChatMessage').forEach(msg => {
                    msg.addEventListener('mouseenter', () => {
                        msg.style.transform = 'translateX(5px)';
                        msg.style.boxShadow = '0 0 15px rgba(0, 234, 255, 0.2)';
                    });
                    msg.addEventListener('mouseleave', () => {
                        msg.style.transform = 'translateX(0)';
                        msg.style.boxShadow = 'none';
                    });
                });
            });
        </script>
        """,
        unsafe_allow_html=True
)

# --- JARVIS Header ---
st.markdown("""
<div style='
    background: rgba(17, 34, 64, 0.8);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 30px;
    border: 1px solid rgba(0, 234, 255, 0.3);
    box-shadow: 0 0 20px rgba(0, 234, 255, 0.1);
    animation: fadeIn 0.5s ease-out;
'>
    <div style='display:flex;align-items:center;gap:1.5em;margin-bottom:15px;'>
        <div style='
            width: 70px;
            height: 70px;
            border-radius: 50%;
            background: rgba(0, 234, 255, 0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            animation: pulse 2s infinite;
        '>
            <img src='https://upload.wikimedia.org/wikipedia/commons/6/6a/Iron_Man_Mark_III_helmet.png' 
                width='60' 
                style='transform: scale(0.9);transition: transform 0.3s ease;'
                onmouseover="this.style.transform='scale(1.1)'"
                onmouseout="this.style.transform='scale(0.9)'"
            >
        </div>
        <div>
            <h1 style='
                font-family:Orbitron,sans-serif;
                color:#00eaff;
                margin:0;
                font-size:2.5em;
                text-shadow: 0 0 10px rgba(0, 234, 255, 0.5);
            '>J.A.R.V.I.S. AI</h1>
            <div style='
                font-family:Orbitron,sans-serif;
                color:#64ffda;
                font-size:1.1em;
                margin-top:5px;
            '>
                <b>Just A Rather Very Intelligent System</b>
            </div>
        </div>
    </div>
    
    <div style='
        color:#8892b0;
        font-size:1em;
        background: rgba(17, 34, 64, 0.5);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #64ffda;
        margin-top: 15px;
    '>
        <div style='margin-bottom:8px;'>
            <span style='color:#00eaff;'>💡 Quick Commands:</span>
        </div>
        <div style='display:grid;grid-template-columns:auto auto;gap:10px;'>
            <div>
                <b style='color:#64ffda;'>Web Search:</b> 
                <code style='background:rgba(100,255,218,0.1);padding:2px 6px;border-radius:4px;'>
                    search: [query]
                </code>
            </div>
            <div>
                <b style='color:#64ffda;'>Memory:</b> 
                <code style='background:rgba(100,255,218,0.1);padding:2px 6px;border-radius:4px;'>
                    remember: [fact]
                </code>
            </div>
        </div>
    </div>
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
