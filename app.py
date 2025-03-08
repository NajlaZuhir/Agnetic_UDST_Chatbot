# app.py
import streamlit as st
from agnetic_rag_policies import get_detailed_response
from Document import policy_links

# Set page config
st.set_page_config(
    page_title="UDST Policy Chatbot",
    page_icon="📚",
    layout="centered"
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Custom CSS styling
st.markdown("""
<style>
    /* Force all st.button elements to have consistent width and height */
    div.stButton > button {
        width: 100% !important;
        height: 50px !important; /* Adjust to desired height */
        white-space: normal !important; /* Allow text to wrap if needed */
    }

    /* Global Title Styling */
    .global-title {
        color: #2B3A8C;
        font-size: 2.5em !important;
        font-weight: 700;
        text-align: center;
        padding: 20px 0;
        border-bottom: 3px solid #2B3A8C;
        margin-bottom: 25px;
    }

    /* Chat container */
    .chat-history {
        max-height: 60vh;
        overflow-y: auto;
        padding-right: 10px;
        margin-bottom: 20px;
    }

    /* Input container */
    .input-container {
        position: fixed;
        bottom: 80px;
        width: 85%;
        display: flex;
        gap: 10px;
        z-index: 999;
    }

    /* Clear button styling */
    .clear-btn {
        background-color: #f8f9fa;
        border: 1px solid #ced4da;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with numbered policies
with st.sidebar:
    st.header("📜 20 UDST Policies")
    for i, policy_name in enumerate(policy_links.keys(), start=1):
        url = policy_links[policy_name]
        st.markdown(f"{i}. [{policy_name}]({url})", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("UDST Policy Chatbot")

# Main interface
st.markdown('<h1 class="global-title">📚🤖 UDST POLICY ASSISTANT V2</h1>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align: center; font-size: 20px; color: #555;">
  Providing quick answers to your UDST policy questions.
</p>
""", unsafe_allow_html=True)

# Chat history container
with st.container():
    st.markdown('<div class="chat-history">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    st.markdown('</div>', unsafe_allow_html=True)

# --- SAMPLE QUESTIONS SECTION ---
if not st.session_state.messages:
    st.subheader("💬 FAQ:")
    cols = st.columns(2)
    sample_questions = [
        "How many absences are allowed before I risk failing?",
        "What’s the process for registering each semester?",
        "Does UDST offer scholarships or financial aid?",
        "Are there special requirements for international students?",
        "How can I appeal a grade I disagree with?",
        "What happens if a student behaves inappropriately on campus?"
    ]

    for i, question in enumerate(sample_questions):
        if cols[i % 2].button(question, use_container_width=True, key=f"sample_{i}"):
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })
            with st.spinner("Searching policies..."):
                response_text = get_detailed_response(question)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text
            })
            st.rerun()

# --- INPUT SECTION ---
input_container = st.container()
with input_container:
    cols = st.columns([5, 1])
    with cols[0]:
        prompt = st.chat_input("Enter your question about UDST policies:")
    with cols[1]:
        if st.button(
            "Clear Chat",
            key="clear_chat",
            use_container_width=True,
            help="Clear conversation history"
        ):
            st.session_state.messages = []
            st.rerun()

# --- PROCESS USER QUERY ---
if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    with st.spinner("🔍 Searching university policies..."):
        response_text = get_detailed_response(prompt)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text
    })
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: grey;">
    <p>Note: Responses are based on official UDST policies.</p>
</div>
""", unsafe_allow_html=True)
