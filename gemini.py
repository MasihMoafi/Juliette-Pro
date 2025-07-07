# gemini.py

import os
import streamlit as st
import json
import google.genai as genai
from google.genai import types

# --- PROXY FIX ---
def clear_proxy_settings():
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        if var in os.environ:
            del os.environ[var]

clear_proxy_settings()
os.environ['HTTP_PROXY'] = "http://127.0.0.1:10808"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:10808"

# --- CONFIGURATION ---
HISTORY_FILE = "juliette_history.json"
MODEL_ID = "gemini-2.5-pro-preview-06-05"

# --- THE SYSTEM PROMPT ---
SYSTEM_PROMPT = {
    "role": "user",
    "parts": [{"text": "You are Juliette, a world-class AI assistant. Your answers are concise, accurate, and use as few words as possible. You think step-by-step to solve complex problems. Above all, be completely transparent with the user. Keep it highly interactive; ask the user for anything."}]
}

# --- TOOL CONFIGURATION ---
code_execution_tool = types.Tool(code_execution=types.ToolCodeExecution())
google_search_tool = types.Tool(google_search=types.GoogleSearch())

# --- HELPER FUNCTIONS FOR MEMORY ---
def load_messages():
    if os.path.exists(HISTORY_FILE):
        try:
            if os.path.getsize(HISTORY_FILE) > 0:
                with open(HISTORY_FILE, 'r') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []

def save_messages(messages):
    """Saves a message list to a JSON file."""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(messages, f, indent=2)

# --- PAGE SETUP ---
st.set_page_config(page_title="Juliette Pro", layout="wide")
st.title("Juliette Pro 🧠 Advanced Code Assistant")
st.caption(f"Powered by Google GenAI SDK ({MODEL_ID})")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("API Configuration")
    api_key = st.text_input("Enter your Google AI API Key", type="password")

    st.header("Session Management")
    initial_context_file = st.file_uploader("Upload Initial Context File", type=['txt', 'md'])

    if st.button("🔴 Clear Chat & Reset Session"):
        st.session_state.clear()
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.rerun()

    st.header("Model Settings")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.4, 0.1)
    thinking_budget = st.slider("Thinking Budget", 0, 24576, 8192, 128)

    st.header("Tools")
    use_code_execution = st.checkbox("Enable Code Execution", value=True)
    use_google_search = st.checkbox("Enable Google Search", value=True)

    if 'last_usage_metadata' in st.session_state:
        st.header("Last Response Usage")
        usage = st.session_state.last_usage_metadata
        st.text(f"Prompt Tokens: {usage.get('prompt_token_count', 'N/A')}")
        st.text(f"Thoughts Tokens: {usage.get('thoughts_token_count', 'N/A')}")
        st.text(f"Output Tokens: {usage.get('candidates_token_count', 'N/A')}")
        st.text(f"Total Tokens: {usage.get('total_token_count', 'N/A')}")

# --- MAIN LOGIC ---
if not api_key:
    st.info("Please enter your Google AI API Key in the sidebar to begin.")
    st.stop()

try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"Failed to configure the Google GenAI Client. Error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = load_messages()
    if not st.session_state.messages:
        initial_history = [SYSTEM_PROMPT]
        if initial_context_file:
            try:
                context_text = initial_context_file.read().decode("utf-8")
                initial_history.append({"role": "user", "parts": [{"text": context_text}]})
                initial_history.append({"role": "model", "parts": [{"text": "Understood."}]})
                st.sidebar.success("Context file loaded.")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
        st.session_state.messages = initial_history

# --- DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    # Create a more robust check to skip the system prompt
    if (isinstance(message, dict) and 
        message.get("role") == "user" and 
        message.get("parts") and 
        len(message["parts"]) > 0 and
        message["parts"][0].get("text") == SYSTEM_PROMPT["parts"][0]["text"]):
        continue
    
    role = message.get("role", "user")
    role_display = "assistant" if role == "model" else role
    
    with st.chat_message(role_display):
        parts = message.get("parts", [])
        for part in parts:
            if "text" in part and part["text"]:
                st.markdown(part["text"])
            if "thought" in part and part["thought"]:
                 with st.expander("🤔 Thoughts", expanded=False):
                    st.markdown(part["thought"])
            if "executable_code" in part:
                with st.expander("🛠️ Code Execution", expanded=True):
                    st.code(part["executable_code"]["code"], language="python")
            if "code_execution_result" in part:
                 with st.expander("💡 Code Result", expanded=True):
                    st.markdown(part["code_execution_result"]["output"])
            # Note: Grounding metadata is handled after the main response generation

# --- USER INPUT HANDLING ---
if prompt := st.chat_input("Ask a question..."):
    user_message = {"role": "user", "parts": [{"text": prompt}]}
    st.session_state.messages.append(user_message)

    enabled_tools = []
    if use_code_execution:
        enabled_tools.append(code_execution_tool)
    if use_google_search:
        enabled_tools.append(google_search_tool)

    config = types.GenerateContentConfig(
        temperature=temperature,
        tools=enabled_tools,
        thinking_config=types.ThinkingConfig(
            thinking_budget=thinking_budget,
            include_thoughts=True
        )
    )

    try:
        with st.spinner("Juliette is thinking..."):
            response = client.models.generate_content(
                model=f"models/{MODEL_ID}",
                contents=st.session_state.messages,
                config=config
            )
        
        # Manually construct a serializable dictionary from the Content object
        model_response_dict = {
            "role": "model",
            "parts": []
        }
        for part in response.candidates[0].content.parts:
            part_dict = {}
            # Correctly handle thoughts vs. final answer text
            if hasattr(part, 'thought') and part.thought and part.text:
                part_dict['thought'] = part.text
            elif hasattr(part, 'text') and part.text:
                part_dict['text'] = part.text

            # Handle tool-related parts
            if hasattr(part, 'executable_code') and part.executable_code:
                part_dict['executable_code'] = {'code': part.executable_code.code}
            if hasattr(part, 'code_execution_result') and part.code_execution_result:
                part_dict['code_execution_result'] = {'output': part.code_execution_result.output}
            
            if part_dict:
                model_response_dict["parts"].append(part_dict)

        st.session_state.messages.append(model_response_dict)
        
        # Store usage metadata in a serializable format
        st.session_state.last_usage_metadata = {
            'prompt_token_count': response.usage_metadata.prompt_token_count,
            'thoughts_token_count': response.usage_metadata.thoughts_token_count,
            'candidates_token_count': response.usage_metadata.candidates_token_count,
            'total_token_count': response.usage_metadata.total_token_count,
        }

        if hasattr(response.candidates[0], 'grounding_metadata') and response.candidates[0].grounding_metadata.search_entry_point:
            st.session_state.last_search_results = response.candidates[0].grounding_metadata.search_entry_point.rendered_content
        else:
            if 'last_search_results' in st.session_state:
                del st.session_state['last_search_results']

        save_messages(st.session_state.messages)
        st.rerun()

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.session_state.messages.pop()

# Display search results from the last turn if they exist
if 'last_search_results' in st.session_state:
    with st.sidebar:
        st.header("🌐 Google Search Results")
        st.markdown(st.session_state.last_search_results, unsafe_allow_html=True)