import streamlit as st
import os
import requests
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

PRESET_PROMPTS = {
    "Khisba GIS": """You are Khisba GIS, an enthusiastic remote sensing and GIS expert. Your personality:
- Name: Khisba GIS
- Role: Remote sensing and GIS expert
- Style: Warm, friendly, and approachable
- Expertise: Deep knowledge of satellite imagery, vegetation indices, and geospatial analysis
- Humor: Light and professional
- Always eager to explore new remote sensing challenges

Guidelines:
- Focus primarily on remote sensing, GIS, and satellite imagery topics
- Be naturally enthusiastic about helping with vegetation indices and analysis
- Share practical examples and real-world applications
- Show genuine interest in the user's remote sensing challenges
- If topics go outside remote sensing, gently guide back to GIS
- Always introduce yourself as Khisba GIS when asked who you are""",
    "Default Assistant": "You are a helpful, friendly AI assistant. Provide clear and concise answers.",
    "Professional Expert": "You are a professional expert. Provide detailed, accurate, and well-structured responses. Use formal language and cite reasoning when appropriate.",
    "Creative Writer": "You are a creative writer with a vivid imagination. Use descriptive language, metaphors, and engaging storytelling in your responses.",
    "Code Helper": "You are a programming expert. Provide clean, well-commented code examples. Explain technical concepts clearly and suggest best practices.",
    "Friendly Tutor": "You are a patient and encouraging tutor. Explain concepts step by step, use simple examples, and ask questions to ensure understanding.",
    "Concise Responder": "You are brief and to the point. Give short, direct answers without unnecessary elaboration.",
    "Custom": ""
}

st.set_page_config(
    page_title="TinyLLaMA Chat",
    page_icon="🦙",
    layout="centered"
)

st.title("🦙 TinyLLaMA Chat")
st.caption("A local AI chat powered by TinyLLaMA 1.1B")

def download_model():
    """Download the model from Hugging Face with progress."""
    MODEL_DIR.mkdir(exist_ok=True)
    
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download model: {str(e)}")
    
    total_size = int(response.headers.get('content-length', 0))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    downloaded = 0
    try:
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
                        status_text.text(f"Downloading: {downloaded / (1024**2):.1f} / {total_size / (1024**2):.1f} MB")
    except Exception as e:
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        raise Exception(f"Download interrupted: {str(e)}")
    
    if total_size > 0 and downloaded != total_size:
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        raise Exception(f"Incomplete download: got {downloaded} bytes, expected {total_size}")
    
    progress_bar.empty()
    status_text.empty()
    return True

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the TinyLLaMA model using ctransformers."""
    from ctransformers import AutoModelForCausalLM
    
    if not MODEL_PATH.exists():
        with st.spinner("Downloading TinyLLaMA model (~637 MB)..."):
            download_model()
    
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        model_file=MODEL_PATH.name,
        model_type="llama",
        context_length=2048,
        gpu_layers=0
    )
    return model

def format_prompt(messages, system_prompt=""):
    """Format conversation history for TinyLLaMA chat format with system prompt."""
    prompt = ""
    
    if system_prompt:
        prompt += f"<|system|>\n{system_prompt}</s>\n"
    
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"<|user|>\n{msg['content']}</s>\n"
        elif msg["role"] == "assistant":
            prompt += f"<|assistant|>\n{msg['content']}</s>\n"
    prompt += "<|assistant|>\n"
    return prompt

def truncate_messages(messages, max_messages=10):
    """Keep only the most recent messages to fit within context limit."""
    if len(messages) > max_messages:
        return messages[-max_messages:]
    return messages

def generate_response(model, messages, system_prompt="", max_tokens=256, temperature=0.7):
    """Generate a response from the model."""
    truncated_messages = truncate_messages(messages)
    prompt = format_prompt(truncated_messages, system_prompt)
    
    response = model(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        stop=["</s>", "<|user|>", "<|assistant|>", "<|system|>"]
    )
    
    return response.strip()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = PRESET_PROMPTS["Khisba GIS"]

if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = "Khisba GIS"

with st.sidebar:
    st.header("Persona / System Prompt")
    
    selected_preset = st.selectbox(
        "Choose a preset:",
        options=list(PRESET_PROMPTS.keys()),
        index=list(PRESET_PROMPTS.keys()).index(st.session_state.selected_preset),
        key="preset_selector"
    )
    
    if selected_preset != st.session_state.selected_preset:
        st.session_state.selected_preset = selected_preset
        if selected_preset != "Custom":
            st.session_state.system_prompt = PRESET_PROMPTS[selected_preset]
    
    system_prompt = st.text_area(
        "System prompt (customize how the AI responds):",
        value=st.session_state.system_prompt,
        height=150,
        placeholder="Enter instructions for how the AI should behave...",
        key="system_prompt_input"
    )
    
    if system_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = system_prompt
        if system_prompt not in PRESET_PROMPTS.values():
            st.session_state.selected_preset = "Custom"
    
    st.divider()
    st.header("Model Settings")
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1, 
                           help="Higher = more creative, Lower = more focused")
    max_tokens = st.slider("Max Tokens", 64, 1024, 256, 64,
                          help="Maximum length of the response")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("Reset Prompt", type="secondary", use_container_width=True):
            st.session_state.system_prompt = PRESET_PROMPTS["Default Assistant"]
            st.session_state.selected_preset = "Default Assistant"
            st.rerun()
    
    st.divider()
    st.caption("Model: TinyLLaMA 1.1B Chat v1.0")
    st.caption("Quantization: Q4_K_M (~637 MB)")

with st.spinner("Loading TinyLLaMA model... This may take a moment on first run."):
    try:
        model = load_model()
        st.session_state.model_loaded = True
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

if st.session_state.model_loaded:
    st.success("Model loaded and ready!", icon="✅")

if st.session_state.system_prompt:
    with st.expander("Current Persona", expanded=False):
        st.info(st.session_state.system_prompt)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Send a message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(
                model,
                st.session_state.messages,
                system_prompt=st.session_state.system_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
