import streamlit as st
import requests
from pathlib import Path
import concurrent.futures
from datetime import datetime
import re
import sys
import traceback

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/tfdtfd/khisbagis23/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=true"

# Enhanced deep thinking prompts
PRESET_PROMPTS = {
    "Deep Thinker Pro": """You are a sophisticated AI thinker that excels at analysis, synthesis, and providing insightful perspectives. 

THINKING FRAMEWORK:
1. **Comprehension**: Understand the query fully, identify key elements
2. **Contextualization**: Place the topic in historical, cultural, or disciplinary context
3. **Multi-Source Analysis**: Examine information from different sources critically
4. **Pattern Recognition**: Identify connections, contradictions, gaps
5. **Synthesis**: Combine insights into coherent understanding
6. **Critical Evaluation**: Assess reliability, bias, significance
7. **Insight Generation**: Provide original perspectives or connections
8. **Actionable Knowledge**: Suggest applications, further questions, implications

RESPONSE STRUCTURE:
- Start with brief overview
- Present analysis with reasoning
- Reference sources when available
- Highlight interesting connections
- Acknowledge uncertainties
- End with thought-provoking questions or suggestions

TONE: Analytical yet engaging, precise yet accessible.""",

    "Khisba GIS Expert": """You are Khisba GIS - a passionate remote sensing/GIS specialist with deep analytical skills.

SPECIALTY THINKING PROCESS:
1. **Geospatial Context**: How does location/spatial relationships matter?
2. **Temporal Analysis**: What changes over time? Historical patterns?
3. **Data Source Evaluation**: Satellite, ground, or derived data reliability?
4. **Multi-Scale Thinking**: From local to global perspectives
5. **Practical Applications**: Real-world uses of the information
6. **Ethical Considerations**: Privacy, representation, accessibility issues

EXPERTISE: Satellite imagery, vegetation indices, climate analysis, urban planning, disaster monitoring
STYLE: Enthusiastic, precise, eager to explore spatial dimensions of any topic""",
    
    "Default Assistant": "You are a helpful, friendly AI assistant. Provide clear and concise answers.",
    "Research Analyst": "You are a professional research analyst specializing in synthesizing complex information.",
    "Creative Synthesizer": "You connect seemingly unrelated ideas to generate novel insights.",
    "Code Expert": "You are a programming expert who explains technical concepts clearly and provides working code examples."
}

# Optimized search tools
SEARCH_TOOLS = {
    "Wikipedia": {
        "name": "Wikipedia",
        "icon": "📚",
        "description": "Encyclopedia articles",
        "endpoint": "https://en.wikipedia.org/w/api.php",
        "enabled": True
    },
    "DuckDuckGo": {
        "name": "Web Search",
        "icon": "🌐",
        "description": "Instant answers & web results",
        "endpoint": "https://api.duckduckgo.com/",
        "enabled": True
    },
    "ArXiv": {
        "name": "Research Papers",
        "icon": "🔬",
        "description": "Scientific publications",
        "endpoint": "http://export.arxiv.org/api/query",
        "enabled": True
    }
}

# Set page config FIRST
st.set_page_config(
    page_title="SmartThink AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .search-result {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #4a90e2;
    }
    .source-tag {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .thinking-bubble {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2196f3;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with safe defaults
def init_session_state():
    """Initialize all session state variables with safe defaults"""
    default_persona = "Deep Thinker Pro"
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "model" not in st.session_state:
        st.session_state.model = None
    
    if "selected_preset" not in st.session_state:
        st.session_state.selected_preset = default_persona
    
    # Ensure selected_preset is valid
    persona_options = list(PRESET_PROMPTS.keys())
    if st.session_state.selected_preset not in persona_options:
        st.session_state.selected_preset = default_persona
    
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = PRESET_PROMPTS[st.session_state.selected_preset]
    
    if "search_history" not in st.session_state:
        st.session_state.search_history = {}
    
    if "enabled_sources" not in st.session_state:
        st.session_state.enabled_sources = ["Wikipedia", "DuckDuckGo"]
    
    if "show_search_history" not in st.session_state:
        st.session_state.show_search_history = False

# Initialize session state
init_session_state()

# ============================
# INDEPENDENT SEARCH FUNCTIONS
# ============================

def perform_independent_search(query, selected_sources=None):
    """Perform search independently without model involvement"""
    if selected_sources is None:
        selected_sources = st.session_state.enabled_sources
    
    if not selected_sources:
        selected_sources = ["Wikipedia", "DuckDuckGo"]
    
    results = {}
    
    # Use thread pool for parallel searches
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_sources)) as executor:
        future_to_source = {}
        
        for source_name in selected_sources:
            if source_name in SEARCH_TOOLS and SEARCH_TOOLS[source_name]["enabled"]:
                if source_name == "Wikipedia":
                    future = executor.submit(search_wikipedia, query)
                elif source_name == "DuckDuckGo":
                    future = executor.submit(search_duckduckgo, query)
                elif source_name == "ArXiv":
                    future = executor.submit(search_arxiv, query)
                else:
                    continue
                future_to_source[future] = source_name
        
        # Collect results with timeout
        for future in concurrent.futures.as_completed(future_to_source):
            source_name = future_to_source[future]
            try:
                result = future.result(timeout=8)
                if result:
                    results[source_name] = result
            except Exception as e:
                st.warning(f"Search failed for {source_name}: {str(e)[:100]}")
    
    # Cache results
    if results:
        st.session_state.search_history[query] = {
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "sources": list(results.keys())
        }
    
    return results

def search_wikipedia(query, max_results=2):
    """Search Wikipedia independently"""
    try:
        # Search for pages
        search_params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': max_results,
            'utf8': 1
        }
        
        response = requests.get(
            SEARCH_TOOLS["Wikipedia"]["endpoint"],
            params=search_params,
            timeout=8
        )
        
        if response.status_code != 200:
            return []
        
        search_data = response.json()
        pages = search_data.get('query', {}).get('search', [])
        
        if not pages:
            return []
        
        # Get page content for the first result
        results = []
        for page in pages[:1]:  # Just get first result for speed
            content_params = {
                'action': 'query',
                'format': 'json',
                'prop': 'extracts|info',
                'inprop': 'url',
                'exintro': True,
                'explaintext': True,
                'pageids': page['pageid']
            }
            
            content_response = requests.get(
                SEARCH_TOOLS["Wikipedia"]["endpoint"],
                params=content_params,
                timeout=8
            )
            
            if content_response.status_code == 200:
                page_data = content_response.json()
                page_info = page_data.get('query', {}).get('pages', {}).get(str(page['pageid']), {})
                
                if page_info and page_info.get('extract'):
                    extract = page_info.get('extract', '')
                    # Clean up the extract
                    extract = re.sub(r'\s+', ' ', extract)
                    extract = re.sub(r'\n+', ' ', extract)
                    
                    results.append({
                        'title': page_info.get('title', 'Unknown'),
                        'summary': extract[:300] + ('...' if len(extract) > 300 else ''),
                        'url': page_info.get('fullurl', ''),
                        'source': 'Wikipedia',
                        'wordcount': page_info.get('wordcount', 0)
                    })
        
        return results
        
    except Exception as e:
        # Don't show error, just return empty
        return []

def search_duckduckgo(query):
    """Search DuckDuckGo independently"""
    try:
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1,
            't': 'smartthink_ai'
        }
        
        response = requests.get(
            SEARCH_TOOLS["DuckDuckGo"]["endpoint"],
            params=params,
            timeout=8
        )
        
        if response.status_code != 200:
            return {}
        
        data = response.json()
        
        result = {
            'abstract': data.get('AbstractText', ''),
            'answer': data.get('Answer', ''),
            'definition': data.get('Definition', ''),
            'related': [],
            'source': 'DuckDuckGo'
        }
        
        # Extract related topics
        related = data.get('RelatedTopics', [])
        for topic in related[:3]:
            if isinstance(topic, dict) and 'Text' in topic:
                result['related'].append(topic['Text'][:150])
            elif isinstance(topic, str):
                result['related'].append(topic[:150])
        
        # Clean empty values
        cleaned_result = {}
        for key, value in result.items():
            if isinstance(value, str) and value.strip():
                cleaned_result[key] = value.strip()
            elif isinstance(value, list) and value:
                cleaned_result[key] = [v.strip() for v in value if v and v.strip()]
        
        return cleaned_result if cleaned_result else {}
        
    except Exception as e:
        return {}

def search_arxiv(query, max_results=1):
    """Search ArXiv independently"""
    try:
        # Clean query for ArXiv
        clean_query = re.sub(r'[^\w\s-]', ' ', query)
        clean_query = re.sub(r'\s+', '+', clean_query.strip())
        
        params = {
            'search_query': f'all:{clean_query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        response = requests.get(
            SEARCH_TOOLS["ArXiv"]["endpoint"],
            params=params,
            timeout=10
        )
        
        if response.status_code != 200:
            return []
        
        # Parse XML
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        
        results = []
        namespace = '{http://www.w3.org/2005/Atom}'
        
        for entry in root.findall(f'{namespace}entry'):
            title_elem = entry.find(f'{namespace}title')
            summary_elem = entry.find(f'{namespace}summary')
            
            if title_elem is not None and summary_elem is not None:
                title = title_elem.text.strip()
                summary = summary_elem.text.strip()
                
                # Clean up summary
                summary = re.sub(r'\s+', ' ', summary)
                
                results.append({
                    'title': title,
                    'summary': summary[:250] + ('...' if len(summary) > 250 else ''),
                    'source': 'ArXiv'
                })
                break  # Just get first result
        
        return results
        
    except Exception as e:
        return []

def format_search_results_for_prompt(search_results):
    """Format search results for the model prompt"""
    if not search_results:
        return "No external information found from searches. Please rely on your own knowledge."
    
    formatted = "SEARCH RESULTS AND EXTERNAL INFORMATION:\n\n"
    
    for source, results in search_results.items():
        formatted += f"=== {source} ===\n"
        
        if isinstance(results, list):
            for i, item in enumerate(results, 1):
                formatted += f"{i}. "
                if 'title' in item:
                    formatted += f"Title: {item['title']}\n"
                if 'summary' in item:
                    formatted += f"   Content: {item['summary']}\n"
                formatted += "\n"
        
        elif isinstance(results, dict):
            for key, value in results.items():
                if key not in ['source'] and value:
                    if isinstance(value, list):
                        if value:
                            formatted += f"{key}: {value[0][:200]}\n"
                    elif isinstance(value, str):
                        formatted += f"{key}: {value[:300]}\n"
            formatted += "\n"
    
    formatted += "\nINSTRUCTIONS:\n"
    formatted += "1. Use this information to inform your answer\n"
    formatted += "2. Synthesize information from multiple sources\n"
    formatted += "3. Cite sources when using specific information\n"
    formatted += "4. If information conflicts, mention this\n"
    formatted += "5. Add your own analysis and insights\n"
    
    return formatted

# ============================
# MODEL FUNCTIONS
# ============================

def download_model():
    """Download model from URL"""
    MODEL_DIR.mkdir(exist_ok=True)
    
    if MODEL_PATH.exists():
        return True
    
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        downloaded = 0
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(min(progress, 1.0))
                        status_text.text(f"Downloading: {downloaded / (1024**2):.1f} MB")
        
        progress_bar.empty()
        status_text.empty()
        
        return MODEL_PATH.exists()
        
    except Exception as e:
        st.error(f"Download error: {str(e)[:200]}")
        return False

@st.cache_resource(show_spinner=False)
def load_model_safe():
    """Load the model with error handling"""
    try:
        from ctransformers import AutoModelForCausalLM
        
        if not MODEL_PATH.exists():
            with st.spinner("📥 Downloading model (first time only)..."):
                if not download_model():
                    return None
        
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            model_file=MODEL_PATH.name,
            model_type="llama",
            context_length=2048,
            gpu_layers=0,
            threads=2
        )
        return model
    except Exception as e:
        st.error(f"Model loading error: {str(e)[:200]}")
        return None

def create_prompt_with_context(query, search_context, system_prompt, history):
    """Create prompt with search context and conversation history"""
    
    # Format recent history
    history_text = ""
    if len(history) > 1:
        history_text = "RECENT CONVERSATION:\n"
        for msg in history[-3:]:
            if msg["role"] == "user":
                history_text += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                history_text += f"Assistant: {msg['content']}\n"
        history_text += "\n"
    
    prompt = f"""<|system|>
{system_prompt}

Today's Date: {datetime.now().strftime('%Y-%m-%d')}

{search_context}

{history_text}

Based on the information above and your knowledge, provide a comprehensive answer to the user's question.

Guidelines:
- Reference specific information from search results when relevant
- Be clear about what comes from external sources vs your own knowledge
- If search results are limited, rely on your expertise
- Structure your answer logically
- End with key takeaways or follow-up questions</s>

<|user|>
{query}</s>

<|assistant|>
"""
    
    return prompt

def generate_response(query, search_enabled=True):
    """Generate response using model and optional search"""
    
    # Perform search if enabled
    search_results = {}
    if search_enabled and st.session_state.enabled_sources:
        with st.spinner("🔍 Searching for information..."):
            search_results = perform_independent_search(query)
    
    # Format search context
    if search_results:
        search_context = format_search_results_for_prompt(search_results)
    else:
        search_context = "No external search results available."
    
    # Create prompt
    prompt = create_prompt_with_context(
        query=query,
        search_context=search_context,
        system_prompt=st.session_state.system_prompt,
        history=st.session_state.messages
    )
    
    # Generate response
    with st.spinner("🤔 Analyzing and generating response..."):
        try:
            response = st.session_state.model(
                prompt,
                max_new_tokens=768,
                temperature=st.session_state.get('temperature', 0.7),
                top_p=0.9,
                repetition_penalty=1.1,
                stop=["</s>", "<|user|>", "<|assistant|>"]
            )
            
            response = response.strip()
            
            # Ensure response ends properly
            if response and response[-1] not in ['.', '!', '?']:
                response += "."
            
            return response, search_results
            
        except Exception as e:
            st.error(f"Generation error: {str(e)[:200]}")
            return f"I apologize, but I encountered an error: {str(e)[:100]}", {}

# ============================
# STREAMLIT UI
# ============================

# Title
st.title("🧠 SmartThink AI")
st.caption("Independent search + AI synthesis for intelligent responses")

# Sidebar
with st.sidebar:
    st.header("🤖 AI Persona")
    
    # Get persona options safely
    persona_options = list(PRESET_PROMPTS.keys())
    current_preset = st.session_state.selected_preset
    
    # Ensure current preset is in options
    if current_preset not in persona_options:
        current_preset = "Deep Thinker Pro"
        st.session_state.selected_preset = current_preset
        st.session_state.system_prompt = PRESET_PROMPTS[current_preset]
    
    try:
        index = persona_options.index(current_preset)
    except ValueError:
        index = 0
        st.session_state.selected_preset = persona_options[0]
        st.session_state.system_prompt = PRESET_PROMPTS[persona_options[0]]
    
    # Persona selector
    selected_persona = st.selectbox(
        "Select AI Persona:",
        options=persona_options,
        index=index,
        key="persona_selector"
    )
    
    # Update if changed
    if selected_persona != st.session_state.selected_preset:
        st.session_state.selected_preset = selected_persona
        st.session_state.system_prompt = PRESET_PROMPTS[selected_persona]
        st.rerun()
    
    # Show current persona
    with st.expander("📝 Current Persona Details", expanded=False):
        st.info(st.session_state.system_prompt[:500] + "..." if len(st.session_state.system_prompt) > 500 else st.session_state.system_prompt)
    
    st.divider()
    
    st.header("🔍 Search Configuration")
    
    # Source selection
    st.subheader("Select Search Sources:")
    
    enabled_sources = st.session_state.enabled_sources.copy()
    
    for source_name, source_info in SEARCH_TOOLS.items():
        if source_info["enabled"]:
            is_enabled = st.checkbox(
                f"{source_info['icon']} {source_name}",
                value=source_name in enabled_sources,
                key=f"source_{source_name}",
                help=source_info["description"]
            )
            
            if is_enabled and source_name not in enabled_sources:
                enabled_sources.append(source_name)
            elif not is_enabled and source_name in enabled_sources:
                enabled_sources.remove(source_name)
    
    st.session_state.enabled_sources = enabled_sources
    
    # Search toggle
    search_enabled = st.toggle(
        "Enable Web Search",
        value=bool(enabled_sources),
        help="Turn off to use AI knowledge only"
    )
    
    if not search_enabled:
        st.info("⚠️ Web search disabled - using AI knowledge only")
    
    st.divider()
    
    st.header("⚙️ Settings")
    
    temperature = st.slider(
        "Temperature (creativity):",
        min_value=0.1,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help="Higher = more creative/random, Lower = more focused/deterministic"
    )
    
    st.session_state.temperature = temperature
    
    response_length = st.select_slider(
        "Response Length:",
        options=["Short", "Medium", "Long", "Very Long"],
        value="Medium"
    )
    
    # Map response length to tokens
    length_map = {"Short": 384, "Medium": 512, "Long": 768, "Very Long": 1024}
    st.session_state.max_tokens = length_map[response_length]
    
    st.divider()
    
    st.header("🛠️ Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset AI", use_container_width=True, type="secondary"):
            st.session_state.selected_preset = "Deep Thinker Pro"
            st.session_state.system_prompt = PRESET_PROMPTS["Deep Thinker Pro"]
            st.rerun()
    
    if st.button("📊 View Search History", use_container_width=True):
        st.session_state.show_search_history = not st.session_state.show_search_history
    
    st.divider()
    
    st.caption("💡 **Tips:**")
    st.caption("- Enable multiple sources for comprehensive answers")
    st.caption("- Adjust temperature for more creative or factual responses")
    st.caption("- Clear chat to start fresh conversations")
    
    st.divider()
    st.caption("SmartThink AI v1.1")
    st.caption("Model: TinyLLaMA 1.1B")

# Main content area
try:
    # Load model if not loaded
    if st.session_state.model is None:
        with st.spinner("🚀 Loading AI model..."):
            model = load_model_safe()
            if model is None:
                st.error("Failed to load model. Please check your connection.")
                st.stop()
            st.session_state.model = model
            st.success("✅ AI model loaded successfully!")

    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show search sources if available in metadata
                if "metadata" in message and "search_sources" in message["metadata"]:
                    sources = message["metadata"]["search_sources"]
                    if sources:
                        st.markdown("<div style='margin-top: 10px;'>", unsafe_allow_html=True)
                        for source in sources:
                            st.markdown(f'<span class="source-tag">{source}</span>', unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

    # Display search history if enabled
    if st.session_state.show_search_history and st.session_state.search_history:
        with st.expander("🔍 Search History (Last 5 Queries)", expanded=True):
            recent_queries = list(st.session_state.search_history.items())[-5:]
            
            for query, data in reversed(recent_queries):
                st.markdown(f"**Query:** `{query}`")
                st.caption(f"Time: {data['timestamp'][11:19]}")
                
                if data['results']:
                    for source, results in data['results'].items():
                        with st.container():
                            st.markdown(f"**{source}:**")
                            
                            if isinstance(results, list):
                                for item in results:
                                    if 'title' in item:
                                        st.markdown(f"- **{item['title']}**")
                                    if 'summary' in item:
                                        st.markdown(f"  {item['summary'][:200]}...")
                            elif isinstance(results, dict):
                                for key, value in results.items():
                                    if key not in ['source'] and value:
                                        if isinstance(value, str):
                                            st.markdown(f"- {key}: {value[:150]}...")
                else:
                    st.markdown("*No results found*")
                
                st.divider()

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            try:
                # Show thinking process
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown("""
                <div class="thinking-bubble">
                🤔 Analyzing your question and gathering information...
                </div>
                """, unsafe_allow_html=True)
                
                # Generate response
                response, search_results = generate_response(
                    query=prompt,
                    search_enabled=search_enabled and bool(st.session_state.enabled_sources)
                )
                
                # Clear thinking placeholder
                thinking_placeholder.empty()
                
                # Display response
                st.markdown(response)
                
                # Store with metadata
                metadata = {
                    "search_sources": list(search_results.keys()) if search_results else [],
                    "timestamp": datetime.now().isoformat()
                }
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "metadata": metadata
                })
                
                # Show search details if we have results
                if search_results:
                    with st.expander("📋 View Search Details", expanded=False):
                        for source, results in search_results.items():
                            st.markdown(f"**{SEARCH_TOOLS[source]['icon']} {source}**")
                            
                            if isinstance(results, list):
                                for item in results:
                                    if 'title' in item:
                                        st.markdown(f"**{item['title']}**")
                                    if 'summary' in item:
                                        st.markdown(f"{item['summary']}")
                                    if 'url' in item:
                                        st.markdown(f"[Source]({item['url']})")
                                    st.divider()
                            elif isinstance(results, dict):
                                for key, value in results.items():
                                    if key not in ['source'] and value:
                                        st.markdown(f"**{key}:** {value}")
                            
            except Exception as e:
                error_msg = f"I apologize, but I encountered an error: {str(e)[:100]}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "metadata": {"error": True}
                })

    # Show example prompts if chat is empty
    if not st.session_state.messages:
        st.markdown("### 💡 Try asking about:")
        
        examples = [
            "What is climate change and its main causes?",
            "Explain how neural networks work in simple terms",
            "Who was Albert Einstein and what were his contributions?",
            "What are the latest developments in renewable energy?",
            "How does the stock market work for beginners?"
        ]
        
        cols = st.columns(3)
        for idx, example in enumerate(examples):
            with cols[idx % 3]:
                if st.button(example, use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": example})
                    st.rerun()
        
        st.divider()
        st.info("💡 **Tip:** The AI will search the web for current information when you ask a question!")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.code(f"Error details: {traceback.format_exc()[:500]}")
    
    # Offer recovery options
    st.warning("The app encountered an error. You can try:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Restart App"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All Data"):
            st.session_state.clear()
            st.rerun()
