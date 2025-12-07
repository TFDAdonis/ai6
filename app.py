import streamlit as st
import requests
from pathlib import Path
import concurrent.futures
from datetime import datetime
import re
import json

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/tfdtfd/khisbagis23/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=true"

# Enhanced deep thinking prompts from first code
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
    "Research Analyst": """You are a professional research analyst specializing in synthesizing complex information.""",
    "Creative Synthesizer": """You connect seemingly unrelated ideas to generate novel insights."""
}

# Optimized search tools from first code
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
    },
    "NewsAPI": {
        "name": "News",
        "icon": "📰",
        "description": "Recent news articles",
        "endpoint": "https://newsapi.org/v2/everything",
        "enabled": False,  # Needs API key
        "api_key": ""
    }
}

st.set_page_config(
    page_title="SmartThink AI",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = None

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = PRESET_PROMPTS["Deep Thinker Pro"]

if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = "Deep Thinker Pro"

if "search_history" not in st.session_state:
    st.session_state.search_history = {}

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# ============================
# INDEPENDENT SEARCH FUNCTIONS
# ============================

def perform_independent_search(query, selected_sources=None):
    """Perform search independently without model involvement"""
    if selected_sources is None:
        selected_sources = ["Wikipedia", "DuckDuckGo"]
    
    results = {}
    search_tasks = []
    
    # Prepare search tasks
    for source_name in selected_sources:
        if source_name in SEARCH_TOOLS and SEARCH_TOOLS[source_name]["enabled"]:
            search_tasks.append((source_name, query))
    
    # Execute searches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(search_tasks)) as executor:
        future_to_source = {}
        
        for source_name, query in search_tasks:
            if source_name == "Wikipedia":
                future = executor.submit(search_wikipedia, query)
                future_to_source[future] = source_name
            elif source_name == "DuckDuckGo":
                future = executor.submit(search_duckduckgo, query)
                future_to_source[future] = source_name
            elif source_name == "ArXiv":
                future = executor.submit(search_arxiv, query)
                future_to_source[future] = source_name
            elif source_name == "NewsAPI":
                future = executor.submit(search_news, query)
                future_to_source[future] = source_name
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_source):
            source_name = future_to_source[future]
            try:
                result = future.result(timeout=10)
                if result:
                    results[source_name] = result
            except Exception as e:
                st.warning(f"Search failed for {source_name}: {str(e)}")
    
    # Cache results
    st.session_state.search_history[query] = {
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    
    return results

def search_wikipedia(query, max_results=3):
    """Search Wikipedia independently"""
    try:
        # Search for pages
        search_params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': max_results
        }
        
        response = requests.get(
            SEARCH_TOOLS["Wikipedia"]["endpoint"],
            params=search_params,
            timeout=10
        )
        
        if response.status_code != 200:
            return []
        
        search_data = response.json()
        pages = search_data.get('query', {}).get('search', [])
        
        results = []
        for page in pages:
            # Get page content
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
                timeout=10
            )
            
            if content_response.status_code == 200:
                page_data = content_response.json()
                page_info = page_data.get('query', {}).get('pages', {}).get(str(page['pageid']), {})
                
                if page_info:
                    results.append({
                        'title': page_info.get('title', 'Unknown'),
                        'summary': page_info.get('extract', '')[:500],
                        'url': page_info.get('fullurl', ''),
                        'source': 'Wikipedia',
                        'relevance_score': page.get('score', 0),
                        'wordcount': page_info.get('wordcount', 0),
                        'timestamp': page_info.get('touched', '')
                    })
        
        return results[:max_results]
        
    except Exception as e:
        st.error(f"Wikipedia search error: {e}")
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
            timeout=10
        )
        
        if response.status_code != 200:
            return {}
        
        data = response.json()
        
        result = {
            'abstract': data.get('AbstractText', ''),
            'answer': data.get('Answer', ''),
            'definition': data.get('Definition', ''),
            'related_topics': [],
            'source': 'DuckDuckGo'
        }
        
        # Extract related topics
        related = data.get('RelatedTopics', [])
        for topic in related[:5]:
            if isinstance(topic, dict) and 'Text' in topic:
                result['related_topics'].append(topic['Text'][:200])
        
        # Clean empty values
        result = {k: v for k, v in result.items() if v and (not isinstance(v, str) or v.strip())}
        
        return result if result else {}
        
    except Exception as e:
        st.error(f"DuckDuckGo search error: {e}")
        return {}

def search_arxiv(query, max_results=3):
    """Search ArXiv independently"""
    try:
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        response = requests.get(
            SEARCH_TOOLS["ArXiv"]["endpoint"],
            params=params,
            timeout=15
        )
        
        if response.status_code != 200:
            return []
        
        # Parse XML
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        
        results = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
            summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
            published_elem = entry.find('{http://www.w3.org/2005/Atom}published')
            
            if title_elem is not None and summary_elem is not None:
                results.append({
                    'title': title_elem.text.strip(),
                    'summary': summary_elem.text.strip()[:400],
                    'published': published_elem.text[:10] if published_elem is not None else '',
                    'source': 'ArXiv',
                    'relevance_score': 1.0
                })
        
        return results
        
    except Exception as e:
        st.error(f"ArXiv search error: {e}")
        return []

def search_news(query, max_results=3):
    """Search news independently (requires NewsAPI key)"""
    if not SEARCH_TOOLS["NewsAPI"]["api_key"]:
        return []
    
    try:
        params = {
            'q': query,
            'apiKey': SEARCH_TOOLS["NewsAPI"]["api_key"],
            'pageSize': max_results,
            'language': 'en',
            'sortBy': 'relevance'
        }
        
        response = requests.get(
            SEARCH_TOOLS["NewsAPI"]["endpoint"],
            params=params,
            timeout=10
        )
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        articles = data.get('articles', [])
        
        results = []
        for article in articles[:max_results]:
            results.append({
                'title': article.get('title', ''),
                'summary': article.get('description', '')[:300],
                'url': article.get('url', ''),
                'source': article.get('source', {}).get('name', 'News'),
                'published': article.get('publishedAt', ''),
                'relevance_score': 0.8
            })
        
        return results
        
    except Exception as e:
        st.error(f"News search error: {e}")
        return []

def analyze_search_results_for_prompt(query, search_results):
    """Analyze and format search results for the model prompt"""
    if not search_results:
        return "No search results available. Please rely on your own knowledge."
    
    formatted_context = "SEARCH RESULTS FOR YOUR ANALYSIS:\n\n"
    
    for source, results in search_results.items():
        formatted_context += f"=== {source.upper()} RESULTS ===\n"
        
        if isinstance(results, list):
            for i, item in enumerate(results[:2], 1):
                formatted_context += f"{i}. "
                if 'title' in item:
                    formatted_context += f"Title: {item['title']}\n"
                if 'summary' in item:
                    formatted_context += f"   Summary: {item['summary']}\n"
                if 'url' in item:
                    formatted_context += f"   URL: {item['url']}\n"
                formatted_context += "\n"
        
        elif isinstance(results, dict):
            for key, value in results.items():
                if key not in ['source', 'relevance_score'] and value:
                    if isinstance(value, list):
                        formatted_context += f"{key}: {', '.join(str(v) for v in value[:3])}\n"
                    else:
                        formatted_context += f"{key}: {value}\n"
            formatted_context += "\n"
    
    formatted_context += "\nANALYSIS INSTRUCTIONS:\n"
    formatted_context += "1. Review these search results carefully\n"
    formatted_context += "2. Identify the most reliable information\n"
    formatted_context += "3. Note any contradictions between sources\n"
    formatted_context += "4. Synthesize information into a coherent answer\n"
    formatted_context += "5. Cite your sources when referencing specific information\n"
    formatted_context += "6. If information is missing, acknowledge the gaps\n"
    
    return formatted_context

# ============================
# MODEL FUNCTIONS
# ============================

def download_model():
    """Download model from URL"""
    MODEL_DIR.mkdir(exist_ok=True)
    
    if MODEL_PATH.exists():
        file_size = MODEL_PATH.stat().st_size / (1024**3)
        st.info(f"Model already downloaded ({file_size:.2f} GB)")
        return True
    
    with st.spinner("Downloading model... This may take a while (~637 MB)"):
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
                            status_text.text(f"Downloaded: {downloaded / (1024**2):.1f} MB")
            
            progress_bar.empty()
            status_text.empty()
            
            if MODEL_PATH.exists():
                file_size = MODEL_PATH.stat().st_size / (1024**3)
                st.success(f"Model downloaded successfully! ({file_size:.2f} GB)")
                return True
            else:
                st.error("Download completed but file not found!")
                return False
                
        except Exception as e:
            st.error(f"Download failed: {str(e)}")
            if MODEL_PATH.exists():
                MODEL_PATH.unlink()
            return False

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the model using ctransformers"""
    from ctransformers import AutoModelForCausalLM
    
    if not MODEL_PATH.exists():
        if not download_model():
            raise Exception("Model download failed")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            model_file=MODEL_PATH.name,
            model_type="llama",
            context_length=4096,
            gpu_layers=0,
            threads=4
        )
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        raise

def create_smart_prompt(query, search_context, system_prompt, conversation_history):
    """Create prompt that includes search results and thinking instructions"""
    
    # Format conversation history
    history_text = ""
    if conversation_history:
        history_text = "PREVIOUS CONVERSATION:\n"
        for msg in conversation_history[-3:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
        history_text += "\n"
    
    # Create the complete prompt
    prompt = f"""<|system|>
{system_prompt}

CURRENT DATE: {datetime.now().strftime('%B %d, %Y')}

{search_context}

{history_text}

INSTRUCTIONS FOR YOUR RESPONSE:
1. Analyze the search results provided above
2. Synthesize information from multiple sources
3. Provide clear, well-reasoned analysis
4. Reference sources when using specific information
5. Acknowledge any uncertainties or gaps
6. Maintain your persona's style and expertise

IMPORTANT: Your response should demonstrate deep thinking and synthesis of the search results.</s>

<|user|>
{query}</s>

<|assistant|>
"""
    
    return prompt

def generate_response_with_search(model, query, system_prompt, conversation_history, temperature=0.7, max_tokens=1024):
    """Generate response using independent search results"""
    
    # Step 1: Perform independent search
    with st.spinner("🔍 Searching for information..."):
        search_results = perform_independent_search(query)
    
    # Step 2: Format search results for prompt
    search_context = analyze_search_results_for_prompt(query, search_results)
    
    # Step 3: Create enhanced prompt
    prompt = create_smart_prompt(query, search_context, system_prompt, conversation_history)
    
    # Step 4: Generate response
    with st.spinner("🧠 Analyzing and synthesizing information..."):
        try:
            response = model(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                stop=["</s>", "<|user|>", "<|assistant|>", "\n\nUser:", "###"]
            )
            
            # Clean up response
            response = response.strip()
            
            # Ensure response is complete
            if response and response[-1] not in ['.', '!', '?', ':', '"', "'"]:
                response += "..."
            
            # Add source citations if we have search results
            if search_results:
                response += f"\n\n📚 *Information synthesized from: {', '.join(search_results.keys())}*"
            
            return response, search_results
            
        except Exception as e:
            st.error(f"Generation error: {str(e)}")
            return "I apologize, but I encountered an error while generating a response. Please try again.", {}

# ============================
# STREAMLIT UI
# ============================

st.title("🧠 SmartThink AI")
st.caption("Independent search + LLM synthesis for intelligent responses")

# Sidebar configuration
with st.sidebar:
    st.header("🤖 AI Persona")
    
    persona_options = list(PRESET_PROMPTS.keys())
    selected_persona = st.selectbox(
        "Select Persona:",
        options=persona_options,
        index=persona_options.index(st.session_state.selected_preset)
    )
    
    if selected_persona != st.session_state.selected_preset:
        st.session_state.selected_preset = selected_persona
        st.session_state.system_prompt = PRESET_PROMPTS[selected_persona]
    
    st.divider()
    
    st.header("🔧 Search Configuration")
    
    enabled_sources = []
    for source_name, source_info in SEARCH_TOOLS.items():
        if source_info["enabled"]:
            enabled = st.checkbox(
                f"{source_info['icon']} {source_name}",
                value=True,
                help=source_info["description"]
            )
            if enabled:
                enabled_sources.append(source_name)
    
    if not enabled_sources:
        enabled_sources = ["Wikipedia", "DuckDuckGo"]
    
    st.divider()
    
    st.header("⚙️ Generation Settings")
    
    temperature = st.slider(
        "Temperature:",
        min_value=0.1,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help="Higher = more creative, Lower = more focused"
    )
    
    max_tokens = st.slider(
        "Response Length:",
        min_value=256,
        max_value=2048,
        value=1024,
        step=256,
        help="Maximum tokens in response"
    )
    
    search_depth = st.selectbox(
        "Search Depth:",
        ["Quick", "Moderate", "Deep"],
        index=1
    )
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.search_history = {}
            st.rerun()
    
    with col2:
        if st.button("📊 View Search History", use_container_width=True):
            st.session_state.show_search_history = not st.session_state.get('show_search_history', False)
    
    st.divider()
    st.caption("SmartThink AI v1.0")
    st.caption("Independent search + LLM synthesis")

# Load model
if st.session_state.model is None:
    with st.spinner("🚀 Loading AI model..."):
        try:
            st.session_state.model = load_model()
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show search sources if available
        if "search_sources" in message.get("metadata", {}):
            sources = message["metadata"]["search_sources"]
            if sources:
                st.caption(f"📚 Sources: {', '.join(sources)}")

# Search history viewer
if st.session_state.get('show_search_history', False) and st.session_state.search_history:
    with st.expander("🔍 Search History", expanded=True):
        for query, data in list(st.session_state.search_history.items())[-5:]:
            st.subheader(f"Query: '{query}'")
            st.caption(f"Time: {data['timestamp']}")
            
            for source, results in data['results'].items():
                st.write(f"**{source}:**")
                if isinstance(results, list):
                    for item in results[:2]:
                        if 'title' in item:
                            st.write(f"- {item['title']}")
                elif isinstance(results, dict):
                    for key, value in results.items():
                        if key not in ['source'] and value:
                            st.write(f"- {key}: {value}")
            st.divider()

# Chat input
if prompt := st.chat_input("Ask me anything (I'll search and analyze):"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        try:
            response, search_results = generate_response_with_search(
                model=st.session_state.model,
                query=prompt,
                system_prompt=st.session_state.system_prompt,
                conversation_history=st.session_state.messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            st.markdown(response)
            
            # Store with metadata
            metadata = {
                "search_sources": list(search_results.keys()) if search_results else [],
                "timestamp": datetime.now().isoformat(),
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "metadata": metadata
            })
            
            # Show detailed search results
            if search_results:
                with st.expander("📊 View Search Results", expanded=False):
                    for source, results in search_results.items():
                        st.subheader(f"{SEARCH_TOOLS[source]['icon']} {source}")
                        
                        if isinstance(results, list):
                            for i, item in enumerate(results[:3], 1):
                                st.write(f"**{i}. {item.get('title', 'Result')}**")
                                if 'summary' in item:
                                    st.write(item['summary'])
                                if 'url' in item:
                                    st.caption(f"[Source]({item['url']})")
                                st.divider()
                        elif isinstance(results, dict):
                            for key, value in results.items():
                                if key not in ['source'] and value:
                                    st.write(f"**{key}:** {value}")
                        
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I apologize, but I encountered an error. Please try again.",
                "metadata": {"error": str(e)}
            })

# Example prompts
if not st.session_state.messages:
    st.markdown("### 💡 Example Questions to Try:")
    
    examples = [
        "What are the latest developments in renewable energy technology?",
        "Explain the impact of climate change on coastal cities",
        "Who was Leonardo da Vinci and what were his most important contributions?",
        "What is quantum computing and how does it differ from classical computing?",
        "Compare different approaches to artificial intelligence ethics"
    ]
    
    cols = st.columns(3)
    for idx, example in enumerate(examples):
        with cols[idx % 3]:
            if st.button(example, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()
