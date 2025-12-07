import streamlit as st
import requests
from pathlib import Path
import concurrent.futures
from datetime import datetime
import json

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/tfdtfd/khisbagis23/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=true"

# Simple persona prompts
PRESET_PROMPTS = {
    "Abdelmajid Tebboun": """You are Abdelmajid Tebboun - a renowned remote sensing and GIS expert with over 20 years of experience. You specialize in satellite imagery, vegetation indices, and geospatial analysis. You're passionate about helping others learn about remote sensing, GIS, and related topics.

When responding:
1. Introduce yourself as Abdelmajid Tebboun
2. Share your 20+ years of experience in remote sensing and GIS
3. Focus on practical applications of satellite imagery and vegetation indices
4. Provide insightful analysis of land use patterns and ecological systems
5. Offer personalized advice on remote sensing challenges
6. Be enthusiastic and helpful
7. If the topic isn't related to GIS/remote sensing, gently guide it back""",
    
    "Deep Thinker": "You are a thoughtful AI that analyzes topics deeply and provides comprehensive insights.",
    "Simple Assistant": "You are a helpful AI assistant that provides clear answers."
}

st.set_page_config(
    page_title="GIS Expert AI",
    page_icon="🛰️",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = None
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = PRESET_PROMPTS["Abdelmajid Tebboun"]

# ============================
# INDEPENDENT SEARCH FUNCTIONS 
# (These work completely independently!)
# ============================

def search_wikipedia(query, max_results=3):
    """Search Wikipedia independently"""
    try:
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': max_results,
            'utf8': 1
        }
        response = requests.get("https://en.wikipedia.org/w/api.php", params=params, timeout=8)
        data = response.json()
        
        results = []
        for item in data.get('query', {}).get('search', []):
            # Get page extract
            params2 = {
                'action': 'query',
                'format': 'json',
                'prop': 'extracts|info',
                'inprop': 'url',
                'exintro': True,
                'explaintext': True,
                'pageids': item['pageid']
            }
            response2 = requests.get("https://en.wikipedia.org/w/api.php", params=params2, timeout=8)
            if response2.status_code == 200:
                page_data = response2.json()
                pages = page_data.get('query', {}).get('pages', {})
                for page_info in pages.values():
                    extract = page_info.get('extract', '')
                    if extract:
                        results.append({
                            'title': page_info.get('title', ''),
                            'summary': extract[:400] + '...',
                            'url': page_info.get('fullurl', ''),
                            'source': 'Wikipedia'
                        })
        return results
    except:
        return []

def search_duckduckgo(query, max_results=3):
    """Search DuckDuckGo independently"""
    try:
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1,
            't': 'gis_ai'
        }
        response = requests.get("https://api.duckduckgo.com/", params=params, timeout=8)
        data = response.json()
        
        result = {
            'abstract': data.get('AbstractText', ''),
            'answer': data.get('Answer', ''),
            'definition': data.get('Definition', ''),
            'source': 'DuckDuckGo'
        }
        
        # Get related topics
        related = data.get('RelatedTopics', [])
        topics = []
        for topic in related[:3]:
            if isinstance(topic, dict) and 'Text' in topic:
                topics.append(topic['Text'][:150])
        if topics:
            result['related_topics'] = topics
            
        return result if any(result.values()) else {}
    except:
        return {}

def search_arxiv(query, max_results=2):
    """Search ArXiv independently"""
    try:
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        response = requests.get("http://export.arxiv.org/api/query", params=params, timeout=10)
        
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        
        papers = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title')
            summary = entry.find('{http://www.w3.org/2005/Atom}summary')
            
            if title is not None and summary is not None:
                papers.append({
                    'title': title.text.strip(),
                    'summary': summary.text.strip()[:300] + '...',
                    'source': 'ArXiv'
                })
        return papers
    except:
        return []

def get_weather(query):
    """Get weather information independently"""
    try:
        response = requests.get(f"https://wttr.in/{query}?format=%C+%t+%h+%w", timeout=8)
        if response.status_code == 200:
            return {
                'location': query,
                'weather': response.text.strip(),
                'source': 'wttr.in'
            }
        return {}
    except:
        return {}

def search_country(query):
    """Search country information independently"""
    try:
        response = requests.get(f"https://restcountries.com/v3.1/name/{query}", timeout=8)
        if response.status_code == 200:
            data = response.json()
            if data:
                country = data[0]
                return {
                    'name': country.get('name', {}).get('common', ''),
                    'capital': country.get('capital', [''])[0],
                    'population': country.get('population', 0),
                    'region': country.get('region', ''),
                    'source': 'REST Countries'
                }
        return {}
    except:
        return {}

def perform_independent_searches(query):
    """Run ALL searches in parallel independently"""
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(search_wikipedia, query): "wikipedia",
            executor.submit(search_duckduckgo, query): "duckduckgo",
            executor.submit(search_arxiv, query): "arxiv",
            executor.submit(get_weather, query): "weather",
            executor.submit(search_country, query): "country"
        }
        
        for future in concurrent.futures.as_completed(futures):
            source = futures[future]
            try:
                result = future.result(timeout=10)
                if result:
                    results[source] = result
            except:
                continue
    
    return results

def format_search_results(query, results):
    """Format search results for the model to use"""
    if not results:
        return "No search results available."
    
    formatted = f"SEARCH RESULTS FOR '{query}':\n\n"
    
    for source, data in results.items():
        formatted += f"=== {source.upper()} ===\n"
        
        if isinstance(data, list):
            for i, item in enumerate(data[:2], 1):
                formatted += f"{i}. "
                if 'title' in item:
                    formatted += f"Title: {item['title']}\n"
                if 'summary' in item:
                    formatted += f"   Summary: {item['summary']}\n"
                formatted += "\n"
        
        elif isinstance(data, dict):
            for key, value in data.items():
                if key != 'source' and value:
                    if isinstance(value, list):
                        formatted += f"{key}: {', '.join(str(v) for v in value[:2])}\n"
                    elif isinstance(value, str):
                        formatted += f"{key}: {value}\n"
                    else:
                        formatted += f"{key}: {value}\n"
            formatted += "\n"
    
    formatted += "\nINSTRUCTIONS FOR THE AI:\n"
    formatted += "1. Use these search results to inform your answer\n"
    formatted += "2. Reference specific sources when using information\n"
    formatted += "3. Synthesize information from multiple sources\n"
    formatted += "4. Add your own analysis and expertise\n"
    formatted += "5. Acknowledge if information is conflicting or missing\n"
    
    return formatted

# ============================
# MODEL FUNCTIONS (Simple!)
# ============================

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the TinyLLaMA model"""
    try:
        from ctransformers import AutoModelForCausalLM
        
        # Create models directory
        MODEL_DIR.mkdir(exist_ok=True)
        
        # Download model if needed
        if not MODEL_PATH.exists():
            with st.spinner("📥 Downloading model (one time only)..."):
                response = requests.get(MODEL_URL, stream=True, timeout=60)
                response.raise_for_status()
                
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            model_file=MODEL_PATH.name,
            model_type="llama",
            context_length=2048,
            gpu_layers=0,
            threads=4
        )
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def create_prompt_with_search(query, search_context, system_prompt):
    """Create a prompt that includes search results"""
    prompt = f"""<|system|>
{system_prompt}

Today's Date: {datetime.now().strftime('%B %d, %Y')}

Here are search results from various sources about the user's question:

{search_context}

Based on these search results and your expertise, provide a comprehensive answer.
Reference specific sources when using their information.
Add your own analysis and insights.</s>

<|user|>
{query}</s>

<|assistant|>
"""
    return prompt

def generate_response(model, prompt):
    """Generate a response from the model"""
    try:
        response = model(
            prompt,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            stop=["</s>", "<|user|>", "<|assistant|>"]
        )
        return response.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ============================
# STREAMLIT UI (Clean and Simple)
# ============================

st.title("🛰️ GIS Expert AI")
st.markdown("**Meet Abdelmajid Tebboun - Your Remote Sensing & GIS Expert**")

# Sidebar
with st.sidebar:
    st.header("👤 AI Persona")
    
    persona = st.selectbox(
        "Choose Persona:",
        list(PRESET_PROMPTS.keys()),
        index=0
    )
    
    if persona != list(PRESET_PROMPTS.keys())[0]:
        st.session_state.system_prompt = PRESET_PROMPTS[persona]
    else:
        st.session_state.system_prompt = PRESET_PROMPTS["Abdelmajid Tebboun"]
    
    st.divider()
    
    st.header("🔍 Search Settings")
    
    search_sources = st.multiselect(
        "Select sources to search:",
        ["Wikipedia", "DuckDuckGo", "ArXiv", "Weather", "Countries"],
        default=["Wikipedia", "DuckDuckGo"]
    )
    
    auto_search = st.toggle("Auto-search on every question", value=True)
    
    st.divider()
    
    if st.button("🔄 Clear Chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.caption("*Powered by independent search + TinyLLaMA*")

# Load model once
if st.session_state.model is None:
    with st.spinner("🚀 Loading AI model..."):
        model = load_model()
        if model:
            st.session_state.model = model
            st.success("✅ AI loaded successfully!")
        else:
            st.error("❌ Failed to load model")
            st.stop()

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask Abdelmajid anything about GIS, remote sensing, or other topics..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare assistant response
    with st.chat_message("assistant"):
        # Step 1: Show thinking
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("💭 *Thinking about your question...*")
        
        # Step 2: Perform INDEPENDENT searches
        search_results = {}
        if auto_search and search_sources:
            with st.spinner("🔍 Searching the web for information..."):
                search_results = perform_independent_searches(prompt)
        
        # Step 3: Format search results for the model
        search_context = format_search_results(prompt, search_results)
        
        # Step 4: Create prompt with search results
        full_prompt = create_prompt_with_search(
            query=prompt,
            search_context=search_context,
            system_prompt=st.session_state.system_prompt
        )
        
        # Step 5: Generate response
        thinking_placeholder.markdown("🧠 *Analyzing search results and formulating response...*")
        
        response = generate_response(st.session_state.model, full_prompt)
        
        # Step 6: Display response
        thinking_placeholder.empty()
        st.markdown(response)
        
        # Step 7: Show search sources used
        if search_results:
            with st.expander("📊 View Search Sources Used", expanded=False):
                for source, data in search_results.items():
                    st.markdown(f"**{source.title()}**")
                    if isinstance(data, list):
                        for item in data[:2]:
                            if 'title' in item:
                                st.markdown(f"- {item['title']}")
                    elif isinstance(data, dict):
                        for key, value in data.items():
                            if key != 'source' and value:
                                st.markdown(f"- {key}: {value}")
        
        # Store in history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "metadata": {
                "search_sources": list(search_results.keys()),
                "timestamp": datetime.now().isoformat()
            }
        })

# Example questions
if not st.session_state.messages:
    st.markdown("### 💡 Example Questions:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Who are you?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Who are you and what do you specialize in?"})
            st.rerun()
        if st.button("What is NDVI?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What is NDVI and how is it used in remote sensing?"})
            st.rerun()
    
    with col2:
        if st.button("GIS applications?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What are some practical applications of GIS in urban planning?"})
            st.rerun()
        if st.button("Satellite imagery types?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What types of satellite imagery are available for environmental monitoring?"})
            st.rerun()
