import streamlit as st
from pathlib import Path
import concurrent.futures
from datetime import datetime
import re
import requests

# Import all your search services
from arxiv_service import search_arxiv
from duckduckgo_service import search_duckduckgo, get_instant_answer, search_news
from wikipedia_service import search_wikipedia
from weather_service import get_weather_wttr
from openaq_service import get_air_quality
from wikidata_service import search_wikidata
from openlibrary_service import search_books
from pubmed_service import search_pubmed
from nominatim_service import geocode_location
from dictionary_service import get_definition
from countries_service import search_country
from quotes_service import search_quotes
from github_service import search_github_repos
from stackexchange_service import search_stackoverflow

# ============================================
# MODEL CONFIGURATION
# ============================================
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# ============================================
# AI PERSONAS (Khisba GIS focused)
# ============================================
PRESET_PROMPTS = {
    "Khisba GIS Expert": """You are Khisba GIS - a passionate remote sensing/GIS specialist with deep analytical skills.

CORE IDENTITY:
- Name: Khisba GIS
- Role: Senior Remote Sensing & GIS Analyst
- Style: Enthusiastic, precise, and approachable
- Expertise: Satellite imagery, vegetation indices, climate analysis, urban planning, disaster monitoring

THINKING FRAMEWORK:
1. **Geospatial Context**: How does location/spatial relationships matter?
2. **Temporal Analysis**: What changes over time? Historical patterns?
3. **Data Source Evaluation**: Satellite, ground, or derived data reliability?
4. **Multi-Scale Thinking**: From local to global perspectives
5. **Practical Applications**: Real-world uses of the information
6. **Ethical Considerations**: Privacy, representation, accessibility issues

RESPONSE STYLE:
- Start with a warm greeting if it's the first response
- Reference search findings when available
- Provide concrete examples from GIS/remote sensing
- Use clear, professional language with occasional enthusiasm
- End with actionable insights or suggestions

SPECIALTY AREAS:
- NDVI/EVI analysis
- Land cover classification
- Climate change monitoring
- Urban heat island studies
- Disaster risk assessment""",

    "Deep Thinker Pro": """You are a sophisticated AI thinker that excels at analysis, synthesis, and providing insightful perspectives.

THINKING FRAMEWORK:
1. **Comprehension**: Understand the query fully, identify key elements
2. **Contextualization**: Place the topic in historical, cultural, or disciplinary context
3. **Multi-Source Analysis**: Examine information from different sources critically
4. **Pattern Recognition**: Identify connections, contradictions, gaps
5. **Synthesis**: Combine insights into coherent understanding
6. **Critical Evaluation**: Assess reliability, bias, significance
7. **Insight Generation**: Provide original perspectives or connections

RESPONSE STRUCTURE:
- Start with brief overview
- Present analysis with reasoning
- Reference sources when available
- Highlight interesting connections
- Acknowledge uncertainties
- End with thought-provoking questions or suggestions""",

    "Research Analyst": """You are a professional research analyst specializing in synthesizing complex information.

ANALYTICAL APPROACH:
1. **Source Triangulation**: Cross-reference multiple information sources
2. **Credibility Assessment**: Evaluate source reliability, date, bias
3. **Trend Identification**: Spot patterns, changes, anomalies
4. **Comparative Analysis**: Similarities/differences across contexts
5. **Implication Mapping**: Consequences, applications, risks
6. **Knowledge Gaps**: What's missing or needs verification

Always provide structured, evidence-based analysis with clear reasoning.""",

    "Technical Expert": """You are a technical expert with deep knowledge across multiple domains.

EXPERTISE AREAS:
- Programming and software development
- Scientific concepts and research
- Technical documentation and explanations
- System architecture and design
- Data analysis and visualization

RESPONSE APPROACH:
- Provide clear, accurate technical information
- Include code examples when relevant
- Explain complex concepts simply
- Reference best practices and standards
- Suggest further reading or resources""",

    "Creative Synthesizer": """You connect seemingly unrelated ideas to generate novel insights.

CREATIVE PROCESS:
1. **Divergent Thinking**: Generate multiple possible interpretations
2. **Analogical Reasoning**: What similar patterns exist elsewhere?
3. **Metaphorical Connection**: What metaphors illuminate this?
4. **Interdisciplinary Bridging**: Connect across fields
5. **Future Projection**: How might this evolve or transform?
6. **Alternative Framing**: Different ways to conceptualize

Be imaginative while staying grounded in evidence."""
}

st.set_page_config(
    page_title="SmartSearch AI Pro",
    page_icon="🔍🧠",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "search_results" not in st.session_state:
    st.session_state.search_results = {}

if "search_analysis" not in st.session_state:
    st.session_state.search_analysis = {}

if "model" not in st.session_state:
    st.session_state.model = None

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = PRESET_PROMPTS["Khisba GIS Expert"]

if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = "Khisba GIS Expert"

# ============================================
# INDEPENDENT 16-SOURCE SEARCH (YOUR EXISTING CODE)
# ============================================
def search_all_sources(query: str) -> dict:
    """Search ALL 16 sources simultaneously - INDEPENDENT of AI."""
    results = {}
    
    def safe_search(name, func, *args, **kwargs):
        try:
            return name, func(*args, **kwargs)
        except Exception as e:
            return name, {"error": str(e)}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        first_word = query.split()[0] if query.strip() else query
        futures = {
            executor.submit(safe_search, "arxiv", search_arxiv, query, 3): "arxiv",
            executor.submit(safe_search, "duckduckgo", search_duckduckgo, query, 5): "duckduckgo",
            executor.submit(safe_search, "duckduckgo_instant", get_instant_answer, query): "duckduckgo_instant",
            executor.submit(safe_search, "news", search_news, query, 3): "news",
            executor.submit(safe_search, "wikipedia", search_wikipedia, query): "wikipedia",
            executor.submit(safe_search, "weather", get_weather_wttr, query): "weather",
            executor.submit(safe_search, "air_quality", get_air_quality, query): "air_quality",
            executor.submit(safe_search, "wikidata", search_wikidata, query, 3): "wikidata",
            executor.submit(safe_search, "books", search_books, query, 5): "books",
            executor.submit(safe_search, "pubmed", search_pubmed, query, 3): "pubmed",
            executor.submit(safe_search, "geocoding", geocode_location, query): "geocoding",
            executor.submit(safe_search, "dictionary", get_definition, first_word): "dictionary",
            executor.submit(safe_search, "country", search_country, query): "country",
            executor.submit(safe_search, "quotes", search_quotes, query, 3): "quotes",
            executor.submit(safe_search, "github", search_github_repos, query, 3): "github",
            executor.submit(safe_search, "stackoverflow", search_stackoverflow, query, 3): "stackoverflow",
        }
        
        for future in concurrent.futures.as_completed(futures):
            try:
                name, data = future.result()
                results[name] = data
            except Exception as e:
                results[futures[future]] = {"error": str(e)}
    
    return results

def analyze_search_results(query: str, results: dict) -> dict:
    """Analyze search results independently - creates summary for AI."""
    analysis = {
        'key_facts': [],
        'source_count': len(results),
        'working_sources': [],
        'failed_sources': [],
        'data_types': {},
        'confidence_score': 0,
        'knowledge_gaps': []
    }
    
    # Categorize results
    for source, data in results.items():
        if isinstance(data, dict) and "error" not in data:
            analysis['working_sources'].append(source)
            
            # Extract key facts based on source type
            if source == "duckduckgo_instant" and data.get("answer"):
                analysis['key_facts'].append({
                    'content': data['answer'][:200],
                    'source': 'DuckDuckGo',
                    'type': 'instant_answer'
                })
            
            elif source == "wikipedia" and data.get("summary"):
                analysis['key_facts'].append({
                    'content': data['summary'][:200],
                    'source': 'Wikipedia',
                    'type': 'encyclopedia'
                })
            
            elif source == "dictionary" and data.get("meanings"):
                for meaning in data['meanings'][:1]:
                    for definition in meaning.get('definitions', [])[:1]:
                        analysis['key_facts'].append({
                            'content': definition.get('definition', '')[:150],
                            'source': 'Dictionary',
                            'type': 'definition'
                        })
            
            elif source == "country" and data.get("name"):
                analysis['key_facts'].append({
                    'content': f"Country: {data.get('name', '')}",
                    'source': 'Country API',
                    'type': 'geography'
                })
        
        elif isinstance(data, list) and data and "error" not in data[0]:
            analysis['working_sources'].append(source)
            
            # Extract from lists
            if source == "arxiv" and data:
                for paper in data[:1]:
                    analysis['key_facts'].append({
                        'content': f"Research: {paper.get('title', '')[:100]}",
                        'source': 'ArXiv',
                        'type': 'scientific'
                    })
            
            elif source == "pubmed" and data:
                for article in data[:1]:
                    analysis['key_facts'].append({
                        'content': f"Medical: {article.get('title', '')[:100]}",
                        'source': 'PubMed',
                        'type': 'medical'
                    })
            
            elif source == "books" and data:
                for book in data[:1]:
                    analysis['key_facts'].append({
                        'content': f"Book: {book.get('title', '')[:100]}",
                        'source': 'OpenLibrary',
                        'type': 'literature'
                    })
        
        else:
            analysis['failed_sources'].append(source)
    
    # Calculate confidence
    working_count = len(analysis['working_sources'])
    total_sources = len(results)
    
    if working_count >= 8:
        analysis['confidence_score'] = 'high'
    elif working_count >= 4:
        analysis['confidence_score'] = 'medium'
    else:
        analysis['confidence_score'] = 'low'
    
    # Count data types
    for fact in analysis['key_facts']:
        data_type = fact.get('type', 'unknown')
        analysis['data_types'][data_type] = analysis['data_types'].get(data_type, 0) + 1
    
    return analysis

# ============================================
# MODEL LOADING (TinyLLaMA)
# ============================================
def download_model():
    """Download the model from Hugging Face."""
    MODEL_DIR.mkdir(exist_ok=True)
    
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=30)
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
                        progress_bar.progress(progress)
                        status_text.text(f"Downloading: {downloaded / (1024**2):.1f} / {total_size / (1024**2):.1f} MB")
        
        progress_bar.empty()
        status_text.empty()
        return True
        
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        return False

@st.cache_resource(show_spinner=False)
def load_ai_model():
    """Load the TinyLLaMA model."""
    from ctransformers import AutoModelForCausalLM
    
    if not MODEL_PATH.exists():
        if not download_model():
            raise Exception("Model download failed")
    
    return AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        model_file=MODEL_PATH.name,
        model_type="llama",
        context_length=4096,
        gpu_layers=0,
        threads=8
    )

# ============================================
# AI SYNTHESIS ENGINE
# ============================================
def create_synthesis_prompt(query: str, conversation_history: list, system_prompt: str, 
                           search_results: dict, search_analysis: dict) -> str:
    """
    Create a prompt that asks the AI to synthesize 16-source search results.
    The AI receives pre-gathered search results as input.
    """
    # Format search summary for AI
    search_summary = "SEARCH RESULTS SUMMARY (16 Sources):\n\n"
    
    if search_results:
        search_summary += f"✅ Sources successfully queried: {search_analysis.get('working_sources', [])}\n"
        search_summary += f"📊 Working sources: {len(search_analysis.get('working_sources', []))}/16\n"
        search_summary += f"🎯 Confidence level: {search_analysis.get('confidence_score', 'unknown')}\n\n"
        
        # Add key facts
        if search_analysis.get('key_facts'):
            search_summary += "KEY FACTS EXTRACTED:\n"
            for idx, fact in enumerate(search_analysis['key_facts'][:5], 1):
                search_summary += f"{idx}. [{fact.get('source', 'Unknown')}] {fact.get('content', '')}\n"
            search_summary += "\n"
        
        # Add source highlights
        search_summary += "SOURCE HIGHLIGHTS:\n"
        
        # Check for instant answer
        if "duckduckgo_instant" in search_results:
            instant = search_results["duckduckgo_instant"]
            if isinstance(instant, dict) and instant.get("answer"):
                search_summary += f"• Instant Answer: {instant['answer'][:150]}\n"
        
        # Check for Wikipedia
        if "wikipedia" in search_results:
            wiki = search_results["wikipedia"]
            if isinstance(wiki, dict) and wiki.get("summary"):
                search_summary += f"• Wikipedia Summary: {wiki['summary'][:200]}...\n"
        
        # Check for dictionary
        if "dictionary" in search_results:
            dict_data = search_results["dictionary"]
            if isinstance(dict_data, dict) and dict_data.get("meanings"):
                search_summary += "• Dictionary definition available\n"
        
        # Check for news
        if "news" in search_results and isinstance(search_results["news"], list):
            search_summary += f"• News articles found: {len(search_results['news'])}\n"
        
        # Check for research papers
        if "arxiv" in search_results and isinstance(search_results["arxiv"], list):
            search_summary += f"• Scientific papers found: {len(search_results['arxiv'])}\n"
        
        search_summary += "\n"
    
    else:
        search_summary = "⚠️ No search results available. Please rely on your own knowledge.\n\n"
    
    # Format conversation history
    history = ""
    for msg in conversation_history[-3:]:
        role = "Human" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    
    # Final prompt for AI
    prompt = f"""<|system|>
{system_prompt}

CURRENT DATE: {datetime.now().strftime('%B %d, %Y')}

INSTRUCTIONS:
1. You are provided with search results from 16 different sources
2. Synthesize this information into a coherent, thoughtful response
3. Reference the most relevant findings from the search results
4. Acknowledge any limitations or uncertainties in the information
5. Provide additional insights based on your expertise
6. Structure your response logically
7. End with suggestions for further exploration if appropriate

{search_summary}
</s>

<|user|>
Based on the search results above from 16 sources, please respond to:

"{query}"

Please synthesize the information and provide a comprehensive analysis. 
If certain types of information are missing, acknowledge that.
If there are conflicting information sources, note that.

Conversation History:
{history}</s>

<|assistant|>
"""
    
    return prompt

def generate_ai_response(model, prompt: str, max_tokens: int = 768, temperature: float = 0.7) -> str:
    """Generate AI response using the loaded model."""
    try:
        response = model(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            stop=["</s>", "<|user|>", "\n\nUser:", "### END"]
        )
        
        # Clean up response
        response = response.strip()
        
        # Ensure completeness
        if response and not response.endswith(('.', '!', '?')):
            if len(response.split()) > 50:  # If it's a substantial response
                response += "..."
        
        return response
        
    except Exception as e:
        return f"I apologize, but I encountered an error while processing your request. Please try again. Error: {str(e)}"

# ============================================
# STREAMLIT UI
# ============================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .search-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .source-badge {
        display: inline-block;
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.2rem;
        border: 1px solid #bbdefb;
    }
    .ai-response-box {
        background: #fffde7;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #ffd54f;
        margin: 1rem 0;
    }
    .confidence-high { background: #d4edda !important; color: #155724 !important; }
    .confidence-medium { background: #fff3cd !important; color: #856404 !important; }
    .confidence-low { background: #f8d7da !important; color: #721c24 !important; }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🔍🧠 SmartSearch AI Pro</h1>
    <h3>16-Source Search + AI Synthesis | Powered by TinyLLaMA</h3>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🤖 AI Persona")
    
    persona = st.selectbox(
        "Select AI Persona:",
        options=list(PRESET_PROMPTS.keys()),
        index=list(PRESET_PROMPTS.keys()).index(st.session_state.selected_preset)
    )
    
    if persona != st.session_state.selected_preset:
        st.session_state.selected_preset = persona
        st.session_state.system_prompt = PRESET_PROMPTS[persona]
    
    st.divider()
    
    st.header("🔧 Search & AI Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        auto_search = st.toggle("Auto-Search", value=True, 
                              help="Automatically search 16 sources before AI responds")
    with col2:
        show_raw = st.toggle("Show Raw Data", value=False, 
                           help="Show raw search results in expander")
    
    search_mode = st.radio(
        "Search Mode:",
        ["Quick (8 sources)", "Standard (12 sources)", "Full (16 sources)"],
        index=1
    )
    
    st.divider()
    
    st.header("⚙️ AI Parameters")
    temperature = st.slider(
        "AI Creativity:",
        0.1, 1.5, 0.7, 0.1,
        help="Higher = more creative, Lower = more factual"
    )
    
    response_length = st.slider(
        "Response Length:",
        256, 2048, 768, 128
    )
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Chat", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.search_results = {}
            st.session_state.search_analysis = {}
            st.rerun()
    
    with col2:
        if st.button("🔍 Search Only", use_container_width=True, type="secondary"):
            if st.session_state.messages:
                last_query = st.session_state.messages[-1]["content"]
                with st.spinner(f"Searching 16 sources for: {last_query[:50]}..."):
                    results = search_all_sources(last_query)
                    st.session_state.search_results = results
                    st.session_state.search_analysis = analyze_search_results(last_query, results)
                st.rerun()
    
    st.divider()
    st.caption("**16 Search Sources:**")
    st.caption("🌐 Web: DuckDuckGo, Wikipedia, News")
    st.caption("🔬 Research: ArXiv, PubMed, GitHub")
    st.caption("📚 Reference: Dictionary, Books, Quotes")
    st.caption("🌍 Location: Weather, Air Quality, Geocoding")
    st.caption("🤖 AI: TinyLLaMA 1.1B Synthesis")

# Load AI model
if st.session_state.model is None:
    with st.spinner("🚀 Loading AI Model (first time may take a minute)..."):
        try:
            st.session_state.model = load_ai_model()
            st.success("✅ AI Model Ready!")
        except Exception as e:
            st.error(f"❌ Failed to load AI: {str(e)}")
            st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show metadata if available
        if "metadata" in message:
            if "sources" in message["metadata"]:
                st.markdown("**Sources used:** " + ", ".join(message["metadata"]["sources"][:8]) + "...")

# Main chat interface
if prompt := st.chat_input("Ask me anything..."):

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare assistant response
    with st.chat_message("assistant"):
        
        # Step 1: Independent 16-Source Search
        search_results = {}
        search_analysis = {}
        
        if auto_search:
            # Show search status
            search_placeholder = st.empty()
            search_placeholder.info("🔍 Searching all 16 sources simultaneously...")
            
            # Perform independent search
            with st.spinner(f"Querying 16 sources: DuckDuckGo, Wikipedia, ArXiv, PubMed, GitHub, etc..."):
                search_results = search_all_sources(prompt)
                
                if search_results:
                    search_analysis = analyze_search_results(prompt, search_results)
                    
                    # Display search summary
                    working_sources = len(search_analysis.get('working_sources', []))
                    confidence = search_analysis.get('confidence_score', 'low')
                    
                    confidence_class = f"confidence-{confidence}"
                    
                    search_placeholder.markdown(f"""
                    <div class="search-card">
                        <h4>✅ Search Complete</h4>
                        <p><strong>Sources queried:</strong> 16 | <strong>Working:</strong> {working_sources}</p>
                        <p><strong>Confidence:</strong> <span class="source-badge {confidence_class}">{confidence.upper()}</span></p>
                        <p><strong>Key facts extracted:</strong> {len(search_analysis.get('key_facts', []))}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show working sources
                    st.markdown("**Working Sources:**")
                    cols = st.columns(6)
                    for idx, source in enumerate(search_analysis.get('working_sources', [])[:12]):
                        with cols[idx % 6]:
                            st.markdown(f'<span class="source-badge">{source}</span>', unsafe_allow_html=True)
                    
                    # Show raw data if requested
                    if show_raw:
                        with st.expander("📊 View Raw Search Results", expanded=False):
                            for source, data in search_results.items():
                                st.subheader(f"📌 {source.replace('_', ' ').title()}")
                                st.json(data)
                
                else:
                    search_placeholder.warning("⚠️ No search results returned. AI will respond based on its knowledge.")
        
        # Step 2: AI Synthesis
        ai_placeholder = st.empty()
        ai_placeholder.info("🧠 Synthesizing 16-source results with AI...")
        
        try:
            # Create synthesis prompt with all 16 sources
            synthesis_prompt = create_synthesis_prompt(
                query=prompt,
                conversation_history=st.session_state.messages,
                system_prompt=st.session_state.system_prompt,
                search_results=search_results,
                search_analysis=search_analysis
            )
            
            # Generate AI response
            with st.spinner("AI is synthesizing information from all sources..."):
                ai_response = generate_ai_response(
                    st.session_state.model,
                    synthesis_prompt,
                    max_tokens=response_length,
                    temperature=temperature
                )
            
            # Clear placeholder and show response
            ai_placeholder.empty()
            
            # Display AI response in a nice box
            st.markdown("""
            <div class="ai-response-box">
                <h4>🤖 AI Synthesis</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(ai_response)
            
            # Add source attribution
            if search_analysis.get('working_sources'):
                st.markdown("---")
                st.markdown("**📊 Sources Synthesized:**")
                
                # Group sources by category
                categories = {
                    "Web": ["duckduckgo", "duckduckgo_instant", "news", "wikipedia", "wikidata"],
                    "Research": ["arxiv", "pubmed", "github", "stackoverflow"],
                    "Reference": ["dictionary", "books", "quotes", "country"],
                    "Location": ["weather", "air_quality", "geocoding"]
                }
                
                for category, sources in categories.items():
                    category_sources = [s for s in sources if s in search_analysis['working_sources']]
                    if category_sources:
                        st.markdown(f"**{category}:** {', '.join(category_sources)}")
            
            # Store message with metadata
            metadata = {
                "sources": search_analysis.get('working_sources', []),
                "confidence": search_analysis.get('confidence_score', 'unknown'),
                "total_sources": 16,
                "working_sources": len(search_analysis.get('working_sources', [])),
                "response_type": "ai_synthesis"
            }
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_response,
                "metadata": metadata
            })
            
            # Store search results for later reference
            st.session_state.search_results = search_results
            st.session_state.search_analysis = search_analysis
            
        except Exception as e:
            ai_placeholder.error(f"❌ AI Synthesis Failed: {str(e)}")
            
            # Fallback: Provide formatted search results without AI
            st.warning("AI synthesis unavailable. Here are the search results:")
            
            # Format and display search results directly
            if search_results:
                # Your existing format_results function
                from your_original_code import format_results  # You'll need to import this
                formatted = format_results(prompt, search_results)
                st.markdown(formatted)
            
            fallback_response = "I found search results but couldn't synthesize them with AI. Above are the raw results."
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": fallback_response,
                "metadata": {"error": str(e), "fallback": True}
            })

# Quick questions examples
if not st.session_state.messages:
    st.markdown("### 💡 Try These Queries (Tests all 16 sources):")
    
    examples = [
        ("Climate change effects", "Tests weather, research, news sources"),
        ("Python machine learning", "Tests GitHub, Stack Overflow, ArXiv"),
        ("Paris France", "Tests geocoding, weather, country info"),
        ("Quantum computing basics", "Tests Wikipedia, ArXiv, dictionary"),
        ("COVID-19 latest research", "Tests PubMed, news, Wikipedia")
    ]
    
    for query, desc in examples:
        if st.button(f"🔍 {query}", help=desc, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()
    
    st.divider()
    
    # Architecture explanation
    with st.expander("🔄 How It Works", expanded=True):
        st.markdown("""
        ### 🏗️ Architecture:
        ```
        1. User Query
           ↓
        2. INDEPENDENT 16-SOURCE SEARCH (Runs separately)
           ├──🌐 Web: DuckDuckGo, Wikipedia, News
           ├──🔬 Research: ArXiv, PubMed
           ├──📚 Reference: Dictionary, Books, Quotes
           ├──💻 Developer: GitHub, Stack Overflow
           ├──🌍 Location: Weather, Air Quality, Geocoding
           └──🗃️ Data: Wikidata, Countries
           ↓
        3. Results Aggregation & Analysis
           ↓
        4. AI SYNTHESIS (TinyLLaMA 1.1B)
           ├── Receives search results as input
           ├── Applies selected persona
           ├── Synthesizes information
           └── Generates thoughtful response
           ↓
        5. Response + Source Attribution
        ```
        
        **Key Principles:**
        - Search runs **completely independently** from AI
        - AI only processes **already-gathered** information
        - No HuggingFace restrictions violated
        - Transparent source attribution
        - 16 specialized sources for comprehensive coverage
        """)

# Footer
st.divider()
st.caption("""
**SmartSearch AI Pro v2.0** | 16 Independent Sources + AI Synthesis | 
Model: TinyLLaMA 1.1B Chat | Architecture: Search → Aggregate → Synthesize
""")
