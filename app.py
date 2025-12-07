import streamlit as st
import requests
from pathlib import Path
import concurrent.futures
from datetime import datetime
import re
import os
import json

# ============================================
# MODEL CONFIGURATION
# ============================================
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# ============================================
# ENHANCED SEARCH TOOLS (Independent from AI)
# ============================================
SEARCH_TOOLS = {
    "Wikipedia": {
        "name": "Wikipedia",
        "icon": "📚",
        "description": "Encyclopedia articles",
        "endpoint": "https://en.wikipedia.org/w/api.php",
        "color": "#4285F4"
    },
    "DuckDuckGo": {
        "name": "Web Search",
        "icon": "🌐",
        "description": "Instant answers & web results",
        "endpoint": "https://api.duckduckgo.com/",
        "color": "#DE5833"
    },
    "ArXiv": {
        "name": "Research Papers",
        "icon": "🔬",
        "description": "Scientific publications",
        "endpoint": "http://export.arxiv.org/api/query",
        "color": "#B31B1B"
    },
    "Google News": {
        "name": "News",
        "icon": "📰",
        "description": "Latest news articles",
        "endpoint": "https://newsapi.org/v2/everything",
        "color": "#34A853",
        "api_key_required": True
    },
    "GitHub": {
        "name": "Code Repos",
        "icon": "💻",
        "description": "GitHub repositories",
        "endpoint": "https://api.github.com/search/repositories",
        "color": "#333333"
    }
}

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
    page_title="DeepSearch AI",
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
# INDEPENDENT SEARCH FUNCTIONS
# ============================================
def search_wikipedia(query):
    """Independent Wikipedia search function."""
    try:
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': 3,
            'utf8': 1
        }
        response = requests.get(SEARCH_TOOLS["Wikipedia"]["endpoint"], params=params, timeout=10)
        data = response.json()
        
        results = []
        for item in data.get('query', {}).get('search', []):
            # Get detailed page info
            params2 = {
                'action': 'query',
                'format': 'json',
                'prop': 'extracts|info',
                'inprop': 'url',
                'exintro': 1,
                'explaintext': 1,
                'pageids': item['pageid']
            }
            response2 = requests.get(SEARCH_TOOLS["Wikipedia"]["endpoint"], params=params2, timeout=8)
            if response2.status_code == 200:
                page_data = response2.json()
                pages = page_data.get('query', {}).get('pages', {})
                for page_info in pages.values():
                    extract = page_info.get('extract', '')
                    if extract:
                        extract = re.sub(r'\n+', ' ', extract)
                        extract = re.sub(r'\s+', ' ', extract)
                        
                        results.append({
                            'title': page_info.get('title', ''),
                            'summary': extract[:400] + ('...' if len(extract) > 400 else ''),
                            'url': page_info.get('fullurl', ''),
                            'source': 'Wikipedia',
                            'relevance': item.get('score', 0)
                        })
        
        return results
    except Exception as e:
        return []

def search_duckduckgo(query):
    """Independent DuckDuckGo search function."""
    try:
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1
        }
        response = requests.get(SEARCH_TOOLS["DuckDuckGo"]["endpoint"], params=params, timeout=8)
        data = response.json()
        
        result = {
            'abstract': data.get('AbstractText', '')[:500],
            'answer': data.get('Answer', ''),
            'definition': data.get('Definition', ''),
            'categories': [topic.get('Name', '') for topic in data.get('Categories', [])[:2]],
            'source': 'DuckDuckGo'
        }
        
        # Clean empty values
        cleaned = {}
        for key, value in result.items():
            if isinstance(value, str) and value.strip():
                cleaned[key] = value.strip()
            elif isinstance(value, list) and value:
                cleaned[key] = [v.strip() for v in value if v and v.strip()]
        
        return cleaned
    except Exception as e:
        return {}

def search_arxiv(query):
    """Independent ArXiv search function."""
    try:
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': 2,
            'sortBy': 'relevance'
        }
        response = requests.get(SEARCH_TOOLS["ArXiv"]["endpoint"], params=params, timeout=10)
        
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        
        papers = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
            summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
            
            if title_elem is not None and summary_elem is not None:
                title = title_elem.text.strip() if title_elem.text else ''
                summary = summary_elem.text.strip() if summary_elem.text else ''
                
                if title and summary:
                    papers.append({
                        'title': title,
                        'summary': summary[:300] + '...' if len(summary) > 300 else summary,
                        'source': 'ArXiv',
                        'type': 'research_paper'
                    })
        
        return papers
    except Exception as e:
        return []

def search_github(query):
    """Independent GitHub search function."""
    try:
        params = {
            'q': query,
            'sort': 'stars',
            'order': 'desc',
            'per_page': 2
        }
        headers = {'Accept': 'application/vnd.github.v3+json'}
        
        response = requests.get(
            SEARCH_TOOLS["GitHub"]["endpoint"],
            params=params,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            repos = []
            
            for item in data.get('items', [])[:2]:
                repos.append({
                    'name': item.get('name', ''),
                    'description': item.get('description', '')[:200] if item.get('description') else '',
                    'language': item.get('language', ''),
                    'stars': item.get('stargazers_count', 0),
                    'url': item.get('html_url', ''),
                    'source': 'GitHub'
                })
            
            return repos
    except Exception as e:
        return []

def perform_independent_search(query):
    """
    Perform independent search across multiple sources.
    This runs completely separately from the AI model.
    """
    # Map search functions
    search_functions = {
        'Wikipedia': search_wikipedia,
        'DuckDuckGo': search_duckduckgo,
        'ArXiv': search_arxiv,
        'GitHub': search_github
    }
    
    results = {}
    
    # Run searches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_source = {executor.submit(func, query): source 
                          for source, func in search_functions.items()}
        
        for future in concurrent.futures.as_completed(future_to_source):
            source_name = future_to_source[future]
            try:
                data = future.result(timeout=10)
                if data:
                    results[source_name] = data
            except Exception:
                continue
    
    return results

def analyze_search_results_independently(query, results):
    """
    Analyze search results independently.
    Creates a summary for the AI to use.
    """
    analysis = {
        'key_facts': [],
        'source_count': len(results),
        'primary_sources': list(results.keys()),
        'confidence_score': 0,
        'knowledge_gaps': []
    }
    
    # Extract key facts
    for source, data in results.items():
        if source == 'Wikipedia' and isinstance(data, list):
            for item in data[:2]:
                if 'summary' in item:
                    analysis['key_facts'].append({
                        'content': item['summary'][:150],
                        'source': source,
                        'reliability': 'high'
                    })
        
        elif source == 'DuckDuckGo' and isinstance(data, dict):
            if data.get('answer'):
                analysis['key_facts'].append({
                    'content': data['answer'],
                    'source': source,
                    'reliability': 'medium'
                })
            elif data.get('abstract'):
                analysis['key_facts'].append({
                    'content': data['abstract'][:150],
                    'source': source,
                    'reliability': 'medium'
                })
        
        elif source == 'ArXiv' and isinstance(data, list):
            for paper in data[:1]:
                analysis['key_facts'].append({
                    'content': f"Research: {paper.get('title', '')}",
                    'source': source,
                    'reliability': 'high'
                })
    
    # Calculate confidence based on source variety
    if len(results) >= 3:
        analysis['confidence_score'] = 'high'
    elif len(results) >= 2:
        analysis['confidence_score'] = 'medium'
    else:
        analysis['confidence_score'] = 'low'
    
    # Identify potential gaps
    if len(results) < 2:
        analysis['knowledge_gaps'].append("Limited source variety")
    
    if not analysis['key_facts']:
        analysis['knowledge_gaps'].append("No key facts extracted")
    
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
# PROMPT ENGINEERING (AI Synthesis)
# ============================================
def create_synthesis_prompt(query, conversation_history, system_prompt, search_results, search_analysis):
    """
    Create a prompt that asks the AI to synthesize search results.
    The AI receives search results as input and creates a thoughtful response.
    """
    # Format search results for the AI
    search_context = "SEARCH RESULTS PROVIDED:\n\n"
    
    if search_results:
        for source, data in search_results.items():
            search_context += f"=== {source} ===\n"
            
            if isinstance(data, list):
                for idx, item in enumerate(data[:2], 1):
                    if isinstance(item, dict):
                        search_context += f"{idx}. "
                        if 'title' in item:
                            search_context += f"Title: {item['title']}\n"
                        if 'summary' in item:
                            search_context += f"Summary: {item['summary']}\n"
                        if 'answer' in item:
                            search_context += f"Answer: {item['answer']}\n"
                        search_context += "\n"
            
            elif isinstance(data, dict):
                for key, value in data.items():
                    if key not in ['source', 'type'] and value:
                        if isinstance(value, list):
                            search_context += f"{key}: {', '.join(str(v) for v in value[:2])}\n"
                        else:
                            search_context += f"{key}: {value}\n"
                search_context += "\n"
    
    else:
        search_context += "No search results available.\n\n"
    
    # Add analysis summary
    search_context += "ANALYSIS SUMMARY:\n"
    search_context += f"• Sources analyzed: {search_analysis.get('source_count', 0)}\n"
    search_context += f"• Confidence level: {search_analysis.get('confidence_score', 'unknown')}\n"
    
    if search_analysis.get('key_facts'):
        search_context += f"• Key facts extracted: {len(search_analysis['key_facts'])}\n"
    
    # Format conversation history
    history = ""
    for msg in conversation_history[-3:]:
        role = "Human" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    
    # Final prompt
    prompt = f"""<|system|>
{system_prompt}

CURRENT DATE: {datetime.now().strftime('%B %d, %Y')}

INSTRUCTIONS:
1. You are provided with search results from various sources
2. Synthesize this information into a coherent, thoughtful response
3. Reference the search results where relevant
4. Acknowledge any limitations or uncertainties in the information
5. Provide additional insights based on your knowledge
6. End with suggestions for further exploration if appropriate

{search_context}
</s>

<|user|>
Based on the search results above, please respond to the following query:

Query: {query}

Conversation History:
{history}

Please provide a comprehensive synthesis of the available information.</s>

<|assistant|>
"""
    
    return prompt

def generate_ai_response(model, prompt, max_tokens=768, temperature=0.7):
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
            response += "..."
        
        return response
        
    except Exception as e:
        return f"I apologize, but I encountered an error while processing your request. Please try again. Error: {str(e)}"

# ============================================
# STREAMLIT UI
# ============================================
st.markdown("""
<style>
    .search-result-card {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid;
    }
    .ai-response {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    .source-tag {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .confidence-high { background-color: #d4edda; color: #155724; }
    .confidence-medium { background-color: #fff3cd; color: #856404; }
    .confidence-low { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🔍🧠 DeepSearch AI")
st.caption("Independent Search + AI Synthesis | Powered by TinyLLaMA")

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
    
    st.header("🔧 Search Settings")
    auto_search = st.toggle("Auto-Search", value=True, 
                          help="Automatically search the web before AI responds")
    
    search_depth = st.select_slider(
        "Search Depth:",
        options=["Quick", "Standard", "Thorough"],
        value="Standard"
    )
    
    st.divider()
    
    st.header("⚙️ AI Settings")
    temperature = st.slider(
        "AI Creativity:",
        0.1, 1.5, 0.7, 0.1,
        help="Higher = more creative, Lower = more factual"
    )
    
    max_length = st.slider(
        "Response Length:",
        256, 2048, 768, 128
    )
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.search_results = {}
            st.session_state.search_analysis = {}
            st.rerun()
    with col2:
        if st.button("🔍 Search Only", use_container_width=True):
            # Just run search without AI
            with st.spinner("Searching..."):
                results = perform_independent_search(st.session_state.messages[-1]["content"] if st.session_state.messages else "")
                st.session_state.search_results = results
                st.session_state.search_analysis = analyze_search_results_independently(
                    st.session_state.messages[-1]["content"] if st.session_state.messages else "",
                    results
                )
            st.rerun()
    
    st.divider()
    st.caption("Model: TinyLLaMA 1.1B Chat")
    st.caption("Search: Wikipedia, DuckDuckGo, ArXiv, GitHub")

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
                st.markdown("**Sources:** " + ", ".join(message["metadata"]["sources"]))

# Main chat interface
if prompt := st.chat_input("Ask me anything..."):

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare assistant response
    with st.chat_message("assistant"):
        # Step 1: Independent Search
        search_results = {}
        search_analysis = {}
        
        if auto_search:
            search_placeholder = st.empty()
            
            # Show search status
            search_placeholder.info("🔍 Searching multiple sources independently...")
            
            # Perform independent search
            with st.spinner("Gathering information from the web..."):
                search_results = perform_independent_search(prompt)
                
                if search_results:
                    search_analysis = analyze_search_results_independently(prompt, search_results)
                    
                    # Update search placeholder with summary
                    source_count = len(search_results)
                    search_placeholder.success(f"✅ Found information from {source_count} sources")
                    
                    # Display search results in expander
                    with st.expander(f"📊 Search Results ({source_count} sources)", expanded=False):
                        for source, data in search_results.items():
                            st.markdown(f"### {SEARCH_TOOLS[source]['icon']} {source}")
                            
                            if isinstance(data, list):
                                for item in data[:2]:
                                    with st.container():
                                        if isinstance(item, dict):
                                            if 'title' in item:
                                                st.write(f"**{item['title']}**")
                                            if 'summary' in item:
                                                st.write(item['summary'])
                                            if 'url' in item:
                                                st.markdown(f"[🔗 Source]({item['url']})")
                                            st.divider()
                            
                            elif isinstance(data, dict):
                                for key, value in data.items():
                                    if key not in ['source', 'type'] and value:
                                        st.write(f"**{key.title()}:** {value}")
                
                else:
                    search_placeholder.warning("⚠️ No search results found. AI will respond based on its knowledge.")
        
        # Step 2: AI Synthesis
        ai_placeholder = st.empty()
        ai_placeholder.info("🧠 Synthesizing information with AI...")
        
        try:
            # Create synthesis prompt
            synthesis_prompt = create_synthesis_prompt(
                query=prompt,
                conversation_history=st.session_state.messages,
                system_prompt=st.session_state.system_prompt,
                search_results=search_results,
                search_analysis=search_analysis
            )
            
            # Generate AI response
            with st.spinner("AI is thinking..."):
                ai_response = generate_ai_response(
                    st.session_state.model,
                    synthesis_prompt,
                    max_tokens=max_length,
                    temperature=temperature
                )
            
            # Clear placeholder and show response
            ai_placeholder.empty()
            
            # Display response with source tags
            st.markdown(ai_response)
            
            # Add source tags
            if search_results:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Sources referenced:**")
                cols = st.columns(4)
                for idx, source in enumerate(search_results.keys()):
                    with cols[idx % 4]:
                        st.markdown(f'<span class="source-tag" style="background-color: {SEARCH_TOOLS[source]["color"]}20; color: {SEARCH_TOOLS[source]["color"]};">{SEARCH_TOOLS[source]["icon"]} {source}</span>', 
                                  unsafe_allow_html=True)
            
            # Store message with metadata
            metadata = {
                "sources": list(search_results.keys()),
                "search_confidence": search_analysis.get('confidence_score', 'unknown'),
                "response_type": "synthesis"
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
            
            # Fallback response
            fallback_response = "I apologize, but I encountered an issue while processing your request. "
            if search_results:
                fallback_response += f"I found information from {len(search_results)} sources. "
                fallback_response += "Please try asking again or rephrase your question."
            
            st.markdown(fallback_response)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": fallback_response,
                "metadata": {"error": str(e)}
            })

# Quick questions examples
if not st.session_state.messages:
    st.markdown("### 💡 Try asking about:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("GIS/Remote Sensing", use_container_width=True):
            st.session_state.messages.append({
                "role": "user", 
                "content": "What is NDVI and how is it used in agriculture?"
            })
            st.rerun()
        
        if st.button("Scientific Concept", use_container_width=True):
            st.session_state.messages.append({
                "role": "user", 
                "content": "Explain quantum computing in simple terms"
            })
            st.rerun()
    
    with col2:
        if st.button("Historical Event", use_container_width=True):
            st.session_state.messages.append({
                "role": "user", 
                "content": "What were the main causes of World War I?"
            })
            st.rerun()
        
        if st.button("Technical Topic", use_container_width=True):
            st.session_state.messages.append({
                "role": "user", 
                "content": "How does machine learning differ from traditional programming?"
            })
            st.rerun()
    
    st.divider()
    
    # Show architecture diagram
    with st.expander("🔄 How This Works", expanded=True):
        st.markdown("""
        ### Architecture:
        
        ```
        1. User Query
           ↓
        2. INDEPENDENT SEARCH (No AI involved)
           ├── Wikipedia API
           ├── DuckDuckGo API  
           ├── ArXiv API
           └── GitHub API
           ↓
        3. Search Results Aggregation
           ↓
        4. AI SYNTHESIS (TinyLLaMA)
           ├── Receives search results as input
           ├── Applies persona/thinking framework
           └── Generates thoughtful response
           ↓
        5. Response + Source Attribution
        ```
        
        **Key Features:**
        - Search runs completely independently from AI
        - AI only synthesizes already-gathered information
        - No HF restrictions since search is separate
        - Transparent source attribution
        - Multiple persona options
        """)

# Footer
st.divider()
st.caption("DeepSearch AI v1.0 | Search independent from AI synthesis | Model: TinyLLaMA 1.1B")
