import streamlit as st
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
import concurrent.futures
import requests
from pathlib import Path

# ========== AI Model Configuration ==========
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

# ========== Streamlit Page Config ==========
st.set_page_config(
    page_title="AI Search Assistant",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Multi-Source Search Assistant")
st.markdown("*Searches all sources simultaneously, then gets AI-enhanced analysis*")

# ========== Session State Initialization ==========
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = PRESET_PROMPTS["Khisba GIS"]

if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = "Khisba GIS"

# ========== Sidebar ==========
with st.sidebar:
    st.header("🔧 AI Configuration")
    
    # Model settings in sidebar
    st.subheader("AI Model Settings")
    
    selected_preset = st.selectbox(
        "Choose AI Persona:",
        options=list(PRESET_PROMPTS.keys()),
        index=list(PRESET_PROMPTS.keys()).index(st.session_state.selected_preset),
        key="preset_selector"
    )
    
    if selected_preset != st.session_state.selected_preset:
        st.session_state.selected_preset = selected_preset
        if selected_preset != "Custom":
            st.session_state.system_prompt = PRESET_PROMPTS[selected_preset]
    
    system_prompt = st.text_area(
        "System Prompt:",
        value=st.session_state.system_prompt,
        height=150,
        placeholder="Enter instructions for how the AI should behave...",
        key="system_prompt_input"
    )
    
    if system_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = system_prompt
        if system_prompt not in PRESET_PROMPTS.values():
            st.session_state.selected_preset = "Custom"
    
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1, 
                           help="Higher = more creative, Lower = more focused")
    max_tokens = st.slider("Max Tokens", 64, 1024, 256, 64,
                          help="Maximum length of the AI response")
    
    st.divider()
    
    # Sources information (original unchanged)
    st.header("📊 16 Sources Searched")
    st.markdown("""
    **Web & Knowledge:**
    - DuckDuckGo Web Search
    - DuckDuckGo Instant Answers
    - DuckDuckGo News
    - Wikipedia
    - Wikidata
    
    **Science & Research:**
    - ArXiv (Scientific Papers)
    - PubMed (Medical Research)
    
    **Reference:**
    - OpenLibrary (Books)
    - Dictionary API
    - REST Countries
    - Quotable (Quotes)
    
    **Developer:**
    - GitHub Repositories
    - Stack Overflow Q&A
    
    **Location & Environment:**
    - Nominatim (Geocoding)
    - wttr.in (Weather)
    - OpenAQ (Air Quality)
    """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset AI"):
            st.session_state.system_prompt = PRESET_PROMPTS["Default Assistant"]
            st.session_state.selected_preset = "Default Assistant"
            st.rerun()

# ========== Display Chat History ==========
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ========== ORIGINAL FUNCTIONS (Unchanged) ==========
def search_all_sources(query: str) -> dict:
    """Search ALL sources simultaneously."""
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

def format_results(query: str, results: dict) -> str:
    """Format all search results into a readable response."""
    output = [f"## Search Results for: *{query}*\n"]
    
    if "duckduckgo_instant" in results:
        instant = results["duckduckgo_instant"]
        if isinstance(instant, dict) and instant.get("answer"):
            output.append(f"### 💡 Quick Answer\n{instant['answer']}\n")
    
    if "wikipedia" in results:
        wiki = results["wikipedia"]
        if isinstance(wiki, dict) and wiki.get("exists"):
            output.append(f"### 📚 Wikipedia: {wiki.get('title', 'N/A')}")
            output.append(f"{wiki.get('summary', 'No summary')[:500]}...")
            output.append(f"[Read more]({wiki.get('url', '')})\n")
    
    if "duckduckgo" in results:
        ddg = results["duckduckgo"]
        if isinstance(ddg, list) and ddg and "error" not in ddg[0]:
            output.append("### 🌐 Web Results")
            for item in ddg[:3]:
                output.append(f"- **{item.get('title', 'N/A')}**")
                output.append(f"  {item.get('body', '')[:150]}...")
                if item.get('url'):
                    output.append(f"  [Link]({item.get('url')})")
            output.append("")
    
    if "arxiv" in results:
        arxiv_data = results["arxiv"]
        if isinstance(arxiv_data, list) and arxiv_data and "error" not in arxiv_data[0]:
            output.append("### 🔬 Scientific Papers (ArXiv)")
            for paper in arxiv_data[:3]:
                authors = ", ".join(paper.get("authors", [])[:2])
                output.append(f"- **{paper.get('title', 'N/A')}**")
                output.append(f"  Authors: {authors} | Published: {paper.get('published', 'N/A')}")
                output.append(f"  {paper.get('summary', '')[:200]}...")
                if paper.get('url'):
                    output.append(f"  [View Paper]({paper.get('url')})")
            output.append("")
    
    if "pubmed" in results:
        pubmed_data = results["pubmed"]
        if isinstance(pubmed_data, list) and pubmed_data and "error" not in pubmed_data[0] and "message" not in pubmed_data[0]:
            output.append("### 🏥 Medical Research (PubMed)")
            for article in pubmed_data[:3]:
                authors = ", ".join(article.get("authors", [])[:2])
                output.append(f"- **{article.get('title', 'N/A')}**")
                output.append(f"  Authors: {authors} | Year: {article.get('year', 'N/A')}")
                output.append(f"  {article.get('abstract', '')[:200]}...")
                if article.get('url'):
                    output.append(f"  [View Article]({article.get('url')})")
            output.append("")
    
    if "books" in results:
        books_data = results["books"]
        if isinstance(books_data, list) and books_data and "error" not in books_data[0]:
            output.append("### 📖 Books (OpenLibrary)")
            for book in books_data[:3]:
                authors = ", ".join(book.get("authors", [])[:2])
                output.append(f"- **{book.get('title', 'N/A')}**")
                output.append(f"  Authors: {authors} | First Published: {book.get('first_publish_year', 'N/A')}")
                if book.get('url'):
                    output.append(f"  [View Book]({book.get('url')})")
            output.append("")
    
    if "wikidata" in results:
        wikidata = results["wikidata"]
        if isinstance(wikidata, list) and wikidata and "error" not in wikidata[0]:
            output.append("### 🗃️ Wikidata Entities")
            for entity in wikidata[:3]:
                output.append(f"- **{entity.get('label', 'N/A')}**: {entity.get('description', 'No description')}")
                if entity.get('url'):
                    output.append(f"  [View]({entity.get('url')})")
            output.append("")
    
    if "weather" in results:
        weather = results["weather"]
        if isinstance(weather, dict) and "error" not in weather:
            output.append("### 🌤️ Weather")
            output.append(f"- Location: {weather.get('location', 'N/A')}")
            output.append(f"- Temperature: {weather.get('temperature_c', 'N/A')}°C / {weather.get('temperature_f', 'N/A')}°F")
            output.append(f"- Condition: {weather.get('condition', 'N/A')}")
            output.append(f"- Humidity: {weather.get('humidity', 'N/A')}%")
            output.append("")
    
    if "air_quality" in results:
        aq = results["air_quality"]
        if isinstance(aq, dict) and "error" not in aq and aq.get("data"):
            output.append("### 🌬️ Air Quality")
            output.append(f"- City: {aq.get('city', 'N/A')}")
            for loc in aq.get("data", [])[:2]:
                output.append(f"- Location: {loc.get('location', 'N/A')}")
                for m in loc.get("measurements", [])[:3]:
                    output.append(f"  - {m.get('parameter', 'N/A')}: {m.get('value', 'N/A')} {m.get('unit', '')}")
            output.append("")
    
    if "geocoding" in results:
        geo = results["geocoding"]
        if isinstance(geo, dict) and "error" not in geo:
            output.append("### 📍 Location Info")
            output.append(f"- {geo.get('display_name', 'N/A')}")
            output.append(f"- Coordinates: {geo.get('latitude', 'N/A')}, {geo.get('longitude', 'N/A')}")
            if geo.get('osm_url'):
                output.append(f"- [View on Map]({geo.get('osm_url')})")
            output.append("")
    
    if "news" in results:
        news_data = results["news"]
        if isinstance(news_data, list) and news_data and "error" not in news_data[0] and "message" not in news_data[0]:
            output.append("### 📰 News")
            for article in news_data[:3]:
                output.append(f"- **{article.get('title', 'N/A')}**")
                if article.get('source'):
                    output.append(f"  Source: {article.get('source')} | {article.get('date', '')}")
                output.append(f"  {article.get('body', '')[:150]}...")
                if article.get('url'):
                    output.append(f"  [Read Article]({article.get('url')})")
            output.append("")
    
    if "dictionary" in results:
        dictionary = results["dictionary"]
        if isinstance(dictionary, dict) and "error" not in dictionary and "message" not in dictionary:
            output.append(f"### 📖 Dictionary: {dictionary.get('word', 'N/A')}")
            phonetics = dictionary.get('phonetics', [])
            if phonetics:
                output.append(f"*Pronunciation: {', '.join(phonetics)}*")
            for meaning in dictionary.get('meanings', [])[:2]:
                output.append(f"**{meaning.get('part_of_speech', '')}**")
                for defn in meaning.get('definitions', [])[:2]:
                    output.append(f"- {defn.get('definition', '')}")
                    if defn.get('example'):
                        output.append(f"  *Example: \"{defn.get('example')}\"*")
            output.append("")
    
    if "country" in results:
        country = results["country"]
        if isinstance(country, dict) and "error" not in country and "message" not in country:
            output.append(f"### 🌍 Country: {country.get('name', 'N/A')} {country.get('flag_emoji', '')}")
            output.append(f"- **Official Name**: {country.get('official_name', 'N/A')}")
            output.append(f"- **Capital**: {country.get('capital', 'N/A')}")
            output.append(f"- **Region**: {country.get('region', 'N/A')} / {country.get('subregion', 'N/A')}")
            output.append(f"- **Population**: {country.get('population', 'N/A'):,}" if isinstance(country.get('population'), int) else f"- **Population**: {country.get('population', 'N/A')}")
            languages = country.get('languages', [])
            if languages:
                output.append(f"- **Languages**: {', '.join(languages[:3])}")
            currencies = country.get('currencies', [])
            if currencies:
                output.append(f"- **Currencies**: {', '.join(currencies[:2])}")
            if country.get('map_url'):
                output.append(f"- [View on Map]({country.get('map_url')})")
            output.append("")
    
    if "quotes" in results:
        quotes_data = results["quotes"]
        if isinstance(quotes_data, list) and quotes_data and "error" not in quotes_data[0] and "message" not in quotes_data[0]:
            output.append("### 💬 Quotes")
            for quote in quotes_data[:3]:
                output.append(f"> \"{quote.get('content', '')}\"")
                output.append(f"> — *{quote.get('author', 'Unknown')}*")
                output.append("")
    
    if "github" in results:
        github_data = results["github"]
        if isinstance(github_data, list) and github_data and "error" not in github_data[0] and "message" not in github_data[0]:
            output.append("### 💻 GitHub Repositories")
            for repo in github_data[:3]:
                output.append(f"- **{repo.get('name', 'N/A')}** ⭐ {repo.get('stars', 0):,}")
                output.append(f"  {repo.get('description', 'No description')[:100]}...")
                output.append(f"  Language: {repo.get('language', 'N/A')} | Forks: {repo.get('forks', 0):,}")
                if repo.get('url'):
                    output.append(f"  [View Repository]({repo.get('url')})")
            output.append("")
    
    if "stackoverflow" in results:
        so_data = results["stackoverflow"]
        if isinstance(so_data, list) and so_data and "error" not in so_data[0] and "message" not in so_data[0]:
            output.append("### 🔧 Stack Overflow")
            for q in so_data[:3]:
                answered_emoji = "✅" if q.get('is_answered') else "❓"
                output.append(f"- {answered_emoji} **{q.get('title', 'N/A')}**")
                output.append(f"  Score: {q.get('score', 0)} | Answers: {q.get('answer_count', 0)} | Views: {q.get('view_count', 0):,}")
                tags = q.get('tags', [])[:3]
                if tags:
                    output.append(f"  Tags: {', '.join(tags)}")
                if q.get('url'):
                    output.append(f"  [View Question]({q.get('url')})")
            output.append("")
    
    return "\n".join(output)

# ========== NEW AI FUNCTIONS ==========
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
    try:
        from ctransformers import AutoModelForCausalLM
    except ImportError:
        st.error("ctransformers not installed. Please install with: pip install ctransformers==0.2.27")
        return None
    
    if not MODEL_PATH.exists():
        with st.spinner("Downloading TinyLLaMA model (~637 MB)..."):
            try:
                download_model()
            except Exception as e:
                st.error(f"Failed to download model: {str(e)}")
                return None
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            model_file=MODEL_PATH.name,
            model_type="llama",
            context_length=2048,
            gpu_layers=0
        )
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def format_results_for_ai(query: str, results: dict) -> str:
    """Format search results into a concise text for AI processing."""
    context_parts = []
    
    context_parts.append(f"User query: {query}")
    context_parts.append("\n=== SEARCH RESULTS SUMMARY ===\n")
    
    if "duckduckgo_instant" in results:
        instant = results["duckduckgo_instant"]
        if isinstance(instant, dict) and instant.get("answer"):
            context_parts.append(f"Quick Answer: {instant['answer']}")
    
    if "wikipedia" in results:
        wiki = results["wikipedia"]
        if isinstance(wiki, dict) and wiki.get("exists"):
            context_parts.append(f"Wikipedia Summary: {wiki.get('summary', '')[:300]}")
    
    if "duckduckgo" in results:
        ddg = results["duckduckgo"]
        if isinstance(ddg, list) and ddg and "error" not in ddg[0]:
            context_parts.append("Top Web Results:")
            for i, item in enumerate(ddg[:3], 1):
                context_parts.append(f"  {i}. {item.get('title', '')} - {item.get('body', '')[:150]}")
    
    if "arxiv" in results:
        arxiv_data = results["arxiv"]
        if isinstance(arxiv_data, list) and arxiv_data and "error" not in arxiv_data[0]:
            context_parts.append("Scientific Papers:")
            for paper in arxiv_data[:2]:
                authors = ", ".join(paper.get("authors", [])[:2])
                context_parts.append(f"  - {paper.get('title', '')} by {authors}")
    
    if "pubmed" in results:
        pubmed_data = results["pubmed"]
        if isinstance(pubmed_data, list) and pubmed_data and "error" not in pubmed_data[0]:
            context_parts.append("Medical Research:")
            for article in pubmed_data[:2]:
                context_parts.append(f"  - {article.get('title', '')}")
    
    if "books" in results:
        books_data = results["books"]
        if isinstance(books_data, list) and books_data and "error" not in books_data[0]:
            context_parts.append("Books:")
            for book in books_data[:2]:
                authors = ", ".join(book.get("authors", [])[:2])
                context_parts.append(f"  - {book.get('title', '')} by {authors}")
    
    if "weather" in results:
        weather = results["weather"]
        if isinstance(weather, dict) and "error" not in weather:
            context_parts.append(f"Weather: {weather.get('location', '')} - {weather.get('temperature_c', '')}°C, {weather.get('condition', '')}")
    
    if "country" in results:
        country = results["country"]
        if isinstance(country, dict) and "error" not in country:
            context_parts.append(f"Country Info: {country.get('name', '')}, Capital: {country.get('capital', '')}")
    
    if "news" in results:
        news_data = results["news"]
        if isinstance(news_data, list) and news_data and "error" not in news_data[0]:
            context_parts.append("Recent News:")
            for article in news_data[:2]:
                context_parts.append(f"  - {article.get('title', '')}")
    
    context_parts.append("\n=== END OF SEARCH RESULTS ===\n")
    context_parts.append(f"Based on the search results above, please provide a comprehensive answer to the user's query: '{query}'")
    
    return "\n".join(context_parts)

def generate_ai_response(model, search_context, query, system_prompt="", max_tokens=256, temperature=0.7):
    """Generate AI response based on search results."""
    from ctransformers import AutoModelForCausalLM
    
    # Create a special prompt for the AI with search results
    ai_prompt = f"<|system|>\n{system_prompt}\n\nYou have access to search results from multiple sources. Use these results to answer the user's question. If the search results don't contain enough information, say so and provide general knowledge based on your training.</s>\n"
    ai_prompt += f"<|user|>\nHere are the search results for '{query}':\n\n{search_context}\n\nBased on these results, please answer: {query}</s>\n"
    ai_prompt += "<|assistant|>\n"
    
    response = model(
        ai_prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        stop=["</s>", "<|user|>", "<|assistant|>", "<|system|>"]
    )
    
    return response.strip()

# ========== Load Model ==========
model = None
try:
    with st.spinner("Loading AI model..."):
        model = load_model()
        if model:
            st.session_state.model_loaded = True
except Exception as e:
    st.warning(f"AI model not available: {str(e)}. Search functionality will work, but AI analysis won't be available.")

# ========== Main Chat Interface ==========
if prompt := st.chat_input("Search anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        # Search all sources first
        with st.spinner("🔎 Searching all 16 sources simultaneously..."):
            search_results = search_all_sources(prompt)
        
        # Create tabs
        if model and st.session_state.model_loaded:
            tab1, tab2, tab3 = st.tabs(["🤖 AI Analysis", "📊 Search Results", "📈 Raw Data"])
        else:
            tab1, tab2 = st.tabs(["📊 Search Results", "📈 Raw Data"])
        
        # AI Analysis Tab
        if model and st.session_state.model_loaded:
            with tab1:
                with st.spinner("🤖 AI is analyzing search results..."):
                    # Format search results for AI
                    search_context = format_results_for_ai(prompt, search_results)
                    
                    # Generate AI response
                    ai_response = generate_ai_response(
                        model,
                        search_context,
                        prompt,
                        system_prompt=st.session_state.system_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    
                    st.markdown("### AI Analysis Based on Search Results")
                    st.markdown(ai_response)
        
        # Search Results Tab (original formatting)
        with tab2 if model else tab1:
            response = format_results(prompt, search_results)
            st.markdown(response)
        
        # Raw Data Tab
        with tab3 if model else tab2:
            st.markdown("### 📊 View Raw Data")
            for source, data in search_results.items():
                with st.expander(f"📌 {source.replace('_', ' ').title()}"):
                    st.json(data)
        
        # Store combined response
        if model and st.session_state.model_loaded:
            combined_response = f"## 🤖 AI Analysis\n{ai_response}\n\n---\n\n## 📊 Search Results Summary\n{response[:500]}..."
        else:
            combined_response = response
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": combined_response
    })
