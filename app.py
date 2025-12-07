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
from datetime import datetime

# MODEL CONFIG - KEEP IT SIMPLE!
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/tfdtfd/khisbagis23/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=true"

st.set_page_config(
    page_title="AI Search Assistant PRO",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI-Powered Search Assistant")
st.markdown("**Search 16 sources → Get synthesized AI response**")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = None

# ============================
# YOUR ORIGINAL SCRAPPER FUNCTIONS - UNCHANGED!
# ============================

def search_all_sources(query: str) -> dict:
    """Search ALL sources simultaneously - YOUR ORIGINAL CODE"""
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

def format_raw_results(query: str, results: dict) -> str:
    """Format raw search results for the AI to read - YOUR ORIGINAL FORMATTING"""
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

# ============================
# MINIMAL MODEL FUNCTIONS
# ============================

def load_ai_model():
    """Simple model loader - only loads if user wants AI synthesis"""
    try:
        from ctransformers import AutoModelForCausalLM
        
        # Download if needed
        MODEL_DIR.mkdir(exist_ok=True)
        if not MODEL_PATH.exists():
            with st.spinner("Downloading AI model (one time only)..."):
                response = requests.get(MODEL_URL, stream=True, timeout=60)
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
            gpu_layers=0
        )
        return model
    except:
        return None

def generate_ai_response(query: str, search_results_text: str, model) -> str:
    """Let the AI synthesize the search results into a better answer"""
    prompt = f"""<|system|>
You are a helpful AI assistant. The user has asked: "{query}"

Below are search results from multiple sources (Wikipedia, news, scientific papers, etc.). 
Please synthesize this information into a coherent, well-structured answer.

Rules:
1. Use the search results below as your primary source
2. Cite which source you're using (e.g., "According to Wikipedia...")
3. If search results conflict, mention this
4. If information is missing, say so
5. Keep the answer clear and organized
6. Add brief analysis/synthesis

SEARCH RESULTS:
{search_results_text[:3000]}  # Limit to avoid overflow

Now provide a synthesized answer:</s>

<|user|>
{query}</s>

<|assistant|>
"""
    
    try:
        response = model(
            prompt,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            stop=["</s>", "<|user|>", "<|assistant|>"]
        )
        return response.strip()
    except:
        return None

# ============================
# STREAMLIT UI
# ============================

with st.sidebar:
    st.header("⚙️ Settings")
    
    # AI toggle
    use_ai = st.toggle("🤖 Enable AI Synthesis", value=True, 
                      help="Let AI synthesize search results into better answers")
    
    if use_ai and st.session_state.model is None:
        with st.spinner("Loading AI model..."):
            model = load_ai_model()
            if model:
                st.session_state.model = model
                st.success("AI loaded!")
            else:
                st.error("Could not load AI model")
                use_ai = False
    
    st.divider()
    
    st.header("📊 16 Search Sources")
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
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat interface
if prompt := st.chat_input("Search anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        # Step 1: Run YOUR ORIGINAL SCRAPPER
        with st.spinner("🔍 Searching across 16 sources..."):
            search_results = search_all_sources(prompt)
        
        # Step 2: Format raw results (YOUR ORIGINAL FORMAT)
        raw_results_text = format_raw_results(prompt, search_results)
        
        # Step 3: Decide output format
        if use_ai and st.session_state.model:
            # Show AI is thinking
            st.caption("🤖 AI is synthesizing search results...")
            
            # Let AI synthesize the results
            ai_response = generate_ai_response(prompt, raw_results_text, st.session_state.model)
            
            if ai_response:
                # Display AI-synthesized response
                st.markdown(ai_response)
                
                # Show raw results in expander
                with st.expander("📊 View Original Search Results", expanded=False):
                    st.markdown(raw_results_text)
            else:
                # Fallback to raw results if AI fails
                st.markdown(raw_results_text)
                st.caption("⚠️ AI synthesis failed, showing raw results")
        else:
            # Just show raw results (no AI)
            st.markdown(raw_results_text)
        
        # Show raw data in expander
        with st.expander("🔍 View Raw Search Data", expanded=False):
            for source, data in search_results.items():
                st.subheader(f"📌 {source.replace('_', ' ').title()}")
                st.json(data)
    
    # Store in history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": ai_response if (use_ai and ai_response) else raw_results_text
    })

# Add info about the app
st.sidebar.divider()
st.sidebar.caption("**Workflow:**")
st.sidebar.caption("1. Search 16 sources in parallel")
st.sidebar.caption("2. Format results")
st.sidebar.caption("3. AI synthesizes (if enabled)")
st.sidebar.caption("4. Show final answer + raw data")
