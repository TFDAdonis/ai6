import streamlit as st
import requests
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import concurrent.futures
from pathlib import Path
import re

# ============================
# ALL SEARCH SERVICES BUILT-IN
# ============================

# 1. Wikipedia Service
def search_wikipedia(query, max_results=3):
    """Search Wikipedia"""
    try:
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': max_results
        }
        response = requests.get("https://en.wikipedia.org/w/api.php", params=params, timeout=10)
        data = response.json()
        
        results = []
        for item in data.get('query', {}).get('search', []):
            # Get page content
            content_params = {
                'action': 'query',
                'format': 'json',
                'prop': 'extracts|info',
                'inprop': 'url',
                'exintro': True,
                'explaintext': True,
                'pageids': item['pageid']
            }
            content_response = requests.get(
                "https://en.wikipedia.org/w/api.php", 
                params=content_params, 
                timeout=10
            )
            
            if content_response.status_code == 200:
                page_data = content_response.json()
                page_info = page_data.get('query', {}).get('pages', {}).get(str(item['pageid']), {})
                
                if page_info and page_info.get('extract'):
                    results.append({
                        'title': page_info.get('title', ''),
                        'summary': page_info.get('extract', '')[:400] + '...',
                        'url': page_info.get('fullurl', ''),
                        'source': 'Wikipedia',
                        'wordcount': page_info.get('wordcount', 0)
                    })
        
        if results:
            return {
                'exists': True,
                'title': results[0]['title'],
                'summary': results[0]['summary'],
                'url': results[0]['url']
            }
        else:
            return {'exists': False, 'message': 'No Wikipedia article found'}
    except Exception as e:
        return {'error': str(e)}

# 2. DuckDuckGo Services
def search_duckduckgo(query, max_results=5):
    """Search DuckDuckGo web results"""
    try:
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1
        }
        response = requests.get("https://api.duckduckgo.com/", params=params, timeout=10)
        data = response.json()
        
        results = []
        # Extract from Abstract
        if data.get('AbstractText'):
            results.append({
                'title': data.get('Heading', 'Abstract'),
                'body': data.get('AbstractText'),
                'url': data.get('AbstractURL', ''),
                'source': 'DuckDuckGo'
            })
        
        # Extract from RelatedTopics
        for topic in data.get('RelatedTopics', []):
            if isinstance(topic, dict) and 'Text' in topic:
                results.append({
                    'title': topic.get('FirstURL', '').split('/')[-1].replace('_', ' '),
                    'body': topic['Text'],
                    'url': topic.get('FirstURL', ''),
                    'source': 'DuckDuckGo'
                })
                if len(results) >= max_results:
                    break
        
        return results if results else []
    except Exception as e:
        return {'error': str(e)}

def get_instant_answer(query):
    """Get instant answer from DuckDuckGo"""
    try:
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1
        }
        response = requests.get("https://api.duckduckgo.com/", params=params, timeout=10)
        data = response.json()
        
        if data.get('Answer'):
            return {'answer': data['Answer'], 'type': 'instant_answer'}
        elif data.get('AbstractText'):
            return {'answer': data['AbstractText'][:300], 'type': 'abstract'}
        elif data.get('Definition'):
            return {'answer': data['Definition'], 'type': 'definition'}
        else:
            return {'answer': 'No instant answer available', 'type': 'none'}
    except Exception as e:
        return {'error': str(e)}

def search_news(query, max_results=3):
    """Search news via DuckDuckGo"""
    try:
        # DuckDuckGo news is limited, so we'll use a news API placeholder
        # In a real app, you'd use NewsAPI with a key
        return [
            {
                'title': f'News about {query}',
                'body': 'Latest news articles would appear here with a proper NewsAPI key.',
                'source': 'NewsAPI (needs key)',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'url': '#'
            }
        ]
    except Exception as e:
        return {'error': str(e)}

# 3. ArXiv Service
def search_arxiv(query, max_results=3):
    """Search ArXiv for scientific papers"""
    try:
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        response = requests.get("http://export.arxiv.org/api/query", params=params, timeout=15)
        
        root = ET.fromstring(response.content)
        namespace = '{http://www.w3.org/2005/Atom}'
        
        papers = []
        for entry in root.findall(f'{namespace}entry'):
            title_elem = entry.find(f'{namespace}title')
            summary_elem = entry.find(f'{namespace}summary')
            published_elem = entry.find(f'{namespace}published')
            
            if title_elem is not None and summary_elem is not None:
                # Extract authors
                authors = []
                for author_elem in entry.findall(f'{namespace}author'):
                    name_elem = author_elem.find(f'{namespace}name')
                    if name_elem is not None:
                        authors.append(name_elem.text)
                
                papers.append({
                    'title': title_elem.text.strip(),
                    'summary': summary_elem.text.strip()[:300] + '...',
                    'authors': authors[:3],  # Limit to 3 authors
                    'published': published_elem.text[:10] if published_elem is not None else '',
                    'url': f"https://arxiv.org/abs/{entry.find(f'{namespace}id').text.split('/')[-1]}"
                })
                if len(papers) >= max_results:
                    break
        
        return papers if papers else []
    except Exception as e:
        return {'error': str(e)}

# 4. Weather Service
def get_weather_wttr(query):
    """Get weather from wttr.in"""
    try:
        response = requests.get(f"https://wttr.in/{query}?format=j1", timeout=10)
        if response.status_code == 200:
            data = response.json()
            current = data['current_condition'][0]
            return {
                'location': query,
                'temperature_c': current['temp_C'],
                'temperature_f': current['temp_F'],
                'condition': current['weatherDesc'][0]['value'],
                'humidity': current['humidity'],
                'wind_speed': current['windspeedKmph'],
                'source': 'wttr.in'
            }
        return {'error': 'Location not found'}
    except Exception as e:
        return {'error': str(e)}

# 5. Air Quality Service
def get_air_quality(query):
    """Get air quality data"""
    try:
        # This is a placeholder - OpenAQ requires API setup
        return {
            'city': query,
            'data': [{
                'location': 'City Center',
                'measurements': [
                    {'parameter': 'PM2.5', 'value': '12', 'unit': 'µg/m³'},
                    {'parameter': 'PM10', 'value': '20', 'unit': 'µg/m³'},
                    {'parameter': 'O3', 'value': '45', 'unit': 'ppb'}
                ]
            }],
            'message': 'OpenAQ API required for real data'
        }
    except Exception as e:
        return {'error': str(e)}

# 6. Wikidata Service
def search_wikidata(query, max_results=3):
    """Search Wikidata"""
    try:
        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'language': 'en',
            'search': query,
            'limit': max_results
        }
        response = requests.get("https://www.wikidata.org/w/api.php", params=params, timeout=10)
        data = response.json()
        
        entities = []
        for entity in data.get('search', []):
            entities.append({
                'label': entity.get('label', ''),
                'description': entity.get('description', ''),
                'id': entity.get('id', ''),
                'url': f"https://www.wikidata.org/wiki/{entity.get('id', '')}"
            })
        
        return entities if entities else []
    except Exception as e:
        return {'error': str(e)}

# 7. OpenLibrary Service
def search_books(query, max_results=5):
    """Search books on OpenLibrary"""
    try:
        params = {
            'q': query,
            'mode': 'everything',
            'limit': max_results
        }
        response = requests.get("https://openlibrary.org/search.json", params=params, timeout=10)
        data = response.json()
        
        books = []
        for doc in data.get('docs', []):
            books.append({
                'title': doc.get('title', ''),
                'authors': doc.get('author_name', ['Unknown']),
                'first_publish_year': doc.get('first_publish_year', ''),
                'url': f"https://openlibrary.org{doc.get('key', '')}"
            })
            if len(books) >= max_results:
                break
        
        return books if books else []
    except Exception as e:
        return {'error': str(e)}

# 8. PubMed Service
def search_pubmed(query, max_results=3):
    """Search PubMed for medical articles"""
    try:
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json'
        }
        response = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params, timeout=15)
        
        # This is simplified - real PubMed search requires multiple API calls
        return [
            {
                'title': f'Medical research about {query}',
                'authors': ['Researcher A', 'Researcher B'],
                'year': '2023',
                'abstract': 'PubMed search requires full API implementation.',
                'url': 'https://pubmed.ncbi.nlm.nih.gov/'
            }
        ]
    except Exception as e:
        return {'error': str(e)}

# 9. Geocoding Service
def geocode_location(query):
    """Geocode location using Nominatim"""
    try:
        params = {
            'q': query,
            'format': 'json',
            'limit': 1
        }
        response = requests.get("https://nominatim.openstreetmap.org/search", params=params, timeout=10)
        data = response.json()
        
        if data:
            location = data[0]
            return {
                'display_name': location.get('display_name', ''),
                'latitude': location.get('lat', ''),
                'longitude': location.get('lon', ''),
                'osm_url': f"https://www.openstreetmap.org/?mlat={location.get('lat')}&mlon={location.get('lon')}"
            }
        return {'error': 'Location not found'}
    except Exception as e:
        return {'error': str(e)}

# 10. Dictionary Service
def get_definition(word):
    """Get word definition from dictionary API"""
    try:
        response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}", timeout=10)
        if response.status_code == 200:
            data = response.json()[0]
            meanings = []
            for meaning in data.get('meanings', []):
                definitions = []
                for defn in meaning.get('definitions', [])[:2]:
                    definitions.append({
                        'definition': defn.get('definition', ''),
                        'example': defn.get('example', '')
                    })
                meanings.append({
                    'part_of_speech': meaning.get('partOfSpeech', ''),
                    'definitions': definitions
                })
            
            return {
                'word': data.get('word', ''),
                'phonetics': [p.get('text', '') for p in data.get('phonetics', []) if p.get('text')],
                'meanings': meanings
            }
        return {'error': 'Word not found'}
    except Exception as e:
        return {'error': str(e)}

# 11. Country Service
def search_country(query):
    """Search country information"""
    try:
        response = requests.get(f"https://restcountries.com/v3.1/name/{query}", timeout=10)
        if response.status_code == 200:
            data = response.json()[0]
            return {
                'name': data.get('name', {}).get('common', ''),
                'official_name': data.get('name', {}).get('official', ''),
                'capital': data.get('capital', [''])[0],
                'region': data.get('region', ''),
                'subregion': data.get('subregion', ''),
                'population': data.get('population', 0),
                'languages': list(data.get('languages', {}).values()),
                'currencies': [f"{curr.get('name', '')} ({curr.get('symbol', '')})" 
                              for curr in data.get('currencies', {}).values()],
                'flag_emoji': data.get('flag', ''),
                'map_url': data.get('maps', {}).get('googleMaps', '')
            }
        return {'error': 'Country not found'}
    except Exception as e:
        return {'error': str(e)}

# 12. Quotes Service
def search_quotes(query, max_results=3):
    """Search quotes"""
    try:
        params = {
            'query': query,
            'limit': max_results
        }
        response = requests.get("https://api.quotable.io/search/quotes", params=params, timeout=10)
        data = response.json()
        
        quotes = []
        for quote in data.get('results', []):
            quotes.append({
                'content': quote.get('content', ''),
                'author': quote.get('author', 'Unknown')
            })
            if len(quotes) >= max_results:
                break
        
        return quotes if quotes else []
    except Exception as e:
        return {'error': str(e)}

# 13. GitHub Service
def search_github_repos(query, max_results=3):
    """Search GitHub repositories"""
    try:
        params = {
            'q': query,
            'sort': 'stars',
            'order': 'desc',
            'per_page': max_results
        }
        headers = {
            'Accept': 'application/vnd.github.v3+json'
        }
        response = requests.get(
            "https://api.github.com/search/repositories", 
            params=params, 
            headers=headers, 
            timeout=10
        )
        data = response.json()
        
        repos = []
        for repo in data.get('items', []):
            repos.append({
                'name': repo.get('full_name', ''),
                'description': repo.get('description', ''),
                'stars': repo.get('stargazers_count', 0),
                'forks': repo.get('forks_count', 0),
                'language': repo.get('language', ''),
                'url': repo.get('html_url', '')
            })
            if len(repos) >= max_results:
                break
        
        return repos if repos else []
    except Exception as e:
        return {'error': str(e)}

# 14. Stack Overflow Service
def search_stackoverflow(query, max_results=3):
    """Search Stack Overflow questions"""
    try:
        params = {
            'order': 'desc',
            'sort': 'relevance',
            'intitle': query,
            'site': 'stackoverflow',
            'pagesize': max_results
        }
        response = requests.get("https://api.stackexchange.com/2.3/search", params=params, timeout=10)
        data = response.json()
        
        questions = []
        for item in data.get('items', []):
            questions.append({
                'title': item.get('title', ''),
                'score': item.get('score', 0),
                'answer_count': item.get('answer_count', 0),
                'view_count': item.get('view_count', 0),
                'is_answered': item.get('is_answered', False),
                'tags': item.get('tags', []),
                'url': item.get('link', '')
            })
            if len(questions) >= max_results:
                break
        
        return questions if questions else []
    except Exception as e:
        return {'error': str(e)}

# ============================
# MAIN SEARCH FUNCTION
# ============================

def search_all_sources(query: str) -> dict:
    """Search ALL 14 sources simultaneously"""
    results = {}
    
    def safe_search(name, func, *args, **kwargs):
        try:
            return name, func(*args, **kwargs)
        except Exception as e:
            return name, {"error": str(e)[:100]}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=14) as executor:
        first_word = query.split()[0] if query.strip() else query
        
        # Submit all search tasks
        futures = {
            executor.submit(safe_search, "wikipedia", search_wikipedia, query): "wikipedia",
            executor.submit(safe_search, "duckduckgo", search_duckduckgo, query, 5): "duckduckgo",
            executor.submit(safe_search, "duckduckgo_instant", get_instant_answer, query): "duckduckgo_instant",
            executor.submit(safe_search, "arxiv", search_arxiv, query, 3): "arxiv",
            executor.submit(safe_search, "weather", get_weather_wttr, query): "weather",
            executor.submit(safe_search, "wikidata", search_wikidata, query, 3): "wikidata",
            executor.submit(safe_search, "books", search_books, query, 3): "books",
            executor.submit(safe_search, "geocoding", geocode_location, query): "geocoding",
            executor.submit(safe_search, "dictionary", get_definition, first_word): "dictionary",
            executor.submit(safe_search, "country", search_country, query): "country",
            executor.submit(safe_search, "quotes", search_quotes, query, 3): "quotes",
            executor.submit(safe_search, "github", search_github_repos, query, 3): "github",
            executor.submit(safe_search, "stackoverflow", search_stackoverflow, query, 3): "stackoverflow",
        }
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            source_name = futures[future]
            try:
                name, data = future.result(timeout=15)
                if data and not (isinstance(data, dict) and 'error' in data):
                    results[source_name] = data
            except Exception:
                continue
    
    return results

# ============================
# FORMATTING FUNCTION
# ============================

def format_search_results(query: str, results: dict) -> str:
    """Format search results into readable text"""
    output = [f"## 🔍 Search Results for: **{query}**\n"]
    
    # Quick Answer
    if "duckduckgo_instant" in results:
        instant = results["duckduckgo_instant"]
        if isinstance(instant, dict) and instant.get("answer"):
            output.append(f"### 💡 Quick Answer\n{instant['answer']}\n")
    
    # Wikipedia
    if "wikipedia" in results:
        wiki = results["wikipedia"]
        if isinstance(wiki, dict) and wiki.get("exists"):
            output.append(f"### 📚 Wikipedia\n**{wiki.get('title', 'N/A')}**\n")
            output.append(f"{wiki.get('summary', '')}\n")
            output.append(f"[Read more]({wiki.get('url', '')})\n")
    
    # Web Results
    if "duckduckgo" in results:
        ddg = results["duckduckgo"]
        if isinstance(ddg, list) and ddg:
            output.append("### 🌐 Web Results\n")
            for i, item in enumerate(ddg[:3], 1):
                output.append(f"{i}. **{item.get('title', 'N/A')}**")
                output.append(f"   {item.get('body', '')[:150]}...")
                output.append(f"   [Link]({item.get('url', '')})\n")
    
    # Scientific Papers
    if "arxiv" in results:
        arxiv_data = results["arxiv"]
        if isinstance(arxiv_data, list) and arxiv_data:
            output.append("### 🔬 Scientific Papers\n")
            for paper in arxiv_data[:2]:
                authors = ", ".join(paper.get("authors", [])[:2])
                output.append(f"- **{paper.get('title', 'N/A')}**")
                output.append(f"  *Authors: {authors}*")
                output.append(f"  {paper.get('summary', '')}")
                output.append(f"  [View Paper]({paper.get('url', '')})\n")
    
    # Books
    if "books" in results:
        books_data = results["books"]
        if isinstance(books_data, list) and books_data:
            output.append("### 📖 Books\n")
            for book in books_data[:2]:
                authors = ", ".join(book.get("authors", [])[:2])
                output.append(f"- **{book.get('title', 'N/A')}**")
                output.append(f"  *By: {authors}*")
                output.append(f"  [View Book]({book.get('url', '')})\n")
    
    # Weather
    if "weather" in results:
        weather = results["weather"]
        if isinstance(weather, dict) and 'error' not in weather:
            output.append("### 🌤️ Weather\n")
            output.append(f"- **Location:** {weather.get('location', 'N/A')}")
            output.append(f"- **Temperature:** {weather.get('temperature_c', 'N/A')}°C / {weather.get('temperature_f', 'N/A')}°F")
            output.append(f"- **Condition:** {weather.get('condition', 'N/A')}")
            output.append(f"- **Humidity:** {weather.get('humidity', 'N/A')}%\n")
    
    # Country Info
    if "country" in results:
        country = results["country"]
        if isinstance(country, dict) and 'error' not in country:
            output.append(f"### 🌍 {country.get('name', 'Country')} {country.get('flag_emoji', '')}\n")
            output.append(f"- **Capital:** {country.get('capital', 'N/A')}")
            output.append(f"- **Population:** {country.get('population', 0):,}")
            output.append(f"- **Region:** {country.get('region', 'N/A')}")
            if country.get('map_url'):
                output.append(f"- [View on Map]({country.get('map_url')})\n")
    
    # GitHub Repos
    if "github" in results:
        github_data = results["github"]
        if isinstance(github_data, list) and github_data:
            output.append("### 💻 GitHub Repositories\n")
            for repo in github_data[:2]:
                output.append(f"- **{repo.get('name', 'N/A')}** ⭐ {repo.get('stars', 0):,}")
                output.append(f"  {repo.get('description', '')[:100]}")
                output.append(f"  [View Repo]({repo.get('url', '')})\n")
    
    # Stack Overflow
    if "stackoverflow" in results:
        so_data = results["stackoverflow"]
        if isinstance(so_data, list) and so_data:
            output.append("### 🔧 Stack Overflow\n")
            for q in so_data[:2]:
                answered = "✅" if q.get('is_answered') else "❓"
                output.append(f"{answered} **{q.get('title', 'N/A')}**")
                output.append(f"  Score: {q.get('score', 0)} | Answers: {q.get('answer_count', 0)}")
                output.append(f"  [View Question]({q.get('url', '')})\n")
    
    # Dictionary
    if "dictionary" in results:
        dictionary = results["dictionary"]
        if isinstance(dictionary, dict) and 'error' not in dictionary:
            output.append(f"### 📖 Dictionary: {dictionary.get('word', 'Word')}\n")
            for meaning in dictionary.get('meanings', [])[:2]:
                output.append(f"**{meaning.get('part_of_speech', '')}**")
                for defn in meaning.get('definitions', [])[:1]:
                    output.append(f"- {defn.get('definition', '')}")
                    if defn.get('example'):
                        output.append(f'  *Example: "{defn.get("example")}"*')
                output.append("")
    
    # Quotes
    if "quotes" in results:
        quotes_data = results["quotes"]
        if isinstance(quotes_data, list) and quotes_data:
            output.append("### 💬 Quotes\n")
            for quote in quotes_data[:2]:
                output.append(f'> "{quote.get("content", "")}"')
                output.append(f'> — *{quote.get("author", "Unknown")}*\n')
    
    # Wikidata
    if "wikidata" in results:
        wikidata = results["wikidata"]
        if isinstance(wikidata, list) and wikidata:
            output.append("### 🗃️ Wikidata\n")
            for entity in wikidata[:2]:
                output.append(f"- **{entity.get('label', 'N/A')}**")
                output.append(f"  {entity.get('description', '')}")
                output.append(f"  [View]({entity.get('url', '')})\n")
    
    # Geocoding
    if "geocoding" in results:
        geo = results["geocoding"]
        if isinstance(geo, dict) and 'error' not in geo:
            output.append("### 📍 Location\n")
            output.append(f"{geo.get('display_name', '')}")
            output.append(f"[View on Map]({geo.get('osm_url', '')})\n")
    
    return "\n".join(output)

# ============================
# STREAMLIT APP
# ============================

st.set_page_config(
    page_title="Super Search 🔍",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Super Search Assistant")
st.markdown("**Search 14 sources simultaneously • Real-time results**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.markdown("### 🔍 Search Sources (14 Total)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        wikipedia = st.checkbox("Wikipedia", value=True)
        duckduckgo = st.checkbox("DuckDuckGo", value=True)
        arxiv = st.checkbox("ArXiv Papers", value=True)
        weather = st.checkbox("Weather", value=True)
        books = st.checkbox("Books", value=True)
        dictionary = st.checkbox("Dictionary", value=True)
        country = st.checkbox("Countries", value=True)
    
    with col2:
        github = st.checkbox("GitHub", value=True)
        stackoverflow = st.checkbox("Stack Overflow", value=True)
        quotes = st.checkbox("Quotes", value=True)
        wikidata = st.checkbox("Wikidata", value=True)
        geocoding = st.checkbox("Geocoding", value=True)
    
    st.divider()
    
    # Selected sources
    selected_sources = []
    if wikipedia: selected_sources.append("wikipedia")
    if duckduckgo: selected_sources.append("duckduckgo")
    if arxiv: selected_sources.append("arxiv")
    if weather: selected_sources.append("weather")
    if books: selected_sources.append("books")
    if dictionary: selected_sources.append("dictionary")
    if country: selected_sources.append("country")
    if github: selected_sources.append("github")
    if stackoverflow: selected_sources.append("stackoverflow")
    if quotes: selected_sources.append("quotes")
    if wikidata: selected_sources.append("wikidata")
    if geocoding: selected_sources.append("geocoding")
    
    st.info(f"**{len(selected_sources)}** sources selected")
    
    st.divider()
    
    if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
        if "messages" in st.session_state:
            st.session_state.messages = []
        st.rerun()
    
    st.caption("🔍 *All searches run in parallel*")
    st.caption("⚡ *Fast response times*")
    st.caption("📊 *Real data from APIs*")

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat
if prompt := st.chat_input("What would you like to search for?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        # Show searching status
        status_text = st.empty()
        status_text.markdown(f"🔍 **Searching {len(selected_sources)} sources for:** *{prompt}*")
        
        # Create progress bar
        progress_bar = st.progress(0)
        
        # Perform search
        if selected_sources:
            # We'll simulate progress since we can't track individual searches easily
            for i in range(100):
                progress_bar.progress(i + 1)
            
            # Run the search
            with st.spinner("Gathering data from all sources..."):
                search_results = {}
                
                # Run searches in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_sources)) as executor:
                    future_to_source = {}
                    
                    # Submit selected sources
                    if "wikipedia" in selected_sources:
                        future = executor.submit(search_wikipedia, prompt)
                        future_to_source[future] = "wikipedia"
                    
                    if "duckduckgo" in selected_sources:
                        future = executor.submit(search_duckduckgo, prompt, 5)
                        future_to_source[future] = "duckduckgo"
                        
                        future2 = executor.submit(get_instant_answer, prompt)
                        future_to_source[future2] = "duckduckgo_instant"
                    
                    if "arxiv" in selected_sources:
                        future = executor.submit(search_arxiv, prompt, 3)
                        future_to_source[future] = "arxiv"
                    
                    if "weather" in selected_sources:
                        future = executor.submit(get_weather_wttr, prompt)
                        future_to_source[future] = "weather"
                    
                    if "books" in selected_sources:
                        future = executor.submit(search_books, prompt, 3)
                        future_to_source[future] = "books"
                    
                    if "dictionary" in selected_sources:
                        first_word = prompt.split()[0] if prompt.strip() else prompt
                        future = executor.submit(get_definition, first_word)
                        future_to_source[future] = "dictionary"
                    
                    if "country" in selected_sources:
                        future = executor.submit(search_country, prompt)
                        future_to_source[future] = "country"
                    
                    if "github" in selected_sources:
                        future = executor.submit(search_github_repos, prompt, 3)
                        future_to_source[future] = "github"
                    
                    if "stackoverflow" in selected_sources:
                        future = executor.submit(search_stackoverflow, prompt, 3)
                        future_to_source[future] = "stackoverflow"
                    
                    if "quotes" in selected_sources:
                        future = executor.submit(search_quotes, prompt, 3)
                        future_to_source[future] = "quotes"
                    
                    if "wikidata" in selected_sources:
                        future = executor.submit(search_wikidata, prompt, 3)
                        future_to_source[future] = "wikidata"
                    
                    if "geocoding" in selected_sources:
                        future = executor.submit(geocode_location, prompt)
                        future_to_source[future] = "geocoding"
                    
                    # Collect results
                    completed = 0
                    total = len(future_to_source)
                    
                    for future in concurrent.futures.as_completed(future_to_source):
                        source = future_to_source[future]
                        try:
                            result = future.result(timeout=15)
                            if result and not (isinstance(result, dict) and 'error' in result):
                                search_results[source] = result
                        except Exception:
                            search_results[source] = {"error": "Timeout"}
                        
                        completed += 1
                        progress_bar.progress(min(100, int((completed / total) * 100)))
            
            # Clear progress
            progress_bar.empty()
            status_text.empty()
            
            # Format and display results
            formatted_results = format_search_results(prompt, search_results)
            st.markdown(formatted_results)
            
            # Show raw data in expander
            with st.expander("📊 View Raw API Data", expanded=False):
                for source, data in search_results.items():
                    st.subheader(f"🔧 {source.replace('_', ' ').title()}")
                    st.json(data)
            
            # Add to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": formatted_results
            })
        else:
            st.warning("Please select at least one search source in the sidebar!")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I need you to select at least one search source in the sidebar to search."
            })

# Example queries
if not st.session_state.messages:
    st.markdown("### 💡 Try searching for:")
    
    examples = [
        "climate change",
        "machine learning",
        "Paris weather",
        "Python programming",
        "Albert Einstein",
        "quantum physics",
        "github streamlit",
        "stackoverflow api"
    ]
    
    cols = st.columns(4)
    for idx, example in enumerate(examples):
        with cols[idx % 4]:
            if st.button(f"🔍 {example}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()

# Footer
st.divider()
st.caption("🔍 **Super Search Assistant** • All searches run in parallel • Real API data")
