import streamlit as st
from pathlib import Path
import concurrent.futures
from datetime import datetime
import re
import requests
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any

# ============================================
# MODEL CONFIGURATION
# ============================================
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# ============================================
# AI PERSONAS
# ============================================
PRESET_PROMPTS = {
    "Khisba GIS Expert": """You are Khisba GIS - a passionate remote sensing/GIS specialist with deep analytical skills.

CORE IDENTITY:
- Name: Khisba GIS
- Role: Senior Remote Sensing & GIS Analyst
- Style: Enthusiastic, precise, and approachable
- Expertise: Satellite imagery, vegetation indices, climate analysis, urban planning, disaster monitoring

RESPONSE STYLE:
- Reference search findings when available
- Provide concrete examples from GIS/remote sensing
- Use clear, professional language with occasional enthusiasm
- End with actionable insights or suggestions""",

    "Deep Thinker Pro": """You are a sophisticated AI thinker that excels at analysis, synthesis, and providing insightful perspectives.

THINKING FRAMEWORK:
1. **Comprehension**: Understand the query fully
2. **Contextualization**: Place topic in historical/cultural context
3. **Multi-Source Analysis**: Examine information critically
4. **Pattern Recognition**: Identify connections and gaps
5. **Synthesis**: Combine insights coherently
6. **Critical Evaluation**: Assess reliability and significance""",

    "Research Analyst": """You are a professional research analyst specializing in synthesizing complex information.

ANALYTICAL APPROACH:
1. **Source Triangulation**: Cross-reference multiple sources
2. **Credibility Assessment**: Evaluate source reliability
3. **Trend Identification**: Spot patterns and anomalies
4. **Comparative Analysis**: Similarities/differences across contexts
5. **Implication Mapping**: Consequences and applications
6. **Knowledge Gaps**: Identify what's missing""",

    "Technical Expert": """You are a technical expert with deep knowledge across multiple domains.

EXPERTISE AREAS:
- Programming and software development
- Scientific concepts and research
- Technical documentation and explanations
- System architecture and design
- Data analysis and visualization

RESPONSE APPROACH:
- Provide clear, accurate technical information
- Include examples when relevant
- Explain complex concepts simply
- Reference best practices and standards""",

    "Creative Synthesizer": """You connect seemingly unrelated ideas to generate novel insights.

CREATIVE PROCESS:
1. **Divergent Thinking**: Generate multiple interpretations
2. **Analogical Reasoning**: Find similar patterns elsewhere
3. **Metaphorical Connection**: Use metaphors for illumination
4. **Interdisciplinary Bridging**: Connect across fields
5. **Future Projection**: Explore evolution possibilities
6. **Alternative Framing**: Different conceptualizations"""
}

# ============================================
# SEARCH SERVICE FUNCTIONS (All included directly)
# ============================================

# 1. ArXiv Search
def search_arxiv(query: str, max_results: int = 3) -> List[Dict]:
    """Search ArXiv for scientific papers."""
    try:
        url = "http://export.arxiv.org/api/query"
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        papers = []
        
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
            summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
            
            if title_elem is not None and title_elem.text and summary_elem is not None and summary_elem.text:
                title = title_elem.text.strip().replace('\n', ' ')
                summary = summary_elem.text.strip().replace('\n', ' ')
                
                # Extract authors
                authors = []
                for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                    name_elem = author.find('{http://www.w3.org/2005/Atom}name')
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text.strip())
                
                # Extract published date
                published_elem = entry.find('{http://www.w3.org/2005/Atom}published')
                published = published_elem.text[:10] if published_elem is not None and published_elem.text else ''
                
                # Extract link
                link = ''
                for link_elem in entry.findall('{http://www.w3.org/2005/Atom}link'):
                    if link_elem.get('title') == 'pdf':
                        link = link_elem.get('href', '')
                        break
                
                papers.append({
                    'title': title,
                    'summary': summary[:300] + '...' if len(summary) > 300 else summary,
                    'authors': authors[:3],
                    'published': published,
                    'url': link,
                    'source': 'arxiv'
                })
        
        return papers[:max_results]
    except Exception as e:
        return [{'error': f"ArXiv search failed: {str(e)}"}]

# 2. DuckDuckGo Search
def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict]:
    """Search DuckDuckGo for web results."""
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        
        # Add instant answer if available
        if data.get('AbstractText'):
            results.append({
                'title': 'Instant Answer',
                'body': data['AbstractText'],
                'url': data.get('AbstractURL', ''),
                'source': 'duckduckgo'
            })
        
        # Add related topics
        for topic in data.get('RelatedTopics', [])[:max_results-1]:
            if isinstance(topic, dict) and topic.get('Text') and topic.get('FirstURL'):
                results.append({
                    'title': topic.get('Text', '')[:100],
                    'body': '',
                    'url': topic.get('FirstURL', ''),
                    'source': 'duckduckgo'
                })
        
        return results[:max_results]
    except Exception as e:
        return [{'error': f"DuckDuckGo search failed: {str(e)}"}]

# 3. DuckDuckGo Instant Answer
def get_instant_answer(query: str) -> Dict:
    """Get instant answer from DuckDuckGo."""
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1
        }
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        result = {}
        if data.get('Answer'):
            result['answer'] = data['Answer']
        if data.get('AbstractText'):
            result['abstract'] = data['AbstractText']
        if data.get('Definition'):
            result['definition'] = data['Definition']
        if data.get('Redirect'):
            result['redirect'] = data['Redirect']
        
        return result if result else {'error': 'No instant answer found'}
    except Exception as e:
        return {'error': f"Instant answer failed: {str(e)}"}

# 4. Wikipedia Search
def search_wikipedia(query: str) -> Dict:
    """Search Wikipedia for articles."""
    try:
        # First, search for pages
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': 1
        }
        response = requests.get(search_url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('query', {}).get('search'):
            return {'error': 'No Wikipedia article found'}
        
        page_id = data['query']['search'][0]['pageid']
        title = data['query']['search'][0]['title']
        
        # Get page content
        params = {
            'action': 'query',
            'format': 'json',
            'prop': 'extracts|info',
            'inprop': 'url',
            'exintro': 1,
            'explaintext': 1,
            'pageids': page_id
        }
        response = requests.get(search_url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        pages = data.get('query', {}).get('pages', {})
        if not pages:
            return {'error': 'Could not fetch Wikipedia content'}
        
        page = pages.get(str(page_id), {})
        
        return {
            'exists': True,
            'title': page.get('title', title),
            'summary': page.get('extract', '')[:500],
            'url': page.get('fullurl', f'https://en.wikipedia.org/wiki/{title.replace(" ", "_")}'),
            'source': 'wikipedia'
        }
    except Exception as e:
        return {'error': f"Wikipedia search failed: {str(e)}"}

# 5. Weather Search
def get_weather_wttr(query: str) -> Dict:
    """Get weather information from wttr.in."""
    try:
        url = f"https://wttr.in/{requests.utils.quote(query)}?format=j1"
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        current = data.get('current_condition', [{}])[0]
        location = data.get('nearest_area', [{}])[0]
        
        return {
            'location': location.get('areaName', [{}])[0].get('value', query),
            'temperature_c': current.get('temp_C', 'N/A'),
            'temperature_f': current.get('temp_F', 'N/A'),
            'condition': current.get('weatherDesc', [{}])[0].get('value', 'N/A'),
            'humidity': current.get('humidity', 'N/A'),
            'wind_speed': current.get('windspeedKmph', 'N/A'),
            'source': 'weather'
        }
    except Exception as e:
        return {'error': f"Weather search failed: {str(e)}"}

# 6. Air Quality Search
def get_air_quality(query: str) -> Dict:
    """Get air quality information from OpenAQ."""
    try:
        url = f"https://api.openaq.org/v2/latest"
        params = {
            'limit': 3,
            'city': query
        }
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('results'):
            return {'error': 'No air quality data found'}
        
        locations = []
        for result in data['results'][:2]:
            measurements = []
            for param in result.get('measurements', [])[:3]:
                measurements.append({
                    'parameter': param.get('parameter', ''),
                    'value': param.get('value', ''),
                    'unit': param.get('unit', '')
                })
            
            locations.append({
                'location': result.get('location', ''),
                'city': result.get('city', query),
                'country': result.get('country', ''),
                'measurements': measurements
            })
        
        return {
            'city': query,
            'data': locations,
            'source': 'air_quality'
        }
    except Exception as e:
        return {'error': f"Air quality search failed: {str(e)}"}

# 7. Wikidata Search
def search_wikidata(query: str, max_results: int = 3) -> List[Dict]:
    """Search Wikidata for entities."""
    try:
        url = "https://www.wikidata.org/w/api.php"
        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'language': 'en',
            'search': query,
            'limit': max_results
        }
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        entities = []
        for entity in data.get('search', [])[:max_results]:
            entities.append({
                'label': entity.get('label', ''),
                'description': entity.get('description', ''),
                'id': entity.get('id', ''),
                'url': f"https://www.wikidata.org/wiki/{entity.get('id', '')}",
                'source': 'wikidata'
            })
        
        return entities
    except Exception as e:
        return [{'error': f"Wikidata search failed: {str(e)}"}]

# 8. OpenLibrary Books Search
def search_books(query: str, max_results: int = 5) -> List[Dict]:
    """Search OpenLibrary for books."""
    try:
        url = "https://openlibrary.org/search.json"
        params = {
            'q': query,
            'limit': max_results
        }
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        books = []
        for doc in data.get('docs', [])[:max_results]:
            authors = doc.get('author_name', [])
            if isinstance(authors, list):
                authors = authors[:2]
            
            books.append({
                'title': doc.get('title', ''),
                'authors': authors,
                'first_publish_year': doc.get('first_publish_year', ''),
                'publisher': doc.get('publisher', [])[0] if doc.get('publisher') else '',
                'url': f"https://openlibrary.org{doc.get('key', '')}",
                'source': 'books'
            })
        
        return books
    except Exception as e:
        return [{'error': f"Books search failed: {str(e)}"}]

# 9. PubMed Search
def search_pubmed(query: str, max_results: int = 3) -> List[Dict]:
    """Search PubMed for medical articles."""
    try:
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmode': 'json',
            'retmax': max_results
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        ids = data.get('esearchresult', {}).get('idlist', [])
        if not ids:
            return [{'error': 'No PubMed articles found'}]
        
        # Get details for each article
        details_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params = {
            'db': 'pubmed',
            'id': ','.join(ids),
            'retmode': 'json'
        }
        response = requests.get(details_url, params=params, timeout=10)
        response.raise_for_status()
        details = response.json()
        
        articles = []
        for article_id in ids[:max_results]:
            article_data = details.get('result', {}).get(article_id, {})
            authors = []
            for author in article_data.get('authors', [])[:2]:
                if isinstance(author, dict):
                    authors.append(author.get('name', ''))
            
            articles.append({
                'title': article_data.get('title', ''),
                'authors': authors,
                'year': article_data.get('pubdate', '')[:4],
                'journal': article_data.get('source', ''),
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/",
                'source': 'pubmed'
            })
        
        return articles
    except Exception as e:
        return [{'error': f"PubMed search failed: {str(e)}"}]

# 10. Geocoding (Nominatim)
def geocode_location(query: str) -> Dict:
    """Geocode location using Nominatim."""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': query,
            'format': 'json',
            'limit': 1
        }
        headers = {
            'User-Agent': 'SmartSearchAI/1.0'
        }
        response = requests.get(url, params=params, headers=headers, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return {'error': 'Location not found'}
        
        location = data[0]
        return {
            'display_name': location.get('display_name', ''),
            'latitude': location.get('lat', ''),
            'longitude': location.get('lon', ''),
            'osm_url': f"https://www.openstreetmap.org/?mlat={location.get('lat')}&mlon={location.get('lon')}",
            'source': 'geocoding'
        }
    except Exception as e:
        return {'error': f"Geocoding failed: {str(e)}"}

# 11. Dictionary Definition
def get_definition(word: str) -> Dict:
    """Get dictionary definition."""
    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        response = requests.get(url, timeout=8)
        
        if response.status_code == 404:
            return {'error': 'Word not found in dictionary'}
        
        response.raise_for_status()
        data = response.json()
        
        if not data or not isinstance(data, list):
            return {'error': 'Invalid dictionary response'}
        
        entry = data[0]
        meanings = []
        
        for meaning in entry.get('meanings', [])[:2]:
            definitions = []
            for definition in meaning.get('definitions', [])[:2]:
                definitions.append({
                    'definition': definition.get('definition', ''),
                    'example': definition.get('example', '')
                })
            
            meanings.append({
                'part_of_speech': meaning.get('partOfSpeech', ''),
                'definitions': definitions
            })
        
        phonetics = []
        for phonetic in entry.get('phonetics', []):
            if phonetic.get('text'):
                phonetics.append(phonetic['text'])
        
        return {
            'word': entry.get('word', word),
            'phonetics': phonetics[:3],
            'meanings': meanings,
            'source': 'dictionary'
        }
    except Exception as e:
        return {'error': f"Dictionary search failed: {str(e)}"}

# 12. Country Information
def search_country(query: str) -> Dict:
    """Search for country information."""
    try:
        url = f"https://restcountries.com/v3.1/name/{query}"
        response = requests.get(url, timeout=8)
        
        if response.status_code == 404:
            return {'error': 'Country not found'}
        
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return {'error': 'No country data found'}
        
        country = data[0]
        
        # Get languages
        languages = []
        if country.get('languages'):
            languages = list(country['languages'].values())[:3]
        
        # Get currencies
        currencies = []
        if country.get('currencies'):
            for currency_code, currency_info in country['currencies'].items():
                currencies.append(f"{currency_info.get('name', '')} ({currency_code})")
        
        return {
            'name': country.get('name', {}).get('common', query),
            'official_name': country.get('name', {}).get('official', ''),
            'capital': country.get('capital', [''])[0],
            'region': country.get('region', ''),
            'subregion': country.get('subregion', ''),
            'population': country.get('population', 0),
            'languages': languages,
            'currencies': currencies[:2],
            'flag_emoji': country.get('flag', ''),
            'map_url': f"https://www.google.com/maps?q={country.get('latlng', [0,0])[0]},{country.get('latlng', [0,0])[1]}",
            'source': 'country'
        }
    except Exception as e:
        return {'error': f"Country search failed: {str(e)}"}

# 13. Quotes Search
def search_quotes(query: str, max_results: int = 3) -> List[Dict]:
    """Search for quotes."""
    try:
        url = "https://api.quotable.io/search/quotes"
        params = {
            'query': query,
            'limit': max_results
        }
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        quotes = []
        for quote in data.get('results', [])[:max_results]:
            quotes.append({
                'content': quote.get('content', ''),
                'author': quote.get('author', 'Unknown'),
                'tags': quote.get('tags', [])[:3],
                'source': 'quotes'
            })
        
        return quotes
    except Exception as e:
        return [{'error': f"Quotes search failed: {str(e)}"}]

# 14. GitHub Repositories Search
def search_github_repos(query: str, max_results: int = 3) -> List[Dict]:
    """Search GitHub for repositories."""
    try:
        url = "https://api.github.com/search/repositories"
        params = {
            'q': query,
            'sort': 'stars',
            'order': 'desc',
            'per_page': max_results
        }
        headers = {
            'Accept': 'application/vnd.github.v3+json'
        }
        response = requests.get(url, params=params, headers=headers, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        repos = []
        for repo in data.get('items', [])[:max_results]:
            repos.append({
                'name': repo.get('name', ''),
                'full_name': repo.get('full_name', ''),
                'description': repo.get('description', '')[:150],
                'language': repo.get('language', ''),
                'stars': repo.get('stargazers_count', 0),
                'forks': repo.get('forks_count', 0),
                'url': repo.get('html_url', ''),
                'source': 'github'
            })
        
        return repos
    except Exception as e:
        return [{'error': f"GitHub search failed: {str(e)}"}]

# 15. Stack Overflow Search
def search_stackoverflow(query: str, max_results: int = 3) -> List[Dict]:
    """Search Stack Overflow for questions."""
    try:
        url = "https://api.stackexchange.com/2.3/search"
        params = {
            'order': 'desc',
            'sort': 'relevance',
            'intitle': query,
            'site': 'stackoverflow',
            'pagesize': max_results
        }
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        questions = []
        for question in data.get('items', [])[:max_results]:
            questions.append({
                'title': question.get('title', ''),
                'is_answered': question.get('is_answered', False),
                'score': question.get('score', 0),
                'answer_count': question.get('answer_count', 0),
                'view_count': question.get('view_count', 0),
                'tags': question.get('tags', [])[:3],
                'url': question.get('link', ''),
                'source': 'stackoverflow'
            })
        
        return questions
    except Exception as e:
        return [{'error': f"Stack Overflow search failed: {str(e)}"}]

# 16. News Search (using DuckDuckGo news)
def search_news(query: str, max_results: int = 3) -> List[Dict]:
    """Search for news articles."""
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1
        }
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        news_items = []
        
        # Check for news in the response
        if data.get('AbstractText') and 'news' in query.lower():
            news_items.append({
                'title': 'Latest News',
                'body': data['AbstractText'][:200],
                'url': data.get('AbstractURL', ''),
                'source': 'news',
                'date': datetime.now().strftime('%Y-%m-%d')
            })
        
        # Add related topics as news-like items
        for topic in data.get('RelatedTopics', [])[:max_results]:
            if isinstance(topic, dict) and topic.get('Text'):
                news_items.append({
                    'title': topic.get('Text', '')[:100],
                    'body': '',
                    'url': topic.get('FirstURL', ''),
                    'source': 'news',
                    'date': datetime.now().strftime('%Y-%m-%d')
                })
        
        return news_items[:max_results]
    except Exception as e:
        return [{'error': f"News search failed: {str(e)}"}]

# ============================================
# MAIN SEARCH FUNCTION
# ============================================
def search_all_sources(query: str) -> dict:
    """Search ALL 16 sources simultaneously."""
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
    """Analyze search results independently."""
    analysis = {
        'key_facts': [],
        'source_count': len(results),
        'working_sources': [],
        'failed_sources': [],
        'confidence_score': 0,
        'knowledge_gaps': []
    }
    
    # Check each source
    for source, data in results.items():
        # Check if source has valid data (not an error)
        has_data = False
        
        if isinstance(data, dict):
            has_data = "error" not in data and data.get("error") is None
        elif isinstance(data, list):
            has_data = len(data) > 0 and not ("error" in str(data[0]) if data else False)
        
        if has_data:
            analysis['working_sources'].append(source)
            
            # Extract key facts based on source
            if source == "duckduckgo_instant" and isinstance(data, dict) and data.get("answer"):
                analysis['key_facts'].append({
                    'content': data['answer'][:150],
                    'source': 'DuckDuckGo Instant'
                })
            
            elif source == "wikipedia" and isinstance(data, dict) and data.get("summary"):
                analysis['key_facts'].append({
                    'content': data['summary'][:150],
                    'source': 'Wikipedia'
                })
            
            elif source == "dictionary" and isinstance(data, dict) and data.get("meanings"):
                analysis['key_facts'].append({
                    'content': f"Word definition found for {data.get('word', 'word')}",
                    'source': 'Dictionary'
                })
        else:
            analysis['failed_sources'].append(source)
    
    # Calculate confidence
    working_count = len(analysis['working_sources'])
    if working_count >= 10:
        analysis['confidence_score'] = 'high'
    elif working_count >= 6:
        analysis['confidence_score'] = 'medium'
    else:
        analysis['confidence_score'] = 'low'
    
    # Identify knowledge gaps
    if working_count < 5:
        analysis['knowledge_gaps'].append(f"Only {working_count}/16 sources returned data")
    
    if not analysis['key_facts']:
        analysis['knowledge_gaps'].append("No key facts extracted from search results")
    
    return analysis

# ============================================
# STREAMLIT APP SETUP
# ============================================
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
# MODEL LOADING FUNCTION
# ============================================
def download_model():
    """Download the TinyLLaMA model."""
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
        return False

@st.cache_resource(show_spinner=False)
def load_ai_model():
    """Load the TinyLLaMA model."""
    try:
        from ctransformers import AutoModelForCausalLM
    except ImportError:
        st.error("Please install ctransformers: pip install ctransformers")
        return None
    
    if not MODEL_PATH.exists():
        if not download_model():
            st.error("Model download failed. Please check your internet connection.")
            return None
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            model_file=MODEL_PATH.name,
            model_type="llama",
            context_length=2048,
            gpu_layers=0,
            threads=8
        )
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# ============================================
# AI SYNTHESIS FUNCTIONS - FIXED VERSION
# ============================================
def extract_search_data(search_results: dict, working_sources: list) -> str:
    """Extract and format concrete data from search results."""
    formatted_data = []
    
    for source_name in working_sources:
        data = search_results.get(source_name)
        if not data:
            continue
            
        try:
            if source_name == "wikipedia" and isinstance(data, dict):
                if data.get('summary'):
                    formatted_data.append(f"📚 Wikipedia: {data['summary'][:200]}")
                    
            elif source_name == "arxiv" and isinstance(data, list) and data:
                for item in data[:2]:
                    if isinstance(item, dict) and item.get('summary'):
                        formatted_data.append(f"📄 Research Paper: {item['summary'][:150]}")
                        
            elif source_name == "duckduckgo" and isinstance(data, list) and data:
                for item in data[:2]:
                    if isinstance(item, dict) and item.get('body') and item['body']:
                        formatted_data.append(f"🌐 Web Result: {item['body'][:150]}")
                        
            elif source_name == "duckduckgo_instant" and isinstance(data, dict):
                if data.get('answer'):
                    formatted_data.append(f"💡 Instant Answer: {data['answer'][:150]}")
                elif data.get('abstract'):
                    formatted_data.append(f"💡 Abstract: {data['abstract'][:150]}")
                    
            elif source_name == "dictionary" and isinstance(data, dict):
                if data.get('meanings'):
                    first_meaning = data['meanings'][0]
                    if first_meaning.get('definitions'):
                        def_text = first_meaning['definitions'][0]['definition'][:120]
                        formatted_data.append(f"📖 Definition: {def_text}")
                        
            elif source_name == "news" and isinstance(data, list) and data:
                for item in data[:2]:
                    if isinstance(item, dict) and item.get('body') and item['body']:
                        formatted_data.append(f"📰 News: {item['body'][:150]}")
                        
            elif source_name == "weather" and isinstance(data, dict):
                if data.get('condition') and data.get('condition') != 'N/A':
                    formatted_data.append(f"🌤️ Weather: {data.get('location', 'Location')}: {data.get('condition')} at {data.get('temperature_c')}°C")
                    
        except Exception as e:
            continue
    
    return formatted_data

def create_synthesis_prompt(query: str, conversation_history: list, system_prompt: str, 
                           search_results: dict, search_analysis: dict) -> str:
    """Create prompt for AI to synthesize search results - FORCING SEARCH USAGE."""
    
    working_sources = search_analysis.get('working_sources', [])
    failed_sources = search_analysis.get('failed_sources', [])
    
    # Extract concrete data from search results
    search_data_items = extract_search_data(search_results, working_sources)
    
    # Format search summary
    search_summary = "**SEARCH RESULTS DATA - USE ONLY THIS INFORMATION:**\n\n"
    
    if search_data_items:
        search_summary += "**AVAILABLE INFORMATION FROM SEARCH:**\n"
        for idx, item in enumerate(search_data_items[:6], 1):
            search_summary += f"{idx}. {item}\n"
    else:
        search_summary += "**NO CONCRETE DATA FOUND IN SEARCH**\n"
        search_summary += "Please state: 'Search returned limited or no specific information on this topic.'\n"
    
    search_summary += f"\n**SEARCH METADATA:**\n"
    search_summary += f"• Sources with data: {len(working_sources)}/16\n"
    search_summary += f"• Working sources: {', '.join(working_sources[:5])}\n"
    search_summary += f"• Confidence level: {search_analysis.get('confidence_score', 'low')}\n"
    
    # STRICT INSTRUCTIONS
    instructions = """
**CRITICAL INSTRUCTIONS - FOLLOW EXACTLY:**

1. **USE ONLY SEARCH RESULTS ABOVE** - Do NOT use your own knowledge or training data
2. **IF NO DATA, SAY SO** - If search found nothing, clearly state this
3. **CITE SOURCES** - Mention which search sources provided information
4. **NO INVENTION** - Do not make up facts, dates, numbers, or details
5. **LIMITED SCOPE** - If search was limited, acknowledge this clearly
6. **USE FIRST RESULTS** - When multiple sources exist, prioritize the first/best match

**RESPONSE STRUCTURE:**
1. Start with: "Based on search results from [X] sources:"
2. List key findings from search (only what's in search results)
3. Mention source limitations if applicable
4. End with: "Additional sources would provide more comprehensive information."
"""

    # Add conversation history
    history = ""
    for msg in conversation_history[-2:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    
    # Create final prompt with strict formatting
    prompt = f"""<|system|>
{system_prompt}

You are an assistant that STRICTLY USES ONLY SEARCH RESULTS provided below.
Your task is to report what was found in the search, NOT to generate new information.

{search_summary}

{instructions}
</s>

<|user|>
Query: "{query}"

**IMPORTANT:**
1. What information did the search ACTUALLY find?
2. What specific data points are available from search?
3. Which sources provided this information?
4. How comprehensive was the search coverage?

**DO NOT:**
- Invent facts not in search results
- Use your training knowledge
- Speculate beyond what search found
- Provide details not explicitly in search data

**Previous conversation context:**
{history}

Now provide a response based SOLELY on the search results above.
</s>

<|assistant|>
Based on search results from {len(working_sources)} sources:"""
    
    return prompt

def generate_ai_response(model, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
    """Generate AI response with strict search adherence."""
    try:
        response = model(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.3,  # Low temperature for factual responses
            top_p=0.8,
            repetition_penalty=1.2,
            stop=["</s>", "<|user|>", "\n\nUser:", "###", "**Note:", "\n\nNote:"],
            stream=False
        )
        
        response_text = response.strip()
        
        # Ensure response references search
        if not any(keyword in response_text.lower() for keyword in ['search', 'result', 'found', 'according to', 'source']):
            response_text = "Based on search results: " + response_text
        
        return response_text
    except Exception as e:
        return f"I apologize, but I encountered an error while processing search results: {str(e)}"

# ============================================
# STREAMLIT UI
# ============================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
    .confidence-high { background: #d4edda !important; color: #155724 !important; }
    .confidence-medium { background: #fff3cd !important; color: #856404 !important; }
    .confidence-low { background: #f8d7da !important; color: #721c24 !important; }
    .search-data-box {
        background: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🔍🧠 SmartSearch AI Pro</h1>
    <h4>16-Source Search + AI Synthesis | Search-First Approach</h4>
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
    
    st.header("🔧 Settings")
    
    auto_search = st.toggle("Auto-Search 16 Sources", value=True)
    show_details = st.toggle("Show Search Details", value=False)
    force_search_only = st.toggle("Force Search-Only Mode", value=True, 
                                  help="AI will ONLY use search results, no training knowledge")
    
    st.divider()
    
    st.header("⚙️ AI Parameters")
    
    temperature = st.slider(
        "Creativity:",
        0.1, 1.5, 0.3, 0.1,
        help="Lower = more factual, Higher = more creative"
    )
    
    response_length = st.slider(
        "Response Length:",
        256, 1024, 512, 128
    )
    
    st.divider()
    
    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.search_results = {}
        st.session_state.search_analysis = {}
        st.rerun()
    
    st.divider()
    
    st.markdown("**16 Search Sources:**")
    st.caption("🌐 Web: DuckDuckGo, Wikipedia, News")
    st.caption("🔬 Research: ArXiv, PubMed")
    st.caption("📚 Reference: Dictionary, Books, Quotes")
    st.caption("💻 Developer: GitHub, Stack Overflow")
    st.caption("🌍 Location: Weather, Air Quality, Geocoding")
    
    st.divider()
    
    st.caption("⚠️ AI uses search results only when available")

# Load AI model
if st.session_state.model is None:
    with st.spinner("🚀 Initializing AI..."):
        st.session_state.model = load_ai_model()

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show search metadata if available
        if message["role"] == "assistant" and "metadata" in message:
            metadata = message["metadata"]
            cols = st.columns(3)
            with cols[0]:
                st.caption(f"📊 Sources: {len(metadata.get('sources', []))}/16")
            with cols[1]:
                confidence = metadata.get('confidence', 'unknown')
                st.caption(f"🎯 Confidence: {confidence}")
            with cols[2]:
                if force_search_only:
                    st.caption("🔒 Search-Only Mode")

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        # Step 1: Search
        search_results = {}
        search_analysis = {}
        
        if auto_search:
            search_status = st.empty()
            search_status.info("🔍 Searching 16 sources simultaneously...")
            
            with st.spinner("Querying: Wikipedia, DuckDuckGo, ArXiv, GitHub, Weather, etc..."):
                search_results = search_all_sources(prompt)
                search_analysis = analyze_search_results(prompt, search_results)
            
            working_count = len(search_analysis.get('working_sources', []))
            confidence = search_analysis.get('confidence_score', 'low')
            
            # Display search summary
            cols = st.columns(3)
            with cols[0]:
                st.metric("Sources Found", f"{working_count}/16")
            with cols[1]:
                confidence_color = {
                    'high': 'green',
                    'medium': 'orange',
                    'low': 'red'
                }.get(confidence, 'gray')
                st.metric("Confidence", confidence.capitalize())
            with cols[2]:
                if force_search_only:
                    st.metric("Mode", "🔒 Search-Only")
            
            # Show search data preview
            if show_details and working_count > 0:
                with st.expander("📊 View Raw Search Data", expanded=False):
                    # Display first few data items
                    search_data_items = extract_search_data(search_results, search_analysis.get('working_sources', []))
                    if search_data_items:
                        st.markdown("**Extracted Search Data:**")
                        for item in search_data_items[:3]:
                            st.markdown(f"• {item}")
                    else:
                        st.info("No concrete data extracted from search results")
                    
                    # Show all sources status
                    st.markdown("**All Sources Status:**")
                    all_sources = ["wikipedia", "duckduckgo", "arxiv", "news", "dictionary", 
                                  "weather", "github", "stackoverflow", "pubmed", "books",
                                  "quotes", "country", "geocoding", "air_quality", "wikidata", "duckduckgo_instant"]
                    
                    cols = st.columns(4)
                    for idx, source in enumerate(all_sources):
                        status = "✅" if source in search_analysis.get('working_sources', []) else "❌"
                        with cols[idx % 4]:
                            st.caption(f"{status} {source}")
        
        # Step 2: AI Synthesis
        if st.session_state.model:
            ai_status = st.empty()
            ai_status.info("🧠 AI is synthesizing search results...")
            
            try:
                # Create prompt with search enforcement
                synthesis_prompt = create_synthesis_prompt(
                    prompt,
                    st.session_state.messages,
                    st.session_state.system_prompt,
                    search_results,
                    search_analysis
                )
                
                # Generate response with search-only enforcement
                with st.spinner("Processing search results..."):
                    ai_response = generate_ai_response(
                        st.session_state.model,
                        synthesis_prompt,
                        max_tokens=response_length,
                        temperature=temperature
                    )
                
                ai_status.empty()
                st.markdown(ai_response)
                
                # Store message with metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ai_response,
                    "metadata": {
                        "sources": search_analysis.get('working_sources', []),
                        "confidence": confidence,
                        "search_only": force_search_only,
                        "working_count": working_count
                    }
                })
                
                # Store search data
                st.session_state.search_results = search_results
                st.session_state.search_analysis = search_analysis
                
            except Exception as e:
                ai_status.error(f"AI Error: {str(e)}")
                
                # Fallback: Show formatted search results
                st.info("Showing raw search results due to AI error...")
                
                if search_results and working_count > 0:
                    st.markdown("### Search Results Summary:")
                    
                    # Show available data
                    search_data_items = extract_search_data(search_results, search_analysis.get('working_sources', []))
                    if search_data_items:
                        for item in search_data_items[:5]:
                            st.markdown(f"• {item}")
                    else:
                        st.info("Search returned limited or no specific information.")
                else:
                    st.warning("⚠️ Search returned no data from any sources.")
        else:
            st.warning("⚠️ AI model not loaded. Showing search results only.")
            
            if search_results and working_count > 0:
                st.markdown("### Search Results Summary:")
                search_data_items = extract_search_data(search_results, search_analysis.get('working_sources', []))
                for item in search_data_items[:5]:
                    st.markdown(f"• {item}")

# Quick examples
if not st.session_state.messages:
    st.markdown("### 💡 Example Queries:")
    
    examples = [
        "Climate change effects",
        "Python machine learning tutorials",
        "Weather in Tokyo",
        "Latest AI research papers",
        "GitHub repositories for web development"
    ]
    
    cols = st.columns(3)
    for idx, example in enumerate(examples):
        with cols[idx % 3]:
            if st.button(example, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()

# Footer
st.divider()
st.caption("SmartSearch AI Pro | 16 Search Sources + TinyLLaMA AI | 🔒 Search-First Approach")
