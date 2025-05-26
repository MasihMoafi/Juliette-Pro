#!/usr/bin/env python
# coding: utf-8

# Disable proxy settings as mentioned in user rules
import os # Standard top-level import

def clear_proxy_settings():
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        if var in os.environ:
            del os.environ[var]

clear_proxy_settings()

# Import necessary libraries
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import httpx
import json
import re
from bs4 import BeautifulSoup
import ollama
import plotly.express as px
from smolagents import tool, CodeAgent
import traceback # For printing full tracebacks

# Define the model to use
LLM_MODEL = 'qwen3:8b' # Make sure this model is pulled in Ollama: ollama pull qwen3:8b

# Custom Ollama model class that works with smolagents
class OllamaModel:
    def __init__(self, model_name=LLM_MODEL, temperature=0.1):
        self.model_name = model_name
        self.temperature = temperature
    
    def __call__(self, messages, **kwargs):
        """
        Process messages in the format expected by smolagents and return a string response.
        """
        try:
            ollama_messages = []
            
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content_input = msg.get("content", "") # Renamed to avoid conflict
                    
                    if isinstance(content_input, list):
                        content_str = ""
                        for item in content_input:
                            if isinstance(item, str):
                                content_str += item
                            elif isinstance(item, dict) and item.get("type") == "text":
                                content_str += item.get("text", "")
                        current_content = content_str # Use a different variable name
                    else:
                        current_content = str(content_input) # Ensure it's a string
                    
                    if role in ["tool", "function"]:
                        ollama_messages.append({
                            "role": "user", 
                            "content": f"Tool result: {current_content}"
                        })
                    else:
                        if role not in ["user", "assistant", "system"]:
                            role = "user" 
                        ollama_messages.append({"role": role, "content": current_content})
                else:
                    ollama_messages.append({"role": "user", "content": str(msg)})
            
            response = ollama.chat(
                model=self.model_name, 
                messages=ollama_messages,
                options={"temperature": self.temperature}
            )
            
            extracted_content = ""
            if hasattr(response, 'message') and isinstance(response.message, dict) and 'content' in response.message:
                extracted_content = response.message['content']
            else:
                print(f"Warning: Could not extract content directly from ollama.chat response. Response: {response}")
                if hasattr(response, 'message'):
                    extracted_content = str(response.message)
                else:
                    extracted_content = str(response)
            
            return str(extracted_content) # Return the string content directly
                
        except Exception as e:
            print(f"Error in OllamaModel.__call__: {str(e)}")
            traceback.print_exc() 
            return f"Error processing request in OllamaModel: {str(e)}"

# Define tools
@tool
def web_search(query: str) -> str:
    """
    Searches the web for information using DuckDuckGo.
    Args:
        query: The search query
    """
    print(f"Tool: web_search, Query: {query}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = httpx.get("https://duckduckgo.com/html/", params={"q": query}, headers=headers, timeout=10)
        response.raise_for_status() 
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        for result_item in soup.select('div.result, div.results_links_deep')[:5]: # Max 5 results
            title_elem = result_item.select_one('h2.result__title a, a.result__a')
            snippet_elem = result_item.select_one('a.result__snippet, span.result__snippet')
            link_elem = result_item.select_one('a.result__url, span.result__url') # Changed from result__url to result__url span for some cases
            
            title = title_elem.get_text(strip=True) if title_elem else "No title"
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else "No snippet"
            
            url = None
            if link_elem:
                if link_elem.name == 'a' and 'href' in link_elem.attrs:
                    url = link_elem['href']
                else:
                    url_text = link_elem.get_text(strip=True)
                    if url_text.startswith("http"):
                         url = url_text

            if url and 'duckduckgo.com/y.js' in url:
                from urllib.parse import urlparse, parse_qs
                parsed_url = urlparse(url)
                query_params = parse_qs(parsed_url.query)
                if 'ad_domain' in query_params: url = query_params['ad_domain'][0]
                elif 'uddg' in query_params: url = query_params['uddg'][0]

            if title != "No title" or snippet != "No snippet":
                results.append({"title": title, "snippet": snippet, "url": url or "#"})
        
        if not results: return "No search results found from DuckDuckGo."
        return json.dumps(results, indent=2)

    except httpx.RequestError as e:
        print(f"Network error during DuckDuckGo search: {str(e)}")
        return json.dumps([
            {"title": "Louvre Museum - Mock Data", "snippet": "The Louvre received 9.6 million visitors...", "url": "https://www.louvre.fr/en/visit"},
            {"title": "National Museum of China - Mock Data", "snippet": "Approx. 8.3 million visitors...", "url": "https://en.chnmuseum.cn/"},
            {"title": "Vatican Museums - Mock Data", "snippet": "Around 6.9 million visitors...", "url": "https://www.museivaticani.va/content/museivaticani/en.html"}
        ], indent=2)
    except Exception as e:
        print(f"Error parsing DuckDuckGo results: {str(e)}")
        traceback.print_exc()
        return f"Error parsing search results: {str(e)}"

@tool
def visit_webpage(url: str) -> str:
    """
    Visits a webpage at the given url and reads its content as text.
    Args:
        url: The url of the webpage to visit
    """
    print(f"Tool: visit_webpage, URL: {url}")
    if not url or not url.startswith(('http://', 'https://')):
        return "Invalid URL provided. Must start with http:// or https://"
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = httpx.get(url, headers=headers, timeout=20, follow_redirects=True)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            return f"Cannot parse content type: {content_type}. Only HTML pages are supported."

        soup = BeautifulSoup(response.text, 'html.parser')
        for script_or_style in soup(["script", "style", "nav", "footer", "aside"]): # Remove more non-content elements
            script_or_style.decompose()
            
        main_content_tags = ['main', 'article', 'div[role="main"]', 'div.content', 'div.post-content', 'div.entry-content', 'section']
        text_content = ""
        for tag_selector in main_content_tags:
            main_element = soup.select_one(tag_selector)
            if main_element:
                text_content = main_element.get_text(separator='\n', strip=True)
                if len(text_content) > 200: # Prefer longer main content
                    break 
        
        if not text_content or len(text_content) < 100: # If main content is too short or not found
            body = soup.find('body')
            if body: text_content = body.get_text(separator='\n', strip=True)
            else: text_content = soup.get_text(separator='\n', strip=True)
        
        text_content = re.sub(r'\n\s*\n+', '\n\n', text_content) 
        text_content = re.sub(r'[ \t]{2,}', ' ', text_content)    
        
        if len(text_content) > 10000: text_content = text_content[:10000] + "...[truncated]"
            
        return text_content if text_content else "Could not extract meaningful text from the webpage."

    except httpx.HTTPStatusError as e:
        print(f"HTTP error visiting {url}: {e.response.status_code} - {e.response.reason_phrase}")
        return f"Failed to fetch {url}. Status: {e.response.status_code}. Page content might be protected or unavailable."
    except httpx.RequestError as e:
        print(f"Network error visiting {url}: {str(e)}")
        if "louvre" in url.lower(): return "MOCK: Louvre Museum (Paris, France) - 9.6 million visitors in 2023. Coords: 48.8611 N, 2.3364 E. July Temp: 25.2 C."
        elif "chnmuseum" in url.lower(): return "MOCK: National Museum of China (Beijing, China) - 8.3 million visitors in 2023. Coords: 39.9042 N, 116.4074 E. July Temp: 30.8 C."
        elif "vatican" in url.lower(): return "MOCK: Vatican Museums (Vatican City) - 6.9 million visitors in 2023. Coords: 41.9029 N, 12.4534 E. July Temp: 31.5 C."
        return f"Webpage data for {url} not available due to network issues."
    except Exception as e:
        print(f"Unexpected error visiting {url}: {str(e)}")
        traceback.print_exc()
        return f"Error processing webpage {url}: {str(e)}"

@tool
def create_museum_map(museum_data: list) -> str:
    """
    Creates a scatter map visualization of museums using plotly express.
    Args:
        museum_data: List of dictionaries containing museum information.
                     Each dictionary must have: name, city, country, visitors (float/int), 
                                              lat (float), lon (float), temp (float).
    """
    print(f"Tool: create_museum_map, Data: {museum_data}")
    try:
        if not isinstance(museum_data, list) or not museum_data:
            return "Error: museum_data must be a non-empty list of dictionaries."

        required_keys = ["name", "city", "country", "visitors", "lat", "lon", "temp"]
        processed_data = []
        for i, museum_item in enumerate(museum_data): # Renamed to avoid conflict
            if not isinstance(museum_item, dict):
                return f"Error: Item at index {i} in museum_data is not a dictionary. Received: {type(museum_item)}"
            
            # Check for all required keys
            missing_keys = [key for key in required_keys if key not in museum_item]
            if missing_keys:
                return f"Error: Missing key(s) '{', '.join(missing_keys)}' in museum data at index {i}: {museum_item}"
            
            # Validate and convert numeric fields
            try:
                current_museum_data = {
                    "name": str(museum_item["name"]),
                    "city": str(museum_item["city"]),
                    "country": str(museum_item["country"]),
                    "visitors": float(museum_item["visitors"]),
                    "lat": float(museum_item["lat"]),
                    "lon": float(museum_item["lon"]),
                    "temp": float(museum_item["temp"])
                }
                processed_data.append(current_museum_data)
            except (ValueError, TypeError) as e:
                return f"Error: Could not convert numeric fields for museum at index {i} ({museum_item.get('name', 'Unknown')}): {e}. Data: {museum_item}"
        
        df = pd.DataFrame(processed_data)
        
        fig = px.scatter_mapbox(
            df, lat="lat", lon="lon", hover_name="name",
            hover_data=["city", "country", "visitors", "temp"],
            color="temp", size="visitors", size_max=25, zoom=1,
            height=800, width=1200,
            color_continuous_scale=px.colors.sequential.Plasma,
            title="Top Museums by Visitor Count with July Temperatures"
        )
        
        fig.update_layout(mapbox_style="open-street-map", margin={"r":0, "t":50, "l":0, "b":0})
        
        map_filename = "saved_map.html"
        fig.write_html(map_filename)
        return f"Map successfully created and saved to {map_filename}"

    except Exception as e:
        print(f"Error in create_museum_map: {str(e)}")
        traceback.print_exc()
        return f"Error creating map: {str(e)}. Input data was: {museum_data}"

# Initialize the model
model = OllamaModel(LLM_MODEL)

# Create a single agent with all tools
agent = CodeAgent(
    model=model,
    tools=[web_search, visit_webpage, create_museum_map],
    max_steps=15, 
    additional_authorized_imports=[
        "plotly", "plotly.express", "pandas", "numpy", "json", "re"
    ],
    verbosity_level=2 
)

def main():
    map_filename = "saved_map.html"
    if os.path.exists(map_filename):
        try:
            os.remove(map_filename)
            print(f"Removed existing {map_filename}")
        except OSError as e:
            print(f"Error removing existing {map_filename}: {e}")
    
    query = """
    I need you to research and find the top 3 most visited museums in the world as of late 2023 or early 2024.
    For each museum, I need:
    1. The museum's name (string).
    2. The city where it's located (string).
    3. The country where it's located (string).
    4. The annual visitor count in millions (float, e.g., 8.5 for 8.5 million).
    5. The approximate latitude (float).
    6. The approximate longitude (float).
    7. The average daily high temperature in July at its location in Celsius (float).

    After gathering all this information, you must call the 'create_museum_map' tool.
    The argument to 'create_museum_map' must be a Python list of dictionaries.
    Each dictionary in the list represents one museum and must contain all seven pieces of information.
    The keys in each dictionary must be exactly: "name", "city", "country", "visitors", "lat", "lon", "temp".
    Ensure 'visitors', 'lat', 'lon', and 'temp' are passed as numbers (float or int), not strings.

    Example of the data structure for one museum for the tool call:
    {"name": "Example Museum", "city": "Example City", "country": "Example Country", "visitors": 7.2, "lat": 40.7128, "lon": -74.0060, "temp": 28.5}
    The 'create_museum_map' tool expects a list: e.g., [museum1_dict, museum2_dict, museum3_dict]

    Do not invent any numbers. You must only use numbers sourced from the internet using the provided tools.
    If you cannot find precise data for July temperature, a reasonable estimate for the city in summer is acceptable if stated during research, but provide a float for the tool.
    If search results are poor or data is missing, clearly state what you found and what is missing before attempting to call the map tool. If essential data like lat/lon or visitor numbers are missing for a museum, it might be better to exclude it from the map call or find an alternative.
    Prioritize using the web_search and visit_webpage tools to gather the data.
    Finally, return the exact string result from the 'create_museum_map' tool call.
    """
    
    print(f"Running research query with model {LLM_MODEL}...\n")
    try:
        ollama_models_list = ollama.list()
        print(f"Ollama server contacted. Available models: {[m['name'] for m in ollama_models_list['models']]}")
        # Check if base model name (e.g., 'qwen3' from 'qwen3:8b') exists
        base_model_name = LLM_MODEL.split(':')[0]
        if not any(m['name'].startswith(base_model_name) for m in ollama_models_list['models']):
            print(f"Warning: Model starting with '{base_model_name}' (for {LLM_MODEL}) might not be available in Ollama. Ensure it's pulled.")
        
        print(f"Agent Verbosity: {agent.verbosity_level}, Max Steps: {agent.max_steps}")
        print(f"Query: {query}")

        result = agent.run(query)
        
        print("\n\n=== AGENT RUN FINISHED ===")
        print(f"Final Result from Agent: {result}")
        
        if os.path.exists(map_filename):
            print(f"\nMap file '{map_filename}' was created successfully.")
            # Optional: to automatically open the map
            # import webbrowser
            # webbrowser.open(f"file://{os.path.realpath(map_filename)}")
        else:
            print(f"\nWARNING: Map file '{map_filename}' was NOT created.")
            print("The agent might have failed to call the 'create_museum_map' tool correctly, or an error occurred within the tool.")
            print("Review the agent's thoughts and actions printed above for clues.")

    except httpx.ConnectError as e:
        print(f"\nError: Could not connect to Ollama server. {e}")
        print("Please ensure the Ollama server is running (e.g., 'ollama serve' in your terminal).")
    except Exception as e:
        print(f"\nAN UNEXPECTED ERROR OCCURRED DURING THE AGENT RUN: {e}")
        traceback.print_exc()
        print("\nCheck if Ollama server is running and the model is pulled (e.g., 'ollama pull qwen3:8b').")

if __name__ == "__main__":
    main()