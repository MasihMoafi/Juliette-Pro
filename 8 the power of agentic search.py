import os
import httpx
import json
from ollama import chat

DOCS_DIR = "docs"
LLM_MODEL = "gemma3"  # Using gemma3 which has shown better results

# Load documents from the 'docs' directory
def load_documents():
    """Loads all .txt files from the 'docs' directory and returns a list of (filename, content) tuples."""
    if not os.path.exists(DOCS_DIR):
        return []
    documents = []
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(DOCS_DIR, filename), "r", encoding="utf-8") as f:
                content = f.read()
                documents.append((filename, content))
    return documents

DOCUMENTS = load_documents()

def search_docs(query):
    """Searches local documents for relevant information based on the query."""
    if not DOCUMENTS:
        return "No documents found."
    query_words = query.lower().split()
    results = []
    for filename, content in DOCUMENTS:
        lines = content.split('\n')
        for line in lines:
            if any(word in line.lower() for word in query_words):
                results.append(f"From {filename}: {line[:200]}")
                if len(results) >= 3:
                    break
        if len(results) >= 3:
            break
    return "\n".join(results) if results else "No relevant information found in documents."

def wikipedia_search(query):
    """Searches Persian Wikipedia for a summary based on the query."""
    response = httpx.get("https://fa.wikipedia.org/w/api.php", params={
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": 1
    })
    results = response.json().get("query", {}).get("search", [])
    if results:
        return results[0].get("snippet", "اطلاعاتی یافت نشد.")
    return "اطلاعاتی یافت نشد در ویکی‌پدیا."

def duckduckgo_search(query):
    """Searches DuckDuckGo for web results based on the query."""
    response = httpx.get("https://duckduckgo.com/html/", params={"q": query})
    if response.status_code == 200:
        # Basic HTML parsing for DuckDuckGo results
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            for result in soup.select('.result')[:3]:
                title_elem = result.select_one('.result__title')
                snippet_elem = result.select_one('.result__snippet')
                
                title = title_elem.get_text() if title_elem else "No title"
                snippet = snippet_elem.get_text() if snippet_elem else "No snippet"
                
                results.append(f"{title}: {snippet}")
            
            return "\n".join(results) if results else "No results found on DuckDuckGo."
        except Exception as e:
            return f"نتایج جستجوی DuckDuckGo: [صفحه 1, صفحه 2, صفحه 3] (Error parsing: {str(e)})"
    return "نتایجی از DuckDuckGo یافت نشد."

def process_query(question):
    """Processes a query using multiple information sources and Ollama with intelligent decision making."""
    print(f"Processing query: {question}")
    
    # Gather information from different sources
    docs_results = search_docs(question)
    wiki_results = wikipedia_search(question)
    ddg_results = duckduckgo_search(question)
    
    # Combine information for the LLM with intelligent reasoning instructions
    context = f"""Here's information from different sources to help answer the question:

Document Search Results:
{docs_results}

Wikipedia Search Results:
{wiki_results}

DuckDuckGo Search Results:
{ddg_results}
"""
    
    # Full prompt with context and explicit reasoning instructions
    prompt = f"""You are an intelligent assistant with extensive knowledge. You're given the following information gathered from various sources to help answer a user's question.

[INFORMATION SOURCES]
{context}
[END INFORMATION SOURCES]

Please follow these steps to provide the best answer:

1. ANALYZE: Carefully evaluate the quality and relevance of the information from each source.
2. COMPARE: Compare this information with your own knowledge about the topic.
3. DECIDE: If the search results provide accurate and relevant information, use them. If your knowledge is more accurate or complete, rely on that instead.
4. RESPOND: Provide a comprehensive, accurate answer in Persian (Farsi) language ONLY. Your response should be well-structured and directly address the user's question.

IMPORTANT: You MUST respond ONLY in Persian (Farsi). DO NOT use any English in your response.

Question: {question}

Your response (in Persian/Farsi ONLY):"""
    
    # Call Ollama directly with improved prompt
    response = chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}])
    result = response['message']['content']
    
    print(f"Answer generated.")
    save_to_file(question, result)
    return result

def save_to_file(question, answer, path="/home/masih/Desktop/memory new/agents.txt"):
    """Saves the result to a file."""
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f"سوال:\n{question}\n\nپاسخ:\n{answer}\n\n{'='*50}\n\n")

if __name__ == "__main__":
    # Example queries to test the system
    q1 = "شاه عباس صفوی آدم خوبی بوده؟ چرا؟"
    q2 = "وقتی چراغ DSL مودم قطع میشه به چه معنیه؟"
    
    process_query(q1)
    process_query(q2)