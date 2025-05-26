# --- Dependencies --- 
# pip install langchain langchain-core langchain-ollama faiss-cpu sentence-transformers 

import datetime 
import os 
from langchain_ollama import ChatOllama, OllamaEmbeddings 
from langchain.memory import ConversationBufferMemory 
from langchain_community.vectorstores import FAISS 
from langchain.prompts import PromptTemplate 
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel 
from langchain_core.output_parsers import StrOutputParser 
from langchain.schema import Document 

# --- Config ---
FAISS_INDEX_PATH = "my_chatbot_memory_index" # Directory to save/load FAISS index 
# --- Ollama LLM & Embeddings Setup ---
# Run in terminal: ollama pull gemma3 
# Run in terminal: ollama pull nomic-embed-text 
OLLAMA_LLM_MODEL = 'gemma3' 
OLLAMA_EMBED_MODEL = 'nomic-embed-text' # Recommended embedding model for Ollama 

try: 
    llm = ChatOllama(model=OLLAMA_LLM_MODEL) 
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL) 
    print(f"Successfully initialized Ollama: LLM='{OLLAMA_LLM_MODEL}', Embeddings='{OLLAMA_EMBED_MODEL}'") 
    # Optional tests removed for brevity 
except Exception as e: 
    print(f"Error initializing Ollama components: {e}") 
    print(f"Ensure Ollama is running & models pulled (e.g., 'ollama pull {OLLAMA_LLM_MODEL}' and 'ollama pull {OLLAMA_EMBED_MODEL}').") 
    exit() 

# --- Vector Store (Episodic Memory) Setup --- Persisted! 
try: 
    if os.path.exists(FAISS_INDEX_PATH): 
        print(f"Loading existing FAISS index from: {FAISS_INDEX_PATH}") 
        vectorstore = FAISS.load_local( 
            FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True # Required for FAISS loading 
        ) 
        retriever = vectorstore.as_retriever(search_kwargs=dict(k=3)) 
        print("FAISS vector store loaded successfully.") 
    else:
        print(f"No FAISS index found at {FAISS_INDEX_PATH}. Initializing new store.") 
        # FAISS needs at least one text to initialize. 
        vectorstore = FAISS.from_texts( 
            ["Initial conversation context placeholder - Bot created"],
            embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs=dict(k=3))
        # Save the initial empty index
        vectorstore.save_local(FAISS_INDEX_PATH)
        print("New FAISS vector store initialized and saved.")

except Exception as e:
    print(f"Error initializing/loading FAISS: {e}")
    print("Check permissions or delete the index directory if corrupted.")
    exit()

# --- Conversation Buffer (Short-Term) Memory Setup ---
# memory_key must match the input variable in the prompt
# return_messages=True formats history as suitable list of BaseMessages
buffer_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
# <<< ADDED: Clear buffer at the start of each script run >>>
buffer_memory.clear()

# --- Define the Prompt Template ---
# Now includes chat_history for the buffer memory
template = """You are a helpful chatbot assistant with episodic memory (from past sessions) and conversational awareness (from the current session).
Use the following relevant pieces of information:
1. Episodic Memory (Knowledge from *previous* chat sessions):
{semantic_context}

2. Chat History (What we've discussed in the *current* session):
{chat_history}

Combine this information with the current user input to generate a coherent and contextually relevant answer.
If recalling information from Episodic Memory, you can mention it stems from a past conversation if appropriate.
If no relevant context or history is found, just respond naturally to the current input.

Current Input:
User: {input}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["semantic_context", "chat_history", "input"],
    template=template
)

# --- Helper Function for Formatting Retrieved Docs (Episodic Memory) ---
# Formats the retrieved documents (past interactions) for the prompt
def format_retrieved_docs(docs):
    # Simplified formatting: Extract core content only and label explicitly
    formatted = []
    for doc in docs:
        content = doc.page_content
        # Basic check to remove placeholder
        if content not in ["Initial conversation context placeholder - Bot created"]:
             # Attempt to strip "Role (timestamp): " prefix if present
             if "):":
                 content = content.split("):", 1)[-1].strip()
             if content: # Ensure content is not empty after stripping
                formatted.append(f"Recalled from a past session: {content}")
    # Use a double newline to separate recalled memories clearly
    return "\n\n".join(formatted) if formatted else "No relevant memories found from past sessions."


# --- Chain Definition using LCEL ---

# Function to load episodic memory (FAISS context)
def load_episodic_memory(input_dict):
    query = input_dict.get("input", "")
    docs = retriever.invoke(query)
    return format_retrieved_docs(docs)

# Function to save episodic memory (and persist FAISS index)
def save_episodic_memory_step(inputs_outputs):
    user_input = inputs_outputs.get("input", "")
    llm_output = inputs_outputs.get("output", "")

    if user_input and llm_output:
         timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
         docs_to_add = [
             Document(page_content=f"User ({timestamp}): {user_input}"),
             Document(page_content=f"Assistant ({timestamp}): {llm_output}")
         ]
         vectorstore.add_documents(docs_to_add)
         vectorstore.save_local(FAISS_INDEX_PATH) # Persist index after adding
         # print(f"DEBUG: Saved to FAISS index: {FAISS_INDEX_PATH}")
    return inputs_outputs # Pass the dict through for potential further steps


# Define the core chain logic
chain_core = (
    RunnablePassthrough.assign(
        semantic_context=RunnableLambda(load_episodic_memory),
        chat_history=RunnableLambda(lambda x: buffer_memory.load_memory_variables(x)['chat_history'])
    )
    | prompt
    | llm
    | StrOutputParser()
)

# Wrap the core logic to handle memory updates
def run_chain(input_dict):
    user_input = input_dict['input']

    # Invoke the core chain to get the response
    llm_response = chain_core.invoke({"input": user_input})

    # Prepare data for saving
    save_data = {"input": user_input, "output": llm_response}

    # Save to episodic memory (FAISS)
    save_episodic_memory_step(save_data)

    # Save to buffer memory
    buffer_memory.save_context({"input": user_input}, {"output": llm_response})

    return llm_response


# --- Chat Loop ---
print(f"\nChatbot Ready! Using Ollama ('{OLLAMA_LLM_MODEL}' chat, '{OLLAMA_EMBED_MODEL}' embed)")
print(f"Episodic memory stored in: {FAISS_INDEX_PATH}")
print("Type 'quit', 'exit', or 'bye' to end the conversation.")

while True:
    user_text = input("You: ")
    if user_text.lower() in ["quit", "exit", "bye"]:
        # Optionally clear buffer memory on exit if desired
        # buffer_memory.clear()
        print("Chatbot: Goodbye!")
        break
    if not user_text:
        continue

    try:
        # Use the wrapper function to handle the chain invocation and memory updates
        response = run_chain({"input": user_text})
        print(f"Chatbot: {response}")

        # Optional debug: View buffer memory
        # print("DEBUG: Buffer Memory:", buffer_memory.load_memory_variables({}))
        # Optional debug: Check vector store size
        # print(f"DEBUG: Vector store size: {vectorstore.index.ntotal}")

    except Exception as e:
        print(f"\nAn error occurred during the chat chain: {e}")
        # Add more detailed error logging if needed
        import traceback
        print(traceback.format_exc())

# --- End of Script ---
