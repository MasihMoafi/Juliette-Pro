
#DISABLE PROXY
import os

def clear_proxy_settings():
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        if var in os.environ:
            del os.environ[var]

clear_proxy_settings()

import ollama
import chromadb
import sqlite3
import uuid
import time
import json
import re
import os
from typing import List, Dict, Any, Optional

# --- Configuration ---
LLM_MODEL = 'qwen3:8b'
LLM_EMBEDDING_MODEL = 'nomic-embed-text'
SYSTEM_MESSAGE = """You are a helpful AI assistant. Please keep your responses concise. If you use information learned from previous interactions (provided as 'IMPORTANT CONTEXT UPDATE'), briefly acknowledge this."""
LOG_FILE_PATH = "chat_log_chroma_persistent.txt" # Changed log file name for clarity
CHROMA_PERSIST_PATH_DEFAULT = "./chroma_db_persist_concise" # Keep this for Chroma
SQLITE_DB_PATH_DEFAULT = "qwen_memory_chroma_concise.db"   # Keep this for SQLite

# --- Utility function for logging ---
def log_interaction_to_file(user_query: str, agent_response: str):
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(f"User Query:\n{user_query}\n\nAgent Response:\n{agent_response}\n\n{'='*50}\n\n")
    except Exception as e:
        print(f"Error writing to log file {LOG_FILE_PATH}: {e}")

class MemoryDB:
    def __init__(self, sqlite_db_path=SQLITE_DB_PATH_DEFAULT, chroma_persist_path=CHROMA_PERSIST_PATH_DEFAULT, collection_name="semantic_memories"):
        self.sqlite_db_path = sqlite_db_path
        self.sqlite_conn = None
        self._connect_sqlite_db() # Connects to SQLite, ensuring persistence

        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_path)
        self.chroma_collection_name = collection_name
        try:
            # This will load the existing collection if the path and name match
            self.chroma_collection = self.chroma_client.get_or_create_collection(name=self.chroma_collection_name)
            print(f"ChromaDB collection '{self.chroma_collection_name}' loaded/created. Count: {self.chroma_collection.count()}")
        except Exception as e:
            print(f"Fatal Error: Could not initialize ChromaDB: {e}")
            raise

    def _get_embedding(self, text: str, model: str = LLM_EMBEDDING_MODEL) -> Optional[List[float]]:
        try:
            if not text or text.isspace(): return None
            response = ollama.embeddings(model=model, prompt=text)
            return response["embedding"]
        except Exception as e:
            print(f"Error generating embedding for text '{text[:50]}...': {e}")
            return None

    # --- MODIFIED FOR PERSISTENCE ---
    def _connect_sqlite_db(self):
        # For persistent memory, we don't want to rename/backup the DB on every startup.
        # We want to connect to the existing one if it's there.
        # A separate backup mechanism could be implemented if needed (e.g., a manual function or periodic).
        
        # print(f"Connecting to SQLite DB: {self.sqlite_db_path}") # Optional: for debugging
        db_exists = os.path.exists(self.sqlite_db_path)

        try:
            self.sqlite_conn = sqlite3.connect(self.sqlite_db_path)
            self.sqlite_conn.execute("PRAGMA foreign_keys = ON;")
            if not db_exists:
                print(f"SQLite DB not found at {self.sqlite_db_path}, creating new one.")
            self._create_sqlite_tables() # This will create tables IF THEY DON'T ALREADY EXIST
        except sqlite3.Error as e:
            print(f"Fatal Error: Could not connect/initialize SQLite DB '{self.sqlite_db_path}': {e}")
            raise
    # --- END OF MODIFICATION ---

    def _create_sqlite_tables(self): # No change here, CREATE TABLE IF NOT EXISTS is correct
        if not self.sqlite_conn: return
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY, content TEXT NOT NULL, memory_type TEXT NOT NULL,
                importance REAL DEFAULT 0.5, metadata TEXT, created_at REAL NOT NULL, last_accessed REAL NOT NULL
            )''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY, title TEXT, created_at REAL NOT NULL, last_updated REAL NOT NULL
            )''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, role TEXT NOT NULL,
                content TEXT NOT NULL, created_at REAL NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
            )''')
            self.sqlite_conn.commit()
        except sqlite3.Error as e:
            print(f"SQLite error during table creation: {e}")

    def add_memory(self, content: str, memory_type: str = "semantic", importance: float = 0.5, metadata: Optional[Dict] = None) -> str:
        if not self.sqlite_conn or not self.chroma_collection:
            print("Error: DB not fully initialized for adding memory.")
            return ""

        memory_id = str(uuid.uuid4())
        current_time = time.time()
        metadata_json = json.dumps(metadata or {})

        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute(
                "INSERT INTO memories (id, content, memory_type, importance, metadata, created_at, last_accessed) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (memory_id, content, memory_type, importance, metadata_json, current_time, current_time)
            )
            self.sqlite_conn.commit()
        except sqlite3.Error as e:
            print(f"SQLite error in add_memory: {e}")
            # If SQLite fails, we probably shouldn't add to Chroma either, or handle inconsistency.
            return ""

        embedding = self._get_embedding(content)
        if embedding:
            try:
                chroma_meta = {"memory_type": memory_type, "importance": importance, "created_at_ts": current_time}
                if metadata:
                    for k, v in metadata.items():
                        if isinstance(v, (str, int, float, bool)): chroma_meta[k] = v
                self.chroma_collection.add(ids=[memory_id], embeddings=[embedding], documents=[content], metadatas=[chroma_meta])
                return memory_id
            except Exception as e:
                print(f"ChromaDB error in add_memory for ID {memory_id}: {e}")
                # Memory is in SQLite, but not Chroma. This is an inconsistency.
                # For robustness, one might consider removing from SQLite or marking as unindexed.
                return memory_id # Still return ID as it's in SQLite for now
        else:
            print(f"Failed to generate embedding for memory ID {memory_id}. Not added to ChromaDB.")
            return memory_id # In SQLite, but not Chroma

    def update_memory_access(self, memory_id: str):
        if not self.sqlite_conn: return
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("UPDATE memories SET last_accessed = ? WHERE id = ?", (time.time(), memory_id))
            self.sqlite_conn.commit()
        except sqlite3.Error as e:
            print(f"SQLite error updating access time for memory {memory_id}: {e}")

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        if not self.sqlite_conn: return None
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("SELECT id, content, memory_type, importance, metadata, created_at, last_accessed FROM memories WHERE id = ?", (memory_id,))
            result = cursor.fetchone()
            if not result: return None
            self.update_memory_access(memory_id)
            metadata_dict = json.loads(result[4] or '{}')
            return {"id": result[0], "content": result[1], "memory_type": result[2], "importance": result[3],
                    "metadata": metadata_dict, "created_at": result[5], "last_accessed": result[6]}
        except (sqlite3.Error, json.JSONDecodeError) as e:
            print(f"Error getting/parsing memory {memory_id}: {e}")
            return None

    def search_memories(self, query: str, limit: int = 5, max_distance_threshold: Optional[float] = 1.5) -> List[Dict[str, Any]]:
        if not self.chroma_collection or not query: return []
        query_embedding = self._get_embedding(query)
        if not query_embedding: return []

        try:
            results = self.chroma_collection.query(query_embeddings=[query_embedding], n_results=limit)
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return []

        memories = []
        if results and results.get('ids') and results['ids'][0]:
            for i, memory_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i] if results.get('distances') and results['distances'][0] else float('inf')
                if max_distance_threshold is not None and distance > max_distance_threshold:
                    continue
                memory = self.get_memory(memory_id) # Fetch full data from SQLite
                if memory:
                    memory["relevance_score"] = 1.0 / (1.0 + distance) if distance is not None else 0.0
                    memories.append(memory)
                # else:
                    # print(f"Memory ID {memory_id} found in Chroma but not in SQLite. Possible data inconsistency.")
        return memories

    def add_conversation(self, title: str = "") -> str: # No change
        if not self.sqlite_conn: return ""
        conversation_id = str(uuid.uuid4())
        current_time = time.time()
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("INSERT INTO conversations (id, title, created_at, last_updated) VALUES (?, ?, ?, ?)",
                           (conversation_id, title, current_time, current_time))
            self.sqlite_conn.commit()
            return conversation_id
        except sqlite3.Error as e:
            print(f"SQLite error adding conversation: {e}")
            return ""

    def add_message(self, conversation_id: str, role: str, content: str) -> str: # No change
        if not self.sqlite_conn: return ""
        message_id = str(uuid.uuid4())
        current_time = time.time()
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                           (message_id, conversation_id, role, content, current_time))
            cursor.execute("UPDATE conversations SET last_updated = ? WHERE id = ?", (current_time, conversation_id))
            self.sqlite_conn.commit()
            return message_id
        except sqlite3.Error as e:
            print(f"SQLite error adding message to conversation {conversation_id}: {e}")
            return ""

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]: # No change
        if not self.sqlite_conn: return None
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("SELECT id, title, created_at, last_updated FROM conversations WHERE id = ?", (conversation_id,))
            conv_result = cursor.fetchone()
            if not conv_result: return None
            conversation = {"id": conv_result[0], "title": conv_result[1], "created_at": conv_result[2],
                            "last_updated": conv_result[3], "messages": []}
            cursor.execute("SELECT id, role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC", (conversation_id,))
            for msg in cursor.fetchall():
                conversation["messages"].append({"id": msg[0], "role": msg[1], "content": msg[2], "created_at": msg[3]})
            return conversation
        except sqlite3.Error as e:
             print(f"SQLite error getting conversation {conversation_id}: {e}")
             return None

    def close_db(self): # No change
        if self.sqlite_conn:
            self.sqlite_conn.close()
            self.sqlite_conn = None
            print("SQLite connection closed.")

class QwenAgent: # No changes needed in QwenAgent itself for this fix
    def __init__(self, model_name=LLM_MODEL, embedding_model_name=LLM_EMBEDDING_MODEL):
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        # MemoryDB now handles persistence correctly due to the change in _connect_sqlite_db
        self.db = MemoryDB()
        self.conversation_id = None

    def start_conversation(self, title=""):
        self.conversation_id = self.db.add_conversation(title)
        print(f"Started new conversation: {self.conversation_id} (Title: '{title}')")
        return self.conversation_id

    def _add_message_to_history(self, message: Dict[str, str]):
        if not self.conversation_id:
            self.start_conversation("Implicit Conversation") # Auto-start if no active conversation
        self.db.add_message(self.conversation_id, message["role"], message["content"])

    def _get_conversation_history(self) -> List[Dict[str, str]]:
        if not self.conversation_id: return []
        conversation = self.db.get_conversation(self.conversation_id)
        if not conversation: return []
        return [{"role": msg["role"], "content": msg["content"]} for msg in conversation["messages"]]

    def recall(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        return self.db.search_memories(query, limit=limit, max_distance_threshold=1.5)

    def _reflect_on_memories(self, query: str, memories: List[Dict[str, Any]]) -> Optional[str]:
        if not memories: return None
        memory_texts = [
            (f"Memory {i+1} (ID: {mem.get('id', 'N/A')[:8]}, Type: {mem.get('memory_type', 'N/A')}, "
             f"Score: {mem.get('relevance_score', 0.0):.2f}):\n{mem['content']}")
            for i, mem in enumerate(memories)
        ]
        memories_str = "\n\n".join(memory_texts)
        reflection_prompt = f"USER_QUERY: \"{query}\"\n\nRETRIEVED_MEMORIES:\n{memories_str}\n\nTask: Analyze memories for relevance to USER_QUERY. Extract contradictions, corrections, or key facts. Be concise. Conclude with 'Key takeaways for current query:'."
        system_message = {"role": "system", "content": "You are a reflective AI. Analyze memories strictly for relevance to the current query. Be concise."}
        try:
            response = ollama.chat(model=self.model_name, messages=[system_message, {"role": "user", "content": reflection_prompt}], stream=False)
            return response['message']['content']
        except Exception as e:
            print(f"Error during reflection: {e}")
            return None

    def _extract_insights(self, reflection: str, query: str) -> Optional[str]:
        if not reflection: return None
        extraction_prompt = f"USER_QUERY: \"{query}\"\n\nREFLECTION:\n{reflection}\n\nTask: Extract 1-3 most CRITICAL facts, corrections, or answers relevant to USER_QUERY from REFLECTION. Numbered list. Concise. If none, state 'No critical insights derived.'."
        system_message = {"role": "system", "content": "You extract critical, concise insights from reflections, strictly relevant to the current query."}
        try:
            response = ollama.chat(model=self.model_name, messages=[system_message, {"role": "user", "content": extraction_prompt}], stream=False)
            extracted = response['message']['content']
            return None if "No critical insights derived." in extracted else extracted
        except Exception as e:
            print(f"Error during insight extraction: {e}")
            return None

    def _detect_correction(self, message: str) -> bool:
        correction_indicators = [
            "actually", "in fact", "the truth is", "correction:", "my mistake", "i was wrong",
            "that's not right", "that's incorrect", "no,"
        ]
        message_lower = message.lower()
        for indicator in correction_indicators:
            if indicator in message_lower:
                if not any(f"{neg}{indicator}" in message_lower for neg in ["not ", "isn't "]):
                     return True
        return False

    def _save_to_memory(self, conversation_history: List[Dict[str, str]], insights: Optional[str] = None):
        if not conversation_history or len(conversation_history) < 2: return

        current_turn_messages = conversation_history[-2:]
        turn_content = "\n".join([f"{msg['role']}: {msg['content']}" for msg in current_turn_messages])
        self.db.add_memory(content=turn_content, memory_type="episodic_turn", importance=0.6,
                           metadata={"conversation_id": self.conversation_id})

        last_user_message_content = current_turn_messages[0]['content']
        if self._detect_correction(last_user_message_content):
            self.db.add_memory(content=last_user_message_content, memory_type="semantic_correction", importance=0.95,
                               metadata={"conversation_id": self.conversation_id, "source": "user_correction"})

        if insights and "No critical insights derived." not in insights:
            self.db.add_memory(content=insights, memory_type="semantic_insight", importance=0.8,
                               metadata={"conversation_id": self.conversation_id, "source": "ai_extracted_insight"})

    def chat(self, message: str) -> Dict[str, Any]:
        # If no conversation is active, start one. This is useful for interactive mode.
        if not self.conversation_id:
            self.start_conversation("Interactive Session")
            
        user_message_dict = {"role": "user", "content": message}
        self._add_message_to_history(user_message_dict)

        current_history = self._get_conversation_history()
        memories = self.recall(message, limit=5)

        reflection, insights = None, None
        if memories:
            print(f"Retrieved {len(memories)} memories for query: '{message}'") # Debug: show retrieved memories
            # for mem_idx, mem_item in enumerate(memories):
            #    print(f"  Mem {mem_idx+1} (Score: {mem_item.get('relevance_score',0):.2f}): {mem_item.get('content', '')[:100]}...")
            reflection = self._reflect_on_memories(message, memories)
            if reflection:
                # print(f"Reflection: {reflection}") # Debug: show reflection
                insights = self._extract_insights(reflection, message)
                # if insights: print(f"Insights: {insights}") # Debug: show insights

        llm_messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
        if insights:
            directive = f"IMPORTANT CONTEXT UPDATE:\n{insights}\nAddress user query: '{message}' using this."
            llm_messages.insert(1, {"role": "system", "content": directive})
        llm_messages.extend(current_history)

        assistant_response_content = ""
        try:
            response = ollama.chat(model=self.model_name, messages=llm_messages, stream=False)
            assistant_response_content = response['message']['content']
            assistant_message_dict = {"role": "assistant", "content": assistant_response_content}
            self._add_message_to_history(assistant_message_dict)
            self._save_to_memory(current_history + [assistant_message_dict], insights)

            result = {"content": assistant_response_content}
            if insights: result["insights"] = insights
            log_interaction_to_file(user_query=message, agent_response=assistant_response_content)
            return result
        except Exception as e:
            print(f"Error during LLM call: {e}")
            error_msg = "Error processing request."
            if hasattr(e, 'status_code') and e.status_code == 404:
                 error_msg = f"Model '{self.model_name}' or '{self.embedding_model_name}' not found. Pull via Ollama."
            self._add_message_to_history({"role": "assistant", "content": error_msg})
            log_interaction_to_file(user_query=message, agent_response=error_msg)
            return {"content": error_msg}

    def close(self):
        self.db.close_db()

# --- Main Execution Block for Interactive Chat (using the corrected MemoryDB) ---
if __name__ == "__main__":
    # Clear the log file for a fresh session log, or append if desired
    if os.path.exists(LOG_FILE_PATH):
        try: 
            # os.remove(LOG_FILE_PATH) # Uncomment to clear log on each start
            print(f"Log file '{LOG_FILE_PATH}' will be appended to or created.")
        except Exception as e: print(f"Warning: Could not clear log file: {e}")

    try:
        ollama.list()
        print(f"Using chat model: {LLM_MODEL}, embedding model: {LLM_EMBEDDING_MODEL}")
    except Exception as e:
        print(f"Ollama connection error: {e}. Ensure Ollama is running and models are pulled.\n  ollama pull {LLM_MODEL}\n  ollama pull {LLM_EMBEDDING_MODEL}")
        exit(1)

    # The agent will now use the default persistent paths defined at the top
    # and the modified MemoryDB._connect_sqlite_db will ensure data is loaded.
    print(f"Attempting to load/create SQLite DB at: {SQLITE_DB_PATH_DEFAULT}")
    print(f"Attempting to load/create ChromaDB at: {CHROMA_PERSIST_PATH_DEFAULT}")
    agent = QwenAgent()
    print("Memory system ready. Previous memories should be loaded if they exist.")
    print("Type 'quit' to exit.")
    
    # Check current memory count in Chroma
    try:
        initial_chroma_count = agent.db.chroma_collection.count()
        print(f"Initial ChromaDB memory count: {initial_chroma_count}")
    except Exception as e:
        print(f"Could not get initial ChromaDB count: {e}")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        if not user_input.strip(): # Skip empty input
            continue
            
        response = agent.chat(user_input)
        print(f"Assistant: {response['content']}")
        if response.get('insights'):
            print(f"Insights applied: {response['insights']}")

    agent.close()
    print("Chat session ended.")