#!/usr/bin/env python
# coding: utf-8

from qwen_memory import QwenAgent
import sys
import os

def test_memory_persistence():
    """Test if memory persists across sessions by starting a fresh conversation."""
    # Open file for writing
    with open("persistence_results.txt", "w") as f:
        # Redirect stdout to file
        original_stdout = sys.stdout
        sys.stdout = f

        print("===== TESTING MEMORY PERSISTENCE ACROSS SESSIONS =====")
        
        # Create a new agent instance (simulating a new chat session)
        agent = QwenAgent(model_name="qwen3:8b", embedding_model_name="nomic-embed-text")
        
        # Check the existing memory in ChromaDB
        print(f"ChromaDB collection count: {agent.db.chroma_collection.count()}")
        
        # First, directly search the memory to see what's available
        print("\n=== Direct Memory Search Before Query ===")
        query = "DNS button modem"
        memories = agent.recall(query)
        print(f"Retrieved {len(memories)} memories for query: '{query}'")
        for i, memory in enumerate(memories):
            print(f"\nMemory {i+1}:")
            print(f"  ID: {memory.get('id', 'N/A')}")
            print(f"  Type: {memory.get('memory_type', 'N/A')}")
            print(f"  Score: {memory.get('relevance_score', 0.0):.4f}")
            print(f"  Content: {memory['content']}")
            
        # Test reflection on memories manually
        if memories:
            print("\n=== Reflection on Memories ===")
            reflection = agent._reflect_on_memories(query, memories)
            print(reflection)
            
            if reflection:
                print("\n=== Extracted Insights ===")
                insights = agent._extract_insights(reflection, query)
                print(insights)
        
        # Start a completely new conversation
        print("\n=== NEW SESSION: Testing Memory Persistence ===")
        agent.start_conversation("Memory persistence test")
        
        # Ask a question similar to the correction in the previous test
        dns_query = "The DNS buttons on my modem aren't working as they usually do. Why?"
        print(f"User: {dns_query}")
        response = agent.chat(dns_query)
        print(f"Agent: {response['content']}")
        
        # Show insights if any were used
        if response.get('insights'):
            print("\nInsights applied in response:")
            print(response['insights'])
        else:
            print("\nNo insights were applied in the response")
            
        # Close agent
        agent.close()
        
        # Restore original stdout
        sys.stdout = original_stdout
        print(f"Test complete. Results saved to persistence_results.txt")

if __name__ == "__main__":
    test_memory_persistence() 