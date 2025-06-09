# Code Agent Project using Hugging Face smolagents (Local Ollama Setup)

This project is an implementation and exploration of concepts from the DeepLearning.AI short course: "[Building Code Agents with Hugging Face smolagents](https://www.deeplearning.ai/short-courses/building-code-agents-with-hugging-face-smolagents/)", created in collaboration with Hugging Face. It has been specifically configured to run with a local LLM via Ollama.

## Overview

The project focuses on "Code Agents," a novel approach to agentic systems where Large Language Models (LLMs) generate a single block of code encompassing a complete plan of action. This contrasts with traditional tool-calling agents that make sequential function calls. The `smolagents` framework by Hugging Face, particularly leveraging `smolagents.CodeAgent`, is utilized here to build and manage these code agents.

The core idea is to leverage LLMs to write code that, when executed, performs complex tasks. This method aims for more efficient and reliable task completion, and this repository demonstrates how to achieve this with a locally hosted model.

## Key Concepts & Features

This repository demonstrates or allows for the exploration of:

*   **Code Agent Architecture:** Understanding how LLMs can generate entire scripts as actionable plans.
*   **`smolagents` Framework:** Utilizing Hugging Face's lightweight framework, specifically `smolagents.CodeAgent`.
*   **Local LLM Integration:** Custom integration with Ollama to use locally hosted language models (e.g., `qwen3:8b`).
*   **Secure Code Execution:** Implementing or exploring techniques for safely running LLM-generated code.
*   **Task Handling:** Building agents capable of tasks such as web browsing, data extraction, and visualization.
*   **Agent Monitoring & Evaluation:** Methods to trace, debug, and assess code agent behavior.

## Local Setup and Configuration

This project requires a specific local setup to run correctly, primarily centered around using Ollama for local LLM hosting.

1.  **Install Ollama:**
    *   Download and install Ollama from [ollama.com](https://ollama.com/).
    *   Ensure the Ollama server is running (e.g., by executing `ollama serve` in your terminal if it's not already running as a service).

2.  **Pull the LLM Model:**
    *   The default model used in `Museum_map_local.py` is `qwen3:8b`. Pull this model using the Ollama CLI:
        ```bash
        ollama pull qwen3:8b
        ```
    *   You can modify `LLM_MODEL` in the script to use other Ollama-hosted models, but ensure they are pulled first.

3.  **Python Environment and Dependencies:**
    *   It's highly recommended to use a Python virtual environment.
    *   Install the necessary Python libraries. Create a `requirements.txt` file with the following content:
        ```txt
        smolagents
        ollama
        pandas
        plotly
        httpx
        beautifulsoup4
        numpy
        # Add other potential implicit dependencies if discovered during use
        ```
    *   Then install them:
        ```bash
        pip install -r requirements.txt
        ```
    *   **Note on `numpy`**: While not explicitly imported at the top level in `Museum_map_local.py` by the user, it's listed in `additional_authorized_imports` for the `CodeAgent` and libraries like `pandas` depend on it. Including it in `requirements.txt` is good practice.

4.  **Proxy Configuration:**
    *   The script `Museum_map_local.py` includes a function (`clear_proxy_settings()`) that attempts to disable system proxy settings (`HTTP_PROXY`, `HTTPS_PROXY`). This is to prevent interference with local network requests, especially to the Ollama server. If you are in a corporate environment with mandatory proxies, this might require further adjustment or disabling this function if it causes issues with legitimate external access needed by tools.

5.  **Custom Ollama Model Class (`OllamaModel`):**
    *   `Museum_map_local.py` contains a custom `OllamaModel` class. This class acts as a bridge, adapting the `ollama` library's chat interface to be compatible with the `smolagents.CodeAgent`'s expectations for a model. This is a key part of the local model integration.

6.  **Tool Fallbacks (Mock Data):**
    *   The `web_search` and `visit_webpage` tools in `Museum_map_local.py` have built-in fallbacks that return mock data if live internet requests fail. This is useful for testing or running in environments with limited/no internet access, but be aware that the agent will operate on this mock data in such scenarios.

## Running the Example

Once the setup is complete:
1.  Navigate to the repository directory.
2.  Execute the main script:
    ```bash
    python Museum_map_local.py
    ```
    The script will attempt to find the top 3 most visited museums, gather data, and then generate an HTML map file (`saved_map.html`) using Plotly.

## Further Exploration

*   Explore the official **`smolagents` GitHub repository**: [https://github.com/huggingface/smolagents](https://github.com/huggingface/smolagents)
*   Refer to the **`smolagents` documentation**, specifically for `CodeAgent`: [https://huggingface.co/docs/smolagents/reference/agents#smolagents.CodeAgent](https://huggingface.co/docs/smolagents/reference/agents#smolagents.CodeAgent)
*   Review the **DeepLearning.AI course materials**: "[Building Code Agents with Hugging Face smolagents](https://www.deeplearning.ai/short-courses/building-code-agents-with-hugging-face-smolagents/)"

---

*This README has been tailored to the specifics found in `Museum_map_local.py`, emphasizing local Ollama setup. Adjust model names or Python dependencies as needed if you modify the scripts.*
