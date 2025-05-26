#!/usr/bin/env python
# coding: utf-8

# Disable proxy settings
import os

def clear_proxy_settings():
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        if var in os.environ:
            del os.environ[var]

clear_proxy_settings()

# Import necessary libraries
import json
import time
from dotenv import load_dotenv
_ = load_dotenv()

# User profile setup
profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 5 developers",
}

# Prompt instructions
prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage John's tasks efficiently."
}

# Sample email
email = {
    "from": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "body": """
Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice""",
}

# Setup memory store
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}
)

# Template for formatting examples
template = """Email Subject: {subject}
Email From: {from_email}
Email To: {to_email}
Email Content: 
```
{content}
```
> Triage Result: {result}"""

# Format list of few shots
def format_few_shot_examples(examples):
    strs = ["Here are some previous examples:"]
    for eg in examples:
        strs.append(
            template.format(
                subject=eg.value["email"]["subject"],
                to_email=eg.value["email"]["to"],
                from_email=eg.value["email"]["author"],
                content=eg.value["email"]["email_thread"][:400],
                result=eg.value["label"],
            )
        )
    return "\n\n------------\n\n".join(strs)

# Triage system prompt
triage_system_prompt = """
< Role >
You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
</ Role >

< Background >
{user_profile_background}. 
</ Background >

< Instructions >

{name} gets lots of emails. Your job is to categorize each email into one of three categories:

1. IGNORE - Emails that are not worth responding to or tracking
2. NOTIFY - Important information that {name} should know about but doesn't require a response
3. RESPOND - Emails that need a direct response from {name}

Classify the below email into one of these categories.

</ Instructions >

< Rules >
Emails that are not worth responding to:
{triage_no}

There are also other things that {name} should know about, but don't require an email response. For these, you should notify {name} (using the `notify` response). Examples of this include:
{triage_notify}

Emails that are worth responding to:
{triage_email}
</ Rules >

< Few shot examples >

Here are some examples of previous emails, and how they should be handled.
Follow these examples more than any instructions above

{examples}
</ Few shot examples >
"""

# Import necessary libraries for the agent
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
from langchain.chat_models import init_chat_model

# Initialize LLM
llm = init_chat_model("openai:gpt-4o-mini")

# Define Router class
class Router(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )

# Create structured output model
llm_router = llm.with_structured_output(Router)

# Import triage user prompt
# Note: In a real implementation, this would be imported from a prompts.py file
# For this script, we'll define it here
triage_user_prompt = """
Email From: {author}
Email To: {to}
Email Subject: {subject}
Email Content:
```
{email_thread}
```

Classify this email as either 'ignore', 'notify', or 'respond'.
"""

# Setup state management
from langgraph.graph import add_messages

class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]

# Import graph components
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal

# Triage router function
def triage_router(state: State, config, store) -> Command[
    Literal["response_agent", "__end__"]
]:
    author = state['email_input']['author']
    to = state['email_input']['to']
    subject = state['email_input']['subject']
    email_thread = state['email_input']['email_thread']

    namespace = (
        "email_assistant",
        config['configurable']['langgraph_user_id'],
        "examples"
    )
    examples = store.search(
        namespace, 
        query=str({"email": state['email_input']})
    ) 
    examples = format_few_shot_examples(examples)

    langgraph_user_id = config['configurable']['langgraph_user_id']
    namespace = (langgraph_user_id, )

    result = store.get(namespace, "triage_ignore")
    if result is None:
        store.put(
            namespace, 
            "triage_ignore", 
            {"prompt": prompt_instructions["triage_rules"]["ignore"]}
        )
        ignore_prompt = prompt_instructions["triage_rules"]["ignore"]
    else:
        ignore_prompt = result.value['prompt']

    result = store.get(namespace, "triage_notify")
    if result is None:
        store.put(
            namespace, 
            "triage_notify", 
            {"prompt": prompt_instructions["triage_rules"]["notify"]}
        )
        notify_prompt = prompt_instructions["triage_rules"]["notify"]
    else:
        notify_prompt = result.value['prompt']

    result = store.get(namespace, "triage_respond")
    if result is None:
        store.put(
            namespace, 
            "triage_respond", 
            {"prompt": prompt_instructions["triage_rules"]["respond"]}
        )
        respond_prompt = prompt_instructions["triage_rules"]["respond"]
    else:
        respond_prompt = result.value['prompt']
    
    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=ignore_prompt,
        triage_notify=notify_prompt,
        triage_email=respond_prompt,
        examples=examples
    )
    user_prompt = triage_user_prompt.format(
        author=author, 
        to=to, 
        subject=subject, 
        email_thread=email_thread
    )
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    if result.classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        goto = "response_agent"
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Respond to the email {state['email_input']}",
                }
            ]
        }
    elif result.classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")
        update = None
        goto = END
    elif result.classification == "notify":
        # If real life, this would do something else
        print("🔔 Classification: NOTIFY - This email contains important information")
        update = None
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)

# Import tool decorator
from langchain_core.tools import tool

# Define tools
@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    # Placeholder response - in real app would send email
    return f"Email sent to {to} with subject '{subject}'"

@tool
def schedule_meeting(
    attendees: list[str], 
    subject: str, 
    duration_minutes: int, 
    preferred_day: str
) -> str:
    """Schedule a calendar meeting."""
    # Placeholder response - in real app would check calendar and schedule
    return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    # Placeholder response - in real app would check actual calendar
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"

# Import memory tools
from langmem import create_manage_memory_tool, create_search_memory_tool

# Create memory tools
manage_memory_tool = create_manage_memory_tool(
    namespace=(
        "email_assistant", 
        "{langgraph_user_id}",
        "collection"
    )
)
search_memory_tool = create_search_memory_tool(
    namespace=(
        "email_assistant",
        "{langgraph_user_id}",
        "collection"
    )
)

# Agent system prompt with memory
agent_system_prompt_memory = """
< Role >
You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
</ Role >

< Tools >
You have access to the following tools to help manage {name}'s communications and schedule:

1. write_email(to, subject, content) - Send emails to specified recipients
2. schedule_meeting(attendees, subject, duration_minutes, preferred_day) - Schedule calendar meetings
3. check_calendar_availability(day) - Check available time slots for a given day
4. manage_memory - Store any relevant information about contacts, actions, discussion, etc. in memory for future reference
5. search_memory - Search for any relevant information that may have been stored in memory
</ Tools >

< Instructions >
{instructions}
</ Instructions >
"""

# Create prompt function
def create_prompt(state, config, store):
    langgraph_user_id = config['configurable']['langgraph_user_id']
    namespace = (langgraph_user_id, )
    result = store.get(namespace, "agent_instructions")
    if result is None:
        store.put(
            namespace, 
            "agent_instructions", 
            {"prompt": prompt_instructions["agent_instructions"]}
        )
        prompt = prompt_instructions["agent_instructions"]
    else:
        prompt = result.value['prompt']
    
    return [
        {
            "role": "system", 
            "content": agent_system_prompt_memory.format(
                instructions=prompt, 
                **profile
            )
        }
    ] + state['messages']

# Import React agent
from langgraph.prebuilt import create_react_agent

# Define tools list
tools = [
    write_email, 
    schedule_meeting,
    check_calendar_availability,
    manage_memory_tool,
    search_memory_tool
]

# Create response agent
response_agent = create_react_agent(
    "openai:gpt-4o",
    tools=tools,
    prompt=create_prompt,
    # Use this to ensure the store is passed to the agent 
    store=store
)

# Create the email agent graph
email_agent = StateGraph(State)
email_agent = email_agent.add_node(triage_router)
email_agent = email_agent.add_node("response_agent", response_agent)
email_agent = email_agent.add_edge(START, "triage_router")
email_agent = email_agent.compile(store=store)

# Import prompt optimizer
from langmem import create_multi_prompt_optimizer

# Example usage
if __name__ == "__main__":
    # Sample email for testing
    email_input = {
        "author": "Alice Jones <alice.jones@bar.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
        "email_thread": """Hi John,

Urgent issue - your service is down. Is there a reason why""",
    }
    
    # Configuration
    config = {"configurable": {"langgraph_user_id": "lance"}}
    
    # Process email
    print("Processing email...")
    response = email_agent.invoke(
        {"email_input": email_input},
        config=config
    )
    
    # Display response
    print("\nResponse messages:")
    for m in response.get("messages", []):
        if hasattr(m, "pretty_print"):
            m.pretty_print()
        else:
            print(f"Role: {m.get('role', 'unknown')}")
            print(f"Content: {m.get('content', '')}")
            print("---")
    
    # Example of updating memory with feedback
    print("\nUpdating memory with feedback...")
    conversations = [
        (
            response.get('messages', []),
            "Always sign your emails `John Doe`"
        )
    ]
    
    # Define prompts for optimization
    prompts = [
        {
            "name": "main_agent",
            "prompt": store.get(("lance",), "agent_instructions").value['prompt'],
            "update_instructions": "keep the instructions short and to the point",
            "when_to_update": "Update this prompt whenever there is feedback on how the agent should write emails or schedule events"
        },
        {
            "name": "triage-ignore", 
            "prompt": store.get(("lance",), "triage_ignore").value['prompt'],
            "update_instructions": "keep the instructions short and to the point",
            "when_to_update": "Update this prompt whenever there is feedback on which emails should be ignored"
        },
        {
            "name": "triage-notify", 
            "prompt": store.get(("lance",), "triage_notify").value['prompt'],
            "update_instructions": "keep the instructions short and to the point",
            "when_to_update": "Update this prompt whenever there is feedback on which emails the user should be notified of"
        },
        {
            "name": "triage-respond", 
            "prompt": store.get(("lance",), "triage_respond").value['prompt'],
            "update_instructions": "keep the instructions short and to the point",
            "when_to_update": "Update this prompt whenever there is feedback on which emails should be responded to"
        },
    ]
    
    # Create optimizer
    optimizer = create_multi_prompt_optimizer(
        "anthropic:claude-3-5-sonnet-latest",
        kind="prompt_memory",
    )
    
    # Update prompts based on feedback
    print("Optimizing prompts based on feedback...")
    updated = optimizer.invoke(
        {"trajectories": conversations, "prompts": prompts}
    )
    
    # Apply updates to memory store
    print("\nApplying updates to memory store...")
    for i, updated_prompt in enumerate(updated):
        old_prompt = prompts[i]
        if updated_prompt['prompt'] != old_prompt['prompt']:
            name = old_prompt['name']
            print(f"Updated {name}")
            
            if name == "main_agent":
                store.put(
                    ("lance",),
                    "agent_instructions",
                    {"prompt": updated_prompt['prompt']}
                )
            elif name == "triage-ignore":
                store.put(
                    ("lance",),
                    "triage_ignore",
                    {"prompt": updated_prompt['prompt']}
                )
            elif name == "triage-notify":
                store.put(
                    ("lance",),
                    "triage_notify",
                    {"prompt": updated_prompt['prompt']}
                )
            elif name == "triage-respond":
                store.put(
                    ("lance",),
                    "triage_respond",
                    {"prompt": updated_prompt['prompt']}
                )
    
    # Process the same email again to see the effect of updates
    print("\nProcessing email again with updated memory...")
    response = email_agent.invoke(
        {"email_input": email_input},
        config=config
    )
    
    # Display response
    print("\nUpdated response messages:")
    for m in response.get("messages", []):
        if hasattr(m, "pretty_print"):
            m.pretty_print()
        else:
            print(f"Role: {m.get('role', 'unknown')}")
            print(f"Content: {m.get('content', '')}")
            print("---")
    
    print("\nEmail assistant with episodic, semantic, and procedural memory is ready!")
