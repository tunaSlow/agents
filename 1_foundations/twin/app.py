
import os
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from context import TWIN_SYSTEM_PROMPT
from styles import CSS, EXAMPLES, JS
from tools import handle_tool_calls, tools

load_dotenv(override=True)

# Standard Groq model target
MODEL_NAME = "openai/gpt-oss-120b"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

groq_client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)


def sanitize_message(msg):
    """
    Strips internal fields (e.g. 'metadata', 'refusal') from message objects/dicts
    to ensure strict compatibility with Groq's API schema.
    """
    if hasattr(msg, "model_dump"):
        msg = msg.model_dump(exclude_none=True)
    elif hasattr(msg, "dict"):
        msg = msg.dict(exclude_none=True)
    elif not isinstance(msg, dict):
        return None

    role = msg.get("role")
    cleaned = {"role": role}

    # Extract standard payload attributes
    if "content" in msg and msg["content"] is not None:
        cleaned["content"] = msg["content"]

    if role == "assistant" and msg.get("tool_calls"):
        cleaned["tool_calls"] = msg["tool_calls"]
    elif role == "tool":
        if "tool_call_id" in msg:
            cleaned["tool_call_id"] = msg["tool_call_id"]
        if "name" in msg:
            cleaned["name"] = msg["name"]

    return cleaned


def chat(user_message, history):
    messages = [{"role": "system", "content": TWIN_SYSTEM_PROMPT}]

    # Reconstruct history into clean OpenAI/Groq message format
    for turn in history:
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            u_text, a_text = turn
            if u_text:
                messages.append({"role": "user", "content": u_text})
            if a_text:
                messages.append({"role": "assistant", "content": a_text})
        elif isinstance(turn, dict):
            cleaned_turn = sanitize_message(turn)
            if cleaned_turn:
                messages.append(cleaned_turn)

    messages.append({"role": "user", "content": user_message})

    # Initial Agent Invocation
    response = groq_client.chat.completions.create(
        model=MODEL_NAME, messages=messages, tools=tools
    )

    # Tool Execution Loop with Max-Iteration Safety Guard
    max_iterations = 5
    iterations = 0

    while response.choices[0].finish_reason == "tool_calls" and iterations < max_iterations:
        iterations += 1
        assistant_msg = response.choices[0].message
        tool_calls = assistant_msg.tool_calls

        # Execute local tool definitions
        tool_results = handle_tool_calls(tool_calls)

        # Append cleaned assistant reasoning and tool output payload
        messages.append(sanitize_message(assistant_msg))
        for res in tool_results:
            cleaned_res = sanitize_message(res)
            if cleaned_res:
                messages.append(cleaned_res)

        # Re-query model with formatted tool output
        response = groq_client.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=tools
        )

    return response.choices[0].message.content


if __name__ == "__main__":
    gr.ChatInterface(
        chat,
        examples=EXAMPLES,
        title="Digital Twin",
        description="Talk to my AI twin about my career",
        chatbot=gr.Chatbot(show_label=False),
    ).launch(css=CSS, js=JS, theme=gr.themes.Base())