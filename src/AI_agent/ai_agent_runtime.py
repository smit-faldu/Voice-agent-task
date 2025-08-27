import os
import asyncio
from typing import Optional, List, Any
from dotenv import load_dotenv

# AutoGen AgentChat imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

# --- Model client setup ---
# Note: This client class typically targets OpenAI-compatible endpoints.
# Your current code uses a Gemini model name with GEMINI_API_KEY.
# If your stack supports this, keep it; otherwise, set OPENAI_API_KEY and an OpenAI model.
MODEL_NAME = os.getenv("LLM_MODEL", "gemini-1.5-flash")
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")


model_client = OpenAIChatCompletionClient(
    model=MODEL_NAME,
    api_key=API_KEY,
)

# --- Agents ---
manager = AssistantAgent(
    name="manager",
    model_client=model_client,
    handoffs=["math", "story", "general"],
    system_message=(
        "You are a manager agent. Your primary task is to analyze the user's query and hand off to the right agent.\n"
        "1. Identify the language (english, hindi, or hinglish).\n"
        "2. Determine if the topic is 'math', 'story', or 'general'.\n"
        "- math: For math questions.\n"
        "- story: For short kids stories.\n"
        "- general: For anything else (greetings, thanks, casual questions).\n"
        "Always send your brief plan first, then handoff to a SINGLE agent."
    ),
)

math_agent = AssistantAgent(
    name="math",
    model_client=model_client,
    system_message=(
        "You are a friendly math teacher for kids.\n"
        "First, calculate the correct numerical answer. Second, create a simple, kid-friendly explanation.\n"
        "Your response must contain a raw JSON string of the following format: "
        '{"numerical_answer": "4", "explanation": "Imagine you have 2 apples..."}'
        "\nAfter the JSON, on a new line, say TERMINATE."
    ),
)

story_agent = AssistantAgent(
    name="story",
    model_client=model_client,
    system_message=(
        "You are a creative storyteller for kids. ✨\n"
        "Write a very short, engaging story with a positive message. Keep it concise to minimize TTS and token usage.\n"
        "Hard length limits:\n"
        "- title: <= 60 characters\n"
        "- story_text: 2–4 short sentences, total <= 400 characters\n"
        "- moral: <= 120 characters\n"
        "Output requirements:\n"
        "- Respond with RAW JSON only (no markdown, no code fences).\n"
        "- Use exactly these keys: title, story_text, moral.\n"
        "- Keep language simple and kid-friendly.\n"
        "Example JSON format: {\"title\": \"The Brave Rabbit\", \"story_text\": \"Once upon a time...\", \"moral\": \"Bravery comes in all sizes.\"}\n"
        "After the JSON, on a new line, say TERMINATE."
    ),
)

general_agent = AssistantAgent(
    name="general",
    model_client=model_client,
    system_message=(
        "You are a friendly assistant for general conversation (greetings, thanks, small talk, simple Q&A).\n"
        "Respond concisely in a warm, natural tone suitable for TTS.\n"
        "Avoid code blocks and avoid long lists. Keep responses short.\n"
        "Do not output JSON unless explicitly asked by the user.\n"
        "End your final message with the word TERMINATE on a new line."
    ),
)

termination_condition = TextMentionTermination("TERMINATE")


def _new_team() -> Swarm:
    # Build a fresh team per request to avoid state leakage/hangs
    return Swarm(
        participants=[manager, math_agent, story_agent, general_agent],
        termination_condition=termination_condition,
    )


def _clean_json_to_text(raw: str) -> str:
    """Convert raw JSON-ish content to TTS-friendly text.
    - If it looks like math JSON: {"numerical_answer": "4", "explanation": "..."}
    - If it looks like story JSON: {"title": ..., "story_text": ..., "moral": ...}
    Fallback: return trimmed string.
    """
    try:
        import json
        s = raw.strip()
        # Strip code fences if any
        if s.startswith("```"):
            s = s.strip("`\n ")
        # Try parse JSON object from the first {...} region
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(s[start:end+1])
            if isinstance(obj, dict):
                # math format
                if "numerical_answer" in obj and "explanation" in obj:
                    na = str(obj.get("numerical_answer", ""))
                    ex = str(obj.get("explanation", ""))
                    return f"Answer: {na}. {ex}".strip()
                # story format
                if all(k in obj for k in ("title", "story_text", "moral")):
                    title = str(obj.get("title", "")).strip()
                    story = str(obj.get("story_text", "")).strip()
                    moral = str(obj.get("moral", "")).strip()
                    parts = []
                    if title:
                        parts.append(title)
                    if story:
                        parts.append(story)
                    if moral:
                        parts.append(f"Moral: {moral}")
                    return ". ".join(parts).strip()
    except Exception:
        pass
    # Remove TERMINATE token if present
    return raw.replace("TERMINATE", "").strip()


async def get_agent_response(task: str) -> str:
    """Run the swarm and return only the assistant's final text (clean for TTS).
    Builds a fresh team per call to avoid state issues."""
    team = _new_team()
    try:
        result: Any = await team.run(task=task)

        # 1) Prefer explicit messages list if available
        messages = getattr(result, "messages", None)
        if isinstance(messages, list) and messages:
            # Look from the end for last assistant (non-user) text
            for m in reversed(messages):
                content = getattr(m, "content", None)
                source = getattr(m, "source", None)
                if isinstance(content, str) and source and source != "user":
                    return _clean_json_to_text(content)
                # If content is list of parts, join textual parts
                if isinstance(content, list):
                    parts = []
                    for seg in content:
                        if isinstance(seg, str):
                            parts.append(seg)
                        elif isinstance(seg, dict) and "text" in seg:
                            parts.append(str(seg["text"]))
                    if parts and source and source != "user":
                        return _clean_json_to_text("".join(parts))

        # 2) Fallback to top-level content if present
        content = getattr(result, "content", None)
        if isinstance(content, str):
            return _clean_json_to_text(content)
        if isinstance(content, list):
            return _clean_json_to_text("".join(map(str, content)))
        if isinstance(content, dict):
            return _clean_json_to_text(str(content))

        # 3) Last resort: string of result
        return _clean_json_to_text(str(result))
    except Exception as e:
        return f"[Agent error: {e}]"


async def shutdown() -> None:
    """Cleanly close model client connections."""
    try:
        await model_client.close()
    except Exception:
        pass


if __name__ == "__main__":
    # Simple manual test
    async def _test():
        reply = await get_agent_response("what is 2+2")
        print("Agent reply:\n", reply)
        await shutdown()
    asyncio.run(_test())