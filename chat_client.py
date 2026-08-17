import uuid
import anthropic
from pathlib import Path
from dotenv import load_dotenv
import os
from memory.vector_store import add_chat, semantic_search, load_perm_mem, store_perm_mem, keyword_search, rrf

load_dotenv(Path(__file__).parent / "config" / ".env")
api_key=os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

memory_schema = {
    "name": "recall_memory",
    "description": "If user references something from the past, or asks if you remember something, search past chats for relevant context and/or memories. Don't be TOO specific with queries or you lose good candidates, but not TOO general to get a hit on everything. Call this for every NEW topic about the past, even if you have some info on it",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for. Don't make assumptions. Must be based on (but doesn't have to be perfectly worded like) the user prompt"},
            "keywords": {"type": "array", "items": {"type": "string"}, "maxItems": 2, "description": "One word, most important keyword. MUST be relevant to topic. No filler words. Second keyword optional"}
        },
        "required": ["query", "keywords"]
    }
}

perm_mem_schema = {
    "name": "store_perm_mem",
    "description": "If you receive a fact about the user, their preferences, opinion, etc, or the user corrects you on something, store that lesson/fact in permanent memory.",
    "input_schema": {
        "type": "object",
        "properties": {
            "fact": {"type": "string", "description": "Fact to store"}
        },
        "required": ["fact"]
    }
}

lesson_schema = {
    "name": "store_lesson",
    "description": "If the user corrects a mistake you made or tells you to behave differently going forward, store the lesson (what went wrong and what to do instead) so you don't repeat it.",
    "input_schema": {
        "type": "object",
        "properties": {
            "lesson": {"type": "string", "description": "The corrected behavior, stated as a rule to follow going forward"}
        },
        "required": ["lesson"]
    }
}

def format_conversation(messages):
    lines = []
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]

        if isinstance(content, str):
            text = content
        else:
            # list of content blocks (e.g. TextBlock objects from the SDK)
            text = " ".join(
                block.text for block in content
                if hasattr(block, "text")
            )

        # Skip blocks that produce no text (e.g. tool_use / tool_result blocks)
        # so they don't leak blank stub lines into the conversation string.
        if text:
            lines.append(f"{role}: {text}")

    return "\n".join(lines)

def chat(connection, model):

    session_id = uuid.uuid4()
    messages = []

    print("Type 'exit' to end chat\n")

    counter = 0
    while True:

        user = input("User: ")
        print("")

        # Record where this turn starts so we can capture the whole turn later,
        # regardless of how many tool-call round trips occur within it.
        turn_start_index = len(messages)

        # A turn that calls recall_memory is a read of existing memory, not new
        # information -- storing/embedding it would let paraphrased recaps (or
        # "nothing found" misses) outcompete the original facts in future search.
        used_recall_memory = False

        messages.append({"role": "user", "content": user})

        if user == "exit":
            return format_conversation(messages)

        while True:
            response = client.messages.create(
                model="claude-haiku-4-5",
                system=f"Durable facts you've been told to remember (doesn't replace searching for memory): {load_perm_mem(connection, "FACT")}\nLessons you've learned: {load_perm_mem(connection, "LESSON")}",
                messages=messages,
                tools=[memory_schema, perm_mem_schema, lesson_schema],
                max_tokens=1000
            )

            if response.stop_reason == "tool_use":
                # Drop preamble text blocks (e.g. narration like "Let me search my
                # memory...") and keep only the tool_use blocks, so filler text
                # doesn't later get embedded into the exchange via format_conversation.
                assistant_content = [b for b in response.content if b.type == "tool_use"]
            else:
                # end_turn (or any other stop reason): keep content unchanged,
                # since this is the real final answer text.
                assistant_content = response.content

            messages.append({"role": "assistant", "content": assistant_content})


            if response.stop_reason == "end_turn":
                for b in response.content:
                    if b.type == "text": print(f"Claude: {b.text}\n")
                break

            elif response.stop_reason == "tool_use":
                # Every tool_use block in this response needs its own tool_result --
                # collecting into a list (instead of overwriting a single tool_id)
                # keeps responses with multiple simultaneous tool_use blocks valid.
                tool_result_blocks = []
                for object in response.content:
                    if object.type == "tool_use":
                        # Each branch sets its own local result_text so tool calls
                        # never share or leak each other's result content.
                        if object.name == "recall_memory":
                            used_recall_memory = True
                            print("Attempting to pull from memory\n")
                            print(f"DEBUG: raw tool_use input query={object.input['query']!r}, keywords={object.input['keywords']!r}")
                            sem_search_results = semantic_search(connection, model, object.input["query"])
                            print(f"DEBUG: vec search ids = {sem_search_results}")
                            keyword_search_results = keyword_search(connection, object.input["keywords"])
                            print(f"DEBUG: keyword search ids = {keyword_search_results}")
                            combined = rrf(connection, sem_search_results, keyword_search_results)
                            print(f"DEBUG: rrf merged results = {[(sid, ti, sc) for (_, sid, ti, sc) in combined]}")
                            if combined:
                                result_text = "\n---\n".join(
                                    f"[Memory {i+1}, score={score:.4f}]\n{exchange_text}"
                                    for i, (exchange_text, result_session_id, result_turn_index, score) in enumerate(combined)
                                )
                            else:
                                result_text = "No relevant memories found."

                        elif object.name == "store_perm_mem":
                            print(f"Storing permanent memory: {object.input["fact"]}\n")
                            store_perm_mem(connection, object.input["fact"], "FACT")
                            result_text = "Fact stored."

                        elif object.name == "store_lesson":
                            print(f"Storing lesson: {object.input["lesson"]}\n")
                            store_perm_mem(connection, object.input["lesson"], "LESSON")
                            result_text = "Lesson stored"

                        tool_result_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": object.id,
                            "content": result_text
                        })

                messages.append({"role": "user", "content": tool_result_blocks})

        if not used_recall_memory:
            exchange = format_conversation(messages[turn_start_index:])
            embedding = model.encode(exchange)
            counter += 1
            add_chat(connection, exchange, str(session_id), counter, embedding)
