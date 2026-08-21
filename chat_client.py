import uuid
import anthropic
from pathlib import Path
from dotenv import load_dotenv
import os
from memory.vector_store import add_chat, load_perm_mem, store_perm_mem, keyword_search, rrf, add_semantic_memory, semantic_search

load_dotenv(Path(__file__).parent / "config" / ".env")
api_key=os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

episodic_schema = {
    "name": "recall_memory",
    "description": "If user references something from the past, or asks if you remember something, search past chats for relevant context and/or memories. Don't be TOO specific with queries or you lose good candidates, but not TOO general to get a hit on everything. Call this for every NEW topic about the past, even if you have some info on it.",
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
    "description": "If you receive a fact ABOUT THE USER, their preferences, opinion, etc, store that fact in permanent memory.",
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

add_semantic_mem = {
    "name": "store_sem_mem",
    "description": "If the user asks to store a fact that is not necessarily about them or a preference/opinion, store in semantic memory.",
    "input_schema": {
        "type": "object",
        "properties": {
            "memory": {"type": "string", "description": "The memory asked to be stored"}
        },
        "required": ["memory"]
    }
}

search_semantic_mem = {
    "name": "search_semantic",
    "description": "Always use this tool FOR EVERY response to see if there is any relevant memory",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for. Generated based on but not worded exactly like the user prompt"},
            "keywords": {"type": "array", "items": {"type": "string"}, "maxItems": 2, "description": "One word, most important keyword. MUST be relevant to topic. No filler words. Second keyword optional"}
        },
        "required": ["query", 'keywords']
    }
}

def _recall_pipeline(connection, model, query, keywords, exclude_ids, vec_table, keyword_table, content_table, text_column, top_k, session_column=None, turn_column=None):
    sem_ids = semantic_search(connection, model, query, exclude_ids, vec_table, content_table, top_k=top_k)
    kw_ids = keyword_search(connection, keyword_table, exclude_ids, keywords)
    combined = rrf(connection, sem_ids, kw_ids, content_table, text_column, session_column, turn_column)
    return combined


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
    seen_chunk_ids = set()
    seen_exchanges_ids = set()

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
                tools=[episodic_schema, perm_mem_schema, lesson_schema, add_semantic_mem, search_semantic_mem],
                max_tokens=1000
            )

            # This can probably be moved to the main tool use stop reason section
            if response.stop_reason == "tool_use":
                # Drop preamble text blocks (e.g. narration like "Let me search my
                # memory...") and keep only the tool_use blocks, so filler text
                # doesn't later get embedded into the exchange via format_conversation.
                tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

                # Episodic (recall_memory) and semantic (search_semantic) memory are
                # exclusive -- if both were requested, drop search_semantic entirely
                # so it's as if Claude never called it, and it never needs a tool_result.
                tool_use_names = [b.name for b in tool_use_blocks]
                if "recall_memory" in tool_use_names and "search_semantic" in tool_use_names:
                    tool_use_blocks = [b for b in tool_use_blocks if b.name != "search_semantic"]

                assistant_content = tool_use_blocks
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
                print(f"DEBUG: tool_use blocks in this response = {[b.name for b in assistant_content]}")

                # Every tool_use block in this response needs its own tool_result --
                # collecting into a list (instead of overwriting a single tool_id)
                # keeps responses with multiple simultaneous tool_use blocks valid.
                tool_result_blocks = []
                for object in assistant_content:
                    if object.type == "tool_use":
                        # Each branch sets its own local result_text so tool calls
                        # never share or leak each other's result content.
                        if object.name == "recall_memory":
                            used_recall_memory = True
                            print("Attempting to pull from memory\n")
                            print(f"DEBUG: raw tool_use input query={object.input['query']!r}, keywords={object.input['keywords']!r}")
                            combined = _recall_pipeline(connection, model, object.input["query"], object.input["keywords"], seen_exchanges_ids, "vec", "keywords", "exchanges", "exchange", 15, "session_id", "turn_index")
                            if combined:
                                result_text = "\n---\n".join(
                                    f"[Memory {i+1}, score={score:.4f}]\n{exchange_text}"
                                    for i, (exchange_text, result_session_id, result_turn_index, score, result_rowids) in enumerate(combined)
                                )

                                for _, _, _, _, result_rowids in combined:
                                    seen_exchanges_ids.update(result_rowids)

                            else:
                                # Empty here is ambiguous: it could mean nothing relevant
                                # exists, or that the only relevant memories were already
                                # surfaced earlier this conversation and got excluded by
                                # seen_exchanges_ids. Re-run unfiltered (no exclusions) to
                                # tell the two cases apart before answering.
                                unfiltered = _recall_pipeline(connection, model, object.input["query"], object.input["keywords"], set(), "vec", "keywords", "exchanges", "exchange", 15, "session_id", "turn_index")
                                if unfiltered:
                                    result_text = "No new memories -- the only relevant memory/memories were already surfaced earlier in this conversation. Refer back to what was already said instead of claiming nothing was found."
                                else:
                                    result_text = "No relevant memories found."

                        elif object.name == "store_perm_mem":
                            print(f"Storing permanent memory: {object.input["fact"]}\n")
                            error = store_perm_mem(connection, object.input["fact"], "FACT")
                            if error: result_text = error
                            else: result_text = "Fact stored."

                        elif object.name == "store_lesson":
                            print(f"Storing lesson: {object.input["lesson"]}\n")
                            error = store_perm_mem(connection, object.input["lesson"], "LESSON")
                            if error: result_text = error
                            else: result_text = "Lesson stored"

                        elif object.name == "store_sem_mem":
                            print(f"Storing semantic memory: {object.input["memory"]}\n")
                            embedding = model.encode(object.input["memory"])
                            error = add_semantic_memory(connection, object.input["memory"], embedding)
                            if error: result_text = error
                            else: result_text = "Semantic memory stored"

                        elif object.name == "search_semantic":
                            print(f"Searching semantic memory\n")
                            combined = _recall_pipeline(connection, model, object.input["query"], object.input["keywords"], seen_chunk_ids, "sem_vecs", "sem_keywords", "chunks", "chunk", 2)
                            if combined:
                                result_text = "\n---\n".join(
                                    f"[Memory {i+1}, score={score:.4f}]\n{text}"
                                    for i, (text, result_session_id, result_turn_index, score, result_rowids) in enumerate(combined)
                                )

                                for _, _, _, _, result_rowids in combined:
                                    seen_chunk_ids.update(result_rowids)

                            else:
                                # See recall_memory above: distinguish a true miss from
                                # everything relevant already having been surfaced and
                                # excluded via seen_chunk_ids.
                                unfiltered = _recall_pipeline(connection, model, object.input["query"], object.input["keywords"], set(), "sem_vecs", "sem_keywords", "chunks", "chunk", 2)
                                if unfiltered:
                                    result_text = "No new memories -- the only relevant memory/memories were already surfaced earlier in this conversation. Refer back to what was already said instead of claiming nothing was found."
                                else:
                                    result_text = "No relevant memories found."

                        tool_result_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": object.id,
                            "content": result_text
                        })

                messages.append({"role": "user", "content": tool_result_blocks})

        # We don't want to add exchanges that were recalling memories into the db, or those will become
        # the best scoring semantic results
        if not used_recall_memory:
            exchange = format_conversation(messages[turn_start_index:])
            embedding = model.encode(exchange)
            counter += 1
            add_chat(connection, exchange, str(session_id), counter, embedding)
