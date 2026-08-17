# Claude API RAG System with SQLite

## A proof-of-concept chatbot built with the raw Claude Messages API that implements a RAG system on chat history (episodic memory) and persistent facts (semantic memory) using SQLite 

### Intro
This project is an educational and instructive learning exercise for a memory system in an agent harness. It wraps the raw Claude Messages API in a manual tool-calling loop, giving the model two memory tools: a semantic search over past conversation turns stored as embeddings in a SQLite vector store (episodic memory), and a table of durable facts explicitly stored about the user (semantic memory). Each conversation turn is embedded and saved so later turns can recall relevant past exchanges by similarity and keywords instead of keeping the full history in context. The goal is to test raw LLM tool definition and calling using explicit tool schemas to route between storing durable facts and searching episodic history, and vectorization in semantic querying without relying on a higher abstraction orchestration framework as solutions for limited context windows. The flow is as follows:

1. The user sends their first message.
2. Chat is initialized, with all durable facts from semantic memory loaded into the system prompt.
3. The model decides whether the user is referencing a past chat or stating a new durable fact, and calling the correct tool if so.
4. If no memory tool needs to be called, the model sends its response. If a memory tool is needed, the following loop initiates:
    - The model requests to run the tool
    - The function is ran
    - Tool result is sent back as a new message
    - The API is called once again with the tool result

    The model may once again decide a tool call is needed, continuing the loop

### Tool Loop Diagram:

![Tool Diagram](assets/tool_loop.png)

## Claude Messages API

- The Claude Messages API is stateless, meaning it executes one isolated response on its current context, and thus we must manage the context window ourselves. We do this by maintaining a `messages` list of dicts. Every user message, model response, and tool message are appended to this list as the conversation goes on, and that list is injected as the prompt for every API call. 

- We define two tool schemas prior to the model invokation in JSON schema format, `memory_schema` and `perm_mem_schema`. The tools' descriptions are what the model references to decide what tool to use and when. 
    - Part of the complexity is handling the tool loop manually. Claude provides a tool (Agent SDK) that abstracts the tool loop, state management, and stop reasons. This project intentionally skips that abstraction to work at the lower level, trading convenience for a deeper study of how each step (tool requests, dispatch, results, state) actually works.

## SQLite

- Why SQLite
    - SQLite was chosen as the database since it is free, local, and lightweight.
    - Since SQLite is file-based, the entire memory store is a single `.db` file with no server or setup step, which fits the project's scope as a local proof of concept rather than a production data layer.

- Querying
    - The `sqlite-vec` extension adds vector similarity search directly to SQLite through a virtual table (`vec0`), so embeddings can be stored and queried without standing up a separate vector database.
        - We use cosine as the distance metric to be able to intuitively understand whether a score is good or bad.
        - The threshold for returned chats is hardcoded at .75, but there is no right answer. The threshold at .75-.9 is what i found works best for me.
    - The `FTS5` built in extension provides the functionality for keyword searching, using `BM25` for relevance scoring.
    - Four tables back the two memory types: `exchanges` holds the raw text of each conversation turn (one user prompt and one model response), `vec` is the virtual table holding that turn's embedding (joined back to `exchanges` by id) for episodic search, and `perm_memory` holds the durable facts used as semantic memory. The fourth is the `keywords` virtual table that indexes each turn's raw text for literal keyword matching

### Hybrid Querying and RRF
- Running a hybrid style query is beneficial for context retrieval since a 384D semantic search is far from perfect. A keyword search helps plug the holes.
- With hybrid querying, we need a way of combining the results. Reciprocal Rank Fusion, or RRF, is the algorithm commonly used to combine the results of a hybrid query for LLM context management. 
- RRF can't combine the two results by their raw scores directly, since cosine distance and BM25 rank live on incomparable scales. Instead it only looks at each result's *rank* (its position) within its own list:
    - `RRF_score(d) = Σ 1 / (k + rank(d))`, summed over every ranked list `d` shows up in (rank is 1-indexed; a doc missing from a list just contributes nothing for that list). Higher score wins, since a better (lower) rank produces a larger fraction, opposite polarity from distance, where lower is better.
    - `k` is a damping constant (60 here, the standard default from the original RRF paper) that controls how much a rank difference matters. A small `k` makes rank 1 vs rank 2 a huge gap, a large `k` makes the top ranks nearly indistinguishable, which is more forgiving of noisy small differences.
- `semantic_search` and `keyword_search` both return up to `top_k` results. After RRF scoring, adjacent or near adjacent exchanges (determined by $\pm$ 2 their `turn_index`, the ${n}^{th}$ exchange in the conversation) are merged into single entries. This allows for larger conversations to be pulled in.
- Performance is determined on the balancing of: `top_k` ceilings, `max_distance` (semantic threshold), and the adjacency windows for merging

### Embedding Model

- Embeddings are generated locally with `sentence-transformers`' `all-MiniLM-L6-v2` model rather than an embedding API call, keeping the project self-contained and avoiding extra latency/cost on every turn.
- The model outputs 384-dimensional vectors, matching the `float[384]` column defined on the `vec0` virtual table.
- The same model is used both to embed each conversation turn when it's stored (`add_chat`) and to embed the query text at recall time (`semantic_search`), which is required since a query is only comparable to stored vectors that came from the same embedding
space.

## Problems Encountered and Fixed
1. Memory search results primarily returning memory recall turns
    - **The Problem:** If a user were to ask a question about memory that didn't exist yet, the model would (correctly) state that the memory didn't exist. However, after that memory is instilled, if the user were to ask once again, the top result would be the exchange where the user asked the question, and the model responded saying it had no memory. Thus the model "not having the memory" would propogate.
        - Even if the results were the memory recall turns where the model correctly pulled memory, the response itself is a summary, and each subsequent recall of that topic would degrade as it summarizes a summary, which itself could have summarized a summary and so on.
    - **The Fix:** Don't store episodic memories of memory recall exchanges, creating a win-win where the memory is injected into context but not saved as an event since it's a meta exchange on previous information.

2. Model treated the system prompt injected permanent facts as it's only source of memory
    - **The Problem:** The model wouldn't invoke the memory tool because it thought it had all relevant memory already coming in through the system prompt. The permanent facts were preceded with "Permanant facts:" which could have caused this incorrect thinking. This would also cause a meta exchange to be saved to the database, side-stepping the fix to problem 1.
    - **The Fix:** The preceding prompt was tweaked to "Durable facts you've been told to remember (doesn't replace searching for memory):"

3. Unrelated semantic results bled into context, and if that topic was requested to be recalled, the model treated that as the full context for that topic
    - **The Problem:** If we asked to recall topic X and a small bit of topic Y bled through the semantic results, the model would never re-query to retrieve the full context for topic Y
    - **The Fix:** Add to the memory tool description saying "Call this for every NEW topic about the past, even if you have some info on it."

## Still to implement

- Cost / usage tracking
- Third vector store for per turn semantic search
    - Episodic memory promotion
- UI
- Redundant id in perm mem
- Caching
- Compaction