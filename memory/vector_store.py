# install depdendencies:
# pip install sqlite-vec

import sqlite3
import sqlite_vec


def get_connection(db_name):
    try:
        connection = sqlite3.connect(db_name)
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        return connection
    except Exception as e:
        print(f"Error: {e}")
        raise

def create_store(connection):

    # vec virtual table
    exchange_vec_query = """
    CREATE VIRTUAL TABLE IF NOT EXISTS vec USING vec0 (
    id INTEGER PRIMARY KEY,
    embedding float[384] distance_metric=cosine
    )
    """

    # FTS5 virtual table
    exchange_keyword_query = """
    CREATE VIRTUAL TABLE IF NOT EXISTS keywords USING fts5 (exchange)
    """

    exchange_query = """
    CREATE TABLE IF NOT EXISTS exchanges (
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    exchange TEXT,
    PRIMARY KEY (session_id, turn_index)
    )
    """

    perm_mem_query = """
    CREATE TABLE IF NOT EXISTS perm_memory (fact TEXT)
    """

    lesson_query = """
    CREATE TABLE IF NOT EXISTS lessons (lesson TEXT)
    """

    semantic_mem_query = """
    CREATE TABLE IF NOT EXISTS chunks (chunk TEXT)
    """

    semantic_vec_query = """
    CREATE VIRTUAL TABLE IF NOT EXISTS sem_vecs USING vec0 (
    id INTEGER PRIMARY KEY,
    embedding float[384] distance_metric=cosine
    )
    """

    semantic_keyword_query = """
    CREATE VIRTUAL TABLE IF NOT EXISTS sem_keywords USING fts5 (chunk)
    """

    try:
        with connection:
            connection.execute(exchange_query)
            connection.execute(exchange_vec_query)
            connection.execute(perm_mem_query)
            connection.execute(exchange_keyword_query)
            connection.execute(lesson_query)
            connection.execute(semantic_mem_query)
            connection.execute(semantic_vec_query)
            connection.execute(semantic_keyword_query)

        return ""
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}"

def store_perm_mem (connection, fact:str, type:str):
    if type == "FACT":
        query = "INSERT INTO perm_memory (fact) VALUES (?)"
    elif type == "LESSON":
        query = "INSERT INTO lessons (lesson) VALUES (?)"
    try:
        with connection:
            connection.execute(query, (fact,))
        return ""
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}"

def load_perm_mem (connection, type:str):
    if type == "FACT":
        rows = connection.execute(
            "SELECT fact FROM perm_memory"
        ).fetchall()
    elif type == "LESSON":
        rows = connection.execute(
                "SELECT lesson FROM lessons"
            ).fetchall()

    # Flatten the single-column rows into a plain list of fact strings
    return [row[0] for row in rows]

def add_chat(connection, exchange:str, session_id:str, turn_index:int, embedding):
    query = "INSERT INTO exchanges (exchange, session_id, turn_index) VALUES (?, ?, ?)"
    vec_query = "INSERT INTO vec (id, embedding) VALUES (?, ?)"
    keyword_query = "INSERT INTO keywords (rowid, exchange) VALUES (?, ?)"

    try:
        with connection:
            cursor = connection.execute(query, (exchange, session_id, turn_index))
            new_id = cursor.lastrowid
            connection.execute(vec_query, (new_id, embedding))
            connection.execute(keyword_query, (new_id, exchange))
        return ""
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}"

def add_semantic_memory(connection, text:str, embedding):
    query = "INSERT INTO chunks (chunk) VALUES (?)"
    vec_query = "INSERT INTO sem_vecs (id, embedding) VALUES (?,?)"
    keyword_query = "INSERT INTO sem_keywords (rowid, chunk) VALUES (?,?)"

    try:
        with connection:
            cursor = connection.execute(query, (text,))
            new_id = cursor.lastrowid
            connection.execute(vec_query, (new_id, embedding))
            connection.execute(keyword_query, (new_id, text))
        return ""
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}"

def semantic_search(connection, model, text, exclude_ids, vec_table, content_table, top_k, max_distance=.5):
    embedding = model.encode(text)
    exclude_ids = exclude_ids or []
    placeholders = ",".join("?" * len(exclude_ids))

    rows = connection.execute(
        f"""
        SELECT v.distance, {content_table}.rowid
        FROM (SELECT id, distance FROM {vec_table} WHERE embedding MATCH ? AND k = ? AND id NOT IN ({placeholders})) AS v
        JOIN {content_table} ON {content_table}.rowid = v.id
        """,
        (embedding, top_k, *exclude_ids)).fetchall()

    filtered = [(row[1], row[0]) for row in rows if row[0] <= max_distance]
    print(f"DEBUG: {vec_table} matches within threshold (rowid, distance) = {[(rid, round(dist, 4)) for rid, dist in filtered]}")

    return [rid for rid, _ in filtered]


def keyword_search(connection, table:str, exclude_ids, keywords, top_k=15):
    # Defensive guard: FTS5 MATCH treats multiple space-separated terms as an
    # implicit AND, so reduce each keyword to its first whitespace-separated
    # word (falling back to the original, possibly empty, item if it has no
    # non-whitespace characters, to avoid an IndexError on split()[0]), then
    # OR-join the single-word terms so a row matches if it contains any of them.
    terms = [kw.split(maxsplit=1)[0] if kw.strip() else kw for kw in keywords]
    exclude_ids = exclude_ids or []
    placeholders = ",".join("?" * len(exclude_ids))

    if not terms:
        return []

    query_term = " OR ".join(terms)

    rows = connection.execute(
        f"SELECT rowid, rank FROM {table} WHERE {table} MATCH ? AND rowid NOT IN ({placeholders}) ORDER BY rank LIMIT ?",
        (query_term, *exclude_ids, top_k)
    ).fetchall()
    print(f"DEBUG: {table} keyword matches (rowid, bm25 rank) = {[(row[0], round(row[1], 4)) for row in rows]}")

    return [row[0] for row in rows]  # ranked list of ids, best match first

def rrf(connection, vec_ids, keyword_ids, content_table, text_column, session_column=None, turn_column=None, k=60, top_k=8):
    """Combine two ranked id lists (vec search + keyword search) using
    Reciprocal Rank Fusion, then join the top results back to content_table.

    vec_ids and keyword_ids are both lists of content_table rowids ordered
    best match first. Each id's RRF score is the sum of 1/(k + rank)
    over every list it appears in (rank is 1-based within that list).

    Tables with a real conversational sequence (e.g. exchanges) pass
    session_column/turn_column so adjacent rows can be merged below. Tables
    without one (e.g. chunks) omit them -- each row falls back to being its
    own session (via its own rowid) with a constant turn_index, so the
    grouping/merging logic runs unchanged and simply never merges anything.

    Each returned tuple is (text, session_id, turn_index, score, rowids),
    where rowids is the list of content_table rowids that were merged into
    that result -- callers should dedup against rowids, not session_id,
    since session_id may not be a rowid (e.g. exchanges.session_id).
    """
    scores = {}

    for rank, id_ in enumerate(vec_ids, start=1):
        scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank)

    for rank, id_ in enumerate(keyword_ids, start=1):
        scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank)

    # Sort ids by descending RRF score. Truncation to top_k now happens
    # after merging (below), so all scored ids are used for the join/merge.
    ranked_ids = sorted(scores, key=lambda id_: scores[id_], reverse=True)

    if not ranked_ids:
        return []

    session_select = f"{content_table}.{session_column}" if session_column else f"{content_table}.rowid"
    turn_select = f"{content_table}.{turn_column}" if turn_column else "0"

    placeholders = ",".join("?" * len(ranked_ids))
    rows = connection.execute(
        f"""
        SELECT {content_table}.rowid, {content_table}.{text_column}, {session_select}, {turn_select}
        FROM {content_table}
        WHERE {content_table}.rowid IN ({placeholders})
        """,
        ranked_ids
    ).fetchall()

    # Preserve RRF score ordering when joining back to content rows, and
    # carry each row's RRF score along so it can be merged/sorted below.
    rows_by_id = {row[0]: row for row in rows}
    results = []
    for id_ in ranked_ids:
        row = rows_by_id.get(id_)
        if row is not None:
            rowid, text, session_id, turn_index = row
            results.append((text, session_id, turn_index, scores[id_], [rowid]))

    if not results:
        return []

    # Group fused results by session_id
    sessions = {}
    for row in results:
        sessions.setdefault(row[1], []).append(row)

    combined = []
    for session_id, session_rows in sessions.items():
        # Sort this session's rows by turn_index and merge into runs
        # of consecutive turn_index values
        session_rows.sort(key=lambda r: r[2])

        run = [session_rows[0]]
        for row in session_rows[1:]:
            if row[2] - run[-1][2] <= 2:
                # Within 2 of the previous turn_index (tolerates a gap of up
                # to one skipped turn) -> extend current run
                run.append(row)
            else:
                combined.append(_merge_run(run))
                run = [row]
        combined.append(_merge_run(run))

    # Return combined/singleton results, best match (highest RRF score) first
    combined.sort(key=lambda r: r[3], reverse=True)

    # Truncate here so top_k limits the final merged/singleton results,
    # not the number of raw pre-merge ids considered above.
    combined = combined[:top_k]

    print(f"DEBUG: rrf top_k (rowids, score) = {[(r[4], round(r[3], 4)) for r in combined]}")
    print(f"DEBUG: rrf merged = {[r[4] for r in combined if len(r[4]) > 1]}")

    return combined


def _merge_run(run):
    """Merge a run of consecutive-turn_index rows from the same session
    into a single (exchange_text, session_id, turn_index, score, rowids) tuple.

    Runs of size 1 are returned unchanged. Unlike distance (lower is
    better), RRF score is better when higher, so the merged run's score
    is the maximum score among its rows. rowids collects every row's
    rowid(s) that contributed to the run, so callers can dedup against
    every underlying row, not just one.
    """
    if len(run) == 1:
        return run[0]

    # run is already sorted ascending by turn_index
    exchange_text = "\n".join(row[0] for row in run)
    session_id = run[0][1]
    turn_index = run[0][2]
    score = max(row[3] for row in run)
    rowids = [rowid for row in run for rowid in row[4]]

    return (exchange_text, session_id, turn_index, score, rowids)
