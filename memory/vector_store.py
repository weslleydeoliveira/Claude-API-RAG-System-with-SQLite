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
    keyword_query = """
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
    CREATE TABLE IF NOT EXISTS perm_memory (
    id INTEGER PRIMARY KEY,
    fact TEXT
    )
    """

    lesson_query = """
    CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY,
    lesson TEXT
    )
    """

    try:
        with connection:
            connection.execute(exchange_query)
            connection.execute(exchange_vec_query)
            connection.execute(perm_mem_query)
            connection.execute(keyword_query)
            connection.execute(lesson_query)

    except Exception as e:
        print(f"Error: {e}")

def store_perm_mem (connection, fact:str, type:str):
    if type == "FACT":
        query = "INSERT INTO perm_memory (fact) VALUES (?)"
    elif type == "LESSON":
        query = "INSERT INTO lessons (lesson) VALUES (?)"
    try:
        with connection:
            connection.execute(query, (fact,))
    except Exception as e:
        print(f"Error: {e}")

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
    except Exception as e:
        print(f"Error: {e}")

def semantic_search(connection, model, text, top_k=15, max_distance=.75):
    embedding = model.encode(text)

    rows = connection.execute(
        """
        SELECT exchanges.exchange, exchanges.session_id, exchanges.turn_index, v.distance, exchanges.rowid
        FROM (
            SELECT id, distance FROM vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?
            )
        AS v
        JOIN exchanges ON exchanges.rowid = v.id
        """,
        (embedding, top_k)).fetchall()

    # 1) Keep only rows within the distance threshold
    filtered = [row[4] for row in rows if row[3] <= max_distance]
    return filtered

def keyword_search(connection, keywords, top_k=15):
    # Defensive guard: FTS5 MATCH treats multiple space-separated terms as an
    # implicit AND, so reduce each keyword to its first whitespace-separated
    # word (falling back to the original, possibly empty, item if it has no
    # non-whitespace characters, to avoid an IndexError on split()[0]), then
    # OR-join the single-word terms so a row matches if it contains any of them.
    terms = [kw.split(maxsplit=1)[0] if kw.strip() else kw for kw in keywords]

    if not terms:
        return []

    query_term = " OR ".join(terms)

    rows = connection.execute(
        "SELECT rowid FROM keywords WHERE keywords MATCH ? ORDER BY rank LIMIT ?",
        (query_term, top_k)
    ).fetchall()
    return [row[0] for row in rows]  # ranked list of ids, best match first

def rrf(connection, vec_ids, keyword_ids, k=60, top_k=8):
    """Combine two ranked id lists (vec search + keyword search) using
    Reciprocal Rank Fusion, then join the top results back to exchanges.

    vec_ids and keyword_ids are both lists of exchange rowids ordered
    best match first. Each id's RRF score is the sum of 1/(k + rank)
    over every list it appears in (rank is 1-based within that list).
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

    placeholders = ",".join("?" * len(ranked_ids))
    rows = connection.execute(
        f"""
        SELECT rowid, exchange, session_id, turn_index
        FROM exchanges
        WHERE rowid IN ({placeholders})
        """,
        ranked_ids
    ).fetchall()

    # Preserve RRF score ordering when joining back to exchange rows, and
    # carry each row's RRF score along so it can be merged/sorted below.
    rows_by_id = {row[0]: row for row in rows}
    results = []
    for id_ in ranked_ids:
        row = rows_by_id.get(id_)
        if row is not None:
            _, exchange_text, session_id, turn_index = row
            results.append((exchange_text, session_id, turn_index, scores[id_]))

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

    return combined


def _merge_run(run):
    """Merge a run of consecutive-turn_index rows from the same session
    into a single (exchange_text, session_id, turn_index, score) tuple.

    Runs of size 1 are returned unchanged. Unlike distance (lower is
    better), RRF score is better when higher, so the merged run's score
    is the maximum score among its rows.
    """
    if len(run) == 1:
        return run[0]

    # run is already sorted ascending by turn_index
    exchange_text = "\n".join(row[0] for row in run)
    session_id = run[0][1]
    turn_index = run[0][2]
    score = max(row[3] for row in run)

    return (exchange_text, session_id, turn_index, score)
