"""
database.py  —  MediSimplify
============================
All SQLite3 persistence for users, family profiles, and lab results.

Design rules:
  - Every function opens its own connection and closes it in a `finally` block
    to prevent database locking under Streamlit's concurrent re-runs.
  - `create_user` and `verify_user` both return a tuple (bool, str | dict)
    so the caller can display exact error messages without a try/except in
    the UI layer.
  - Lab results use `add_health_record`, `get_records_for_profile`, and
    `delete_health_record` — names that match the imports in main.py exactly.
"""

import sqlite3
import hashlib

DB_PATH = "medisimplify.db"


# ─────────────────────────────────────────────────────────────────────────────
# Connection helper
# ─────────────────────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """Open and return a connection with row-factory enabled."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Schema initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create all tables if they do not already exist.
    Safe to call on every Streamlit re-run (uses IF NOT EXISTS).
    Runs lightweight ALTER TABLE migrations so existing databases gain new
    columns without a manual reset.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # ── users ─────────────────────────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ── profiles ──────────────────────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                name       TEXT NOT NULL,
                age        INTEGER,
                gender     TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # ── lab_results ───────────────────────────────────────────────────────
        # One row = one test reading on one date for one profile.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS lab_results (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id    INTEGER NOT NULL,
                date_recorded TEXT NOT NULL DEFAULT '',
                test_name     TEXT NOT NULL DEFAULT '',
                test_value    REAL NOT NULL DEFAULT 0,
                unit          TEXT,
                notes         TEXT,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (profile_id) REFERENCES profiles(id)
            )
        """)

        # Migration guard: add columns that may be absent in older databases
        cur.execute("PRAGMA table_info(lab_results)")
        existing_cols = {row["name"] for row in cur.fetchall()}
        migrations = [
            ("date_recorded", "TEXT NOT NULL DEFAULT ''"),
            ("test_name",     "TEXT NOT NULL DEFAULT ''"),
            ("test_value",    "REAL NOT NULL DEFAULT 0"),
            ("unit",          "TEXT"),
            ("notes",         "TEXT"),
        ]
        for col_name, col_type in migrations:
            if col_name not in existing_cols:
                cur.execute(f"ALTER TABLE lab_results ADD COLUMN {col_name} {col_type}")

        conn.commit()
    finally:
        if conn:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Password hashing
# ─────────────────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# User management
# ─────────────────────────────────────────────────────────────────────────────

def create_user(username: str, password: str) -> tuple[bool, str]:
    """
    Insert a new user row.

    Returns:
        (True,  "User created successfully.")            on success
        (False, "Username already exists. …")            if username taken
        (False, "Failed to create user: <detail>")       on any other DB error
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username.strip(), hash_password(password))
        )
        conn.commit()
        return True, "User created successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please choose a different username."
    except Exception as exc:
        return False, f"Failed to create user: {exc}"
    finally:
        if conn:
            conn.close()


def verify_user(username: str, password: str) -> tuple[bool, dict | str]:
    """
    Verify login credentials.

    Returns:
        (True,  {"id": int, "username": str})    on success
        (False, "Invalid username or password.")  on bad credentials
        (False, "Database error: <detail>")       on DB error

    Callers should unpack and check the bool before using the second element:
        ok, result = verify_user(u, p)
        if ok:
            user_id = result["id"]
        else:
            st.error(result)
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, username FROM users WHERE username = ? AND password_hash = ?",
            (username.strip(), hash_password(password))
        )
        row = cur.fetchone()
        if row:
            return True, dict(row)
        return False, "Invalid username or password."
    except Exception as exc:
        return False, f"Database error: {exc}"
    finally:
        if conn:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Profile management
# ─────────────────────────────────────────────────────────────────────────────

def create_profile(user_id: int, name: str, age: int, gender: str) -> int | None:
    """
    Insert a new family profile row.

    Returns the new profile id on success, or None on failure.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO profiles (user_id, name, age, gender) VALUES (?, ?, ?, ?)",
            (user_id, name.strip(), int(age), gender)
        )
        conn.commit()
        return cur.lastrowid
    except Exception:
        return None
    finally:
        if conn:
            conn.close()


def get_profiles(user_id: int) -> list[dict]:
    """
    Return all profiles for a user, ordered alphabetically by name.

    Each dict exposes both 'name' and 'patient_name' (alias) so that the
    sidebar in main.py can use p["patient_name"] without a KeyError.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM profiles WHERE user_id = ? ORDER BY name",
            (user_id,)
        )
        rows = []
        for r in cur.fetchall():
            d = dict(r)
            d["patient_name"] = d["name"]   # backward-compatible alias
            rows.append(d)
        return rows
    except Exception:
        return []
    finally:
        if conn:
            conn.close()


def delete_profile(profile_id: int) -> None:
    """Delete a profile and cascade-delete all its lab results."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM lab_results WHERE profile_id = ?", (profile_id,))
        cur.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
        conn.commit()
    finally:
        if conn:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Lab results — function names match imports in main.py exactly
# ─────────────────────────────────────────────────────────────────────────────

def add_health_record(
    profile_id: int,
    test_name: str,
    test_value: float,
    unit: str,
    date_recorded: str,
    notes: str = "",
) -> int | None:
    """
    Insert one lab test result row.

    Returns the new row id (truthy int) on success, or None on failure.
    Callers that do  `if ok: …`  will correctly branch on success/failure.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO lab_results
                (profile_id, date_recorded, test_name, test_value, unit, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            profile_id,
            str(date_recorded),
            test_name.strip(),
            float(test_value),
            (unit or "").strip(),
            notes or "",
        ))
        conn.commit()
        return cur.lastrowid   # always > 0 on success → truthy
    except Exception:
        return None
    finally:
        if conn:
            conn.close()


def get_records_for_profile(profile_id: int) -> list[dict]:
    """
    Return all lab result rows for a profile, most-recent date first.

    Each dict contains:
        id, profile_id, date_recorded, test_name, test_value, unit, notes, created_at
    """
    if not profile_id:
        return []
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM lab_results WHERE profile_id = ? ORDER BY date_recorded DESC",
            (profile_id,)
        )
        return [dict(r) for r in cur.fetchall()]
    except Exception:
        return []
    finally:
        if conn:
            conn.close()


def get_unique_test_names(profile_id: int) -> list[str]:
    """Return a sorted list of distinct test names recorded for a profile."""
    if not profile_id:
        return []
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT test_name FROM lab_results WHERE profile_id = ? ORDER BY test_name",
            (profile_id,)
        )
        return [row["test_name"] for row in cur.fetchall()]
    except Exception:
        return []
    finally:
        if conn:
            conn.close()


def delete_health_record(result_id: int) -> None:
    """Delete a single lab result row by primary key."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM lab_results WHERE id = ?", (result_id,))
        conn.commit()
    finally:
        if conn:
            conn.close()
