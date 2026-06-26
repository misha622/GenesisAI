"""Database layer - SQLite storage with API keys support."""

import sqlite3, uuid, os
from typing import Dict, List, Any, Optional

class GenesisDB:
    def __init__(self, db_path: str = "genesis.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY, title TEXT DEFAULT 'Untitled',
                    created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT NOT NULL,
                    role TEXT NOT NULL, content TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
                    description TEXT DEFAULT '', task TEXT DEFAULT 'other',
                    icon TEXT DEFAULT '🤖', model_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS memory_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT NOT NULL,
                    predicate TEXT NOT NULL, object TEXT NOT NULL,
                    confidence REAL DEFAULT 0.7, created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, user_name TEXT NOT NULL,
                    api_key TEXT UNIQUE NOT NULL, calls_limit INTEGER DEFAULT 100,
                    calls_used INTEGER DEFAULT 0, created_at TEXT DEFAULT (datetime('now')),
                    active INTEGER DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, api_key TEXT NOT NULL,
                    model_id TEXT NOT NULL, timestamp TEXT DEFAULT (datetime('now')),
                    status TEXT DEFAULT 'success'
                );
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;
            """)
            conn.commit()

    # --- Chats ---
    def create_chat(self, chat_id: str, title: str = "Untitled") -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR IGNORE INTO chats (id, title) VALUES (?, ?)", (chat_id, title))
            conn.commit()
        return {"id": chat_id, "title": title}

    def list_chats(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM chats ORDER BY updated_at DESC").fetchall()
            return [{"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]} for r in rows]

    def rename_chat(self, chat_id: str, title: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE chats SET title = ?, updated_at = datetime('now') WHERE id = ?", (title, chat_id))
            conn.commit()

    def delete_chat(self, chat_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            conn.commit()

    # --- Messages ---
    def add_message(self, chat_id: str, role: str, content: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)", (chat_id, role, content))
            conn.execute("UPDATE chats SET updated_at = datetime('now') WHERE id = ?", (chat_id,))
            conn.commit()

    def get_messages(self, chat_id: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT role, content, created_at FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,)).fetchall()
            return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]

    def get_message_count(self, chat_id: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,)).fetchone()
            return row[0] if row else 0

    # --- Projects ---
    def get_projects(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
            return [{"name": r[1], "desc": r[2], "task": r[3], "icon": r[4], "modelCount": r[5], "created": r[6]} for r in rows]

    def add_project(self, name: str, desc: str = "", task: str = "other", icon: str = "🤖") -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO projects (name, description, task, icon) VALUES (?, ?, ?, ?)", (name, desc, task, icon))
            conn.commit()
        return {"name": name, "desc": desc, "task": task, "icon": icon, "modelCount": 0}

    def delete_project(self, name: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM projects WHERE name = ?", (name,))
            conn.commit()

    # --- Memory ---
    def add_fact(self, subject: str, predicate: str, obj: str, confidence: float = 0.7):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO memory_facts (subject, predicate, object, confidence) VALUES (?, ?, ?, ?)", (subject, predicate, str(obj), confidence))
            conn.commit()

    def get_facts(self, subject: str = "user") -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT predicate, object, confidence FROM memory_facts WHERE subject = ? ORDER BY confidence DESC LIMIT 20", (subject,)).fetchall()
            return [{"predicate": r[0], "object": r[1], "confidence": r[2]} for r in rows]

    # --- API Keys ---
    def create_api_key(self, user_name: str, calls_limit: int = 100) -> Dict:
        api_key = f"gen-{uuid.uuid4().hex[:24]}"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO api_keys (user_name, api_key, calls_limit) VALUES (?, ?, ?)", (user_name, api_key, calls_limit))
            conn.commit()
        return {"user_name": user_name, "api_key": api_key, "calls_limit": calls_limit, "calls_used": 0}

    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM api_keys WHERE api_key = ? AND active = 1", (api_key,)).fetchone()
            if row:
                return {"id": row[0], "user_name": row[1], "api_key": row[2], "calls_limit": row[3], "calls_used": row[4]}
            return None

    def use_api_key(self, api_key: str, model_id: str) -> bool:
        """Инкрементит счётчик вызовов и логирует использование."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT calls_used, calls_limit FROM api_keys WHERE api_key = ? AND active = 1", (api_key,)).fetchone()
            if not row or row[0] >= row[1]:
                return False
            conn.execute("UPDATE api_keys SET calls_used = calls_used + 1 WHERE api_key = ?", (api_key,))
            conn.execute("INSERT INTO usage_log (api_key, model_id) VALUES (?, ?)", (api_key, model_id))
            conn.commit()
            return True

    def list_api_keys(self, user_name: str = None) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            if user_name:
                rows = conn.execute("SELECT * FROM api_keys WHERE user_name = ?", (user_name,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM api_keys").fetchall()
            return [{"id": r[0], "user_name": r[1], "api_key": r[2], "calls_limit": r[3], "calls_used": r[4], "active": r[6]} for r in rows]

    def delete_api_key(self, api_key: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM api_keys WHERE api_key = ?", (api_key,))
            conn.commit()

    def get_usage_stats(self, api_key: str = None) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            if api_key:
                rows = conn.execute("SELECT * FROM usage_log WHERE api_key = ? ORDER BY timestamp DESC LIMIT 50", (api_key,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM usage_log ORDER BY timestamp DESC LIMIT 50").fetchall()
            return [{"id": r[0], "api_key": r[1], "model_id": r[2], "timestamp": r[3], "status": r[4]} for r in rows]
