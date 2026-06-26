"""
Database layer for Genesis AI - SQLite storage.
Replaces chats.json and projects.json with proper DB.
"""

import sqlite3
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path


class GenesisDB:
    """SQLite database for chats, projects, and memory."""
    
    def __init__(self, db_path: str = "genesis.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    title TEXT DEFAULT 'Untitled',
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    task TEXT DEFAULT 'other',
                    icon TEXT DEFAULT '🤖',
                    model_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL DEFAULT 0.7,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
        print(f"[OK] Database initialized: {self.db_path}")
    
    # --- Chats ---
    def create_chat(self, chat_id: str, title: str = "Untitled") -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO chats (id, title) VALUES (?, ?)",
                (chat_id, title)
            )
            conn.commit()
        return {"id": chat_id, "title": title}
    
    def get_chat(self, chat_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
            if row:
                return {"id": row[0], "title": row[1], "created_at": row[2], "updated_at": row[3]}
        return None
    
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
            conn.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
                (chat_id, role, content)
            )
            conn.execute("UPDATE chats SET updated_at = datetime('now') WHERE id = ?", (chat_id,))
            conn.commit()
    
    def get_messages(self, chat_id: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM messages WHERE chat_id = ? ORDER BY id ASC",
                (chat_id,)
            ).fetchall()
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
            conn.execute(
                "INSERT INTO projects (name, description, task, icon) VALUES (?, ?, ?, ?)",
                (name, desc, task, icon)
            )
            conn.commit()
        return {"name": name, "desc": desc, "task": task, "icon": icon, "modelCount": 0}
    
    def delete_project(self, name: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM projects WHERE name = ?", (name,))
            conn.commit()
    
    # --- Memory ---
    def add_fact(self, subject: str, predicate: str, obj: str, confidence: float = 0.7):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO memory_facts (subject, predicate, object, confidence) VALUES (?, ?, ?, ?)",
                (subject, predicate, str(obj), confidence)
            )
            conn.commit()
    
    def get_facts(self, subject: str = "user") -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT predicate, object, confidence FROM memory_facts WHERE subject = ? ORDER BY confidence DESC LIMIT 20",
                (subject,)
            ).fetchall()
            return [{"predicate": r[0], "object": r[1], "confidence": r[2]} for r in rows]
