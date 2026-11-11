"""database.py —— SQLite 持久化（V1.4：取代本地 JSON 文件存储）。

- 项目根目录生成 agent.db；表 user_states(session_id 主键, state_json, updated_at)。
- WAL 模式 + busy_timeout：并发读写不互相锁死（SQLite 单写多读、写锁等待重试）。
- 每次保存 = 整条状态序列化为 JSON 字符串覆盖写入（upsert，原子事务）。
- 纯 Python 内置 sqlite3，无额外依赖。
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path

# DB_PATH 环境变量：云部署（如 Render 持久化卷）时指向卷内路径；缺省 = 项目根 agent.db
DB_FILE = Path(os.environ.get("DB_PATH") or (Path(__file__).parent / "agent.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_states (
    session_id TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_reminded_at TEXT
)
"""

_REMIND_COLUMN = "last_reminded_at"


def _ensure_remind_column(conn: sqlite3.Connection) -> None:
    """旧库迁移：user_states 若缺 last_reminded_at 列则补上（幂等）。"""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(user_states)")}
    if _REMIND_COLUMN not in cols:
        conn.execute(f"ALTER TABLE user_states ADD COLUMN {_REMIND_COLUMN} TEXT")


def set_db_path(path) -> None:
    """测试用：把数据库切到临时文件（如 smoke_test 的临时目录）。"""
    global DB_FILE
    DB_FILE = Path(path)


def _connect() -> sqlite3.Connection:
    """每个操作一个短连接：线程安全、写完即放锁，并发时靠 busy_timeout 排队。"""
    conn = sqlite3.connect(DB_FILE, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")     # 读不阻塞写、写不阻塞读
    conn.execute("PRAGMA busy_timeout=5000")    # 写锁等待 5 秒而不是立刻报错
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(_SCHEMA)                        # 幂等建表
    _ensure_remind_column(conn)                  # 幂等补列（V1.4.1）
    return conn


def init_db() -> None:
    conn = _connect()
    try:
        conn.commit()
    finally:
        conn.close()


def save_state(session_id: str, state_json: str) -> None:
    """整条状态覆盖写入（不存在则插入）。updated_at 自动刷新。"""
    now = datetime.now().isoformat(timespec="seconds")
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO user_states (session_id, state_json, updated_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "state_json = excluded.state_json, updated_at = excluded.updated_at",
            (session_id, state_json, now),
        )
        conn.commit()
    finally:
        conn.close()


def load_state(session_id: str) -> str | None:
    """读取某会话的完整状态 JSON 字符串；不存在返回 None。"""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT state_json FROM user_states WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def iter_sessions() -> list[tuple[str, str]]:
    """列出所有会话：[(session_id, state_json), ...]。供晚间提醒脚本遍历。"""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT session_id, state_json FROM user_states ORDER BY session_id"
        ).fetchall()
        return [(r[0], r[1]) for r in rows]
    finally:
        conn.close()


def get_last_reminded(session_id: str) -> str | None:
    """读取某会话上次被晚间提醒的日期（ISO 字符串或 None）。"""
    conn = _connect()
    try:
        row = conn.execute(
            f"SELECT {_REMIND_COLUMN} FROM user_states WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def set_last_reminded(session_id: str, day: str) -> None:
    """记录某会话的晚间提醒日期（幂等标记，当天重跑直接跳过）。"""
    conn = _connect()
    try:
        conn.execute(
            f"UPDATE user_states SET {_REMIND_COLUMN} = ? WHERE session_id = ?",
            (day, session_id),
        )
        conn.commit()
    finally:
        conn.close()


def import_legacy_files() -> int:
    """一次性迁移：把旧版 state_*.json 文件导入数据库（库里已有的 session 跳过）。
    旧文件保留在磁盘上，不再被读写。返回迁移条数。"""
    n = 0
    for p in sorted(Path(__file__).parent.glob("state_*.json")):
        name = p.name
        if name.endswith(".bak") or name == "state_legacy_backup.json":
            continue
        sid = p.stem[len("state_"):]
        if load_state(sid) is not None:
            continue
        try:
            save_state(sid, p.read_text(encoding="utf-8"))
            n += 1
        except Exception:
            pass
    return n
