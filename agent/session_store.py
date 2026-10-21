"""Wave 12 — 프로세스 밖(SQLite 파일)에 두는 세션 저장소.

서버가 재시작돼도(새 프로세스, 새 인스턴스) 같은 session.db 파일을 다시 열므로
대화 이력이 유지된다. in-memory dict였다면 재시작 즉시 소실됐을 상태다.
"""
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

DB_PATH = Path(__file__).parent.parent / "session.db"


@dataclass(frozen=True)
class PendingTarget:
    """질문 내용이 같아도 서로 다른 확인 요청을 구분한다."""
    request_id: str
    query: str


class SqliteSessionStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            " thread_id TEXT, role TEXT, content TEXT, created TEXT)"
        )
        # 이전 버전의 질문을 보존하면서 식별자 열을 추가한다. 초기화끼리 경합하지 않게 잠근다.
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS pending_targets ("
                " thread_id TEXT PRIMARY KEY, query TEXT NOT NULL, request_id TEXT NOT NULL)"
            )
            columns = {row[1] for row in self.conn.execute("PRAGMA table_info(pending_targets)")}
            if "request_id" not in columns:
                self.conn.execute("ALTER TABLE pending_targets ADD COLUMN request_id TEXT")
                for (thread_id,) in self.conn.execute("SELECT thread_id FROM pending_targets").fetchall():
                    self.conn.execute("UPDATE pending_targets SET request_id=? WHERE thread_id=?",
                                      (uuid4().hex, thread_id))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            self.conn.close()
            raise

    def load(self, thread_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE thread_id=? ORDER BY rowid",
            (thread_id,),
        ).fetchall()
        return [{"role": r, "content": c} for r, c in rows]

    def append(self, thread_id: str, role: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO messages VALUES (?,?,?,?)",
            (thread_id, role, content, datetime.now().isoformat()),
        )
        self.conn.commit()

    def threads(self, prefix: str) -> list[dict]:
        """접두사가 같은 대화를 최근 순으로, 첫 질문을 제목 삼아 돌려준다."""
        rows = self.conn.execute(
            "SELECT thread_id, MAX(created),"
            " (SELECT content FROM messages f WHERE f.thread_id=m.thread_id AND f.role='user'"
            "  ORDER BY rowid LIMIT 1)"
            " FROM messages m WHERE substr(thread_id, 1, length(?))=?"
            " GROUP BY thread_id ORDER BY MAX(rowid) DESC",
            (prefix, prefix),
        ).fetchall()
        return [{"thread_id": t, "updated": u, "title": title or ""} for t, u, title in rows]

    def close(self) -> None:
        self.conn.close()

    def pending_target(self, thread_id: str) -> PendingTarget | None:
        """확인 대기 중인 원래 질문과 요청 식별자를 함께 가져온다."""
        row = self.conn.execute(
            "SELECT request_id, query FROM pending_targets WHERE thread_id=?", (thread_id,),
        ).fetchone()
        return PendingTarget(*row) if row else None

    def set_pending_target(self, thread_id: str, query: str) -> None:
        """재시작 뒤에도 대상 선택으로 원래 질문을 이어갈 수 있도록 저장한다."""
        self.conn.execute(
            "INSERT INTO pending_targets (thread_id, query, request_id) VALUES (?, ?, ?) "
            "ON CONFLICT(thread_id) DO UPDATE SET query=excluded.query, request_id=excluded.request_id",
            (thread_id, query, uuid4().hex),
        )
        self.conn.commit()

    def clear_pending_target(self, thread_id: str, *, expected_request_id: str | None = None) -> None:
        """진단 완료 시 그 요청만 지운다. 같은 문장의 새 확인 요청도 보존한다."""
        if expected_request_id is None:
            self.conn.execute("DELETE FROM pending_targets WHERE thread_id=?", (thread_id,))
        else:
            self.conn.execute("DELETE FROM pending_targets WHERE thread_id=? AND request_id=?",
                              (thread_id, expected_request_id))
        self.conn.commit()
