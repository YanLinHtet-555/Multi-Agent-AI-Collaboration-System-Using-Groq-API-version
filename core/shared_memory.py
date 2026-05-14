import threading
from typing import Any, Dict, List, Optional


class SharedMemory:
    def __init__(self):
        self._store: Dict[str, str] = {}
        self._message_log: List[Dict[str, str]] = []
        self._lock = threading.Lock()

    def write(self, key: str, value: str, author: str = "system") -> None:
        with self._lock:
            self._store[key] = value
            self._message_log.append({"author": author, "action": "write", "key": key})

    def read(self, key: str) -> Optional[str]:
        with self._lock:
            return self._store.get(key)

    def get_all_context(self) -> str:
        with self._lock:
            if not self._store:
                return "Shared memory is empty."
            sections = []
            for key, value in self._store.items():
                sections.append(f"### {key.upper()}\n{value}")
            return "\n\n".join(sections)

    def get_message_log(self) -> List[Dict[str, str]]:
        with self._lock:
            return list(self._message_log)
