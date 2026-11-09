import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine as engine_mod


class ChatNonStreamSnapshotTests(unittest.TestCase):
    def test_chat_returns_final_dom_snapshot_when_react_rewrites_prefix(self):
        cfg = SimpleNamespace(chat_timeout=30, chat_first_token_timeout=10, profile_dir=str(ROOT / "data"))
        eng = object.__new__(engine_mod.MuseEngine)
        eng.cfg = cfg
        eng.ensure_page = lambda *a, **kw: None
        eng.reset_thread = lambda **kw: None
        eng._send = lambda prompt: "clicked"
        eng._composer_unsent = lambda: ""
        eng.page = SimpleNamespace(js=lambda *a, **kw: "")

        polls = iter([
            (0, "", False),  # baseline before send
            (1, '{"ok": true, "items": 1,', True),
            (1, '{"ok": true, "items": [1, 2, 3]}', True),
            (1, '{"ok": true, "items": [1, 2, 3]}', False),
            (1, '{"ok": true, "items": [1, 2, 3]}', False),
            (1, '{"ok": true, "items": [1, 2, 3]}', False),
            (1, '{"ok": true, "items": [1, 2, 3]}', False),
            (1, '{"ok": true, "items": [1, 2, 3]}', False),
        ])
        eng._poll_chat = lambda: next(polls, (1, '{"ok": true, "items": [1, 2, 3]}', False))

        orig_sleep = engine_mod.time.sleep
        try:
            engine_mod.time.sleep = lambda _: None
            result = eng.chat({}, "test prompt")
        finally:
            engine_mod.time.sleep = orig_sleep

        self.assertEqual(result, '{"ok": true, "items": [1, 2, 3]}')


if __name__ == "__main__":
    unittest.main()
