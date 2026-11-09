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

    def test_chat_preserves_multi_bubble_turn_with_since_idx(self):
        cfg = SimpleNamespace(chat_timeout=30, chat_first_token_timeout=10, profile_dir=str(ROOT / "data"))
        eng = object.__new__(engine_mod.MuseEngine)
        eng.cfg = cfg
        reset_calls = []
        eng.ensure_page = lambda *a, **kw: None
        eng.reset_thread = lambda **kw: reset_calls.append(kw)
        eng._send = lambda prompt: "clicked"
        eng._composer_unsent = lambda: ""

        import json as _json
        js_calls = []

        def fake_js(expr, **kw):
            js_calls.append(expr)
            if "document.body.innerText" in expr:
                return ""
            # First poll before send: sinceIdx is null, 1 prior bubble exists
            if expr.endswith("(null)"):
                return _json.dumps({"cnt": 1, "total": 1, "txt": "previous reply", "stop": False})
            # Subsequent polls with sinceIdx=1: simulate Koda posting Bubble 1 (story) + Bubble 2 (short check)
            since_calls = [c for c in js_calls if c.endswith("(1)")]
            idx = len(since_calls)
            if idx == 1:
                return _json.dumps({"cnt": 2, "total": 2, "txt": "FULL STORY TEXT", "stop": True})
            if idx == 2:
                return _json.dumps({"cnt": 3, "total": 3, "txt": "FULL STORY TEXT\n\nVerified before delivery", "stop": True})
            return _json.dumps({"cnt": 3, "total": 3, "txt": "FULL STORY TEXT\n\nVerified before delivery", "stop": False})

        eng.page = SimpleNamespace(js=fake_js)
        orig_sleep = engine_mod.time.sleep
        try:
            engine_mod.time.sleep = lambda _: None
            result = eng.chat({}, "x" * 1600)
        finally:
            engine_mod.time.sleep = orig_sleep

        self.assertEqual(reset_calls[-1], {"for_chat": True, "force": True})
        self.assertIn("FULL STORY TEXT", result)
        self.assertIn("Verified before delivery", result)
        self.assertTrue(any(c.endswith("(1)") for c in js_calls))


if __name__ == "__main__":
    unittest.main()
