"""Regression tests for multimodal vision chat support in muse2api."""
import asyncio
import base64
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine as engine_mod

PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/w8AAwMCAO+X2ZkAAAAASUVORK5CYII="
)
DATA_PNG = f"data:image/png;base64,{PNG_1X1_B64}"


def _load_isolated_app(home: str):
    os.environ.update(
        MUSE2API_HOME=home,
        MUSE2API_PROFILE_ROOT=home,
        MUSE2API_KEY="test-only",
        MUSE2API_REF_IMAGE_LOCAL_FILES="0",
    )
    spec = importlib.util.spec_from_file_location("muse_chat_vision_app", ROOT / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class ChatVisionAppTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = self._tmp.name
        self.app = _load_isolated_app(self.home)

    def tearDown(self):
        self._tmp.cleanup()

    def test_extracts_image_urls_and_indexes_in_order(self):
        msgs = [
            self.app.ChatMessage(
                role="system",
                content="Read comic pages accurately.",
            ),
            self.app.ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Page 1:"},
                    {"type": "image_url", "image_url": {"url": DATA_PNG}},
                    {"type": "input_image", "image_url": DATA_PNG + "#p2"},
                    {"type": "text", "text": "Describe both pages."},
                ],
            ),
        ]
        prompt, imgs = self.app.build_chat_prompt_and_images(msgs)
        self.assertEqual(len(imgs), 2)
        self.assertIn("[Image 1]", prompt)
        self.assertIn("[Image 2]", prompt)
        self.assertLess(prompt.index("Page 1:"), prompt.index("[Image 1]"))
        self.assertLess(prompt.index("[Image 1]"), prompt.index("[Image 2]"))
        self.assertEqual(self.app.build_chat_prompt(msgs), prompt)

    def test_rejects_over_limit_in_single_turn_and_trims_multi_turn_history(self):
        too_many = [{"type": "image_url", "image_url": {"url": DATA_PNG}} for _ in range(11)]
        with self.assertRaises(self.app.HTTPException) as ctx:
            self.app.build_chat_prompt_and_images(
                [self.app.ChatMessage(role="user", content=too_many)]
            )
        self.assertEqual(ctx.exception.status_code, 400)

        turn1 = [{"type": "image_url", "image_url": {"url": f"https://img/{i}.png"}} for i in range(6)]
        turn2 = [{"type": "image_url", "image_url": {"url": f"https://img/{i}.png"}} for i in range(6, 12)]
        msgs = [
            self.app.ChatMessage(role="user", content=turn1),
            self.app.ChatMessage(role="assistant", content="Noted."),
            self.app.ChatMessage(role="user", content=turn2),
        ]
        prompt, kept = self.app.build_chat_prompt_and_images(msgs)
        self.assertEqual(len(kept), self.app.MAX_REFERENCE_IMAGES)
        self.assertEqual(kept[0], "https://img/2.png")
        self.assertEqual(kept[-1], "https://img/11.png")
        self.assertEqual(prompt.count("[历史图片已省略]"), 2)
        self.assertIn("[Image 1]", prompt)
        self.assertIn("[Image 10]", prompt)

    def test_chat_completions_rejects_local_path_before_account_pick(self):
        local_img = str(Path(self.home) / "secret.png")
        Path(local_img).write_bytes(base64.b64decode(PNG_1X1_B64))

        called = {"pick": False}
        orig_pick = self.app.store.pick_account
        try:
            self.app.store.pick_account = lambda **kw: called.update(pick=True) or None
            req = self.app.ChatRequest(
                model="muse-spark",
                messages=[
                    self.app.ChatMessage(
                        role="user",
                        content=[{"type": "image_url", "image_url": {"url": local_img}}],
                    )
                ],
            )
            with self.assertRaises(self.app.HTTPException) as ctx:
                asyncio.run(self.app.chat_completions(req, True))
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertFalse(called["pick"])
        finally:
            self.app.store.pick_account = orig_pick

    def test_chat_completions_and_responses_forward_prepared_images(self):
        captured = []
        orig_pick = self.app.store.pick_account
        orig_safe = self.app.safe_chat_stream

        def fake_pick(**kw):
            return {"id": "acc1", "cookies": {"k": "v"}, "cookies_exp": {}}

        def fake_safe(cookies, prompt, expires, timeout, account_id=None, final_out=None, images=None):
            captured.append({"prompt": prompt, "images": list(images or [])})
            if final_out is not None:
                final_out["text"] = "vision ok"
            yield "vision ok"

        try:
            self.app.store.pick_account = fake_pick
            self.app.safe_chat_stream = fake_safe

            chat_req = self.app.ChatRequest(
                model="muse-spark",
                messages=[
                    self.app.ChatMessage(
                        role="user",
                        content=[
                            {"type": "text", "text": "Inspect page"},
                            {"type": "image_url", "image_url": {"url": DATA_PNG}},
                        ],
                    )
                ],
            )
            res = asyncio.run(self.app.chat_completions(chat_req, True))
            self.assertEqual(res["choices"][0]["message"]["content"], "vision ok")
            self.assertEqual(len(captured[-1]["images"]), 1)
            self.assertTrue(captured[-1]["images"][0].startswith("data:image/png;base64,"))

            resp_req = self.app.ResponsesRequest(
                model="muse-spark",
                input=[
                    {"type": "input_text", "text": "Inspect response image"},
                    {"type": "input_image", "image_url": {"url": DATA_PNG}},
                ],
            )
            r_res = asyncio.run(self.app.responses_api(resp_req, True))
            self.assertEqual(r_res["output"][0]["content"][0]["text"], "vision ok")
            self.assertEqual(len(captured[-1]["images"]), 1)
            self.assertIn("[Image 1]", captured[-1]["prompt"])
        finally:
            self.app.store.pick_account = orig_pick
            self.app.safe_chat_stream = orig_safe


class ChatVisionEngineTests(unittest.TestCase):
    def test_chat_stream_attaches_images_and_cleans_temp_dir_on_success_and_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            created_dirs = []
            attached_payloads = []
            reset_calls = []

            eng = object.__new__(engine_mod.MuseEngine)
            eng.cfg = SimpleNamespace(chat_timeout=30, chat_first_token_timeout=10, profile_dir=tmp)
            eng.ensure_page = lambda *a, **kw: None
            eng.reset_thread = lambda **kw: reset_calls.append(kw)
            eng.page = SimpleNamespace(js=lambda *a, **kw: "")

            def fake_attach(imgs):
                attached_payloads.append(list(imgs))
                d = Path(tmp) / f"upload-{len(created_dirs)}"
                d.mkdir(parents=True, exist_ok=True)
                (d / "ref.png").write_bytes(b"png")
                created_dirs.append(d)
                return {"dir": str(d), "hashes": {"h1"}}

            eng._attach_images = fake_attach
            eng._send = lambda prompt: "clicked"
            eng._composer_unsent = lambda: ""
            polls = iter([
                (0, "", False),
                (1, "Kayden has black hair.", False),
                (1, "Kayden has black hair.", False),
                (1, "Kayden has black hair.", False),
                (1, "Kayden has black hair.", False),
                (1, "Kayden has black hair.", False),
            ])
            eng._poll_chat = lambda **kw: next(polls, (1, "Kayden has black hair.", False))

            orig_sleep = engine_mod.time.sleep
            try:
                engine_mod.time.sleep = lambda _: None
                out = eng.chat({}, "Read [Image 1]", images=[DATA_PNG])
            finally:
                engine_mod.time.sleep = orig_sleep

            self.assertEqual(out, "Kayden has black hair.")
            self.assertEqual(reset_calls[-1], {"for_chat": True, "force": True})
            self.assertEqual(attached_payloads, [[DATA_PNG]])
            self.assertFalse(created_dirs[-1].exists(), "upload dir must be removed on success")

            # Verify cleanup also happens when _send fails
            def fail_send(prompt):
                raise engine_mod.MuseGenerationError("send boom")

            eng._send = fail_send
            with self.assertRaises(engine_mod.MuseGenerationError):
                eng.chat({}, "Read [Image 1]", images=[DATA_PNG])
            self.assertFalse(created_dirs[-1].exists(), "upload dir must be removed on error")


if __name__ == "__main__":
    unittest.main()
