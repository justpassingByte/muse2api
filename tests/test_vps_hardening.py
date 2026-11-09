"""Offline unit tests for 24/7 VPS hardening (log cleanup/rotation, repo parameterization, git config safety)."""
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine as engine_mod


def _import_app_isolated(home: str, extra_env: dict[str, str] | None = None):
    env_patch = {
        "MUSE2API_HOME": home,
        "MUSE2API_PROFILE_ROOT": home,
        "MUSE2API_KEY": "test-only",
    }
    if extra_env:
        env_patch.update(extra_env)
    saved_modules = {name: sys.modules.get(name) for name in ("config", "store", "engine", "cdp")}
    try:
        with patch.dict(os.environ, env_patch, clear=False):
            for name in ("config", "store", "engine", "cdp"):
                sys.modules.pop(name, None)
            spec = importlib.util.spec_from_file_location("muse2api_vps_test_app", ROOT / "app.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    finally:
        for name, mod in saved_modules.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


class EngineVpsHardeningTests(unittest.TestCase):
    def test_stop_closes_chromium_log_handle(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                profile_dir=os.path.join(tmp, "profile"),
                data_dir=tmp,
                download_dir=os.path.join(tmp, "downloads"),
            )
            eng = engine_mod.MuseEngine(cfg)
            log_file = Path(tmp) / "chromium.log"
            eng._log = open(log_file, "ab", buffering=0)
            eng.stop()
            self.assertIsNone(eng._log)

    def test_start_rotates_oversized_chromium_log_and_closes_prior_handle(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                profile_dir=os.path.join(tmp, "profile"),
                data_dir=tmp,
                download_dir=os.path.join(tmp, "downloads"),
                home_dir=tmp,
                extra_path="",
                chromium="chromium",
                cdp_port=19999,
            )
            eng = engine_mod.MuseEngine(cfg)
            prior_handle = MagicMock()
            eng._log = prior_handle

            log_file = Path(tmp) / "chromium.log"
            with open(log_file, "wb") as f:
                f.seek(20 * 1024 * 1024 + 10)
                f.write(b"x")

            fake_proc = MagicMock()
            fake_proc.poll.return_value = None
            with patch.object(engine_mod.subprocess, "Popen", return_value=fake_proc), \
                 patch.object(engine_mod, "http_json", side_effect=[OSError("not running"), {"webSocketDebuggerUrl": "ws://127.0.0.1:19999/dev"}]), \
                 patch.object(engine_mod, "CDP", return_value=MagicMock()):
                eng.start()

            prior_handle.close.assert_called_once()
            self.assertTrue((Path(tmp) / "chromium.log.1").is_file())
            self.assertTrue(log_file.is_file())
            self.assertLess(log_file.stat().st_size, 1024)
            eng.stop()
            self.assertIsNone(eng._log)


class RepoConfigurationTests(unittest.TestCase):
    def test_muse2api_repo_parameterizes_update_and_tarball_urls(self):
        with tempfile.TemporaryDirectory() as home:
            app_mod = _import_app_isolated(home, {"MUSE2API_REPO": "https://github.com/custom-org/custom-muse.git"})
            self.assertEqual(app_mod.MUSE2API_REPO, "custom-org/custom-muse")
            self.assertEqual(app_mod.REPO_URL, "https://github.com/custom-org/custom-muse")

            requested_urls = []

            class FakeResp:
                def __init__(self, status_code, payload=None, content=b""):
                    self.status_code = status_code
                    self._payload = payload
                    self.content = content

                def json(self):
                    return self._payload

            def fake_get(url, headers=None, timeout=None):
                requested_urls.append(url)
                if "version.json" in url:
                    return FakeResp(200, {"version": "9.9.9", "highlights": ["ok"]})
                if "/commits" in url:
                    return FakeResp(200, [{"sha": "abcdef1234567", "commit": {"message": "test commit", "committer": {"date": "2026-11-09T00:00:00Z"}}}])
                if "tar.gz" in url:
                    return FakeResp(502)
                raise AssertionError(f"unexpected url: {url}")

            app_mod._UPDATE_CACHE.update(ts=0.0, data=None)
            with patch("requests.get", side_effect=fake_get):
                status = app_mod._check_update_sync(force=True)

            self.assertEqual(status["repo_url"], "https://github.com/custom-org/custom-muse")
            self.assertTrue(any("raw.githubusercontent.com/custom-org/custom-muse/main/version.json" in u for u in requested_urls))
            self.assertTrue(any("api.github.com/repos/custom-org/custom-muse/commits" in u for u in requested_urls))

            with patch.object(app_mod, "_ensure_git_repo"), \
                 patch.object(app_mod, "_git", return_value=SimpleNamespace(returncode=1, stdout="", stderr="")), \
                 patch("requests.get", side_effect=fake_get):
                with self.assertRaises(app_mod.HTTPException):
                    app_mod._upgrade_from_github_sync()

            self.assertIn("https://codeload.github.com/custom-org/custom-muse/tar.gz/refs/heads/main", requested_urls)

    def test_ensure_git_repo_preserves_existing_git_identity(self):
        with tempfile.TemporaryDirectory() as home:
            app_mod = _import_app_isolated(home, {"MUSE2API_REPO": "justpassingByte/muse2api"})
            calls = []

            def fake_git(args, timeout=30):
                calls.append(args)
                if args == ["config", "user.name"]:
                    return SimpleNamespace(returncode=0, stdout="Existing User\n", stderr="")
                if args == ["config", "user.email"]:
                    return SimpleNamespace(returncode=0, stdout="existing@example.com\n", stderr="")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch("os.path.isdir", return_value=True), \
                 patch.object(app_mod, "_git", side_effect=fake_git):
                app_mod._ensure_git_repo("")

            self.assertIn(["remote", "set-url", "origin", "https://github.com/justpassingByte/muse2api.git"], calls)
            self.assertFalse(any(len(c) >= 3 and c[:2] == ["config", "user.name"] for c in calls))
            self.assertFalse(any(len(c) >= 3 and c[:2] == ["config", "user.email"] for c in calls))


if __name__ == "__main__":
    unittest.main()
