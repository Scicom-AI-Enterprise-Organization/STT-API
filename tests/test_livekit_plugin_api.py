"""The import surface an agent uses: `from stt_api.livekit_plugin import ...`.

One import line for all three plugins, resolved lazily — reaching for the turn
detector must not load onnxruntime, read the dummy TTS audio file, or register an
inference runner for a plugin the agent never touches. And when the extra is
missing, the error has to say which extra, not just "No module named 'livekit'".
"""

import subprocess
import sys
import textwrap

import pytest

import stt_api.livekit_plugin as plugin


def run_python(*sources: str) -> subprocess.CompletedProcess:
    """Run snippets in a fresh interpreter, so sys.modules starts clean.

    Each snippet is dedented on its own and the results are joined, so a prelude
    and a body written at different indentation levels still line up.
    """
    source = "\n".join(textwrap.dedent(s) for s in sources)
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=False,
    )


class TestExportedNames:
    def test_all_lists_every_plugin(self):
        assert set(plugin.__all__) == {
            "WhisperSTT",
            "DropBlankSTT",
            "PulseVAD",
            "PulseVADModel",
            "SemanticVAD",
            "SmartTurnV3",
            "ScicomEoT",
            "MultilingualModel",
            "GTCRN",
            "MODEL_SAMPLE_RATE",
            "DummySTT",
            "DummyLLM",
            "DummyTTS",
        }

    def test_dir_matches_all(self):
        assert dir(plugin) == sorted(plugin.__all__)

    def test_every_exported_name_points_at_a_real_submodule_attribute(self):
        for name, (submodule, attribute) in plugin._EXPORTS.items():
            assert name in plugin.__all__
            module = __import__(
                f"stt_api.livekit_plugin.{submodule}", fromlist=[attribute]
            )
            assert hasattr(module, attribute), f"{submodule}.{attribute} is gone"

    def test_unknown_name_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="has no attribute 'Nope'"):
            plugin.Nope  # noqa: B018 — the lookup is the thing under test

    def test_install_hint_is_the_documented_command(self):
        assert "stt-api[scicom-livekit-plugin]" in plugin.INSTALL_HINT
        assert "git+https://github.com/" in plugin.INSTALL_HINT


class TestLaziness:
    def test_importing_the_package_loads_no_plugin_dependency(self):
        done = run_python(
            """
            import sys
            import stt_api.livekit_plugin  # noqa: F401
            heavy = [m for m in ("livekit", "onnxruntime", "numpy", "soundfile", "transformers")
                     if m in sys.modules]
            assert not heavy, heavy
            print("clean")
            """
        )
        assert done.returncode == 0, done.stderr
        assert "clean" in done.stdout

    def test_the_turn_detector_does_not_drag_in_the_other_plugins(self):
        done = run_python(
            """
            import sys
            from stt_api.livekit_plugin import MultilingualModel  # noqa: F401
            # onnxruntime belongs to noise cancellation, soundfile to the dummy TTS.
            # livekit-agents may import either for its own reasons; what must not
            # happen is our modules pulling them in.
            siblings = [m for m in sys.modules if m.startswith("stt_api.livekit_plugin.")
                        and not m.startswith("stt_api.livekit_plugin.turn_detector")]
            assert not siblings, siblings
            print("clean")
            """
        )
        if done.returncode != 0 and "scicom-livekit-plugin" in done.stderr:
            pytest.skip("livekit extra not installed")
        assert done.returncode == 0, done.stderr
        assert "clean" in done.stdout

    def test_a_resolved_name_is_cached_on_the_module(self):
        pytest.importorskip("livekit.rtc", reason="livekit not installed")
        vars(plugin).pop("GTCRN", None)  # back to the state a fresh import is in
        first = plugin.GTCRN
        # Written into the module's namespace, so every later read is a plain
        # attribute lookup rather than another trip through importlib.
        assert vars(plugin)["GTCRN"] is first
        assert plugin.GTCRN is first


class TestMissingExtra:
    """What an agent sees when it installed plain stt-api by mistake."""

    # Stand in for an environment where the extra was never installed.
    BLOCK_LIVEKIT = """
        import sys

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == "livekit" or name.startswith("livekit."):
                    raise ModuleNotFoundError(f"No module named '{name}'", name=name)
                return None

        sys.meta_path.insert(0, Blocker())
    """

    def test_error_names_the_extra_and_the_install_command(self):
        done = run_python(
            self.BLOCK_LIVEKIT,
            """
            try:
                from stt_api.livekit_plugin import DummySTT  # noqa: F401
            except ModuleNotFoundError as exc:
                print(exc)
            else:
                raise AssertionError("expected ModuleNotFoundError")
            """,
        )
        assert done.returncode == 0, done.stderr
        assert "scicom-livekit-plugin" in done.stdout
        assert "uv pip install" in done.stdout
        assert "livekit" in done.stdout  # the dependency that is actually missing

    def test_each_plugin_resolves_on_its_own(self):
        """Any one name works without the others being importable in this env."""
        pytest.importorskip("livekit.agents", reason="livekit not installed")
        for name in plugin.__all__:
            done = run_python(
                f"""
                from stt_api.livekit_plugin import {name}
                print({name!r}, "ok")
                """
            )
            assert done.returncode == 0, f"{name}: {done.stderr[-500:]}"
            assert f"{name} ok" in done.stdout.replace("'", "")

    def test_error_keeps_the_missing_module_name(self):
        done = run_python(
            self.BLOCK_LIVEKIT,
            """
            try:
                from stt_api.livekit_plugin import GTCRN  # noqa: F401
            except ModuleNotFoundError as exc:
                print(exc.name)
                print(type(exc.__cause__).__name__)
            """,
        )
        assert done.returncode == 0, done.stderr
        # `except ModuleNotFoundError as e: e.name` still works for callers, and
        # the original traceback is chained rather than swallowed.
        assert done.stdout.splitlines()[:2] == ["livekit", "ModuleNotFoundError"]


class TestPluginsStayIndependentOfTheServer:
    def test_no_plugin_module_imports_the_server_side(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent / "stt_api" / "livekit_plugin"
        server_modules = ("stt_api.main", "stt_api.diarization", "stt_api.vad", "stt_api.nemo")
        for path in root.rglob("*.py"):
            source = path.read_text()
            for module in server_modules:
                assert f"import {module}" not in source, f"{path.name} imports {module}"
