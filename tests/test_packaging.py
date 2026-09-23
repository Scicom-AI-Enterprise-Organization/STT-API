"""Packaging tests for the standalone LiveKit plugin install.

The promise this file holds up:

    uv pip install "stt-api[scicom-livekit-plugin] @ git+https://github.com/Scicom-AI-Enterprise-Organization/STT-API.git"

puts the three LiveKit plugins and their runtime into an agent's environment and
nothing else from this repo's stack — no torch, no fastapi, no VAD or diarization
models. It is a packaging promise, so it is checked against the packaging
metadata rather than against whatever happens to be installed in this venv.
"""

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # 3.10
    tomllib = pytest.importorskip("tomli", reason="tomli needed to read pyproject on 3.10")

REPO = Path(__file__).resolve().parent.parent
PLUGIN_EXTRA = "scicom-livekit-plugin"

# Anything here in the plugin extra means an agent is being handed the server's
# stack: several GB of wheels for modules it will never import.
SERVER_ONLY = {
    "torch",
    "torchaudio",
    "fastapi",
    "uvicorn",
    "librosa",
    "silero-vad",
    "kaldiio",
    "kaldi-native-fbank",
    "uroman",
    "unidecode",
    "python-multipart",
    # Shootout and figure tooling lives inside the plugin packages but belongs to
    # [benchmark]; none of it should reach an agent image either.
    "pesq",
    "pystoi",
    "pyrnnoise",
    "soxr",
    "boto3",
    "matplotlib",
    "livekit-plugins-silero",
}


@pytest.fixture(scope="module")
def pyproject():
    with open(REPO / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


@pytest.fixture(scope="module")
def extras(pyproject):
    return pyproject["project"]["optional-dependencies"]


def base_name(requirement: str) -> str:
    """'livekit-agents[turn-detector]~=1.2' -> 'livekit-agents'."""
    return re.split(r"[\[<>=!~; ]", requirement.strip(), maxsplit=1)[0].lower().replace("_", "-")


class TestExtras:
    def test_livekit_plugin_extra_exists(self, extras):
        assert PLUGIN_EXTRA in extras, (
            "the documented install name is stt-api[scicom-livekit-plugin]; renaming it "
            "breaks every agent that pins it"
        )

    def test_livekit_alias_resolves_to_the_same_extra(self, extras):
        # The pre-rename name. A self-reference rather than a copy of the list,
        # so the two can never drift apart.
        assert [base_name(r) for r in extras["livekit"]] == ["stt-api"]
        assert f"[{PLUGIN_EXTRA}]" in extras["livekit"][0]

    def test_no_server_dependency_in_the_plugin_extra(self, extras):
        leaked = {base_name(r) for r in extras[PLUGIN_EXTRA]} & SERVER_ONLY
        assert not leaked, f"{leaked} would be installed into every agent"

    def test_plugin_extra_covers_what_the_plugin_modules_import(self, extras):
        declared = {base_name(r) for r in extras[PLUGIN_EXTRA]}
        assert {
            "livekit-agents",   # all of them
            "numpy",            # pulse_vad, semantic_vad, whisper_stt, noise_cancellation
            "onnxruntime",      # pulse_vad, semantic_vad, noise_cancellation
            "aiohttp",          # turn_detector, whisper_stt
            "transformers",     # turn_detector, semantic_vad
            "huggingface-hub",  # semantic_vad
            "soundfile",        # dummy
        } <= declared

    def test_turn_detector_backend_is_pulled_in(self, extras):
        # MultilingualModel subclasses livekit.plugins.turn_detector.base, which
        # only ships with this extra of livekit-agents.
        agents = [r for r in extras[PLUGIN_EXTRA] if base_name(r) == "livekit-agents"]
        assert agents and "turn-detector" in agents[0]

    def test_base_dependencies_are_empty(self, pyproject):
        # Everything belongs to an extra. Otherwise a plugin install silently
        # carries whatever the server happens to need this month.
        assert pyproject["project"]["dependencies"] == []

    def test_server_extra_still_declares_what_it_imports(self, extras):
        declared = {base_name(r) for r in extras["server"]}
        # Moved out of the base dependencies, so they have to be here now.
        assert {"aiohttp", "transformers"} <= declared
        assert {"fastapi", "torch"} <= declared


class TestShippedResources:
    """Data files the plugins read at runtime have to be in the wheel."""

    def test_package_data_covers_every_non_python_file(self, pyproject):
        patterns = pyproject["tool"]["setuptools"]["package-data"]
        # Expand the globs rather than trusting the literal strings.
        covered = set()
        for package, globs in patterns.items():
            root = REPO / package.replace(".", "/")
            for pattern in globs:
                covered.update(p.resolve() for p in root.glob(pattern))

        plugin_root = REPO / "stt_api" / "livekit_plugin"
        runtime_files = {
            p.resolve()
            for p in plugin_root.rglob("*")
            if p.is_file()
            and p.suffix in {".onnx", ".npy", ".wav", ".mp3", ".json", ".txt", ".bin"}
            and "__pycache__" not in p.parts
        }
        missing = runtime_files - covered
        assert not missing, (
            f"{sorted(str(p.relative_to(REPO)) for p in missing)} would be left out of "
            "the wheel; add a [tool.setuptools.package-data] glob for it"
        )

    def test_every_plugin_package_is_reachable_from_the_top_level(self):
        # A new plugin under stt_api/livekit_plugin is only "installable from the
        # extra" once it is exported here too, which is easy to forget.
        import stt_api.livekit_plugin as plugin

        packages = {
            d.name
            for d in (REPO / "stt_api" / "livekit_plugin").iterdir()
            if d.is_dir() and (d / "__init__.py").is_file() and d.name != "benchmark"
        }
        exported = {submodule for submodule, _ in plugin._EXPORTS.values()}
        assert packages <= exported, f"not exported from stt_api.livekit_plugin: {packages - exported}"

    def test_gtcrn_weights_exist_in_the_checkout(self):
        assert (
            REPO / "stt_api/livekit_plugin/noise_cancellation/resources/gtcrn_simple.onnx"
        ).is_file()


@pytest.mark.slow
class TestBuiltWheel:
    """The end of the chain: build the wheel and read what pip would read.

    Skipped when uv or a package index is out of reach, so an offline checkout
    still runs the rest of the file.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def wheel(cls, tmp_path_factory):
        if shutil.which("uv") is None:
            pytest.skip("uv not installed")
        # Build from a copy: setuptools drops a build/ directory next to the
        # sources it is given, and a test has no business dirtying the checkout.
        source = tmp_path_factory.mktemp("src")
        shutil.copy(REPO / "pyproject.toml", source)
        shutil.copy(REPO / "README.md", source)
        shutil.copytree(
            REPO / "stt_api",
            source / "stt_api",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        out = tmp_path_factory.mktemp("dist")
        built = subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(out), str(source)],
            capture_output=True,
            text=True,
            check=False,
        )
        if built.returncode != 0:
            # No uv cache and no index, most likely. Offline checkouts still run
            # every other test in this file.
            pytest.skip(f"wheel build unavailable: {built.stderr.strip()[-300:]}")
        return zipfile.ZipFile(next(out.glob("*.whl")))

    @pytest.fixture(scope="class")
    @classmethod
    def metadata(cls, wheel):
        name = next(n for n in wheel.namelist() if n.endswith(".dist-info/METADATA"))
        return wheel.read(name).decode()

    def test_extra_is_advertised(self, metadata):
        assert f"Provides-Extra: {PLUGIN_EXTRA}" in metadata
        assert "Provides-Extra: livekit" in metadata

    def test_nothing_is_installed_unconditionally(self, metadata):
        unconditional = [
            line
            for line in metadata.splitlines()
            if line.startswith("Requires-Dist:") and "extra ==" not in line
        ]
        assert unconditional == [], f"installed even without an extra: {unconditional}"

    def test_plugin_extra_pulls_no_server_wheel(self, metadata):
        for line in metadata.splitlines():
            if line.startswith("Requires-Dist:") and f'extra == "{PLUGIN_EXTRA}"' in line:
                assert base_name(line.split(":", 1)[1]) not in SERVER_ONLY, line

    def test_runtime_resources_are_in_the_wheel(self, wheel):
        names = set(wheel.namelist())
        assert "stt_api/livekit_plugin/noise_cancellation/resources/gtcrn_simple.onnx" in names
        assert "stt_api/livekit_plugin/noise_cancellation/resources/LICENSE.gtcrn" in names
        assert "stt_api/livekit_plugin/dummy/audio/tawaran.wav" in names

    def test_plugin_modules_are_in_the_wheel(self, wheel):
        names = set(wheel.namelist())
        for module in ("turn_detector/multilingual", "noise_cancellation/gtcrn", "dummy/stt"):
            assert f"stt_api/livekit_plugin/{module}.py" in names
