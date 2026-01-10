"""Integration tests for diarization feature."""

import pytest
import os
import aiohttp

# Skip if no test audio available
TEST_AUDIO = os.environ.get("TEST_AUDIO", "test_audio/masak.mp3")
API_URL = os.environ.get("STT_API_URL", "http://localhost:9090")


@pytest.fixture
def api_url():
    return API_URL


class TestDiarizationIntegration:
    """Integration tests for /audio/transcriptions with diarization."""

    @pytest.mark.asyncio
    async def test_diarization_none_default(self, api_url):
        """Test that diarization=none returns segments without speaker field."""
        if not os.path.exists(TEST_AUDIO):
            pytest.skip(f"Test audio not found: {TEST_AUDIO}")

        async with aiohttp.ClientSession() as session:
            with open(TEST_AUDIO, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename="test.mp3")
                data.add_field("response_format", "verbose_json")
                data.add_field("diarization", "none")

                async with session.post(
                    f"{api_url}/audio/transcriptions", data=data
                ) as r:
                    assert r.status == 200
                    result = await r.json()

                    # Segments should exist but without speaker field
                    assert "segments" in result
                    if result["segments"]:
                        first_seg = result["segments"][0]
                        # speaker field should not be present
                        assert "speaker" not in first_seg

    @pytest.mark.asyncio
    async def test_diarization_online(self, api_url):
        """Test online diarization returns segments with speaker field."""
        if not os.path.exists(TEST_AUDIO):
            pytest.skip(f"Test audio not found: {TEST_AUDIO}")

        async with aiohttp.ClientSession() as session:
            with open(TEST_AUDIO, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename="test.mp3")
                data.add_field("response_format", "verbose_json")
                data.add_field("diarization", "online")
                data.add_field("speaker_similarity", "0.75")
                data.add_field("speaker_max_n", "5")

                async with session.post(
                    f"{api_url}/audio/transcriptions", data=data
                ) as r:
                    assert r.status == 200
                    result = await r.json()

                    assert "segments" in result
                    if result["segments"]:
                        # All segments should have speaker field
                        for seg in result["segments"]:
                            assert "speaker" in seg
                            assert isinstance(seg["speaker"], int)
                            assert seg["speaker"] >= 0

    @pytest.mark.asyncio
    async def test_diarization_offline(self, api_url):
        """Test offline diarization returns segments with speaker field."""
        if not os.path.exists(TEST_AUDIO):
            pytest.skip(f"Test audio not found: {TEST_AUDIO}")

        async with aiohttp.ClientSession() as session:
            with open(TEST_AUDIO, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename="test.mp3")
                data.add_field("response_format", "verbose_json")
                data.add_field("diarization", "offline")

                async with session.post(
                    f"{api_url}/audio/transcriptions", data=data
                ) as r:
                    # May fail if OSD service is not running
                    if r.status == 503:
                        pytest.skip("OSD service not available")

                    assert r.status == 200
                    result = await r.json()

                    assert "segments" in result
                    if result["segments"]:
                        # All segments should have speaker field
                        for seg in result["segments"]:
                            assert "speaker" in seg
                            assert isinstance(seg["speaker"], int)
                            assert seg["speaker"] >= 0

    @pytest.mark.asyncio
    async def test_diarization_invalid_mode(self, api_url):
        """Test that invalid diarization mode returns 400."""
        if not os.path.exists(TEST_AUDIO):
            pytest.skip(f"Test audio not found: {TEST_AUDIO}")

        async with aiohttp.ClientSession() as session:
            with open(TEST_AUDIO, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename="test.mp3")
                data.add_field("diarization", "invalid")

                async with session.post(
                    f"{api_url}/audio/transcriptions", data=data
                ) as r:
                    assert r.status == 400

    @pytest.mark.asyncio
    async def test_diarization_online_speaker_params(self, api_url):
        """Test online diarization respects speaker parameters."""
        if not os.path.exists(TEST_AUDIO):
            pytest.skip(f"Test audio not found: {TEST_AUDIO}")

        async with aiohttp.ClientSession() as session:
            # Test with max 2 speakers
            with open(TEST_AUDIO, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename="test.mp3")
                data.add_field("response_format", "verbose_json")
                data.add_field("diarization", "online")
                data.add_field("speaker_similarity", "0.5")
                data.add_field("speaker_max_n", "2")

                async with session.post(
                    f"{api_url}/audio/transcriptions", data=data
                ) as r:
                    assert r.status == 200
                    result = await r.json()

                    if result["segments"]:
                        # All speaker IDs should be < speaker_max_n
                        speaker_ids = {seg["speaker"] for seg in result["segments"]}
                        assert all(sid < 2 for sid in speaker_ids)

    @pytest.mark.asyncio
    async def test_diarization_json_format_no_speaker(self, api_url):
        """Test that json format works with diarization (speaker in segments only)."""
        if not os.path.exists(TEST_AUDIO):
            pytest.skip(f"Test audio not found: {TEST_AUDIO}")

        async with aiohttp.ClientSession() as session:
            with open(TEST_AUDIO, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename="test.mp3")
                data.add_field("response_format", "json")
                data.add_field("diarization", "online")

                async with session.post(
                    f"{api_url}/audio/transcriptions", data=data
                ) as r:
                    assert r.status == 200
                    result = await r.json()

                    # json format only returns text
                    assert "text" in result
                    assert "segments" not in result

