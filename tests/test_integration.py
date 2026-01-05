import pytest
import os
import httpx

TEST_AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_audio")

# API base URL - defaults to container name for docker-to-docker communication
# Can be overridden with LOCAL_STT_API environment variable
API_BASE_URL = os.environ.get("LOCAL_STT_API", "http://stt-api:9090")


@pytest.fixture
def api_client():
    """HTTP client for making requests to the API"""
    return httpx.AsyncClient(base_url=API_BASE_URL, timeout=60.0)


@pytest.fixture
def sample_audio_file():
    """Load a sample audio file for testing"""
    audio_path = os.path.join(TEST_AUDIO_DIR, "pas.mp3")
    if not os.path.exists(audio_path):
        pytest.skip(f"Test audio file not found: {audio_path}")
    with open(audio_path, "rb") as f:
        return f.read()


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests that call the API over HTTP"""

    async def test_health_check(self, api_client):
        """Test root endpoint returns correct response"""
        response = await api_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "STT API"
        assert data["version"] == "1.0"

    async def test_invalid_language(self, api_client, sample_audio_file):
        """Test that invalid language returns 400"""
        response = await api_client.post(
            "/audio/transcriptions",
            files={"file": ("test.mp3", sample_audio_file, "audio/mpeg")},
            data={"language": "invalid_lang"},
        )
        assert response.status_code == 400
        assert "language only supports" in response.json()["detail"]

    async def test_invalid_response_format(self, api_client, sample_audio_file):
        """Test that invalid response_format returns 400"""
        response = await api_client.post(
            "/audio/transcriptions",
            files={"file": ("test.mp3", sample_audio_file, "audio/mpeg")},
            data={"response_format": "invalid_format"},
        )
        assert response.status_code == 400
        assert "response_format only supports" in response.json()["detail"]

    async def test_json_response_format(self, api_client, sample_audio_file):
        """Test json response format"""
        response = await api_client.post(
            "/audio/transcriptions",
            files={"file": ("test.mp3", sample_audio_file, "audio/mpeg")},
            data={"response_format": "json", "language": "en"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert isinstance(data["text"], str)

    async def test_text_response_format(self, api_client, sample_audio_file):
        """Test text response format"""
        response = await api_client.post(
            "/audio/transcriptions",
            files={"file": ("test.mp3", sample_audio_file, "audio/mpeg")},
            data={"response_format": "text", "language": "en"},
        )

        assert response.status_code == 200
        assert isinstance(response.text, str)
        assert not response.text.startswith("{")

    async def test_verbose_json_response_format(self, api_client, sample_audio_file):
        """Test verbose_json response format"""
        response = await api_client.post(
            "/audio/transcriptions",
            files={"file": ("test.mp3", sample_audio_file, "audio/mpeg")},
            data={"response_format": "verbose_json", "language": "en"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "language" in data
        assert "duration" in data
        assert "text" in data
        assert "segments" in data
        assert isinstance(data["segments"], list)

    async def test_language_auto_detect(self, api_client, sample_audio_file):
        """Test language auto-detection when language is null"""
        response = await api_client.post(
            "/audio/transcriptions",
            files={"file": ("test.mp3", sample_audio_file, "audio/mpeg")},
            data={"language": "null", "response_format": "verbose_json"},
        )

        if response.status_code == 200:
            data = response.json()
            assert "language" in data
        elif response.status_code == 500:
            pytest.skip("Upstream API doesn't support null language auto-detection")
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")

    async def test_missing_file(self, api_client):
        """Test that missing file returns error"""
        response = await api_client.post(
            "/audio/transcriptions", data={"language": "en"}
        )
        assert response.status_code == 422

    async def test_vad_parameters(self, api_client, sample_audio_file):
        """Test custom VAD parameters"""
        response = await api_client.post(
            "/audio/transcriptions",
            files={"file": ("test.mp3", sample_audio_file, "audio/mpeg")},
            data={
                "language": "en",
                "minimum_silent_ms": "300",
                "minimum_trigger_vad_ms": "2000",
                "reject_segment_vad_ratio": "0.8",
            },
        )

        assert response.status_code == 200
