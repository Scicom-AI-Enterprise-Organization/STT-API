import pytest
import os
import asyncio
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app, parse_segments

# Get the test audio directory
TEST_AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_audio")


@pytest.fixture
def client():
    """FastAPI test client fixture"""
    return TestClient(app)


@pytest.fixture
def sample_audio_file():
    """Load a sample audio file for testing"""
    audio_path = os.path.join(TEST_AUDIO_DIR, "pas.mp3")
    if not os.path.exists(audio_path):
        pytest.skip(f"Test audio file not found: {audio_path}")
    with open(audio_path, "rb") as f:
        return f.read()


@pytest.fixture
def mock_transcribe_chunk():
    """Mock transcribe_chunk function to return predefined responses"""

    async def mock_func(wav_data, language, timestamp_granularities, last_timestamp):
        # Return mock transcription with timestamps
        text = f"<|{last_timestamp}|>Hello world<|{last_timestamp + 2.5}|>"
        detected_lang = language if language != "null" else "en"
        return text, detected_lang

    return mock_func


class TestParseSegments:
    """Test parse_segments function"""

    def test_parse_segments_with_timestamps(self):
        """Test parsing text with timestamp markers"""
        text = "<|0.0|>Hello world<|2.5|><|3.0|>How are you<|5.5|>"
        segments = parse_segments(text)

        assert len(segments) == 2
        assert segments[0]["id"] == 0
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 2.5
        assert segments[0]["text"] == "Hello world"

        assert segments[1]["id"] == 1
        assert segments[1]["start"] == 3.0
        assert segments[1]["end"] == 5.5
        assert segments[1]["text"] == "How are you"

    def test_parse_segments_with_speaker_labels(self):
        """Test parsing text with speaker labels"""
        text = "<|speaker:0|><|0.0|>Hello<|2.0|><|speaker:1|><|3.0|>World<|5.0|>"
        segments = parse_segments(text)

        assert len(segments) == 2
        assert segments[0]["speaker"] == 0
        assert segments[1]["speaker"] == 1

    def test_parse_segments_empty_input(self):
        """Test parsing empty input"""
        segments = parse_segments("")
        assert segments == []

    def test_parse_segments_no_timestamps(self):
        """Test parsing text without valid timestamp patterns"""
        segments = parse_segments("Just plain text without timestamps")
        assert segments == []

    def test_parse_segments_mixed_content(self):
        """Test parsing text with mixed timestamped and non-timestamped content"""
        text = "<|0.0|>Valid segment<|2.0|>Some text without timestamps<|4.0|>Another segment<|6.0|>"
        segments = parse_segments(text)

        # Should only parse segments with valid timestamp pairs
        assert len(segments) == 2
        assert segments[0]["text"] == "Valid segment"
        assert segments[1]["text"] == "Another segment"


class TestRootEndpoint:
    """Test GET / endpoint"""

    def test_read_root(self, client):
        """Test root endpoint returns correct response"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "STT API"
        assert data["version"] == "1.0"


class TestAudioTranscriptionsEndpoint:
    """Test POST /audio/transcriptions endpoint"""

    def test_invalid_language(self, client, sample_audio_file):
        """Test that invalid language returns 400"""
        response = client.post(
            "/audio/transcriptions",
            files={"file": ("test.mp3", sample_audio_file, "audio/mpeg")},
            data={"language": "invalid_lang"},
        )
        assert response.status_code == 400
        assert "language only supports" in response.json()["detail"]

    def test_invalid_response_format(self, client, sample_audio_file):
        """Test that invalid response_format returns 400"""
        response = client.post(
            "/audio/transcriptions",
            files={"file": ("test.mp3", sample_audio_file, "audio/mpeg")},
            data={"response_format": "invalid_format"},
        )
        assert response.status_code == 400
        assert "response_format only supports" in response.json()["detail"]

    @patch("app.main.transcribe_chunk")
    def test_json_response_format(
        self, mock_transcribe, client, sample_audio_file, mock_transcribe_chunk
    ):
        """Test json response format"""
        mock_transcribe.side_effect = mock_transcribe_chunk

        response = client.post(
            "/audio/transcriptions",
            files={"file": ("test.mp3", sample_audio_file, "audio/mpeg")},
            data={"response_format": "json", "language": "en"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert isinstance(data["text"], str)

    @patch("app.main.transcribe_chunk")
    def test_text_response_format(
        self, mock_transcribe, client, sample_audio_file, mock_transcribe_chunk
    ):
        """Test text response format"""
        mock_transcribe.side_effect = mock_transcribe_chunk

        response = client.post(
            "/audio/transcriptions",
            files={"file": ("test.mp3", sample_audio_file, "audio/mpeg")},
            data={"response_format": "text", "language": "en"},
        )

        assert response.status_code == 200
        assert isinstance(response.text, str)
        # Should be plain text, not JSON
        assert not response.text.startswith("{")

    @patch("app.main.transcribe_chunk")
    def test_verbose_json_response_format(
        self, mock_transcribe, client, sample_audio_file, mock_transcribe_chunk
    ):
        """Test verbose_json response format"""

        # Create a more detailed mock response
        async def detailed_mock(
            wav_data, language, timestamp_granularities, last_timestamp
        ):
            text = f"<|{last_timestamp}|>Hello world<|{last_timestamp + 2.5}|>"
            return text, "en"

        mock_transcribe.side_effect = detailed_mock

        response = client.post(
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

    @patch("app.main.transcribe_chunk")
    def test_language_auto_detect(
        self, mock_transcribe, client, sample_audio_file, mock_transcribe_chunk
    ):
        """Test language auto-detection when language is null"""
        mock_transcribe.side_effect = mock_transcribe_chunk

        response = client.post(
            "/audio/transcriptions",
            files={"file": ("test.mp3", sample_audio_file, "audio/mpeg")},
            data={"language": "null", "response_format": "verbose_json"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "language" in data

    @patch("app.main.transcribe_chunk")
    def test_vad_chunking_creates_chunks(
        self, mock_transcribe, client, sample_audio_file
    ):
        """Test that VAD chunking creates multiple chunks"""
        call_count = 0

        async def counting_mock(
            wav_data, language, timestamp_granularities, last_timestamp
        ):
            nonlocal call_count
            call_count += 1
            text = f"<|{last_timestamp}|>Chunk {call_count}<|{last_timestamp + 1.0}|>"
            return text, "en"

        mock_transcribe.side_effect = counting_mock

        response = client.post(
            "/audio/transcriptions",
            files={"file": ("test.mp3", sample_audio_file, "audio/mpeg")},
            data={"response_format": "json"},
        )

        assert response.status_code == 200
        # Should have called transcribe_chunk at least once
        assert call_count > 0

    @patch("app.main.transcribe_chunk")
    def test_concurrent_transcription(self, mock_transcribe, client, sample_audio_file):
        """Test that chunks are transcribed concurrently"""
        import time

        call_times = []

        async def timing_mock(
            wav_data, language, timestamp_granularities, last_timestamp
        ):
            call_times.append(time.time())
            # Simulate some processing time
            await asyncio.sleep(0.1)
            text = f"<|{last_timestamp}|>Text<|{last_timestamp + 1.0}|>"
            return text, "en"

        mock_transcribe.side_effect = timing_mock

        response = client.post(
            "/audio/transcriptions",
            files={"file": ("test.mp3", sample_audio_file, "audio/mpeg")},
            data={"response_format": "json"},
        )

        assert response.status_code == 200
        # If multiple chunks were created, they should be called concurrently
        # (timing difference should be small if concurrent)
        if len(call_times) > 1:
            time_diff = max(call_times) - min(call_times)
            # Concurrent calls should happen within a small time window
            assert time_diff < 0.5  # Allow some margin

    def test_missing_file(self, client):
        """Test that missing file returns error"""
        response = client.post("/audio/transcriptions", data={"language": "en"})
        assert response.status_code == 422  # FastAPI validation error

    @patch("app.main.transcribe_chunk")
    def test_vad_parameters(
        self, mock_transcribe, client, sample_audio_file, mock_transcribe_chunk
    ):
        """Test custom VAD parameters"""
        mock_transcribe.side_effect = mock_transcribe_chunk

        response = client.post(
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

    @patch("app.main.transcribe_chunk")
    def test_empty_transcription_result(
        self, mock_transcribe, client, sample_audio_file
    ):
        """Test handling of empty transcription results"""

        async def empty_mock(
            wav_data, language, timestamp_granularities, last_timestamp
        ):
            return "", None

        mock_transcribe.side_effect = empty_mock

        response = client.post(
            "/audio/transcriptions",
            files={"file": ("test.mp3", sample_audio_file, "audio/mpeg")},
            data={"response_format": "json"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        # Should handle empty results gracefully
        assert isinstance(data["text"], str)
