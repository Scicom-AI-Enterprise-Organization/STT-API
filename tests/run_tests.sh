#!/bin/bash

# Test script for diarization API
# Tests both online and offline diarization with 3 audio files

API_URL="http://localhost:9091/audio/transcriptions"
OUTPUT_DIR="test_outputs"
mkdir -p "$OUTPUT_DIR"

# Test files
FILES=("ks-184.mp3" "muzaha.mp3" "pakar.mp3")

echo "Starting API tests..."
echo "=========================================="

# Test Online Diarization
echo ""
echo "=== TESTING ONLINE DIARIZATION ==="
echo ""

for file in "${FILES[@]}"; do
    echo "Testing: $file (online)"
    echo "----------------------------------------"
    
    output_file="${OUTPUT_DIR}/${file%.*}_online.json"
    time_file="${OUTPUT_DIR}/${file%.*}_online_time.txt"
    
    { time curl -X POST "$API_URL" \
        -F "file=@../test_audio/$file" \
        -F "diarization=online" \
        -F "speaker_similarity=0.3" \
        -F "language=ms" \
        -F "response_format=verbose_json" \
        -o "$output_file" \
        -w "\nHTTP Status: %{http_code}\nTotal Time: %{time_total}s\n" \
        2>&1; } | tee "$time_file"
    
    echo ""
done

# Test Offline Diarization
echo ""
echo "=== TESTING OFFLINE DIARIZATION ==="
echo ""

for file in "${FILES[@]}"; do
    echo "Testing: $file (offline)"
    echo "----------------------------------------"
    
    output_file="${OUTPUT_DIR}/${file%.*}_offline.json"
    time_file="${OUTPUT_DIR}/${file%.*}_offline_time.txt"
    
    { time curl -X POST "$API_URL" \
        -F "file=@../test_audio/$file" \
        -F "diarization=offline" \
        -F "speaker_similarity=0.3" \
        -F "language=ms" \
        -F "response_format=verbose_json" \
        -o "$output_file" \
        -w "\nHTTP Status: %{http_code}\nTotal Time: %{time_total}s\n" \
        2>&1; } | tee "$time_file"
    
    echo ""
done

echo "All tests completed!"
echo "Results saved in: $OUTPUT_DIR"

