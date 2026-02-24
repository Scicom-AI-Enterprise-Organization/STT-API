import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
from dataclasses import dataclass
from packaging import version
from transformers import AutoModelForCTC, AutoTokenizer
from transformers import __version__ as transformers_version
from transformers.utils import is_flash_attn_2_available
from app.force_alignment.text_utils import text_normalize, split_text, get_uroman_tokens
import asyncio
import math
import librosa
import io
import logging
from typing import NamedTuple
from typing import Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np

SAMPLING_FREQ = 16000

DYNAMIC_BATCHING_BATCH_SIZE = int(os.environ.get('DYNAMIC_BATCHING_BATCH_SIZE', 8))
DYNAMIC_BATCHING_MICROSLEEP = float(os.environ.get('DYNAMIC_BATCHING_MICROSLEEP', 1e-4))
DEFAULT_CPU_COUNTS = 6
MODEL_COUNTER = 0
ALIGNMENT_MODEL = None
DEVICE = None
DTYPE = None

step_queue = asyncio.Queue()
_process_pool = None
_gpu_thread_pool = None

DEVICE_LIST = []


def get_gpu_thread_pool():
    """Dedicated thread pool for GPU inference to avoid blocking the default executor."""
    global _gpu_thread_pool
    if _gpu_thread_pool is None:
        _gpu_thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gpu-align")
    return _gpu_thread_pool

def get_process_pool():
    global _process_pool
    if _process_pool is None:
        cores = os.cpu_count() or DEFAULT_CPU_COUNTS
        _process_pool = ProcessPoolExecutor(max_workers=min(cores, 6))
    return _process_pool


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start

def load_global_alignment_model():
    global ALIGNMENT_MODEL, ALIGNMENT_TOKENIZER, DICTIONARY, DEVICE, DTYPE

    if DEVICE is None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    dev = DEVICE
    print(f"Force alignment device: {dev}")

    DTYPE = torch.float16 if dev == "cuda" else torch.float32

    ALIGNMENT_MODEL, ALIGNMENT_TOKENIZER = load_alignment_model(dev, dtype=DTYPE)
    # if dev == "cuda":
    #     ALIGNMENT_MODEL = torch.compile(ALIGNMENT_MODEL, mode="reduce-overhead")

    vocab = ALIGNMENT_TOKENIZER.get_vocab()
    vocab = {k: v for k, v in vocab.items()}
    vocab.pop('|', None)
    model_vocab_size = ALIGNMENT_MODEL.config.vocab_size
    vocab["<star>"] = model_vocab_size
    DICTIONARY = vocab
    logging.info(f"Tokenizer vocab: {len(vocab) - 1}, model vocab: {model_vocab_size}, <star> index: {model_vocab_size}")

def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.shape[0]
    num_tokens = len(tokens)

    trellis = np.full((num_frame + 1, num_tokens + 1), -float('inf'))
    trellis[:, 0] = 0
    for t in range(num_frame):
        trellis[t + 1, 1:] = np.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.shape[1] - 1
    t_start = np.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = np.exp(emission[t - 1, tokens[j - 1] if changed > stayed else 0]).item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError('Failed to align')
    return path[::-1]

def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

def merge_words(segments, separator='<star>'):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2].start, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words

def wav2vec2_lengths(L):
    for k, s in [(10,5),(3,2),(3,2),(3,2),(3,2),(2,2),(2,2)]:
        L = (L - k) // s + 1
    return L

def load_alignment_model(
    device: str = "cuda",
    model_path: str = "MahmoudAshraf/mms-300m-1130-forced-aligner",
    attn_implementation: str = None,
    dtype: torch.dtype = torch.float32,
):
    if attn_implementation is None:
        if version.parse(transformers_version) < version.parse("4.41.0"):
            attn_implementation = "eager"
        elif (
            is_flash_attn_2_available()
            and device == "cuda"
            and dtype in [torch.float16, torch.bfloat16]
        ):
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "sdpa"

    model = (
        AutoModelForCTC.from_pretrained(
            model_path,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def time_to_frame(time):
    stride_msec = 20
    frames_per_sec = 1000 / stride_msec
    return int(time * frames_per_sec)

def generate_emissions(audio_waveforms, batch_size: int = 4, context_length = 2, window_length = 30):
    context = context_length * SAMPLING_FREQ
    window = window_length * SAMPLING_FREQ

    new_batch = []
    extentions = []
    for i in range(len(audio_waveforms)):
        audio_waveform = audio_waveforms[i]
        extention = math.ceil(
            audio_waveform.size(0) / window
        ) * window - audio_waveform.size(0)
        padded_waveform = torch.nn.functional.pad(
            audio_waveform, (context, context + extention)
        )
        input_tensor = padded_waveform.unfold(0, window + 2 * context, window)
        new_batch.append(input_tensor[0])
        extentions.append(extention)
        
    lens = [new_batch[i].shape[0] for i in range(len(new_batch))]
    padded = torch.stack(new_batch).cuda().to(DTYPE)

    with torch.inference_mode():
        emissions_ = ALIGNMENT_MODEL(padded).logits
    emissions = []
    for i in range(emissions_.shape[0]):
        e = emissions_[i][time_to_frame(context_length) : -time_to_frame(context_length) + 1]
        if time_to_frame(extentions[i] / SAMPLING_FREQ) > 0:
            e = e[: -time_to_frame(extentions[i] / SAMPLING_FREQ), :]
        e = torch.log_softmax(e, dim=-1)
        e = torch.cat([e, torch.zeros(e.size(0), 1).to(e.device)], dim=1)
        emissions.append(e)

    return emissions


def preprocess_text(
    text, romanize=False, language="eng", split_size="word", star_frequency="segment"
):
    assert split_size in [
        "sentence",
        "word",
        "char",
    ], "Split size must be sentence, word, or char"
    assert star_frequency in [
        "segment",
        "edges",
    ], "Star frequency must be segment or edges"
    if language in ["jpn", "chi"]:
        split_size = "char"
    text_split = split_text(text, split_size)
    norm_text = [text_normalize(line.strip(), language) for line in text_split]

    if romanize:
        tokens = get_uroman_tokens(norm_text, language)
    else:
        tokens = [" ".join(list(word)) for word in norm_text]

    # add <star> token to the tokens and text
    # it's used extensively here but I found that it produces more accurate results
    # and doesn't affect the runtime
    if star_frequency == "segment":
        tokens_starred = []
        [tokens_starred.extend(["<star>", token]) for token in tokens]

        text_starred = []
        [text_starred.extend(["<star>", chunk]) for chunk in text_split]

    elif star_frequency == "edges":
        tokens_starred = ["<star>"] + tokens + ["<star>"]
        text_starred = ["<star>"] + text_split + ["<star>"]

    return tokens_starred, text_starred

def postprocess_alignment(
    emission_np,          # np.ndarray [num_frames, vocab]
    wav_np,               # np.ndarray [num_samples]
    sr: int,
    transcript: str,
    language: str,
    DICTIONARY: dict[str, int],
) -> dict[str, Any]:

    raw_text = transcript.strip()

    tokens_starred, text_starred = preprocess_text(
        raw_text,
        romanize=True,
        language=language,
    )
    tokens_starred.append("<star>")
    text_starred.append("<star>")

    tokens = tokens_starred
    token_indices = [
        DICTIONARY[c] for c in " ".join(tokens).split(" ") if c in DICTIONARY
    ]

    o = emission_np  # [num_frames, vocab]
    trellis = get_trellis(o, token_indices)
    path = backtrack(trellis, o, token_indices)
    segments = merge_repeats(path, " ".join(tokens).split(" "))
    word_segments = merge_words(segments)

    text_starred_filtered = [t for t in text_starred if t != "<star>"]

    seconds_per_frame = (len(wav_np) / sr) / o.shape[0]

    words_alignment = []
    for i, s in enumerate(word_segments):
        words_alignment.append(
            {
                "text": text_starred_filtered[i],
                "start": round(s.start * seconds_per_frame, 3),
                "end": round(s.end * seconds_per_frame, 3),
                "start_t": s.start,
                "end_t": s.end,
                "score": round(s.score, 3),
            }
        )

    result = {
        "words_alignment": words_alignment,
        "length": len(wav_np) / sr,
    }
    return result


class AlignBatchItem(NamedTuple):
    emission: np.ndarray
    wav_np: np.ndarray
    sr: int
    transcript: str
    language: str

def batch_emission(batch_payloads: list[tuple[bytes, str, str]]):
    assert ALIGNMENT_MODEL is not None and DICTIONARY is not None

    waveforms_1d = []   # list[torch.Tensor[T]]
    wav_nps = []
    srs = []
    transcripts = []
    languages = []

    for audio_bytes, transcript, language in batch_payloads:
        file_like = io.BytesIO(audio_bytes)
        wav_np, sr = librosa.load(file_like, sr=SAMPLING_FREQ, mono=True)
        wav_np = wav_np.astype(np.float32)

        w = torch.from_numpy(wav_np)

        waveforms_1d.append(w)
        wav_nps.append(wav_np)
        srs.append(SAMPLING_FREQ)
        transcripts.append(transcript)
        languages.append(language)

    emissions_list = generate_emissions(
        waveforms_1d,
        batch_size=DYNAMIC_BATCHING_BATCH_SIZE,
    )

    batch_items = []
    for i in range(len(batch_payloads)):
        emissions_np = emissions_list[i].cpu().numpy()
        batch_items.append(
            AlignBatchItem(
                emission=emissions_np,
                wav_np=wav_nps[i],
                sr=srs[i],
                transcript=transcripts[i],
                language=languages[i],
            )
        )

    return batch_items


async def step():
    while True:
        batch = []
        futures = []

        try:
            # Block until at least one item arrives
            item = await step_queue.get()
            batch.append(item)

            # Collect more items that arrived in the meantime (up to batch size)
            deadline = asyncio.get_event_loop().time() + DYNAMIC_BATCHING_MICROSLEEP
            while len(batch) < DYNAMIC_BATCHING_BATCH_SIZE:
                timeout = max(0, deadline - asyncio.get_event_loop().time())
                try:
                    item = await asyncio.wait_for(step_queue.get(), timeout=timeout)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            futures = [b[0] for b in batch]
            payloads = [(b[1], b[2], b[3]) for b in batch]

            batch_items = await asyncio.get_running_loop().run_in_executor(
                get_gpu_thread_pool(), batch_emission, payloads
            )

            loop = asyncio.get_running_loop()
            process_pool = get_process_pool()

            post_tasks = []
            for item in batch_items:
                post_tasks.append(
                    loop.run_in_executor(
                        process_pool,
                        postprocess_alignment,
                        item.emission,
                        item.wav_np,
                        item.sr,
                        item.transcript,
                        item.language,
                        DICTIONARY,
                    )
                )

            results = await asyncio.gather(*post_tasks, return_exceptions=True)

            for fut, res in zip(futures, results):
                if fut.done():
                    continue
                if isinstance(res, Exception):
                    fut.set_exception(res)
                else:
                    fut.set_result(res)

        except Exception as e:
            logging.exception("Error in step_force_align: %s", e)
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)

async def queue_force_align(
    fut: asyncio.Future,
    audio_bytes: bytes,
    transcript: str,
    language: str = "eng",
):
    await step_queue.put((fut, audio_bytes, transcript, language))