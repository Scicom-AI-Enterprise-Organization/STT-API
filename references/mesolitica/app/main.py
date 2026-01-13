from env import *

from typing import Annotated, List, Union
from fastapi import HTTPException, FastAPI, Request, Query, Depends
from fastapi import BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from sse_starlette import EventSourceResponse, ServerSentEvent
from fastapi_utils.tasks import repeat_every
from fastapi import File, Form, UploadFile
from torchaudio.io import StreamReader
from auth import JWTBearer, get_apikey_ws
from db import add_total_tokens, add_total_speech, add_total_retrieval
from manager import ConnectionManager
from malaya_speech.model.clustering import StreamingKMeansMaxCluster
from fastapi_profiler import PyInstrumentProfilerMiddleware
from silero_vad import load_silero_vad
from transformers import AutoTokenizer
from langfuse import Langfuse
import pyloudnorm as pyln
import itertools
import tiktoken
import malaya_speech
import torch
import base64
import numpy as np
import asyncio
import base_model
import fastapi_loki_tempo
import json_logging
import aiohttp
import urllib.parse
import redis
import json
import logging
import re
import time
import traceback
import sentry_sdk
import starlette

starlette.datastructures.UploadFile.spool_max_size = 0

title = 'Mesolitica API'
description = f'{title}, get your API key at <a target="blank_" href="https://playground.mesolitica.com/">https://playground.mesolitica.com/</a>'
__version__ = '0.2'

MODELS_REPRESENTATION = {
    'MaLLaM-🌙-Small': 'mallam-small',
    'MaLLaM-🌙-Tiny': 'mallam-small',
    'mallam-tiny': 'mallam-small',
    'mallam-small-reasoning': 'mallam-small',
    'mesolitica/malaysian-mistral-7b-32k-instructions': 'malaysian-mistral',
    'mesolitica/malaysian-tinyllama-1.1b-16k-instructions': 'malaysian-tinyllama',
}
MODELS = {
    'mallam-small': MALLAM_SMALL_HOST,
    'mallam-tiny': MALLAM_SMALL_HOST,
    'mallam-small-reasoning': MALLAM_SMALL_REASONING_HOST,
}
MODELS_PAID = {
    'mallam-small-roleplay': 'mallam-small',
}
MODELS_SPEECH = {
    'base': AUDIO_BASE_HOST,
    'small': AUDIO_SMALL_HOST,
    'tiny': AUDIO_TINY_HOST,
}
MODELS_TTS = {
    'base': {
        'url': TTS_HOST,
        'speakers': {'husein', 'idayu', 'anwar-ibrahim', 'kp'}
    }
}
MODELS_EMBEDDING = {
    'base': EMBEDDING_HOST,
}
MODELS_RERANKER = {
    'base': RERANKER_HOST,
}
MODELS_TRANSLATION = {
    'base': TRANSLATION_BASE_API,
    'small': TRANSLATION_SMALL_API,
}
TOKENIZERS = {
    'mallam-small': AutoTokenizer.from_pretrained('mesolitica/malaysian-Qwen-Qwen2.5-14B-Instruct'),
    'mallam-small-reasoning': AutoTokenizer.from_pretrained('mesolitica/malaysian-Qwen-Qwen2.5-14B-Instruct'),
    'mallam-tiny': AutoTokenizer.from_pretrained('mesolitica/malaysian-Qwen-Qwen2.5-14B-Instruct')
}

AVAILABLE_HOST = {}
for model, hosts in MODELS.items():
    hosts = [host.strip() for host in hosts.split(',')]
    AVAILABLE_HOST[model] = {no: {'url': host, 'available': False} for no, host in enumerate(hosts)}

redis_db = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
langfuse = Langfuse()
encoding = tiktoken.get_encoding('cl100k_base')
speaker_v = None
webrtc = malaya_speech.vad.webrtc()
silero = load_silero_vad(onnx=True)
frame_size = {
    'webrtc': 480,
    'silero': 512,
}

async def check_available():
    global AVAILABLE_HOST
    timeout = aiohttp.ClientTimeout(total=1)
    for model in AVAILABLE_HOST.keys():
        for i in range(len(AVAILABLE_HOST[model])):
            try:
                url = urllib.parse.urljoin(AVAILABLE_HOST[model][i]['url'], '/health')
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    try:
                        async with session.get(url) as response:
                            AVAILABLE_HOST[model][i]['available'] = response.status == 200
                    except Exception as e:
                        AVAILABLE_HOST[model][i]['available'] = False
            except Exception as e:
                pass

def normalize_value(value, min_value, max_value):
    normalized = (value - min_value) / (max_value - min_value)
    normalized_clamped = max(0, min(1, normalized))
    return normalized_clamped

def calculate_audio_volume(audio, sample_rate=16000):
    block_size = audio.size / sample_rate
    meter = pyln.Meter(sample_rate, block_size=block_size)
    audio = audio / np.max(np.abs(audio))
    loudness = meter.integrated_loudness(audio)
    loudness = normalize_value(loudness, -20, 80)
    return loudness

def exp_smoothing(value: float, prev_value: float, factor: float) -> float:
    return prev_value + factor * (value - prev_value)

def load_speaker_v():
    global speaker_v
    if speaker_v is None:
        speaker_v = malaya_speech.speaker_vector.nemo(model='huseinzol05/nemo-titanet_large')
        _ = speaker_v.eval()

if INITIAL_LOAD:
    logging.info('initial load speaker vector')
    load_speaker_v()

class RequestCancelledMiddleware:
    # https://github.com/tiangolo/fastapi/discussions/11360
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        queue = asyncio.Queue()
        async def message_poller(sentinel, handler_task):
            nonlocal queue
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    handler_task.cancel()
                    return sentinel
                await queue.put(message)

        sentinel = object()
        handler_task = asyncio.create_task(self.app(scope, queue.get, send))
        asyncio.create_task(message_poller(sentinel, handler_task))

        try:
            return await handler_task
        except asyncio.CancelledError:
            logging.warning('Cancelling request due to disconnect')


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=title,
        version=__version__,
        description=description,
        routes=app.routes,
    )
    openapi_schema['info']['x-logo'] = {
        'url': 'https://s3-us-west-2.amazonaws.com/cbi-image-service-prd/modified/55c298b0-b530-4a7c-b11e-edba4bbd1085.png'}
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app = FastAPI(
    title=title,
    description=description,
    version=__version__,
)
app.openapi = custom_openapi
fastapi_loki_tempo.patch(app=app)
app.add_middleware(RequestCancelledMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if ENABLE_PROFILER:
    app.add_middleware(
        PyInstrumentProfilerMiddleware,
        server_app=app,
        profiler_output_type='html',
        is_print_each_request=True,
        open_in_browser=True,
        html_file_name='example_profile.html',
    )

streaming_headers = {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
}

buffer_size = 4096
sample_rate = 16000
segment_length = sample_rate
maxlen = 30
replaces = ['<|startoftranscript|>', '<|endoftext|>', '<|transcribe|>', '<|transcribeprecise|>']
pattern = r'<\|\-?\d+\.?\d*\|>'
pattern_pair = r'<\|(\d+\.\d+)\|>(.*?)<\|(\d+\.\d+)\|>'
MAX_FILE_SIZE = MAX_FILE_MB * 1024 * 1024

manager = ConnectionManager()

@app.get('/', include_in_schema=False)
def hello(request: Request = None):
    """
    Hello from Mesolitica API!
    """
    client_host = request.client.host
    return {
        'message': title,
        'version': __version__,
    }

@app.get('/health')
def health(request: Request = None):
    """
    Health check for Mesolitica LLM Router API
    """
    return Response(status_code=200)

@app.get('/chat/tpm', dependencies=[Depends(JWTBearer())], tags=['Chat Completion'])
async def chat_tpm(request: Request = None):
    """
    Check current Chat Tokens per Minute quota.
    """
    email = request.email
    tpm = redis_db.get(email)
    if tpm is not None:
        tpm = int(tpm)
    else:
        tpm = -1
    return {
        'current': tpm,
        'max': MAX_TPM
    }

@app.post('/chat/completions', dependencies=[Depends(JWTBearer())], tags=['Chat Completion'])
async def chat_completions(
    form: base_model.ChatCompletionForm,
    request: Request = None,
    background_tasks: BackgroundTasks = None,
):
    """
    Chat Completion API, compatible with OpenAI library.
    """
    email = request.email
    model = getattr(form, 'model')
    model = MODELS_REPRESENTATION.get(model, model)
    if model not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f'model `{model}` does not support.',
        )
    stream = getattr(form, 'stream', False)
    data = form.dict()
    continue_mode = data.pop('continue_mode', False)
    data['model'] = model
    if stream:
        data['stream_options'] = {
            "include_usage": True,
            "continuous_usage_stats": True
        }

    tools = getattr(form, 'tools')
    tools_parsed = []
    if tools:
        for no, tool in enumerate(tools):
            if isinstance(tool, str):
                try:
                    tool = json.loads(tool)
                except BaseException:
                    raise HTTPException(
                        status_code=400,
                        detail=f'tools index {no} is not JSON or dictionary.',
                    )
            tools_parsed.append(json.dumps(tool, indent=4))

    if len(tools_parsed):
        fs = '\n\n'.join(tools_parsed)
    else:
        fs = ''

    if len(fs) < 5:
        fs = ''

    data.pop('tools', None)

    if len(fs):
        fs = f"""
Below is functions provided to convert user input into function parameters,
<FUNCTION>
{fs}
</FUNCTION>
Sometime user input is not able to convert to function parameters, so please ignore it and reply as normal chatbot.
""".strip()

    if data['messages'][0]['role'] == 'system':
        data['messages'][0]['content'] = f"{data['messages'][0]['content']}\n\n{fs}"
    else:
        d = {
            'role': 'system',
            'content': fs,
        }
        data['messages'].insert(0, d)

    if data['messages'][0]['role'] == 'system':
        s = data['messages'][0]['content'].strip()
        if len(s) < 1:
            data['messages'].pop(0)

    contents = [data['messages'][i]['content'] for i in range(len(data['messages']))]
    tokens = TOKENIZERS[model](contents, add_special_tokens = False, return_attention_mask = False)['input_ids']
    tokens = set(itertools.chain(*tokens))
    if len(tokens & set(TOKENIZERS[model].all_special_ids)):
        data['messages'] = [{'role': 'user', 'content': 'the user is trying to access you with unapproriate way, please generate refusal response'}]

    logging.info(data['messages'])

    if continue_mode:
        if data['messages'][-1]['role'] != 'assistant':
            raise HTTPException(
                status_code=400,
                detail=f'For continue mode, last role must be the assistant.',
            )
        prompt = TOKENIZERS[model].apply_chat_template(data.pop('messages'), tokenize = False)
        prompt = prompt.split(TOKENIZERS[model].eos_token)[:-1]
        prompt = TOKENIZERS[model].eos_token.join(prompt)
        data['prompt'] = prompt
        append_url = '/v1/completions'

        logging.info(data['prompt'])
    else:
        append_url = '/v1/chat/completions'

    tpm = redis_db.get(email)
    if tpm is not None:
        tpm = int(tpm)
    else:
        tpm = -1
    if email not in SKIP_QUOTA and tpm > MAX_TPM:
        raise HTTPException(
            status_code=429,
            detail='rate limit.',
        )

    url = None
    for i in range(len(AVAILABLE_HOST[model])):
        if AVAILABLE_HOST[model][i]['available']:
            url = urllib.parse.urljoin(AVAILABLE_HOST[model][i]['url'], append_url)
            break
    
    if url is None:
        raise HTTPException(
            status_code=429,
            detail=f'model `{model}` currently on heavy requests, please try again later.',
        )

    async def streaming():
        input_tokens = 0
        output_tokens = 0

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as r:
                    if r.status == 200:
                        try:
                            async for line in r.content:

                                line = line.decode()
                                original_line = line
                                if not line.startswith('data:'):
                                    continue

                                if line:
                                    try:
                                        line = json.loads(line.split('data:')[1])
                                        if not line['choices'][0]['finish_reason']:
                                            output_tokens += 1

                                        else:
                                            input_tokens = line['usage']['prompt_tokens']

                                        yield json.dumps(line)

                                    except BaseException as e:
                                        line = original_line.strip().split('data:')[1].strip()

                        except asyncio.CancelledError as e:
                            yield ServerSentEvent(**{"data": str(e)})

                    else:
                        line = await r.text()
                        raise HTTPException(
                            status_code=r.status, detail=json.loads(line)['message']
                        )
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=str(e)
            )

        redis_db.set(email, str(tpm + input_tokens))
        redis_db.expire(email, EXPIRED_REDIS)

        async def l():
            await add_total_tokens(
                email=email,
                model=MODELS_PAID.get(model, model),
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
        background_tasks.add_task(l)
        
    if stream:
        return EventSourceResponse(streaming(), headers=streaming_headers)

    else:
        async with aiohttp.ClientSession() as session:
            response = await session.post(url, json=data)
            status_code = response.status
            r = await response.json()
            print(r)

            if status_code == 200:
                input_tokens = r['usage']['prompt_tokens']
                output_tokens = r['usage']['completion_tokens']
                if not continue_mode:
                    result = r['choices'][0]['message']['content'].lstrip()
                    r['choices'][0]['message']['content'] = result

                redis_db.set(email, str(tpm + input_tokens))
                redis_db.expire(email, EXPIRED_REDIS)

                async def l():
                    await add_total_tokens(
                        email=email,
                        model=MODELS_PAID.get(model, model),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens
                    )
                background_tasks.add_task(l)

                return r
            else:
                if isinstance(r, dict):
                    if 'message' in r:
                        r = r['message']
                    elif 'detail' in r:
                        r = r['detail']
                    else:
                        r = str(r)
                else:
                    r = str(r)

                raise HTTPException(
                    status_code=400,
                    detail=r,
                )


async def post_audio(
    email,
    model,
    language,
    timestamp_granularities,
    wav_data,
    last_timestamp,
    in_int16=False,
    diarization=None,
    background_tasks=None,
):
    if diarization is not None:
        load_speaker_v()

    url = urllib.parse.urljoin(MODELS_SPEECH[model], '/v1/audio/transcriptions')

    if not in_int16:
        wav_data = np.int16((wav_data / np.max(np.abs(wav_data))) * 32768)

    data = {
        'audio_file': wav_data.tobytes(),
        'language': language,
        'task': timestamp_granularities,
        'is_numpy': '1'
    }
    texts = ''
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as r:
                if r.status == 200:
                    async for line in r.content:
                        line = line.decode()
                        original_line = line
                        if not line.startswith('data:'):
                            continue
                        if line:
                            line = json.loads(line.split('data:')[1])
                            text = line['token']['text']
                            for r in replaces:
                                text = text.replace(r, '')
                            matches = re.findall(pattern, text)
                            for match in matches:
                                timestamp = float(match.split('|')[1])
                                timestamp += last_timestamp
                                timestamp = round(timestamp, 2)
                                timestamp = f'<|{timestamp}|>'
                                text = text.replace(match, timestamp)
                            if len(text):
                                texts += text
                                if diarization is not None:
                                    matches = re.findall(pattern_pair, texts)
                                    if len(matches):
                                        match = matches[0]
                                        if len(match[1]) > 2:
                                            start = int(
                                                (float(match[0]) - last_timestamp) * sample_rate)
                                            end = int(
                                                (float(match[-1]) - last_timestamp) * sample_rate)
                                            sample_wav = wav_data[start:end]
                                            v = speaker_v([sample_wav])[0]
                                            speaker = malaya_speech.diarization.streaming(
                                                v, diarization
                                            )
                                            speaker = f'{speaker}|>'
                                            splitted = text.split('<|')
                                            text = '<|'.join(
                                                splitted[:1] + [speaker] + splitted[1:])

                                        texts = text.split('|>')[-2] + '|>'

                                yield json.dumps({'token': text})
                else:
                    line = await r.text()
                    try:
                        detail = json.loads(line)['detail']
                    except BaseException:
                        detail = line
                    yield json.dumps({'error': detail})
                    yield ServerSentEvent(**{"data": detail})
                    return

    except Exception as e:
        error = str(e)
        yield json.dumps({'error': error})
        yield ServerSentEvent(**{"data": error})
        return

    audio_len = len(wav_data) / sample_rate

    async def l():
        await add_total_speech(
            email=email,
            model=model,
            audio_length=audio_len
        )
        redis_key = f'{email}_speech'
        tpm = redis_db.get(redis_key)
        if tpm is not None:
            tpm = int(float(tpm))
        else:
            tpm = -1

        redis_db.set(redis_key, str(tpm + audio_len))
        redis_db.expire(redis_key, EXPIRED_REDIS)

    background_tasks.add_task(l)


@app.get('/audio/hpm', dependencies=[Depends(JWTBearer())], tags=['Audio Transcription'])
async def audio_hpm(request: Request = None):
    """
    Check current Hour per Minute quota.
    """
    email = request.email
    redis_key = f'{email}_speech'
    tpm = redis_db.get(redis_key)
    if tpm is not None:
        tpm = int(float(tpm)) / 60 / 60
    else:
        tpm = -1
    return {
        'current': tpm,
        'max': int(MAX_SPM / 60 / 60)
    }


def check_audio_parameter(
    model='base',
    language='null',
    timestamp_granularities='segment',
    response_format='text',
    speaker_similarity=0.5,
    speaker_max_n=5,
    chunking_method='naive',
    vad_method='silero',
    minimum_silent_ms=200,
    minimum_trigger_vad_ms=1500,
    reject_segment_vad_ratio=0.9,
):
    if model not in MODELS_SPEECH:
        raise HTTPException(
            status_code=400,
            detail=f'model `{model}` does not support.',
        )

    if language is None:
        language = 'null'

    language = language.lower().strip()

    if language not in {'none', 'null', 'en', 'ms', 'zh', 'ta'}:
        raise HTTPException(
            status_code=400,
            detail=f'`language` only support `none`, `null`, `en`, `ms`, `zh` and `ta` for now.',
        )

    timestamp_granularities = timestamp_granularities.lower().strip()

    if timestamp_granularities not in {'segment', 'word'}:
        raise HTTPException(
            status_code=400,
            detail=f'`timestamp_granularities` only support `segment` and `word`',
        )

    if response_format not in {'text', 'json', 'verbose_json'}:
        raise HTTPException(
            status_code=400,
            detail=f'currently `response_format` only support `text`, `json`, `verbose_json`',
        )

    if not (0.0 < speaker_similarity < 1.0):
        raise HTTPException(
            status_code=400,
            detail='`speaker_similarity` must be greater than 0.0 and less than 1.0',
        )

    if not (1 < speaker_max_n < 100):
        raise HTTPException(
            status_code=400,
            detail='`speaker_max_n` must be greater than 1 and less than 100',
        )

    if chunking_method not in {'naive', 'vad'}:
        raise HTTPException(
            status_code=400,
            detail=f'`chunking_method` only support `naive`, `vad`',
        )

    if vad_method not in {'webrtc', 'silero'}:
        raise HTTPException(
            status_code=400,
            detail=f'`vad_method` only support `webrtc`, `silero`',
        )

    if minimum_silent_ms < 1:
        raise HTTPException(
            status_code=400,
            detail=f'`minimum_silent_ms` must be greater than 0',
        )

    if minimum_trigger_vad_ms < 1:
        raise HTTPException(
            status_code=400,
            detail=f'`minimum_trigger_vad_ms` must be greater than 0',
        )

    if not (0.0 < reject_segment_vad_ratio < 1.0):
        raise HTTPException(
            status_code=400,
            detail='`reject_segment_vad_ratio` must be greater than 0.0 and less than 1.0',
        )

    return language


@app.post(
    '/audio/transcriptions',
    dependencies=[Depends(JWTBearer())],
    tags=['Audio Transcription']
)
async def audio_transcriptions(
    file: bytes = File(),
    model: str = Form('base'),
    language: str = Form(None),
    response_format: str = Form('text'),
    timestamp_granularities: str = Form('segment'),
    enable_diarization: bool = Form(False),
    speaker_similarity: float = Form(0.5),
    speaker_max_n: int = Form(5),
    chunking_method: str = Form('naive'),
    vad_method: str = Form('silero'),
    minimum_silent_ms: int = Form(200),
    minimum_trigger_vad_ms: int = Form(1500),
    reject_segment_vad_ratio: float = Form(0.9),
    stream: bool = Form(False),
    request: Request = None,
    background_tasks: BackgroundTasks = None,
):
    """
    Audio Transcriptions API, compatible with OpenAI library, include streaming.

    Parameters
    ----------
    - **file**: bytes
        The audio file to be processed.

    - **model**: str, optional (default='base')
        The model to be used for processing the audio.

    - **language**: str, optional (default=None)
        The language of the audio. If not specified, the language will be auto-detected.

    - **response_format**: str, optional (default='text')
        The format of the response. Supported formats are:
        - 'text': Plain text format.
        - 'json': JSON format.
        - 'verbose_json': JSON format with additional metadata.

    - **timestamp_granularities**: str, optional (default='segment')
        The granularity of timestamps in the response, `word` level only support for model 1.1 and above.

    - **enable_diarization**: bool, optional (default=False)
        Whether to enable speaker diarization (identifying different speakers).

    - **speaker_similarity**: float, optional (default=0.5)
        The threshold for speaker similarity when diarization is enabled.
        Must be between 0 and 1.

    - **speaker_max_n**: int, optional (default=5)
        The maximum number of speakers to detect in diarization.
        Must be between 1 and 100.

    - **chunking_method**: str, optional (default='naive')
        The method used for chunking the audio. Supported methods are:
        - 'naive': Simple chunking without overlap.
        - 'vad': Chunking based on Voice Activity Detection (VAD).

    - **vad_method**: str, optional (default='webrtc')
        The method used for Voice Activity Detection (VAD). Supported methods are:
        - 'webrtc': Uses the WebRTC VAD.
        - 'silero': Uses the Silero VAD.

    - **minimum_silent_ms**: int, optional (default=200)
        The minimum duration of silence (in milliseconds) required to consider it as a segment boundary.

    - **minimum_trigger_vad_ms**, int, optional (default=2000)
        The minimum duration of current segment to trigger VAD.

    - **reject_segment_vad_ratio**, float, optional (default=0.9)
        If the segment is 90% negative VAD, we will skip to transcribe the segment.

    - **stream**: bool, optional (default=False)
        Whether to stream the response as it is being processed.
    """

    if len(file) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f'maximum size for `file` is {MAX_FILE_MB}MB only.',
        )

    language = check_audio_parameter(
        model=model,
        language=language,
        timestamp_granularities=timestamp_granularities,
        response_format=response_format,
        speaker_similarity=speaker_similarity,
        speaker_max_n=speaker_max_n,
        chunking_method=chunking_method,
        vad_method=vad_method,
        minimum_silent_ms=minimum_silent_ms,
        minimum_trigger_vad_ms=minimum_trigger_vad_ms,
        reject_segment_vad_ratio=reject_segment_vad_ratio,
    )

    email = request.email

    redis_key = f'{email}_speech'
    tpm = redis_db.get(redis_key)
    if tpm is not None:
        tpm = int(float(tpm))
    else:
        tpm = -1

    if email not in SKIP_QUOTA and tpm > MAX_SPM:
        raise HTTPException(
            status_code=429,
            detail='rate limit.',
        )

    async def streaming():
        streamer = StreamReader(
            src=file,
            format=None,
            option=None,
            buffer_size=buffer_size
        )

        if chunking_method == 'naive':
            frames_per_chunk = segment_length
        elif chunking_method == 'vad':
            frames_per_chunk = frame_size[vad_method]

        streamer.add_basic_audio_stream(
            frames_per_chunk=frames_per_chunk,
            sample_rate=sample_rate
        )
        stream_iterator = streamer.stream()
        wav_data = np.array([], dtype=np.float32)
        last_timestamp = 0
        total_silent = 0
        total_silent_frames = 0
        total_frames = 0
        if enable_diarization:
            diarization = StreamingKMeansMaxCluster(
                threshold=speaker_similarity,
                max_clusters=speaker_max_n
            )
        else:
            diarization = None
        try:
            for chunk in stream_iterator:
                frame_pt = chunk[0][:, 0]
                frame = chunk[0][:, 0].numpy()
                total_frames += 1
                if chunking_method == 'vad':

                    if len(frame) < frames_per_chunk:
                        continue

                    if vad_method == 'webrtc':
                        y_int = malaya_speech.astype.float_to_int(frame)
                        vad = webrtc(y_int)

                    elif vad_method == 'silero':
                        vad = silero(frame_pt, sr=sample_rate).numpy()[0][0] > 0.5

                    if vad:
                        total_silent = 0
                    else:
                        total_silent += len(frame)
                        total_silent_frames += 1

                wav_data = np.concatenate([wav_data, frame])
                audio_len = len(wav_data) / sample_rate
                audio_len_ms = audio_len * 1000
                silent_len = (total_silent / sample_rate) * 1000
                negative_ratio = total_silent_frames / total_frames

                vad_trigger = audio_len_ms >= minimum_trigger_vad_ms and silent_len >= minimum_silent_ms
                if vad_trigger or audio_len >= maxlen:
                    if negative_ratio <= reject_segment_vad_ratio:
                        async for t in post_audio(
                            email=email,
                            model=model,
                            language=language,
                            timestamp_granularities=timestamp_granularities,
                            wav_data=wav_data,
                            last_timestamp=last_timestamp,
                            diarization=diarization,
                            background_tasks=background_tasks
                        ):
                            yield t

                    last_timestamp += audio_len
                    total_silent = 0
                    total_silent_frames = 0
                    total_frames = 0

                    wav_data = np.array([], dtype=np.float32)

            if len(wav_data):
                audio_len = len(wav_data) / sample_rate
                negative_ratio = total_silent_frames / total_frames
                if negative_ratio <= reject_segment_vad_ratio:
                    async for t in post_audio(
                        email=email,
                        model=model,
                        language=language,
                        timestamp_granularities=timestamp_granularities,
                        wav_data=wav_data,
                        last_timestamp=last_timestamp,
                        diarization=diarization,
                        background_tasks=background_tasks,
                    ):
                        yield t

        except asyncio.CancelledError as e:
            yield ServerSentEvent(**{"data": str(e)})

    if stream:
        return EventSourceResponse(streaming(), headers=streaming_headers)

    else:

        tokens = []
        async for data in streaming():
            if isinstance(data, str):
                data = json.loads(data)
                if 'error' in data:
                    raise HTTPException(
                        status_code=400,
                        detail=data['error'],
                    )

                tokens.append(data['token'])
        tokens = ''.join(tokens)
        lang = tokens.split('|')[1]

        matches = re.findall(pattern_pair, tokens)
        segments = []
        all_texts = []
        for no, (start, substring, end) in enumerate(matches):
            start_timestamp = float(start)
            end_timestamp = float(end)
            segments.append({
                'id': no,
                'seek': 0,
                'start': start_timestamp,
                'end': end_timestamp,
                'text': substring.strip(),
                'tokens': [],
                'temperature': 0.0,
                'avg_logprob': 0.0,
                'compression_ratio': 1.0,
                'no_speech_prob': 0.0,
            })
            all_texts.append(substring)

        if response_format == 'verbose_json':
            return {
                'task': 'transcribe',
                'language': lang,
                'duration': segments[-1]['end'],
                'text': ''.join(all_texts),
                'segments': segments
            }
        elif response_format == 'json':
            return {
                'text': ''.join(all_texts),
            }
        else:
            return ''.join(all_texts)


@app.websocket('/audio/transcriptions/ws')
async def audio_transcriptions_ws(
    websocket: WebSocket,
    model: Annotated[str, Query()] = 'base',
    language: Annotated[str, Query()] = 'null',
    timestamp_granularities: Annotated[str, Query()] = 'segment',
    enable_diarization: Annotated[bool, Query()] = False,
    speaker_similarity: Annotated[float, Query()] = 0.5,
    speaker_max_n: Annotated[int, Query()] = 5,
    vad_method: Annotated[str, Query()] = 'silero',
    minimum_silent_ms: Annotated[int, Query()] = 200,
    minimum_trigger_vad_ms: Annotated[int, Query()] = 1500,
    reject_segment_vad_ratio: Annotated[float, Query()] = 0.9,
    include_silent_token: Annotated[bool, Query()] = True,
    active_token: Annotated[bool, Query()] = False,
    active_confidence: Annotated[float, Query()] = 0.7,
    active_volume: Annotated[float, Query()] = 0.6,
    active_smoothing_factor: Annotated[float, Query()] = 0.2,
    apikey: Annotated[str, Query()] = 'null',
    background_tasks: BackgroundTasks = None,
):
    """
    Websocket endpoint for audio streaming.
    We expect base64 encoded signed 16-bit audio. If stereo, we will average to become mono channel.
    Please send small chunks like RecordRTC.
    """
    client_id, email = await get_apikey_ws(apikey)
    language = check_audio_parameter(
        model=model,
        language=language,
        timestamp_granularities=timestamp_granularities,
        speaker_similarity=speaker_similarity,
        speaker_max_n=speaker_max_n,
        vad_method=vad_method,
        minimum_silent_ms=minimum_silent_ms,
        minimum_trigger_vad_ms=minimum_trigger_vad_ms,
        reject_segment_vad_ratio=reject_segment_vad_ratio,
    )

    if active_token and vad_method != 'silero':
        raise HTTPException(
            status_code=400,
            detail='silero must be use for VAD method to enable active token.',
        )

    if enable_diarization:
        diarization = StreamingKMeansMaxCluster(
            threshold=speaker_similarity,
            max_clusters=speaker_max_n
        )
    else:
        diarization = None

    frames_per_chunk = frame_size[vad_method]
    await manager.connect(websocket, client_id=client_id)
    manager.diarization[client_id] = diarization
    try:
        while True:
            data = await websocket.receive_text()
            try:
                array = np.frombuffer(base64.b64decode(data), dtype=np.int16)
                array = array.astype(np.float32, order='C') / 32768.0
            except BaseException:
                error = {
                    'error': 'input must be base64 encoded signed 16-bit audio'
                }
                error = json.dumps(error)
                await manager.send_personal_message(error, websocket)
                continue

            a = [manager.wav_data[client_id], array]
            manager.wav_data[client_id] = np.concatenate(a)

            chunks = []
            while True:
                t_ = manager.wav_data[client_id][: frames_per_chunk]
                if len(t_) == frames_per_chunk:
                    manager.wav_data[client_id] = manager.wav_data[client_id][frames_per_chunk:]
                    chunks.append(t_)
                else:
                    break

            for i in range(len(chunks)):
                manager.total_frames[client_id] += 1
                vad_score = 0
                if vad_method == 'webrtc':
                    y_int = malaya_speech.astype.float_to_int(chunks[i])
                    vad = webrtc(y_int)

                elif vad_method == 'silero':
                    vad_score = silero(torch.Tensor(chunks[i]), sr=sample_rate).numpy()[0][0]
                    vad = vad_score > 0.5

                if vad:
                    manager.total_silent[client_id] = 0
                else:
                    manager.total_silent[client_id] += len(chunks[i])
                    manager.total_silent_frames[client_id] += 1

                if active_token:
                    volume = calculate_audio_volume(chunks[i])
                    volume = exp_smoothing(
                        volume,
                        manager.prev_volume[client_id],
                        active_smoothing_factor
                    )
                    manager.prev_volume[client_id] = volume
                    if vad_score < active_confidence and volume < active_volume:
                        t = {'token': '<|active|>'}
                        t = json.dumps(t)
                        await manager.send_personal_message(t, websocket)

                manager.wav_queue[client_id].append(chunks[i])
                audio_len = (len(manager.wav_queue[client_id]) * frames_per_chunk) / sample_rate
                audio_len_ms = audio_len * 1000
                silent_len = (manager.total_silent[client_id] / sample_rate) * 1000
                negative_ratio = manager.total_silent_frames[client_id] / \
                    manager.total_frames[client_id]

                vad_trigger = audio_len_ms >= minimum_trigger_vad_ms and silent_len >= minimum_silent_ms
                if vad_trigger or audio_len >= maxlen:
                    if negative_ratio <= reject_segment_vad_ratio:
                        wav_data = np.concatenate(manager.wav_queue[client_id])
                        async for t in post_audio(
                            email=email,
                            model=model,
                            language=language,
                            timestamp_granularities=timestamp_granularities,
                            wav_data=wav_data,
                            last_timestamp=manager.last_timestamp[client_id],
                            diarization=manager.diarization[client_id],
                            background_tasks=background_tasks
                        ):
                            await manager.send_personal_message(t, websocket)
                    elif include_silent_token:
                        t = {'token': '<|silent|>'}
                        t = json.dumps(t)
                        await manager.send_personal_message(t, websocket)

                    manager.last_timestamp[client_id] += audio_len
                    manager.total_silent[client_id] = 0
                    manager.total_silent_frames[client_id] = 0
                    manager.total_frames[client_id] = 0
                    manager.wav_queue[client_id] = []

            await asyncio.sleep(0)

    except Exception as e:
        await manager.disconnect(client_id)

    except WebSocketDisconnect as e:
        await manager.disconnect(client_id)

@app.post(
    '/audio/speech',
    dependencies=[Depends(JWTBearer())],
    tags=['Create Speech']
)
async def audio_speech(
    form: base_model.Speech,
    request: Request = None,
    background_tasks: BackgroundTasks = None,
):
    """
    Text-to-Speech API, compatible with OpenAI library and natively streaming.
    """
    
    model = form.model
    voice = form.voice
    email = request.email
    
    if form.response_format.lower() != 'wav':
        raise HTTPException(
            status_code=400,
            detail='currently we only support wav format.',
        )

    if not 0.1 <= form.speed <= 2.0:
        raise HTTPException(
            status_code=400,
            detail='speed must be between 0.1 and 2.0',
        )

    if len(form.input) > 1000:
        raise HTTPException(
            status_code=400,
            detail='input cannot more than 1000 characters.',
        )
    
    if model not in MODELS_TTS:
        raise HTTPException(
            status_code=400,
            detail=f'model `{model}` does not support.',
        )

    if voice not in MODELS_TTS[model]['speakers']:
        raise HTTPException(
            status_code=400,
            detail=f'voice `{voice}` does not support for model `{model}`.',
        )
    
    if form.response_format.lower() != 'wav':
        raise HTTPException(
            status_code=400,
            detail=f'currently we only support WAV format.',
        )
    
    url = urllib.parse.urljoin(MODELS_TTS[model]['url'], '/v1/audio/speech')
    data = {
        'input': form.input,
        'response_format': form.response_format.lower(),
        'voice': voice,
    }
    async def process_response():
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as r:
                if r.status == 200:
                    async for chunk in r.content.iter_chunked(128):
                        yield chunk
                    
                else:
                    line = await r.text()
                    try:
                        detail = json.loads(line)['detail']
                    except BaseException:
                        detail = line
                    raise HTTPException(
                        status_code=r.status, detail=detail,
                    )
    
    return StreamingResponse(
        process_response(),
        media_type='audio/wav',
        headers={"Content-Disposition": f"attachment; filename=speech.{form.response_format}"},
    )

@app.get('/embeddings/tpm', dependencies=[Depends(JWTBearer())], tags=['Retrieval'])
async def embeddings_tpm(request: Request = None):
    """
    Check current Retrieval Tokens per Minute quota.
    """
    email = request.email
    redis_key = f'{email}_retrieval'
    tpm = redis_db.get(redis_key)
    if tpm is not None:
        tpm = int(tpm)
    else:
        tpm = -1
    return {
        'current': tpm,
        'max': MAX_TPM
    }


@app.post('/embeddings', dependencies=[Depends(JWTBearer())], tags=['Retrieval'])
async def embeddings(
    form: base_model.Embedding,
    request: Request = None,
    background_tasks: BackgroundTasks = None
):
    """
    Embedding API, compatible with OpenAI library.

    `input` accept,
    - a string
    - an array of strings
    - an array of tokens
    - an array of token arrays

    Currently we do not support `dimension`.
    """

    if isinstance(form.input, list):
        if isinstance(form.input[0], int):
            strings = encoding.decode(form.input)
        elif isinstance(form.input[0], list):
            strings = [encoding.decode(s) for s in form.input]
        else:
            strings = form.input
    else:
        strings = form.input

    model = form.model
    email = request.email

    if model not in MODELS_EMBEDDING:
        raise HTTPException(
            status_code=400,
            detail=f'model `{model}` does not support.',
        )

    redis_key = f'{email}_retrieval'
    tpm = redis_db.get(redis_key)
    if tpm is not None:
        tpm = int(tpm)
    else:
        tpm = -1

    if email not in SKIP_QUOTA and tpm > MAX_TPM:
        raise HTTPException(
            status_code=429,
            detail='rate limit.',
        )

    url = urllib.parse.urljoin(MODELS_EMBEDDING[model], '/v1/embeddings')

    data = {
        'input': strings
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as r:
            if r.status == 200:
                result = await r.json()
                input_tokens = result['usage']['total_tokens']

                redis_db.set(redis_key, str(tpm + input_tokens))
                redis_db.expire(redis_key, EXPIRED_REDIS)

                async def l():
                    await add_total_retrieval(
                        email=email,
                        mode='embedding',
                        model=model,
                        input_tokens=input_tokens
                    )

                background_tasks.add_task(l)

                return result
            else:
                line = await r.text()
                try:
                    detail = json.loads(line)['detail']
                except BaseException:
                    detail = line
                raise HTTPException(
                    status_code=r.status, detail=detail,
                )


@app.post('/reranker', dependencies=[Depends(JWTBearer())], tags=['Retrieval'])
async def reranker(
    form: base_model.Reranker,
    request: Request = None,
    background_tasks: BackgroundTasks = None
):
    model = form.model
    email = request.email

    if model not in MODELS_RERANKER:
        raise HTTPException(
            status_code=400,
            detail=f'model `{model}` does not support.',
        )

    redis_key = f'{email}_retrieval'
    tpm = redis_db.get(redis_key)
    if tpm is not None:
        tpm = int(tpm)
    else:
        tpm = -1

    if email not in SKIP_QUOTA and tpm > MAX_TPM:
        raise HTTPException(
            status_code=429,
            detail='rate limit.',
        )

    url = urllib.parse.urljoin(MODELS_RERANKER[model], '/v1/reranker')

    data = {
        'query': form.query,
        'passages': form.passages,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as r:
            if r.status == 200:
                result = await r.json()
                input_tokens = result['usage']['total_tokens']

                redis_db.set(redis_key, str(tpm + input_tokens))
                redis_db.expire(redis_key, EXPIRED_REDIS)

                async def l():
                    await add_total_retrieval(
                        email=email,
                        mode='reranker',
                        model=model,
                        input_tokens=input_tokens
                    )

                background_tasks.add_task(l)

                return result
            else:
                line = await r.text()
                try:
                    detail = json.loads(line)['detail']
                except BaseException:
                    detail = line
                raise HTTPException(
                    status_code=r.status, detail=detail,
                )


@app.post('/translation', dependencies=[Depends(JWTBearer())], tags=['Translation'])
async def translation(
    form: base_model.Translation,
    request: Request = None,
    background_tasks: BackgroundTasks = None
):
    model = form.model
    email = request.email

    if model not in MODELS_TRANSLATION:
        raise HTTPException(
            status_code=400,
            detail=f'model `{model}` does not support.',
        )

    redis_key = f'{email}_translation'
    tpm = redis_db.get(redis_key)
    if tpm is not None:
        tpm = int(tpm)
    else:
        tpm = -1

    if email not in SKIP_QUOTA and tpm > MAX_TPM_TRANSLATION:
        raise HTTPException(
            status_code=429,
            detail='rate limit.',
        )

    url = urllib.parse.urljoin(MODELS_TRANSLATION[model], '/translate')

    data = form.dict()
    data['text'] = data.pop('input')
    data.pop('model', None)

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as r:
            if r.status == 200:
                result = await r.json()
                input_tokens = result['usage']['total_tokens']

                redis_db.set(redis_key, str(tpm + input_tokens))
                redis_db.expire(redis_key, EXPIRED_REDIS)

                async def l():
                    await add_total_retrieval(
                        email=email,
                        mode='translation',
                        model=model,
                        input_tokens=input_tokens
                    )

                background_tasks.add_task(l)

                return result

            else:
                line = await r.text()
                try:
                    detail = json.loads(line)['detail']
                except BaseException:
                    detail = line
                raise HTTPException(
                    status_code=r.status, detail=detail,
                )


@app.post('/translation/public', tags=['Translation'])
async def translation_public(
    form: base_model.Translation,
    request: Request = None,
):
    """
    Serving https://mesolitica.com/translation with blazingly fast engine and better accuracy.
    Comes with 50k tokens per minute globally rate limit.
    """

    model = form.model

    if model not in MODELS_TRANSLATION:
        raise HTTPException(
            status_code=400,
            detail=f'model `{model}` does not support.',
        )

    redis_key = 'public_translation'
    tpm = redis_db.get(redis_key)
    if tpm is not None:
        tpm = int(tpm)
    else:
        tpm = -1

    if tpm > MAX_PUBLIC_TPM:
        raise HTTPException(
            status_code=429,
            detail='rate limit.',
        )

    url = urllib.parse.urljoin(MODELS_TRANSLATION[model], '/translate')

    data = form.dict()
    data['text'] = data.pop('input')
    data.pop('model', None)

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as r:
            if r.status == 200:
                result = await r.json()
                input_tokens = result['usage']['total_tokens']

                redis_db.set(redis_key, str(tpm + input_tokens))
                redis_db.expire(redis_key, EXPIRED_REDIS)

                return result

            else:
                line = await r.text()
                try:
                    detail = json.loads(line)['detail']
                except BaseException:
                    detail = line
                raise HTTPException(
                    status_code=r.status, detail=detail,
                )


@app.post('/classifier/nsfw', dependencies=[Depends(JWTBearer())], tags=['Classifier'])
async def classifier_nsfw(
    form: base_model.NSFW,
    request: Request = None,
    background_tasks: BackgroundTasks = None
):
    email = request.email

    redis_key = f'{email}_classifier'
    tpm = redis_db.get(redis_key)
    if tpm is not None:
        tpm = int(tpm)
    else:
        tpm = -1

    if email not in SKIP_QUOTA and tpm > MAX_TPM_CLASSIFIER:
        raise HTTPException(
            status_code=429,
            detail='rate limit.',
        )

    url = urllib.parse.urljoin(NSFW_API, '/v1/classify')

    data = {
        'input': form.input,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as r:
            if r.status == 200:
                result = await r.json()
                input_tokens = result['usage']['total_tokens']

                redis_db.set(redis_key, str(tpm + input_tokens))
                redis_db.expire(redis_key, EXPIRED_REDIS)

                async def l():
                    await add_total_retrieval(
                        email=email,
                        mode='classifier',
                        model=model,
                        input_tokens=input_tokens
                    )

                background_tasks.add_task(l)

                return result
            else:
                line = await r.text()
                try:
                    detail = json.loads(line)['detail']
                except BaseException:
                    detail = line
                raise HTTPException(
                    status_code=r.status, detail=detail,
                )


@app.post('/classifier/public/nsfw', tags=['Classifier'])
async def classifier_public_nsfw(
    form: base_model.PublicNSFW,
    request: Request = None,
):
    """
    Serving https://huggingface.co/malaysia-ai/malaysian-sfw-classifier with blazingly fast engine.
    Comes with 1M tokens per minute globally rate limit.
    """

    redis_key = 'public_classifier'
    tpm = redis_db.get(redis_key)
    if tpm is not None:
        tpm = int(tpm)
    else:
        tpm = -1

    if tpm > MAX_TPM_PUBLIC_CLASSIFIER:
        raise HTTPException(
            status_code=429,
            detail='rate limit.',
        )

    url = urllib.parse.urljoin(NSFW_API, '/v1/classify')

    data = {
        'input': form.input,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as r:
            if r.status == 200:
                result = await r.json()
                input_tokens = result['usage']['total_tokens']

                redis_db.set(redis_key, str(tpm + input_tokens))
                redis_db.expire(redis_key, EXPIRED_REDIS)

                return result
            else:
                line = await r.text()
                try:
                    detail = json.loads(line)['detail']
                except BaseException:
                    detail = line
                raise HTTPException(
                    status_code=r.status, detail=detail,
                )


if DEBUG_UPLOAD:
    @app.post('/upload')
    async def upload_file(file: UploadFile = File(...), request: Request = None,):
        import shutil
        with open(f"/tmp/{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return 'a'

@app.on_event('startup')
@repeat_every(seconds=5)
async def lookhead_load_balancing():
    await check_available()