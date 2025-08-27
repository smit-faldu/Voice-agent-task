import argparse
import asyncio
import base64
import numpy as np
import ffmpeg
from time import time
import importlib.util
import sys
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from src.whisper_streaming.whisper_online import backend_factory, online_factory, add_shared_args

# Import AI Agent runtime (path contains a space, so use importlib)
import os as _os
# AGENT_FILE = _os.path.join(_os.path.dirname(__file__), "src", "AI agent", "ai_agent_runtime.py")
# spec = importlib.util.spec_from_file_location("ai_agent_runtime", AGENT_FILE)
# ai_agent_runtime = importlib.util.module_from_spec(spec)
# sys.modules["ai_agent_runtime"] = ai_agent_runtime
# spec.loader.exec_module(ai_agent_runtime)
# get_agent_response = ai_agent_runtime.get_agent_response  # async function

# Try to import the AI agent, with fallback
try:
    from src.AI_agent.ai_agent_runtime import get_agent_response as autogen_agent_response
    print("[AGENT] Loaded AutoGen AI agent")
    _autogen_available = True
except Exception as e:
    print(f"[AGENT] Failed to load AutoGen agent: {e}")
    print("[AGENT] Using simple fallback agent only")
    _autogen_available = False

    # Simple fallback agent
    async def get_agent_response(text: str) -> str:
        """Simple fallback agent for when AutoGen fails"""
        return await get_simple_agent_response(text)

# Simple agent function that can be used as fallback
async def get_simple_agent_response(text: str) -> str:
    """Simple fallback agent for when AutoGen fails"""
    text_lower = text.lower()

    # Math questions
    if "plus" in text_lower or "+" in text:
        if "2" in text and "2" in text:
            return "Answer: 4. When you add 2 plus 2, you get 4! It's like counting: 1, 2, 3, 4!"
        elif "1" in text and "1" in text:
            return "Answer: 2. When you add 1 plus 1, you get 2!"
        else:
            return "I can help with simple math! Try asking me about 2 plus 2."

    # Story requests
    elif "story" in text_lower and "lion" in text_lower:
        return "Leo's Adventure: Once upon a time, there was a brave little lion named Leo. He lived in the savanna and loved to explore. One day, he helped a lost rabbit find its way home. Leo learned that being kind is the greatest strength of all!"

    # Greetings
    elif any(word in text_lower for word in ["hello", "hi", "hey"]):
        return "Hello! I'm your AI assistant. You can ask me math questions or request stories!"

    # Default response
    else:
        return f"I heard you say: '{text}'. I'm a simple assistant that can help with basic math and tell stories about lions!"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


parser = argparse.ArgumentParser(description="Whisper FastAPI Online Server")
parser.add_argument(
    "--host",
    type=str,
    default="localhost",
    help="The host address to bind the server to.",
)
parser.add_argument(
    "--port", type=int, default=8000, help="The port number to bind the server to."
)
parser.add_argument(
    "--warmup-file",
    type=str,
    dest="warmup_file",
    help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .",
)




add_shared_args(parser)
# Server-specific defaults so it "just works" without flags
parser.set_defaults(
    model="tiny.en",
    min_chunk_size=2.0,  # Use 2 seconds for reliable transcription
    vac=False,  # Keep VAC disabled for simplicity
    vac_chunk_size=0.04,
    vad_threshold=0.3,  # Balanced sensitivity
    vad_min_silence_duration_ms=800,  # 800ms silence before considering speech ended
)
args = parser.parse_args()

asr, tokenizer = backend_factory(args)

# Test ASR with a simple audio sample to verify it's working
print("Testing ASR with dummy audio...")
try:
    dummy_audio = np.random.normal(0, 0.1, 16000).astype(np.float32)  # 1 second of noise
    test_result = asr.transcribe(dummy_audio)
    print(f"ASR test result: {test_result}")
except Exception as e:
    print(f"ASR test failed: {e}")




# Load demo HTML for the root endpoint
with open("src/web/live_transcription.html", "r", encoding="utf-8") as f:
    html = f.read()

# --- TTS: Edge TTS (Microsoft Neural Voices) ---
_tts_rate = 16000  # Keep 16kHz for compatibility with frontend

# Try to import edge_tts, with fallback to subprocess call
_edge_tts_module = None
_tts_available = False
_edge_voices = []

try:
    import edge_tts
    _edge_tts_module = edge_tts
    _tts_available = True
    print("[TTS] Edge TTS module imported successfully")
except ImportError:
    print("[TTS] Edge TTS module not found in current environment")
    # Check if edge-tts command is available globally
    try:
        import subprocess
        result = subprocess.run(["edge-tts", "--list-voices"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            _tts_available = True
            print("[TTS] Edge TTS command-line tool found")
        else:
            print("[TTS] Edge TTS command-line tool not found")
    except Exception as e:
        print(f"[TTS] Could not check for edge-tts command: {e}")

if _tts_available:
    # Available high-quality voices
    _edge_voices = [
        "en-US-AriaNeural",      # Female, warm and friendly
    ]

    # Default voice (warm female)
    _current_voice = _edge_voices[0]

    print(f"[TTS] Edge TTS initialized with {len(_edge_voices)} high-quality neural voices")
    print(f"[TTS] Default voice: {_current_voice}")

    # Test Edge TTS connectivity (will be done when server starts)
    async def test_edge_tts():
        try:
            print("[TTS] Testing Edge TTS connectivity...")
            communicate = _edge_tts_module.Communicate("Hello", "en-US-AriaNeural")
            test_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    test_data += chunk["data"]
                    break  # Just get first chunk
            if test_data:
                print("[TTS] Edge TTS connectivity test: SUCCESS")
                return True
            else:
                print("[TTS] Edge TTS connectivity test: FAILED - No audio data")
                return False
        except Exception as e:
            print(f"[TTS] Edge TTS connectivity test: FAILED - {e}")
            return False

else:
    print("[TTS] Edge TTS not available. Please install with: pip install edge-tts")


async def tts_synthesize(text: str) -> Optional[bytes]:
    """Synthesize mono 16k wav bytes from text using Edge TTS. Returns WAV bytes or None."""
    if not text or not _tts_available:
        return None

    try:
        # Clean and prepare text for better synthesis
        clean_text = text.strip()
        if not clean_text:
            return None

        # Improve text for better prosody
        # Add punctuation for natural pauses
        if not clean_text.endswith(('.', '!', '?', ':')):
            clean_text += '.'

        # Ensure proper capitalization
        if clean_text and clean_text[0].islower():
            clean_text = clean_text[0].upper() + clean_text[1:]

        # Validate text length and content
        if len(clean_text.strip()) < 3:
            print(f"[TTS] Text too short for synthesis: '{clean_text}'")
            return None

        # Remove any problematic characters that might cause Edge TTS issues
        clean_text = clean_text.replace('"', "'").replace('\n', ' ').replace('\r', ' ')
        clean_text = ' '.join(clean_text.split())  # Normalize whitespace

        # Select voice based on text hash for consistency
        import hashlib
        text_hash = int(hashlib.md5(clean_text.encode()).hexdigest()[:8], 16)
        voice_idx = text_hash % len(_edge_voices)
        selected_voice = _edge_voices[voice_idx]

        print(f"[TTS] Synthesizing '{clean_text}' with voice: {selected_voice}")

        import tempfile
        import os
        import subprocess

        # Try using the module first, then fall back to command line
        audio_data = None

        if _edge_tts_module:
            try:
                # Try with a more reliable voice first
                reliable_voices = ["en-US-AriaNeural", "en-US-JennyNeural", "en-US-GuyNeural"]

                # If selected voice fails, try reliable ones
                voices_to_try = [selected_voice] + [v for v in reliable_voices if v != selected_voice]

                audio_data = None
                for voice in voices_to_try:
                    try:
                        print(f"[TTS] Trying voice: {voice}")

                        # Use the imported module with timeout
                        communicate = _edge_tts_module.Communicate(clean_text, voice)

                        # Generate audio with timeout
                        audio_data = b""
                        start_time = asyncio.get_event_loop().time()

                        async for chunk in communicate.stream():
                            if chunk["type"] == "audio":
                                audio_data += chunk["data"]

                            # Check timeout manually
                            if asyncio.get_event_loop().time() - start_time > 10:
                                print(f"[TTS] Voice {voice} timed out after 10 seconds")
                                audio_data = None
                                break

                        if audio_data and len(audio_data) > 0:
                            print(f"[TTS] Generated {len(audio_data)} bytes using voice {voice}")
                            break
                        else:
                            print(f"[TTS] Voice {voice} returned empty audio data")
                            audio_data = None

                    except Exception as voice_error:
                        print(f"[TTS] Voice {voice} failed: {voice_error}")
                        audio_data = None
                        continue

                if not audio_data:
                    print("[TTS] All Edge TTS voices failed")

            except Exception as e:
                print(f"[TTS] Module approach failed: {e}")
                audio_data = None

        # Fallback to command line if module failed or not available
        if not audio_data:
            try:
                print("[TTS] Trying command-line approach...")

                # Use command line edge-tts
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
                    temp_mp3_path = temp_mp3.name

                cmd = [
                    "edge-tts",
                    "--voice", selected_voice,
                    "--text", clean_text,
                    "--write-media", temp_mp3_path
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0 and os.path.exists(temp_mp3_path):
                    with open(temp_mp3_path, "rb") as f:
                        audio_data = f.read()
                    print(f"[TTS] Generated {len(audio_data)} bytes using command line")
                else:
                    print(f"[TTS] Command line failed: {result.stderr}")
                    audio_data = None  # Don't return, continue to fallback

            except Exception as e:
                print(f"[TTS] Command line approach failed: {e}")
                audio_data = None  # Don't return, continue to fallback

        if not audio_data or len(audio_data) == 0:
            print("[TTS] No audio data generated from Edge TTS, trying Windows SAPI fallback...")

            # Try Windows SAPI as fallback
            try:
                import pyttsx3

                # Initialize Windows SAPI TTS
                engine = pyttsx3.init()

                # Set properties for better quality
                voices = engine.getProperty('voices')
                if voices:
                    # Use first available voice
                    engine.setProperty('voice', voices[0].id)

                engine.setProperty('rate', 180)  # Speaking rate
                engine.setProperty('volume', 0.9)  # Volume level

                # Save to temporary file
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                    temp_wav_path = temp_wav.name

                engine.save_to_file(clean_text, temp_wav_path)
                engine.runAndWait()

                # Read the generated WAV file
                if os.path.exists(temp_wav_path):
                    with open(temp_wav_path, "rb") as f:
                        sapi_audio = f.read()

                    # Clean up
                    try:
                        os.unlink(temp_wav_path)
                    except:
                        pass

                    if sapi_audio:
                        print(f"[TTS] Generated {len(sapi_audio)} bytes using Windows SAPI")
                        return sapi_audio

            except Exception as sapi_error:
                print(f"[TTS] Windows SAPI fallback failed: {sapi_error}")

            # Final fallback: generate a longer tone that represents speech duration
            print("[TTS] Using tone fallback...")
            import numpy as np
            import io
            import wave

            # Estimate speech duration (roughly 150 words per minute)
            word_count = len(clean_text.split())
            estimated_duration = max(2.0, word_count / 2.5)  # Minimum 2 seconds

            sample_rate = 16000
            t = np.linspace(0, estimated_duration, int(sample_rate * estimated_duration), False)

            # Create a more pleasant multi-tone sound
            tone1 = np.sin(440 * 2 * np.pi * t) * 0.1  # A4
            tone2 = np.sin(554 * 2 * np.pi * t) * 0.1  # C#5
            combined_tone = (tone1 + tone2) * 0.5

            # Add fade in/out
            fade_samples = int(0.1 * sample_rate)  # 0.1 second fade
            combined_tone[:fade_samples] *= np.linspace(0, 1, fade_samples)
            combined_tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)

            # Convert to 16-bit PCM WAV
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                pcm16 = (combined_tone * 32767.0).astype(np.int16)
                wf.writeframes(pcm16.tobytes())

            print(f"[TTS] Generated {estimated_duration:.1f}s fallback tone")
            return buf.getvalue()

        # Edge TTS returns MP3, convert to WAV at 16kHz
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
            temp_mp3.write(audio_data)
            temp_mp3_path = temp_mp3.name

        try:
            # Convert MP3 to 16kHz WAV using ffmpeg
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav_path = temp_wav.name

            # Use ffmpeg to convert MP3 to 16kHz mono WAV
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", temp_mp3_path,
                "-ar", "16000", "-ac", "1", "-f", "wav",
                temp_wav_path
            ]

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[TTS] FFmpeg conversion failed: {result.stderr}")
                return None

            # Read the converted WAV file
            with open(temp_wav_path, "rb") as f:
                wav_data = f.read()

            print(f"[TTS] Successfully synthesized {len(wav_data)} bytes of audio")
            return wav_data

        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_mp3_path)
                os.unlink(temp_wav_path)
            except:
                pass

    except Exception as e:
        print(f"[TTS] Edge TTS synthesis failed: {e}")
        return None


@app.get("/")
async def get():
    return HTMLResponse(html)


SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_SAMPLE = 2  # s16le = 2 bytes per sample

# Allow sub-second chunk sizes without truncation
# Use VAC chunk size when VAC is enabled, otherwise use min_chunk_size
if args.vac:
    CHUNK_SEC = float(args.vac_chunk_size)
else:
    CHUNK_SEC = float(args.min_chunk_size)
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_SEC)
BYTES_PER_CHUNK = SAMPLES_PER_CHUNK * BYTES_PER_SAMPLE

# Read small pieces frequently to reduce latency (default ~50ms)
BYTES_PER_READ = max(4096, int(SAMPLE_RATE * 0.05) * BYTES_PER_SAMPLE)


async def start_ffmpeg_decoder():
    """
    Start an FFmpeg process in async streaming mode that reads WebM from stdin
    and outputs raw s16le PCM on stdout. Returns the process object.
    """
    process = (
        ffmpeg.input("pipe:0", format="webm")
        .output(
            "pipe:1",
            format="s16le",
            acodec="pcm_s16le",
            ac=CHANNELS,
            ar=str(SAMPLE_RATE),
        )
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )
    print(f"[FFMPEG] Started FFmpeg WebM decoder process PID {process.pid}")
    return process



@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection opened.")

    ffmpeg_process = await start_ffmpeg_decoder()
    pcm_buffer = bytearray()
    print("Loading online.")
    online = online_factory(args, asr, tokenizer)
    print("Online loaded.")

    # Test Edge TTS connectivity when connection starts
    if _tts_available and _edge_tts_module:
        await test_edge_tts()



    # Continuously read decoded PCM from ffmpeg stdout in a background task
    async def ffmpeg_stdout_reader():
        nonlocal pcm_buffer
        loop = asyncio.get_event_loop()
        full_transcription = ""
        beg = time()
        
        chunk_history = []  # Will store dicts: {beg, end, text, speaker}
        last_final_text = ""  # Track last text sent to agent to avoid duplicates
        in_flight_agent = None  # asyncio.Task or None
        last_text_time = time()  # Track when we last received text for timeout-based triggering

        # Simple audio buffer for forced transcription
        simple_audio_buffer = []
        last_forced_transcription = time()

        # Conversation state management
        conversation_state = "listening"  # "listening", "processing", "speaking"
        tts_playing = False
        last_agent_response_time = 0
        tts_start_time = 0
        expected_tts_duration = 0
        
        async def _run_agent_and_send(text: str):
            nonlocal in_flight_agent, conversation_state, tts_playing, last_agent_response_time, chunk_history, last_final_text, tts_start_time, expected_tts_duration
            try:
                print(f"Calling agent with '{text}'")
                conversation_state = "processing"

                print("[AGENT] Starting agent call...")

                # For now, use the simple fallback agent directly since AutoGen is hanging
                # TODO: Fix AutoGen integration later
                print("[AGENT] Using simple fallback agent (AutoGen disabled due to hanging)")
                agent_reply = await get_simple_agent_response(text)
                print(f"[AGENT] Fallback agent reply: '{agent_reply}'")

            except Exception as e:
                print(f"[AGENT] All agents failed: {e}")
                agent_reply = f"I'm sorry, I had trouble thinking. Error: {str(e)[:100]}"

            # Prepare payload with conversation state
            payload = {
                "agent_text": agent_reply,
                "conversation_state": "speaking"
            }

            # Synthesize TTS
            conversation_state = "speaking"
            tts_playing = True
            tts_start_time = time()
            wav_bytes = await tts_synthesize(agent_reply)

            if wav_bytes:
                b64 = base64.b64encode(wav_bytes).decode("ascii")
                payload.update({
                    "agent_tts_wav_b64": b64,
                    "sample_rate": 16000,
                })

                # Calculate TTS duration for timing
                audio_duration = len(wav_bytes) / (16000 * 2)  # bytes / (sample_rate * bytes_per_sample)
                payload["tts_duration"] = audio_duration
                expected_tts_duration = audio_duration

            else:
                print("[TTS] No audio generated (TTS not initialized or synth failed)")
                payload["tts_duration"] = 0

            try:
                await websocket.send_json(payload)
                print(f"Agent response sent, TTS duration: {payload.get('tts_duration', 0):.1f}s")

                # Schedule restart for after TTS finishes (non-blocking)
                if wav_bytes:
                    # Create a task to handle the restart after TTS finishes
                    async def restart_after_tts():
                        sleep_duration = payload["tts_duration"] + 1.5
                        print(f"[RESTART] Waiting {sleep_duration:.1f}s for TTS to finish...")
                        await asyncio.sleep(sleep_duration)

                        nonlocal conversation_state, tts_playing, last_agent_response_time, chunk_history, last_final_text

                        print("[RESTART] TTS should be finished, sending restart signal...")

                        # Send restart listening signal
                        restart_payload = {
                            "type": "restart_listening",
                            "conversation_state": "listening",
                            "message": "Ready for next question"
                        }
                        try:
                            await websocket.send_json(restart_payload)
                            print("[RESTART] Sent restart listening signal")

                            conversation_state = "listening"
                            tts_playing = False
                            last_agent_response_time = time()

                            # Reset transcription state for next question
                            chunk_history.clear()
                            last_final_text = ""
                            print("[RESTART] Reset transcription state for next question")

                            # Also reset the ASR state to start fresh
                            try:
                                # Send a reset signal to clear any accumulated state
                                reset_signal = {
                                    "type": "reset_transcription",
                                    "message": "Transcription reset for new question"
                                }
                                await websocket.send_json(reset_signal)
                                print("[RESTART] Sent reset transcription signal")
                            except Exception as e:
                                print(f"[RESTART] Failed to send reset signal: {e}")
                        except Exception as e:
                            print(f"[RESTART] Failed to send restart signal: {e}")
                            # Force state change even if websocket fails
                            conversation_state = "listening"
                            tts_playing = False
                            print("[RESTART] Forced state change to listening")

                    # Start the restart task without waiting for it
                    restart_task = asyncio.create_task(restart_after_tts())
                    print(f"[RESTART] Created restart task for {payload['tts_duration']:.1f}s TTS")
                else:
                    # No TTS, restart immediately
                    conversation_state = "listening"
                    tts_playing = False

            except Exception as _e:
                print(f"Failed to send agent payload: {_e}")
                conversation_state = "listening"
                tts_playing = False
            finally:
                in_flight_agent = None
        
        while True:
            try:
                # Read small amount frequently to reduce latency
                chunk = await loop.run_in_executor(
                    None, ffmpeg_process.stdout.read, BYTES_PER_READ
                )
                if not chunk:  # FFmpeg might have closed
                    await asyncio.sleep(0.02)
                    continue

                pcm_buffer.extend(chunk)

                # Process in CHUNK-sized frames to avoid buffering too long
                while len(pcm_buffer) >= BYTES_PER_CHUNK:
                    frame = pcm_buffer[:BYTES_PER_CHUNK]
                    pcm_buffer = pcm_buffer[BYTES_PER_CHUNK:]

                    # Convert int16 -> float32
                    pcm_array = (
                        np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
                    )

                    # Debug: Check audio data
                    audio_level = np.abs(pcm_array).mean()
                    if not hasattr(online, '_total_audio_processed'):
                        online._total_audio_processed = 0
                    online._total_audio_processed += len(pcm_array)

                    if online._total_audio_processed % 16000 == 0:  # Every second of audio
                        print(f"Processed {online._total_audio_processed/16000:.1f}s of audio, current level: {audio_level:.4f}")

                    # Add to simple buffer for forced transcription
                    simple_audio_buffer.extend(pcm_array)

                    online.insert_audio_chunk(pcm_array)
                    beg_trans, end_trans, trans = online.process_iter()

                    # Force transcription every 4 seconds if no transcription is happening
                    current_time = time()
                    if current_time - last_forced_transcription > 4.0 and len(simple_audio_buffer) > 32000:  # 2+ seconds of audio
                        print(f"Forcing transcription on {len(simple_audio_buffer)} audio samples...")
                        try:
                            # Convert buffer to numpy array and transcribe directly
                            audio_array = np.array(simple_audio_buffer, dtype=np.float32)
                            force_result = asr.transcribe(audio_array)

                            # Extract clean text from the result
                            forced_text = ""
                            if force_result:
                                if isinstance(force_result, dict) and 'text' in force_result:
                                    forced_text = force_result['text'].strip()
                                elif hasattr(force_result, 'text'):
                                    forced_text = force_result.text.strip()
                                elif isinstance(force_result, str):
                                    forced_text = force_result.strip()
                                else:
                                    print(f"Unexpected force_result type: {type(force_result)}")

                            if forced_text:
                                print(f"Forced transcription: '{forced_text}'")
                                # Use the forced transcription as if it came from normal processing
                                trans = forced_text
                                beg_trans = 0
                                end_trans = len(audio_array) / 16000
                                print(f"Setting trans='{trans}' for normal processing flow")

                        except Exception as e:
                            print(f"Force transcription failed: {e}")

                        # Reset buffer and timer
                        simple_audio_buffer = []
                        last_forced_transcription = current_time

                    # Simple debug counter
                    if not hasattr(online, '_debug_counter'):
                        online._debug_counter = 0
                    online._debug_counter += 1

                    # Log every 100 chunks to avoid spam, and always log when we get transcription
                    if online._debug_counter % 100 == 1 or trans:
                        print(f"Processed {online._debug_counter} audio chunks, latest trans: '{trans}'")
                    
                    if trans:
                        # Ensure trans is clean text, not JSON
                        clean_trans = trans
                        if isinstance(trans, dict):
                            clean_trans = trans.get('text', str(trans))
                        elif not isinstance(trans, str):
                            clean_trans = str(trans)

                        print(f"Adding to chunk_history: '{clean_trans}'")
                        chunk_history.append({
                        "beg": beg_trans,
                        "end": end_trans,
                        "text": clean_trans,
                        "speaker": "0"
                        })
                    
                    # Only add clean text to full_transcription
                    if trans and isinstance(trans, str) and not trans.startswith('{'):
                        full_transcription += trans

                    # Simplified buffer logic - for forced transcription, assume buffer is empty
                    try:
                        if args.vac:
                            buffer = online.online.to_flush(
                                online.online.transcript_buffer.buffer
                            )[2]
                        else:
                            buffer = online.to_flush(online.transcript_buffer.buffer)[2]
                        if buffer in full_transcription:
                            buffer = ""
                    except Exception as e:
                        print(f"Buffer calculation failed: {e}, using empty buffer")
                        buffer = ""  # Default to empty buffer if calculation fails
                                        
                    # Single-speaker: use only the most recent complete transcription
                    lines = [
                        {
                            "speaker": "0",
                            "text": "",
                        }
                    ]

                    # Only use the most recent chunk if we have any
                    if chunk_history:
                        # Get the most recent chunk
                        recent_chunk = chunk_history[-1]
                        text_to_add = recent_chunk['text']

                        # Ensure we only add clean text, not JSON objects
                        if isinstance(text_to_add, dict):
                            text_to_add = text_to_add.get('text', '')
                        elif not isinstance(text_to_add, str):
                            text_to_add = str(text_to_add)

                        # Skip empty or JSON-like strings
                        if text_to_add and not text_to_add.startswith('{'):
                            lines[-1]["text"] = text_to_add  # Use = instead of += to avoid concatenation

                    # Always push live transcript update
                    response = {"lines": lines, "buffer": buffer}
                    await websocket.send_json(response)

                    # Simplified agent triggering logic
                    user_text = lines[-1]["text"].strip()
                    current_time = time()

                    # Update last_text_time if we have new text
                    if user_text and user_text != last_final_text:
                        last_text_time = current_time

                    # Check for timeout-based triggering (2 seconds of no new text)
                    text_timeout = current_time - last_text_time > 2.0

                    # Improved triggering logic: require complete sentences or sufficient timeout
                    is_complete_sentence = (buffer == "" and user_text and
                                          (user_text.endswith(('?', '.', '!')) or len(user_text) >= 10))
                    is_timeout_with_content = (text_timeout and user_text and len(user_text) >= 8)

                    should_trigger = is_complete_sentence or is_timeout_with_content

                    # Only trigger if we're in listening state and not already processing
                    can_trigger = (conversation_state == "listening" and not tts_playing and in_flight_agent is None)

                    # Additional check: don't trigger on very short or incomplete text
                    if should_trigger and len(user_text.strip()) < 5:
                        should_trigger = False

                    # Backup restart mechanism: if stuck in speaking state for too long, force restart
                    current_time = time()
                    if (conversation_state == "speaking" and tts_start_time > 0 and
                        current_time - tts_start_time > expected_tts_duration + 5.0):
                        print(f"[BACKUP] TTS stuck for {current_time - tts_start_time:.1f}s, forcing restart...")
                        conversation_state = "listening"
                        tts_playing = False
                        chunk_history.clear()
                        last_final_text = ""
                        tts_start_time = 0
                        expected_tts_duration = 0
                        print("[BACKUP] Forced restart completed")

                    # Debug agent triggering
                    if user_text:
                        print(f"Agent trigger check: user_text='{user_text}', should_trigger={should_trigger}, can_trigger={can_trigger}, state='{conversation_state}', buffer='{buffer}', text_timeout={text_timeout}")

                    if should_trigger and can_trigger and user_text != last_final_text:
                        print(f"Calling agent with '{user_text}' (state: {conversation_state})")

                        # Clear any old transcription state before processing new question
                        if conversation_state == "listening":
                            # Keep only the most recent chunk that triggered the agent
                            if chunk_history:
                                latest_chunk = chunk_history[-1]
                                chunk_history.clear()
                                chunk_history.append(latest_chunk)
                                print("Cleared old chunks, keeping only the triggering chunk")

                        last_final_text = user_text
                        in_flight_agent = asyncio.create_task(_run_agent_and_send(user_text))
                    
            except Exception as e:
                print(f"Exception in ffmpeg_stdout_reader: {e}")
                break

        print("Exiting ffmpeg_stdout_reader...")

    stdout_reader_task = asyncio.create_task(ffmpeg_stdout_reader())

    try:
        message_count = 0
        while True:
            # Receive incoming WebM audio chunks from the client
            message = await websocket.receive_bytes()
            message_count += 1

            # Debug: Log message reception
            if message_count % 10 == 1:  # Every 10th message
                print(f"Received WebSocket message #{message_count}, size: {len(message)} bytes")

            # Only skip extremely small messages
            if len(message) < 10:
                print(f"[DEBUG] Very small message #{message_count}: {len(message)} bytes")
                print(f"[DEBUG] Message content: {message}")
                continue

            # Log first few messages for debugging
            if message_count <= 3:
                print(f"[DEBUG] Message #{message_count}: {len(message)} bytes, format: {message[:20]}")

            # Pass them to ffmpeg via stdin
            try:
                ffmpeg_process.stdin.write(message)
                ffmpeg_process.stdin.flush()
            except Exception as e:
                print(f"[DEBUG] FFmpeg write error: {e}")
                # Don't restart immediately, just continue
                continue

    except WebSocketDisconnect:
        print("WebSocket connection closed.")
    except Exception as e:
        print(f"Error in websocket loop: {e}")
    finally:
        # Clean up ffmpeg and the reader task
        try:
            ffmpeg_process.stdin.close()
        except:
            pass
        stdout_reader_task.cancel()

        try:
            ffmpeg_process.stdout.close()
        except:
            pass

        ffmpeg_process.wait()
        del online




if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "whisper_fastapi_online_server:app", host=args.host, port=args.port, reload=True
    )
