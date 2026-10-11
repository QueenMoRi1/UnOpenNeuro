"""Text-to-speech output and microphone input.

Heavy audio libraries are imported lazily so the core server runs without
them. Install them with:  pip install -e ".[voice,listen]"
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from typing import Any

log = logging.getLogger("unopenneuro.voice")

KOKORO_RATE = 24000
LIPSYNC_FPS = 30


class Speaker:
    """Speaks lines one at a time, optionally driving an avatar's mouth."""

    def __init__(self, cfg: dict[str, Any], avatar=None):
        self.backend = cfg.get("tts", "none")
        self.pitch = float(cfg.get("pitch", 1.0))
        self.voice = cfg.get("kokoro_voice", "af_heart")
        self.avatar = avatar
        self._lock = asyncio.Lock()
        self._pipeline = None
        if self.backend == "say" and not shutil.which("say"):
            raise RuntimeError("tts: say only works on macOS")

    async def __call__(self, text: str) -> None:
        async with self._lock:
            if self.backend == "say":
                await self._say(text)
            elif self.backend == "kokoro":
                await self._kokoro(text)

    async def _say(self, text: str) -> None:
        # macOS default rate is ~175 words per minute.
        rate = str(int(175 * self.pitch))
        proc = await asyncio.create_subprocess_exec("say", "-r", rate, text)
        await proc.wait()

    async def _kokoro(self, text: str) -> None:
        import numpy as np
        import sounddevice as sd

        if self._pipeline is None:
            from kokoro import KPipeline

            self._pipeline = await asyncio.to_thread(KPipeline, lang_code="a")

        def synth():
            chunks = [
                np.asarray(audio, dtype=np.float32)
                for _, _, audio in self._pipeline(text, voice=self.voice)
            ]
            return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

        audio = await asyncio.to_thread(synth)
        if audio.size == 0:
            return
        # Playing faster than it was generated raises the pitch, which is
        # roughly how Neuro's voice is made from a stock TTS voice.
        rate = int(KOKORO_RATE * self.pitch)
        sd.play(audio, samplerate=rate)
        if self.avatar is not None:
            await self._lipsync(audio, rate)
        await asyncio.to_thread(sd.wait)

    async def _lipsync(self, audio, rate: int) -> None:
        import numpy as np

        frame = max(1, rate // LIPSYNC_FPS)
        n = len(audio) // frame
        rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1))
        peak = float(rms.max()) or 1.0
        loop = asyncio.get_running_loop()
        start = loop.time()
        for i, value in enumerate(rms):
            await self.avatar.set_mouth(min(1.0, float(value) / peak * 1.4))
            delay = start + (i + 1) / LIPSYNC_FPS - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
        await self.avatar.set_mouth(0.0)


class Listener:
    """Push-to-talk speech-to-text with faster-whisper."""

    SAMPLE_RATE = 16000

    def __init__(self, cfg: dict[str, Any]):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            cfg.get("whisper_model", "base.en"), device="cpu", compute_type="int8"
        )

    def record_until_enter(self):
        import numpy as np
        import sounddevice as sd

        chunks = []
        with sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=lambda data, *_: chunks.append(data.copy()),
        ):
            input("  (recording... press Enter to stop) ")
        return np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, "float32")

    def transcribe(self, audio) -> str:
        segments, _ = self.model.transcribe(audio, language="en")
        return " ".join(s.text.strip() for s in segments).strip()
