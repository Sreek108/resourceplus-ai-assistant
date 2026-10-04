"""Capture one microphone utterance and compare bilingual Azure STT candidates.

This talks only to Azure Speech. It does not call ResourcePlus or execute HR writes.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.speech.azure_speech import recognize_audio_candidates


def _record_wav(seconds: float) -> bytes:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit("Install sounddevice to use microphone diagnostics.") from exc

    sample_rate = 16_000
    print(f"Recording for {seconds:g} seconds...")
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    target = io.BytesIO()
    with wave.open(target, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(audio.tobytes())
    return target.getvalue()


def _show(label: str, candidate) -> None:
    if candidate is None:
        print(f"{label}: not run")
        return
    print(
        f"{label}: locale={candidate.locale} confidence={candidate.confidence!r} "
        f"status={candidate.result_status} script={candidate.script_class} "
        f"transcript={candidate.transcript!r}"
    )


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--session-language", choices=("en", "ar"))
    args = parser.parse_args()
    audio = await asyncio.to_thread(_record_wav, args.seconds)
    result = await recognize_audio_candidates(
        audio,
        established_language=args.session_language,
        force_fallback=True,
    )
    _show("AutoDetect", result.auto)
    _show("en-US candidate", result.english)
    _show("ar-SA candidate", result.arabic)
    print(f"Selected transcript: {result.selected.transcript!r}")
    print(f"Selected language: {result.selected.language}")
    print(f"Selection reason: {result.selection_reason}")


if __name__ == "__main__":
    asyncio.run(_main())
