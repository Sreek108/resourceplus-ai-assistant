from app.speech.azure_speech import (
    SpeechAudio,
    SpeechConfigurationError,
    SpeechInputError,
    SpeechRecognitionError,
    SpeechServiceError,
    SpeechSynthesisError,
    SpeechTranscript,
    TTSProfile,
    resolve_tts_profile,
    synthesize_speech,
    transcribe_audio,
)
from app.speech.text import markdown_to_speech_text
from app.speech.language import (
    SpokenLanguageResolution,
    count_script_letters,
    detect_text_language,
    resolve_spoken_language,
)
from app.speech.streaming import (
    STREAM_BITS_PER_SAMPLE,
    STREAM_CHANNELS,
    STREAM_SAMPLE_RATE,
    StreamingSpeechRecognizer,
)

__all__ = [
    "SpeechAudio",
    "SpeechConfigurationError",
    "SpeechInputError",
    "SpeechRecognitionError",
    "SpeechServiceError",
    "SpeechSynthesisError",
    "SpeechTranscript",
    "TTSProfile",
    "resolve_tts_profile",
    "synthesize_speech",
    "transcribe_audio",
    "markdown_to_speech_text",
    "SpokenLanguageResolution",
    "count_script_letters",
    "detect_text_language",
    "resolve_spoken_language",
    "STREAM_BITS_PER_SAMPLE",
    "STREAM_CHANNELS",
    "STREAM_SAMPLE_RATE",
    "StreamingSpeechRecognizer",
]
