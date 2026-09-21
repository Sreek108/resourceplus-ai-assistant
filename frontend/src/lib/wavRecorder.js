function mergeBuffers(buffers) {
  const length = buffers.reduce((total, buffer) => total + buffer.length, 0);
  const merged = new Float32Array(length);
  let offset = 0;
  for (const buffer of buffers) {
    merged.set(buffer, offset);
    offset += buffer.length;
  }
  return merged;
}

function writeAscii(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

export class RecordingError extends Error {
  constructor(message, code) {
    super(message);
    this.name = "RecordingError";
    this.code = code;
  }
}

export function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (const sample of samples) {
    const clipped = Math.max(-1, Math.min(1, sample));
    view.setInt16(
      offset,
      clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff,
      true,
    );
    offset += 2;
  }
  return new Blob([view], { type: "audio/wav" });
}

export async function isValidWavBlob(blob, { minDataBytes = 1 } = {}) {
  if (!(blob instanceof Blob) || blob.size < 44) return false;
  const mimeType = blob.type.split(";", 1)[0].trim().toLowerCase();
  if (mimeType && !["audio/wav", "audio/x-wav", "application/octet-stream"].includes(mimeType)) {
    return false;
  }
  const buffer = await blob.arrayBuffer();
  const view = new DataView(buffer);
  const ascii = (offset, length) => String.fromCharCode(
    ...new Uint8Array(buffer, offset, length),
  );
  if (ascii(0, 4) !== "RIFF" || ascii(8, 4) !== "WAVE") return false;
  if (view.getUint32(4, true) + 8 > buffer.byteLength) return false;

  let offset = 12;
  let validPcmFormat = false;
  let usableData = false;
  while (offset + 8 <= buffer.byteLength) {
    const chunkId = ascii(offset, 4);
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkStart = offset + 8;
    const chunkEnd = chunkStart + chunkSize;
    if (chunkEnd > buffer.byteLength) return false;
    if (chunkId === "fmt " && chunkSize >= 16) {
      validPcmFormat = (
        view.getUint16(chunkStart, true) === 1
        && view.getUint16(chunkStart + 2, true) === 1
        && view.getUint32(chunkStart + 4, true) > 0
        && view.getUint16(chunkStart + 14, true) === 16
      );
    } else if (chunkId === "data") {
      usableData = chunkSize >= minDataBytes;
    }
    offset = chunkEnd + (chunkSize % 2);
  }
  return validPcmFormat && usableData;
}

export function encodePcm16(samples, inputSampleRate, outputSampleRate = 16_000) {
  if (!samples.length || inputSampleRate <= 0 || outputSampleRate <= 0) {
    return new ArrayBuffer(0);
  }
  const ratio = inputSampleRate / outputSampleRate;
  const outputLength = Math.max(1, Math.floor(samples.length / ratio));
  const output = new Int16Array(outputLength);
  for (let index = 0; index < outputLength; index += 1) {
    const start = Math.floor(index * ratio);
    const end = Math.max(start + 1, Math.floor((index + 1) * ratio));
    let total = 0;
    let count = 0;
    for (let sourceIndex = start; sourceIndex < end && sourceIndex < samples.length; sourceIndex += 1) {
      total += samples[sourceIndex];
      count += 1;
    }
    const sample = Math.max(-1, Math.min(1, total / Math.max(count, 1)));
    output[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return output.buffer;
}

export async function startWavRecording({ onPcmChunk } = {}) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new RecordingError(
      "Microphone recording isn’t supported in this browser.",
      "recording_unsupported",
    );
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
  } catch (error) {
    const denied = ["NotAllowedError", "PermissionDeniedError"].includes(error?.name);
    throw new RecordingError(
      denied
        ? "Microphone access is blocked. Allow it in your browser and try again."
        : "I couldn’t start the microphone. Please try again.",
      denied ? "permission_denied" : "recording_failed",
    );
  }

  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    stream.getTracks().forEach((track) => track.stop());
    throw new RecordingError(
      "Audio recording isn’t supported in this browser.",
      "recording_unsupported",
    );
  }
  const context = new AudioContextClass();
  if (context.state === "suspended") {
    try {
      await context.resume();
    } catch {
      stream.getTracks().forEach((track) => track.stop());
      await context.close();
      throw new RecordingError(
        "I couldn’t start microphone audio. Please try again.",
        "recording_failed",
      );
    }
  }
  const source = context.createMediaStreamSource(stream);
  const processor = context.createScriptProcessor(4096, 1, 1);
  const silentGain = context.createGain();
  silentGain.gain.value = 0;
  const buffers = [];

  processor.onaudioprocess = (event) => {
    const samples = new Float32Array(event.inputBuffer.getChannelData(0));
    buffers.push(samples);
    if (onPcmChunk) {
      const pcm = encodePcm16(samples, context.sampleRate);
      if (pcm.byteLength) onPcmChunk(pcm);
    }
  };
  source.connect(processor);
  processor.connect(silentGain);
  silentGain.connect(context.destination);

  let stopped = false;
  async function cleanUp() {
    processor.disconnect();
    source.disconnect();
    silentGain.disconnect();
    stream.getTracks().forEach((track) => track.stop());
    await context.close();
  }

  return {
    async stop() {
      if (stopped) throw new Error("Recording has already stopped.");
      stopped = true;
      const sampleRate = context.sampleRate;
      await cleanUp();
      const samples = mergeBuffers(buffers);
      if (!samples.length) {
        throw new RecordingError(
          "I couldn’t hear anything. Hold the mic while you speak.",
          "no_audio",
        );
      }
      return encodeWav(samples, sampleRate);
    },
    async cancel() {
      if (stopped) return;
      stopped = true;
      await cleanUp();
    },
  };
}
