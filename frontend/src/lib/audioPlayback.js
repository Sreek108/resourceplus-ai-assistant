function defaultContextConstructor() {
  return globalThis.AudioContext || globalThis.webkitAudioContext;
}

function completedPlayback() {
  return { started: false, ended: Promise.resolve(), stop() {} };
}

export function createAudioPlaybackManager({
  getContextConstructor = defaultContextConstructor,
  fetchAudio = (...args) => globalThis.fetch(...args),
  createAudio = (url) => globalThis.Audio(url),
} = {}) {
  let context = null;
  let currentPlayback = null;

  async function unlock() {
    const AudioContextConstructor = getContextConstructor();
    if (!AudioContextConstructor) return false;
    try {
      if (!context || context.state === "closed") context = new AudioContextConstructor();
      if (context.state !== "running" && context.state !== "closed") {
        await context.resume();
      }
    } catch {
      return false;
    }
    return context.state === "running";
  }

  function stop() {
    currentPlayback?.stop();
    currentPlayback = null;
  }

  async function playWithWebAudio(url) {
    if (!context) return null;
    if (context.state !== "running" && context.state !== "closed") {
      try {
        await context.resume();
      } catch {
        return null;
      }
    }
    if (context.state !== "running") return null;

    const response = await fetchAudio(url);
    if (!response.ok) throw new Error("Assistant audio could not be loaded.");
    const encodedAudio = await response.arrayBuffer();
    const decodedAudio = await context.decodeAudioData(encodedAudio.slice(0));
    const source = context.createBufferSource();
    source.buffer = decodedAudio;
    source.connect(context.destination);

    let finished = false;
    let resolveEnded;
    const ended = new Promise((resolve) => {
      resolveEnded = resolve;
    });
    const finish = () => {
      if (finished) return;
      finished = true;
      if (currentPlayback === playback) currentPlayback = null;
      resolveEnded();
    };
    const playback = {
      started: true,
      ended,
      stop() {
        if (finished) return;
        try {
          source.stop();
        } catch {
          // An ended AudioBufferSource cannot be stopped twice.
        }
        finish();
      },
    };
    source.onended = finish;
    currentPlayback = playback;
    source.start(0);
    return playback;
  }

  async function playWithHtmlAudio(url) {
    const audio = createAudio(url);
    let finished = false;
    let resolveEnded;
    const ended = new Promise((resolve) => {
      resolveEnded = resolve;
    });
    const finish = () => {
      if (finished) return;
      finished = true;
      if (currentPlayback === playback) currentPlayback = null;
      resolveEnded();
    };
    const playback = {
      started: false,
      ended,
      stop() {
        audio.pause();
        finish();
      },
    };
    audio.onended = finish;
    audio.onerror = finish;
    currentPlayback = playback;
    try {
      await audio.play();
      playback.started = true;
      return playback;
    } catch {
      playback.stop();
      return completedPlayback();
    }
  }

  async function play(url, { userGesture = false } = {}) {
    stop();
    if (userGesture) await unlock();
    if (context) {
      try {
        const playback = await playWithWebAudio(url);
        if (playback) return playback;
      } catch {
        // Loading/decoding failure falls back to the retained Blob URL.
      }
    }
    return playWithHtmlAudio(url);
  }

  async function dispose() {
    stop();
    const existingContext = context;
    context = null;
    if (existingContext && existingContext.state !== "closed") {
      try {
        await existingContext.close();
      } catch {
        // Cleanup must not surface a user-facing voice error.
      }
    }
  }

  return { unlock, play, stop, dispose };
}
