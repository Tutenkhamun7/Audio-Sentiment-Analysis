/**
 * Resamples any audio blob (WebM, Opus, MP4, etc.) to 16kHz 16-bit Mono PCM WAV
 * using native browser Web Audio API (OfflineAudioContext).
 */
export async function convertBlobTo16kHzMonoWav(inputBlob: Blob): Promise<Blob> {
  const arrayBuffer = await inputBlob.arrayBuffer();

  // Create temporary AudioContext to decode compressed input audio stream
  const AudioCtxClass =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;

  if (!AudioCtxClass) {
    throw new Error("Web Audio API is not supported in this browser.");
  }

  const decodeCtx = new AudioCtxClass();
  let decodedBuffer: AudioBuffer;

  try {
    decodedBuffer = await decodeCtx.decodeAudioData(arrayBuffer);
  } finally {
    // Release decode context resources
    if (decodeCtx.state !== "closed") {
      await decodeCtx.close();
    }
  }

  const targetSampleRate = 16000;
  const numChannels = 1; // Standardized Mono
  const duration = decodedBuffer.duration;
  const totalFrames = Math.max(1, Math.ceil(duration * targetSampleRate));

  // Render resampled mono audio using OfflineAudioContext
  const OfflineCtxClass =
    window.OfflineAudioContext ||
    (window as unknown as { webkitOfflineAudioContext: typeof OfflineAudioContext })
      .webkitOfflineAudioContext;

  if (!OfflineCtxClass) {
    throw new Error("OfflineAudioContext is not supported in this browser.");
  }

  const offlineCtx = new OfflineCtxClass(numChannels, totalFrames, targetSampleRate);

  const sourceNode = offlineCtx.createBufferSource();
  sourceNode.buffer = decodedBuffer;
  sourceNode.connect(offlineCtx.destination);
  sourceNode.start(0);

  const renderedBuffer = await offlineCtx.startRendering();
  const channelData = renderedBuffer.getChannelData(0);

  // Encode Float32Array to 16-bit PCM WAV ArrayBuffer
  const wavBuffer = encodeWavPCM(channelData, targetSampleRate);

  return new Blob([wavBuffer], { type: "audio/wav" });
}

function writeString(view: DataView, offset: number, string: string): void {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

function encodeWavPCM(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const numChannels = 1;
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // 1. RIFF chunk descriptor
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true); // ChunkSize: 4 + (8 + SubChunk1Size) + (8 + SubChunk2Size)
  writeString(view, 8, "WAVE");

  // 2. "fmt " sub-chunk
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true); // AudioFormat (1 = PCM)
  view.setUint16(22, numChannels, true); // NumChannels (1 = Mono)
  view.setUint32(24, sampleRate, true); // SampleRate (16000)
  view.setUint32(28, byteRate, true); // ByteRate (16000 * 1 * 2 = 32000)
  view.setUint16(32, blockAlign, true); // BlockAlign (2 bytes)
  view.setUint16(34, bitsPerSample, true); // BitsPerSample (16 bits)

  // 3. "data" sub-chunk
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);

  // Write the PCM audio samples
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    // Clamp sample between -1.0 and 1.0
    const s = Math.max(-1, Math.min(1, samples[i]));
    // Convert float to 16-bit signed PCM integer (-32768 to 32767)
    const int16 = s < 0 ? s * 0x8000 : s * 0x7fff;
    view.setInt16(offset, int16, true);
    offset += 2;
  }

  return buffer;
}
