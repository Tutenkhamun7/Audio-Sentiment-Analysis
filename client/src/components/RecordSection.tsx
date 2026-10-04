import React, { useState, useRef, useEffect, useCallback } from "react";
import { Mic, Square, Loader2, AlertCircle, XCircle } from "lucide-react";
import { convertBlobTo16kHzMonoWav } from "@/lib/wav-encoder";
import { WaveformVisualizer } from "@/components/WaveformVisualizer";

interface RecordSectionProps {
  onRecordingComplete: (wavBlob: Blob) => void;
  isLoading: boolean;
  onCancel?: () => void;
}

type RecordState = "idle" | "requesting" | "recording" | "converting";

export const RecordSection: React.FC<RecordSectionProps> = ({
  onRecordingComplete,
  isLoading,
  onCancel,
}) => {
  const [recordState, setRecordState] = useState<RecordState>("idle");
  const [recordDuration, setRecordDuration] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerIntervalRef = useRef<number | null>(null);

  // Clean up all audio tracks and contexts
  const cleanupAudioStreams = useCallback(() => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (sourceNodeRef.current) {
      try {
        sourceNodeRef.current.disconnect();
      } catch {
        // ignore
      }
      sourceNodeRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    setAnalyserNode(null);
  }, []);

  useEffect(() => {
    return () => {
      cleanupAudioStreams();
    };
  }, [cleanupAudioStreams]);

  const startRecording = async () => {
    setErrorMessage(null);
    setRecordState("requesting");
    chunksRef.current = [];

    try {
      // 1. Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      mediaStreamRef.current = stream;

      // 2. Setup Web Audio Analyser
      const AudioCtxClass =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioCtx = new AudioCtxClass();
      audioContextRef.current = audioCtx;

      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      setAnalyserNode(analyser);

      const source = audioCtx.createMediaStreamSource(stream);
      sourceNodeRef.current = source;
      source.connect(analyser);

      // 3. Setup MediaRecorder
      let mimeType = "";
      if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
        mimeType = "audio/webm;codecs=opus";
      } else if (MediaRecorder.isTypeSupported("audio/webm")) {
        mimeType = "audio/webm";
      } else if (MediaRecorder.isTypeSupported("audio/mp4")) {
        mimeType = "audio/mp4";
      }

      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        setRecordState("converting");
        try {
          const rawBlob = new Blob(chunksRef.current, {
            type: mimeType || "audio/webm",
          });

          // Resample and convert to 16kHz 16-bit Mono PCM WAV
          const wav16kMono = await convertBlobTo16kHzMonoWav(rawBlob);

          // Instant Auto-Submission
          onRecordingComplete(wav16kMono);
        } catch (err: unknown) {
          console.error("WAV conversion error:", err);
          setErrorMessage(
            "Failed to encode audio into 16kHz WAV format: " +
              (err instanceof Error ? err.message : String(err))
          );
        } finally {
          cleanupAudioStreams();
          setRecordState("idle");
          setRecordDuration(0);
        }
      };

      recorder.start(100); // 100ms chunk slices
      setRecordState("recording");
      setRecordDuration(0);

      // Duration counter
      const startTime = Date.now();
      timerIntervalRef.current = window.setInterval(() => {
        setRecordDuration(Math.floor((Date.now() - startTime) / 1000));
      }, 250);
    } catch (err: unknown) {
      cleanupAudioStreams();
      setRecordState("idle");
      console.error("Microphone error:", err);
      setErrorMessage(
        "Could not access microphone. Please grant browser microphone permission and ensure a microphone is connected."
      );
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
    }
  };

  const formatTimer = (totalSeconds: number): string => {
    const mins = Math.floor(totalSeconds / 60);
    const secs = totalSeconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="w-full flex flex-col items-center gap-6">
      {/* Real-time Oscilloscope Waveform Visualizer */}
      <div className="w-full flex flex-col items-center">
        <WaveformVisualizer
          analyserNode={analyserNode}
          isRecording={recordState === "recording"}
          durationText={recordState === "recording" ? formatTimer(recordDuration) : undefined}
          height={160}
        />
      </div>

      {/* Error alert banner */}
      {errorMessage && (
        <div className="w-full flex items-center gap-2.5 p-3 rounded-xl bg-destructive/10 border border-destructive/20 text-destructive text-xs">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Record Control Button Section */}
      <div className="flex flex-col items-center gap-3">
        <div className="relative flex items-center justify-center">
          {/* Subtle pulse ring when recording */}
          {recordState === "recording" && (
            <span className="absolute w-20 h-20 rounded-full bg-rose-500/15 animate-ping pointer-events-none" />
          )}

          {/* Main Record Action Button */}
          <button
            type="button"
            disabled={isLoading || recordState === "requesting" || recordState === "converting"}
            onClick={recordState === "recording" ? stopRecording : startRecording}
            className={`relative z-10 w-16 h-16 rounded-full flex items-center justify-center transition-all duration-200 shadow-xs focus:outline-none cursor-pointer ${
              recordState === "recording"
                ? "bg-rose-600 hover:bg-rose-700 text-white ring-4 ring-rose-500/20 scale-102"
                : recordState === "requesting"
                  ? "bg-muted text-foreground border border-border animate-pulse"
                  : recordState === "converting" || isLoading
                    ? "bg-muted text-muted-foreground cursor-wait"
                    : "bg-foreground text-background hover:opacity-90 active:scale-95"
            }`}
            title={
              recordState === "recording"
                ? "Stop recording & analyze"
                : "Start live voice recording"
            }
          >
            {recordState === "recording" ? (
              <Square className="w-6 h-6 fill-current" />
            ) : recordState === "requesting" ? (
              <Loader2 className="w-6 h-6 animate-spin" />
            ) : recordState === "converting" || isLoading ? (
              <Loader2 className="w-6 h-6 animate-spin" />
            ) : (
              <Mic className="w-6 h-6" />
            )}
          </button>
        </div>

        {/* State Label & Subtext */}
        <div className="text-center">
          <p className="text-xs sm:text-sm font-medium text-foreground">
            {recordState === "recording"
              ? `Press to Stop & Submit (${formatTimer(recordDuration)})`
              : recordState === "requesting"
                ? "Connecting to Microphone..."
                : recordState === "converting"
                  ? "Resampling to 16kHz PCM WAV..."
                  : isLoading
                    ? "Submitting Recording..."
                    : "Click to Speak"}
          </p>
          <p className="text-[11px] text-muted-foreground mt-0.5">
            Auto-converts to 16kHz 16-bit Mono PCM WAV
          </p>

          {isLoading && onCancel && (
            <button
              type="button"
              onClick={onCancel}
              className="mt-3 inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg border border-border bg-card hover:bg-destructive/10 hover:text-destructive hover:border-destructive/30 text-xs font-medium text-muted-foreground transition-all cursor-pointer shadow-2xs"
            >
              <XCircle className="w-3.5 h-3.5" />
              <span>Stop Analysis Request</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
