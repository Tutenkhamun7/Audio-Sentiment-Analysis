import React, { useRef, useEffect, useState, useMemo, useCallback } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  Volume2,
  VolumeX,
  Radio,
  Quote,
  Zap,
  Rewind,
  FastForward,
  AlertCircle,
} from "lucide-react";
import type { TimelineSegment } from "@/types/emotion";
import { formatTime, getEmotionMeta, getSpeakerTheme } from "@/lib/constants";

interface AudioPlayerHUDProps {
  audioBlob: Blob;
  segments: TimelineSegment[];
  seekTime: number | null;
  stopAtTime?: number | null;
  onSeekHandled: () => void;
  onCurrentTimeUpdate?: (time: number) => void;
  activeSegmentId: number | null;
  setActiveSegmentId: (id: number | null) => void;
}

const PLAYBACK_SPEEDS = [1, 1.25, 1.5, 2];

export const AudioPlayerHUD: React.FC<AudioPlayerHUDProps> = ({
  audioBlob,
  segments,
  seekTime,
  stopAtTime,
  onSeekHandled,
  onCurrentTimeUpdate,
  activeSegmentId,
  setActiveSegmentId,
}) => {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const trackRef = useRef<HTMLDivElement | null>(null);

  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(() => {
    if (segments.length > 0) {
      return segments[segments.length - 1].end_time;
    }
    return 0;
  });
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [volume, setVolume] = useState<number>(1);
  const [playbackError, setPlaybackError] = useState<string | null>(null);

  // Dragging & Hover state on custom timeline
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [hoverTime, setHoverTime] = useState<number | null>(null);
  const [hoverPosition, setHoverPosition] = useState<number | null>(null);

  const playUntilTimeRef = useRef<number | null>(null);

  // Derive total duration accurately
  const totalDuration = useMemo(() => {
    if (duration > 0 && isFinite(duration)) return duration;
    if (segments.length > 0) return Math.max(segments[segments.length - 1].end_time, 0.1);
    return 1;
  }, [duration, segments]);

  // Directly attach the audio Blob URL to the native audio element
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !audioBlob) return;

    setPlaybackError(null);
    const url = URL.createObjectURL(audioBlob);
    audio.src = url;
    audio.load();

    return () => {
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
      URL.revokeObjectURL(url);
    };
  }, [audioBlob]);

  // High-frequency smooth 60fps playhead tracking using requestAnimationFrame
  useEffect(() => {
    if (!isPlaying) return;

    let animId: number;
    const tick = () => {
      const audio = audioRef.current;
      if (audio && !isDragging) {
        const t = audio.currentTime;
        setCurrentTime(t);
        if (onCurrentTimeUpdate) {
          onCurrentTimeUpdate(t);
        }

        // Stop if isolated segment playback reached boundary
        if (playUntilTimeRef.current !== null && t >= playUntilTimeRef.current) {
          audio.pause();
          playUntilTimeRef.current = null;
          setIsPlaying(false);
        }
      }
      animId = requestAnimationFrame(tick);
    };

    animId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animId);
  }, [isPlaying, isDragging, onCurrentTimeUpdate]);

  // Handle external seek requests (from Transcript or Segment play button)
  useEffect(() => {
    const audio = audioRef.current;
    if (seekTime !== null && audio) {
      const targetTime = Math.max(0, seekTime);
      playUntilTimeRef.current = typeof stopAtTime === "number" ? stopAtTime : null;

      const performSeekAndPlay = async () => {
        try {
          audio.currentTime = targetTime;
          setCurrentTime(targetTime);
          await audio.play();
          setIsPlaying(true);
          setPlaybackError(null);
        } catch (err: unknown) {
          const errObj = err as Error;
          console.warn("Seek & play playback error:", errObj);
          if (errObj.name === "NotAllowedError") {
            setPlaybackError(
              "Autoplay blocked by browser. Click Play to start audio."
            );
          }
        }
      };

      if (audio.readyState >= 1) {
        performSeekAndPlay();
      } else {
        const onLoaded = () => {
          performSeekAndPlay();
          audio.removeEventListener("loadedmetadata", onLoaded);
        };
        audio.addEventListener("loadedmetadata", onLoaded);
      }

      onSeekHandled();
    }
  }, [seekTime, stopAtTime, onSeekHandled]);

  // Sync active segment with audio playback
  useEffect(() => {
    const current = segments.find(
      (s) => currentTime >= s.start_time && currentTime <= s.end_time
    );

    if (current) {
      if (activeSegmentId !== current.segment_id) {
        setActiveSegmentId(current.segment_id);
      }
    } else {
      if (activeSegmentId !== null) {
        setActiveSegmentId(null);
      }
    }
  }, [currentTime, segments, activeSegmentId, setActiveSegmentId]);

  const activeSegment = useMemo(() => {
    if (activeSegmentId !== null) {
      const found = segments.find((s) => s.segment_id === activeSegmentId);
      if (found) return found;
    }
    return (
      segments.find((s) => currentTime >= s.start_time && currentTime <= s.end_time) || null
    );
  }, [activeSegmentId, currentTime, segments]);

  const updateDuration = () => {
    const audio = audioRef.current;
    if (audio) {
      const d = audio.duration;
      if (!isNaN(d) && isFinite(d) && d > 0) {
        setDuration(d);
      } else if (segments.length > 0) {
        setDuration(segments[segments.length - 1].end_time);
      }
    }
  };

  const togglePlayPause = async () => {
    const audio = audioRef.current;
    if (!audio) return;

    setPlaybackError(null);

    if (isPlaying) {
      audio.pause();
      playUntilTimeRef.current = null;
      setIsPlaying(false);
    } else {
      playUntilTimeRef.current = null; // Continuous play
      try {
        await audio.play();
        setIsPlaying(true);
      } catch (err: unknown) {
        const errObj = err as Error;
        console.error("Audio play failed:", errObj);
        setPlaybackError(
          errObj.name === "NotAllowedError"
            ? "Browser blocked autoplay. Click again to play."
            : `Playback error: ${errObj.message || "Failed to decode audio stream."}`
        );
      }
    }
  };

  const handleReplay = async () => {
    const audio = audioRef.current;
    if (!audio) return;

    setPlaybackError(null);
    playUntilTimeRef.current = null;
    audio.currentTime = 0;
    setCurrentTime(0);

    try {
      await audio.play();
      setIsPlaying(true);
    } catch (err: unknown) {
      console.error("Audio replay failed:", err);
    }
  };

  const handleSkip = (seconds: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    const newTime = Math.min(Math.max(audio.currentTime + seconds, 0), totalDuration);
    audio.currentTime = newTime;
    setCurrentTime(newTime);
  };

  // Timeline track seek calculations
  const calculateTimeFromPointer = useCallback(
    (clientX: number): number => {
      const track = trackRef.current;
      if (!track) return 0;
      const rect = track.getBoundingClientRect();
      const clickX = Math.max(0, Math.min(clientX - rect.left, rect.width));
      const percentage = clickX / rect.width;
      return percentage * totalDuration;
    },
    [totalDuration]
  );

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    setIsDragging(true);
    playUntilTimeRef.current = null;

    const newTime = calculateTimeFromPointer(e.clientX);
    setCurrentTime(newTime);
    if (audioRef.current) {
      audioRef.current.currentTime = newTime;
    }
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const track = trackRef.current;
    if (!track) return;
    const rect = track.getBoundingClientRect();
    const pos = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const time = (pos / rect.width) * totalDuration;

    setHoverPosition(pos);
    setHoverTime(time);

    if (isDragging) {
      const newTime = calculateTimeFromPointer(e.clientX);
      setCurrentTime(newTime);
      if (audioRef.current) {
        audioRef.current.currentTime = newTime;
      }
    }
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (isDragging) {
      try {
        e.currentTarget.releasePointerCapture(e.pointerId);
      } catch {
        // ignore
      }
      setIsDragging(false);
      const newTime = calculateTimeFromPointer(e.clientX);
      if (audioRef.current) {
        audioRef.current.currentTime = newTime;
      }
    }
  };

  const handlePointerLeave = () => {
    if (!isDragging) {
      setHoverTime(null);
      setHoverPosition(null);
    }
  };

  const cyclePlaybackRate = () => {
    const audio = audioRef.current;
    if (!audio) return;
    const nextIdx = (PLAYBACK_SPEEDS.indexOf(playbackSpeed) + 1) % PLAYBACK_SPEEDS.length;
    const nextSpeed = PLAYBACK_SPEEDS[nextIdx];
    audio.playbackRate = nextSpeed;
    setPlaybackSpeed(nextSpeed);
  };

  const toggleMute = () => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setVolume(val);
    const audio = audioRef.current;
    if (audio) {
      audio.volume = val;
      if (val === 0) {
        setIsMuted(true);
      } else if (isMuted) {
        setIsMuted(false);
      }
    }
  };

  // Top emotions for HUD
  const topAcoustic = activeSegment?.acoustic_emotion?.[0];
  const topSemantic = activeSegment?.semantic_emotion?.[0];
  const acousticMeta = topAcoustic ? getEmotionMeta(topAcoustic.label) : null;
  const semanticMeta = topSemantic ? getEmotionMeta(topSemantic.label) : null;
  const speakerTheme = activeSegment ? getSpeakerTheme(activeSegment.speaker) : null;

  const progressPercent = Math.min(Math.max((currentTime / totalDuration) * 100, 0), 100);

  return (
    <div className="w-full flex flex-col gap-4 p-5 sm:p-6 rounded-2xl border border-border/80 bg-card shadow-2xs">
      {/* Native HTML5 Audio Element */}
      <audio
        ref={audioRef}
        preload="auto"
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onTimeUpdate={() => {
          if (!isPlaying && audioRef.current) {
            setCurrentTime(audioRef.current.currentTime);
          }
        }}
        onLoadedMetadata={updateDuration}
        onLoadedData={updateDuration}
        onDurationChange={updateDuration}
        onEnded={() => {
          setIsPlaying(false);
          setCurrentTime(0);
          playUntilTimeRef.current = null;
        }}
        onError={(e) => {
          const mediaErr = audioRef.current?.error;
          console.warn("Audio element error:", mediaErr, e);
          if (mediaErr) {
            setPlaybackError(
              `Audio engine could not decode this audio stream (Code ${mediaErr.code}: ${mediaErr.message || "media source format issue"}).`
            );
          }
        }}
      />

      {/* Playback Error Warning Banner if needed */}
      {playbackError && (
        <div className="flex items-center gap-2 p-3 rounded-xl bg-destructive/10 border border-destructive/20 text-destructive text-xs">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{playbackError}</span>
        </div>
      )}

      {/* Interactive Timeline & Scrubber Bar */}
      <div className="flex flex-col gap-2">
        {/* Scrubber Area */}
        <div className="relative w-full">
          {/* Timeline Track Container */}
          <div
            ref={trackRef}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={handlePointerLeave}
            className="group relative h-9 w-full rounded-xl bg-muted/60 border border-border/70 cursor-pointer overflow-hidden select-none transition-all hover:border-foreground/30"
          >
            {/* Background Segment Regions (shows conversation turns visually) */}
            <div className="absolute inset-0 flex w-full h-full pointer-events-none">
              {segments.map((seg) => {
                const segStart = (seg.start_time / totalDuration) * 100;
                const segWidth = ((seg.end_time - seg.start_time) / totalDuration) * 100;
                const theme = getSpeakerTheme(seg.speaker);
                const isCur = activeSegmentId === seg.segment_id;

                return (
                  <div
                    key={seg.segment_id}
                    style={{
                      left: `${segStart}%`,
                      width: `${segWidth}%`,
                    }}
                    className={`absolute top-0 bottom-0 border-r border-border/50 transition-colors ${
                      isCur
                        ? "bg-foreground/10 dark:bg-foreground/15"
                        : "hover:bg-foreground/5"
                    }`}
                    title={`#${seg.segment_id} ${seg.speaker} (${formatTime(seg.start_time, true)} - ${formatTime(seg.end_time, true)}): "${seg.text.slice(0, 40)}..."`}
                  >
                    {/* Small speaker color pip on segment start */}
                    <span
                      className={`absolute top-1 left-1 w-1.5 h-1.5 rounded-full ${theme.dot} opacity-70`}
                    />
                  </div>
                );
              })}
            </div>

            {/* Filled Progress Bar */}
            <div
              style={{ width: `${progressPercent}%` }}
              className="absolute top-0 bottom-0 left-0 bg-foreground/20 dark:bg-foreground/25 pointer-events-none transition-[width] duration-75"
            />

            {/* Crisp Needle / Playhead */}
            <div
              style={{ left: `${progressPercent}%` }}
              className="absolute top-0 bottom-0 -ml-[1px] w-[2px] bg-foreground shadow-xs pointer-events-none transition-[left] duration-75 z-10"
            >
              {/* Playhead Handle Dot */}
              <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 left-[1px] w-3 h-3 rounded-full bg-foreground border-2 border-background shadow-md scale-90 group-hover:scale-110 transition-transform" />
            </div>

            {/* Hover Guideline & Timestamp Tooltip */}
            {hoverPosition !== null && hoverTime !== null && (
              <div
                style={{ left: `${hoverPosition}px` }}
                className="absolute top-0 bottom-0 -ml-[0.5px] w-[1px] bg-foreground/50 border-r border-dashed border-foreground/70 pointer-events-none z-20"
              >
                <div className="absolute -top-7 -translate-x-1/2 px-1.5 py-0.5 rounded bg-foreground text-background text-[10px] font-mono font-medium shadow-sm whitespace-nowrap">
                  {formatTime(hoverTime, true)}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Timestamps Row */}
        <div className="flex items-center justify-between text-xs font-mono text-muted-foreground px-1">
          <div className="flex items-center gap-1.5">
            <span className="font-semibold text-foreground">
              {formatTime(currentTime, true)}
            </span>
            <span>/</span>
            <span>{formatTime(totalDuration, true)}</span>
          </div>

          <span className="text-[11px] font-sans text-muted-foreground/60">
            Click or drag timeline to seek
          </span>
        </div>
      </div>

      {/* Main Playback & Volume Control Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-1 border-t border-border/60">
        {/* Playback Controls */}
        <div className="flex items-center gap-1.5 sm:gap-2">
          {/* Skip Back 5s */}
          <button
            type="button"
            onClick={() => handleSkip(-5)}
            className="p-2 rounded-lg border border-border/70 bg-card hover:bg-muted text-muted-foreground hover:text-foreground transition-colors active:scale-95 cursor-pointer"
            title="Rewind 5 seconds"
          >
            <Rewind className="w-4 h-4" />
          </button>

          {/* Primary Play / Pause Button */}
          <button
            type="button"
            onClick={togglePlayPause}
            className="w-11 h-11 rounded-full bg-foreground text-background hover:opacity-90 active:scale-95 transition-all flex items-center justify-center cursor-pointer shadow-xs"
            title={isPlaying ? "Pause (Space)" : "Play (Space)"}
          >
            {isPlaying ? (
              <Pause className="w-5 h-5 fill-current" />
            ) : (
              <Play className="w-5 h-5 ml-0.5 fill-current" />
            )}
          </button>

          {/* Skip Forward 5s */}
          <button
            type="button"
            onClick={() => handleSkip(5)}
            className="p-2 rounded-lg border border-border/70 bg-card hover:bg-muted text-muted-foreground hover:text-foreground transition-colors active:scale-95 cursor-pointer"
            title="Fast forward 5 seconds"
          >
            <FastForward className="w-4 h-4" />
          </button>

          {/* Reset / Replay */}
          <button
            type="button"
            onClick={handleReplay}
            className="p-2 rounded-lg border border-border/70 bg-card hover:bg-muted text-muted-foreground hover:text-foreground transition-colors active:scale-95 cursor-pointer"
            title="Restart from beginning"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          {/* Playback Speed Pill */}
          <button
            type="button"
            onClick={cyclePlaybackRate}
            className="px-2.5 py-1.5 rounded-lg border border-border/70 bg-card hover:bg-muted text-xs font-mono font-medium text-foreground transition-colors active:scale-95 cursor-pointer"
            title="Change playback speed"
          >
            {playbackSpeed}x
          </button>
        </div>

        {/* Volume & Mute Controls */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={toggleMute}
            className="p-2 rounded-lg border border-border/70 bg-card hover:bg-muted text-muted-foreground hover:text-foreground transition-colors active:scale-95 cursor-pointer"
            title={isMuted ? "Unmute" : "Mute"}
          >
            {isMuted || volume === 0 ? (
              <VolumeX className="w-4 h-4 text-muted-foreground" />
            ) : (
              <Volume2 className="w-4 h-4" />
            )}
          </button>

          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={isMuted ? 0 : volume}
            onChange={handleVolumeChange}
            className="w-18 sm:w-24 h-1.5 bg-muted rounded-lg appearance-none cursor-pointer accent-foreground focus:outline-none"
            title={`Volume: ${Math.round((isMuted ? 0 : volume) * 100)}%`}
          />
        </div>
      </div>

      {/* Active Segment Live HUD */}
      <div className="pt-3 border-t border-border/60">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2 text-[11px] font-medium tracking-wider uppercase text-muted-foreground">
            <Radio className="w-3 h-3 text-muted-foreground" />
            <span>Active Segment Live HUD</span>
          </div>
          {activeSegment && (
            <span className="text-[11px] font-mono text-muted-foreground">
              Segment #{activeSegment.segment_id} ({formatTime(activeSegment.start_time, true)} →{" "}
              {formatTime(activeSegment.end_time, true)})
            </span>
          )}
        </div>

        {activeSegment ? (
          <div className="p-3.5 rounded-xl border border-border/70 bg-muted/20 flex flex-col gap-2.5 transition-colors">
            {/* Live Badges: Speaker + Acoustic + Semantic */}
            <div className="flex flex-wrap items-center gap-2">
              {/* Speaker pill */}
              <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-medium bg-muted text-foreground border border-border/70">
                <span className={`w-1.5 h-1.5 rounded-full ${speakerTheme?.dot}`} />
                <span>{activeSegment.speaker}</span>
              </div>

              {/* Acoustic Emotion Badge */}
              {acousticMeta && topAcoustic && (
                <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-medium bg-muted/70 text-foreground/90 border border-border/60">
                  <Zap className="w-3 h-3 text-muted-foreground" />
                  <span>
                    Tone: {acousticMeta.emoji} {acousticMeta.displayName} (
                    {Math.round(topAcoustic.score * 100)}%)
                  </span>
                </div>
              )}

              {/* Semantic Emotion Badge */}
              {semanticMeta && topSemantic && (
                <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-medium bg-muted/70 text-foreground/90 border border-border/60">
                  <span>
                    Text: {semanticMeta.emoji} {semanticMeta.displayName} (
                    {Math.round(topSemantic.score * 100)}%)
                  </span>
                </div>
              )}

              {/* Incongruence Conflict warning pill if conflict exists */}
              {activeSegment.is_conflict && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium bg-rose-500/10 text-rose-700 dark:text-rose-300 border border-rose-500/20">
                  ⚠️ Conflict Detected
                </span>
              )}
            </div>

            {/* Live Quote Text */}
            <div className="flex items-start gap-2.5 pt-0.5">
              <Quote className="w-3.5 h-3.5 text-muted-foreground/60 shrink-0 mt-0.5 rotate-180" />
              <p className="text-xs sm:text-sm font-normal text-foreground/90 italic leading-relaxed">
                &ldquo;{activeSegment.text}&rdquo;
              </p>
            </div>
          </div>
        ) : (
          <div className="p-3.5 rounded-xl border border-dashed border-border/50 bg-muted/10 text-center py-4">
            <p className="text-xs text-muted-foreground">
              {isPlaying
                ? "Inter-segment pause or ambient transition..."
                : "Press Play or click a transcript sentence below to listen."}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default AudioPlayerHUD;
