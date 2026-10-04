import { useEffect, useRef } from "react";
import { useTheme } from "@/components/theme-provider";

interface WaveformVisualizerProps {
  analyserNode: AnalyserNode | null;
  isRecording: boolean;
  className?: string;
  height?: number;
  durationText?: string;
}

export function WaveformVisualizer({
  analyserNode,
  isRecording,
  className = "",
  height = 160,
  durationText,
}: WaveformVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameIdRef = useRef<number | null>(null);
  const idlePhaseRef = useRef<number>(0);
  const { theme } = useTheme();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let dataArray: Uint8Array<ArrayBuffer> | null = null;
    if (analyserNode) {
      dataArray = new Uint8Array(new ArrayBuffer(analyserNode.fftSize));
    }

    const render = () => {
      // Handle high-DPI
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = rect.width;
      const h = height;

      if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(h * dpr)) {
        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(h * dpr);
      }

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, width, h);

      const isDark =
        theme === "dark" ||
        (theme === "system" &&
          typeof window !== "undefined" &&
          window.matchMedia("(prefers-color-scheme: dark)").matches);

      // Background subtle grid/lines
      const centerY = h / 2;

      ctx.strokeStyle = isDark ? "rgba(255, 255, 255, 0.05)" : "rgba(255, 255, 255, 0.08)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(width, centerY);
      ctx.stroke();

      // Primary color scheme based on state
      const glowColor = isRecording
        ? isDark
          ? "rgba(239, 68, 68, 0.85)" // Vibrant Red
          : "rgba(220, 38, 38, 0.8)"
        : isDark
          ? "rgba(16, 185, 129, 0.75)" // Emerald
          : "rgba(13, 148, 136, 0.7)";

      const coreColor = isRecording
        ? "#FCA5A5"
        : isDark
          ? "#A7F3D0"
          : "#5EEAD4";

      const ambientColor = isRecording
        ? "rgba(239, 68, 68, 0.25)"
        : isDark
          ? "rgba(16, 185, 129, 0.2)"
          : "rgba(13, 148, 136, 0.15)";

      if (analyserNode && isRecording && dataArray) {
        analyserNode.getByteTimeDomainData(dataArray);

        const bufferLength = analyserNode.fftSize;
        const sliceWidth = width / (bufferLength / 4); // downsample slightly for smoother curve

        // PASS 1: Broad ambient glow
        ctx.save();
        ctx.strokeStyle = ambientColor;
        ctx.lineWidth = 8;
        ctx.shadowBlur = 24;
        ctx.shadowColor = glowColor;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.beginPath();

        let x = 0;
        for (let i = 0; i < bufferLength; i += 4) {
          const v = dataArray[i] / 128.0;
          const y = (v * h) / 2;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
          x += sliceWidth;
        }
        ctx.stroke();
        ctx.restore();

        // PASS 2: Medium glow pass
        ctx.save();
        ctx.strokeStyle = glowColor;
        ctx.lineWidth = 3;
        ctx.shadowBlur = 12;
        ctx.shadowColor = glowColor;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.beginPath();

        x = 0;
        for (let i = 0; i < bufferLength; i += 4) {
          const v = dataArray[i] / 128.0;
          const y = (v * h) / 2;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
          x += sliceWidth;
        }
        ctx.stroke();
        ctx.restore();

        // PASS 3: Crisp core line
        ctx.save();
        ctx.strokeStyle = coreColor;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        x = 0;
        for (let i = 0; i < bufferLength; i += 4) {
          const v = dataArray[i] / 128.0;
          const y = (v * h) / 2;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
          x += sliceWidth;
        }
        ctx.stroke();
        ctx.restore();
      } else {
        // IDLE STATE: Gentle breathing baseline wave
        idlePhaseRef.current += 0.03;
        const phase = idlePhaseRef.current;

        // PASS 1: Soft glow
        ctx.save();
        ctx.strokeStyle = isDark ? "rgba(16, 185, 129, 0.35)" : "rgba(13, 148, 136, 0.35)";
        ctx.lineWidth = 4;
        ctx.shadowBlur = 14;
        ctx.shadowColor = isDark ? "#10B981" : "#0D9488";
        ctx.beginPath();

        const step = 6;
        for (let x = 0; x <= width; x += step) {
          const normX = x / width;
          // Dampen amplitude at left and right edges with windowing function
          const windowMult = Math.sin(normX * Math.PI);
          const y =
            centerY +
            Math.sin(normX * 8 + phase) * 6 * windowMult +
            Math.sin(normX * 16 - phase * 0.7) * 3 * windowMult;

          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.restore();

        // PASS 2: Center core
        ctx.save();
        ctx.strokeStyle = isDark ? "#6EE7B7" : "#5EEAD4";
        ctx.lineWidth = 1.5;
        ctx.beginPath();

        for (let x = 0; x <= width; x += step) {
          const normX = x / width;
          const windowMult = Math.sin(normX * Math.PI);
          const y =
            centerY +
            Math.sin(normX * 8 + phase) * 6 * windowMult +
            Math.sin(normX * 16 - phase * 0.7) * 3 * windowMult;

          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.restore();
      }

      ctx.restore();
      animFrameIdRef.current = requestAnimationFrame(render);
    };

    animFrameIdRef.current = requestAnimationFrame(render);

    return () => {
      if (animFrameIdRef.current !== null) {
        cancelAnimationFrame(animFrameIdRef.current);
      }
    };
  }, [analyserNode, isRecording, theme, height]);

  return (
    <div
      className={`relative w-full overflow-hidden rounded-2xl bg-slate-950 backdrop-blur-md border border-slate-800/80 shadow-inner ${className}`}
    >
      <canvas
        ref={canvasRef}
        className="w-full block"
        style={{ height: `${height}px` }}
      />
      {/* Live recording indicator badge overlay */}
      <div className="absolute top-3 left-4 flex items-center gap-2 pointer-events-none select-none">
        <span
          className={`h-2 w-2 rounded-full ${
            isRecording
              ? "bg-red-500 animate-ping"
              : "bg-emerald-500/80 animate-pulse"
          }`}
        />
        <span className="font-mono text-xs tracking-wider text-slate-300">
          {isRecording ? "Live Audio Stream (16kHz Analyser)" : "Oscilloscope Standby"}
        </span>
      </div>
      {isRecording && (
        <div className="absolute top-3 right-4 flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-red-500/20 border border-red-500/40 text-red-400 font-mono text-xs animate-pulse select-none">
          <span className="h-1.5 w-1.5 rounded-full bg-red-500" />
          <span>{durationText ? `REC ${durationText}` : "REC"}</span>
        </div>
      )}
    </div>
  );
}

export default WaveformVisualizer;
