import React, { useState, useRef } from "react";
import type { DragEvent, ChangeEvent } from "react";
import { UploadCloud, FileAudio, X, Sparkles, Loader2, AlertCircle, XCircle } from "lucide-react";

interface UploadSectionProps {
  onAnalyze: (file: File) => void;
  isLoading: boolean;
  selectedFile: File | null;
  setSelectedFile: (file: File | null) => void;
  onCancel?: () => void;
}

const SUPPORTED_EXTENSIONS = [".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".flac", ".webm"];
const SUPPORTED_MIME_TYPES = [
  "audio/wav",
  "audio/x-wav",
  "audio/mp3",
  "audio/mpeg",
  "audio/mp4",
  "audio/m4a",
  "audio/x-m4a",
  "audio/ogg",
  "audio/flac",
  "audio/webm",
  "video/webm",
  "video/mp4",
];

export const UploadSection: React.FC<UploadSectionProps> = ({
  onAnalyze,
  isLoading,
  selectedFile,
  setSelectedFile,
  onCancel,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndSetFile = (file: File) => {
    setErrorMsg(null);
    const extension = "." + (file.name.split(".").pop()?.toLowerCase() || "");
    const isExtensionValid = SUPPORTED_EXTENSIONS.includes(extension);
    const isMimeValid = SUPPORTED_MIME_TYPES.some((type) => file.type.startsWith(type.split("/")[0]));

    if (!isExtensionValid && !isMimeValid) {
      setErrorMsg(
        `Unsupported audio format (${extension || "unknown"}). Please upload a WAV, MP3, MP4, M4A, OGG, FLAC, or WebM file.`
      );
      return;
    }

    if (file.size > 100 * 1024 * 1024) {
      setErrorMsg("File size exceeds 100MB limit. Please upload a smaller audio clip.");
      return;
    }

    setSelectedFile(file);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const clearFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedFile(null);
    setErrorMsg(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const formatFileSize = (bytes: number): string => {
    const mb = bytes / (1024 * 1024);
    if (mb < 0.1) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${mb.toFixed(2)} MB`;
  };

  return (
    <div className="w-full flex flex-col gap-5">
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept={SUPPORTED_EXTENSIONS.join(",")}
        onChange={handleFileInputChange}
        className="hidden"
        disabled={isLoading}
      />

      {/* Drag and Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isLoading && fileInputRef.current?.click()}
        className={`relative group cursor-pointer rounded-2xl border border-dashed p-8 md:p-12 transition-all duration-200 flex flex-col items-center justify-center text-center ${
          isDragOver
            ? "border-foreground/40 bg-muted/40 shadow-xs"
            : selectedFile
              ? "border-border bg-card shadow-2xs"
              : "border-border/70 hover:border-border bg-card/60 hover:bg-card shadow-2xs"
        }`}
      >
        {!selectedFile ? (
          <div className="flex flex-col items-center gap-3.5 relative z-10">
            <div
              className={`p-3.5 rounded-xl bg-muted text-muted-foreground transition-transform duration-200 group-hover:scale-105 ${
                isDragOver ? "scale-110 text-foreground" : ""
              }`}
            >
              <UploadCloud className="w-8 h-8" />
            </div>

            <div className="space-y-1">
              <h3 className="text-sm sm:text-base font-semibold text-foreground">
                Drop audio file here, or{" "}
                <span className="underline decoration-muted-foreground/40 underline-offset-4 text-foreground">
                  browse
                </span>
              </h3>
              <p className="text-xs text-muted-foreground max-w-sm">
                Speech audio files up to 100MB for multimodal sentiment & emotion breakdown.
              </p>
            </div>

            {/* Supported extensions chips */}
            <div className="flex flex-wrap items-center justify-center gap-1 mt-1">
              {SUPPORTED_EXTENSIONS.map((ext) => (
                <span
                  key={ext}
                  className="px-2 py-0.5 rounded bg-muted/60 text-[10px] font-mono text-muted-foreground uppercase border border-border/40"
                >
                  {ext.replace(".", "")}
                </span>
              ))}
            </div>
          </div>
        ) : (
          /* File Selected Preview Area */
          <div className="w-full max-w-md flex flex-col items-center gap-3 relative z-10">
            <div className="w-full flex items-center justify-between p-3.5 rounded-xl border border-border/70 bg-card shadow-2xs">
              <div className="flex items-center gap-3 min-w-0">
                <div className="p-2.5 rounded-lg bg-muted text-muted-foreground shrink-0">
                  <FileAudio className="w-5 h-5 text-foreground/80" />
                </div>
                <div className="flex flex-col text-left min-w-0">
                  <span className="font-medium text-xs sm:text-sm text-foreground truncate max-w-[220px]">
                    {selectedFile.name}
                  </span>
                  <div className="flex items-center gap-1.5 text-xs text-muted-foreground mt-0.5">
                    <span>{formatFileSize(selectedFile.size)}</span>
                    <span>•</span>
                    <span className="uppercase font-mono text-[10px] px-1 py-0.2 rounded bg-muted/80">
                      {selectedFile.name.split(".").pop() || "AUDIO"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Clear button */}
              <button
                type="button"
                onClick={clearFile}
                disabled={isLoading}
                className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors cursor-pointer"
                title="Remove file"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <p className="text-xs text-muted-foreground">
              Click to replace with a different audio file
            </p>
          </div>
        )}
      </div>

      {/* Error alert if validation failed */}
      {errorMsg && (
        <div className="flex items-center gap-2.5 p-3 rounded-xl bg-destructive/10 border border-destructive/20 text-destructive text-xs">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Action Button: Run Audio Sentiment Analysis & Stop Option */}
      <div className="flex flex-col sm:flex-row items-center justify-end gap-3">
        {isLoading && onCancel && (
          <button
            type="button"
            onClick={onCancel}
            className="w-full sm:w-auto px-4 py-2.5 rounded-xl font-medium text-xs sm:text-sm border border-border bg-card hover:bg-destructive/10 hover:text-destructive hover:border-destructive/30 text-muted-foreground transition-all flex items-center justify-center gap-1.5 cursor-pointer shadow-2xs"
          >
            <XCircle className="w-4 h-4" />
            <span>Stop Request</span>
          </button>
        )}

        <button
          type="button"
          disabled={!selectedFile || isLoading}
          onClick={() => selectedFile && onAnalyze(selectedFile)}
          className={`w-full sm:w-auto min-w-[220px] px-5 py-2.5 rounded-xl font-medium text-xs sm:text-sm shadow-xs transition-all duration-200 flex items-center justify-center gap-2 cursor-pointer ${
            !selectedFile || isLoading
              ? "bg-muted text-muted-foreground cursor-not-allowed opacity-70"
              : "bg-foreground text-background hover:opacity-90 active:scale-[0.99]"
          }`}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Analyzing Sentiment...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              <span>Analyze Audio Sentiment</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};
