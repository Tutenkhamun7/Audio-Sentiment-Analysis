import React from "react";
import { Waves, Moon, Sun, Sparkles, Activity } from "lucide-react";
import { useTheme } from "@/components/theme-provider";

interface HeaderProps {
  onLoadDemo?: () => void;
  hasResults?: boolean;
  onReset?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  onLoadDemo,
  hasResults,
  onReset,
}) => {
  const { theme, setTheme } = useTheme();

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark");
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/60 bg-background/85 backdrop-blur-md transition-colors duration-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-15 flex items-center justify-between">
        {/* Brand Logo & Title */}
        <div
          onClick={onReset}
          className="flex items-center gap-3 cursor-pointer group select-none"
        >
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-foreground text-background shadow-xs group-hover:scale-102 transition-transform duration-200">
            <Waves className="w-4 h-4" />
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-base tracking-tight text-foreground">
                AudioSentiment AI
              </span>
              <span className="hidden sm:inline-flex items-center px-1.5 py-0.2 rounded text-[10px] font-mono font-medium bg-muted text-muted-foreground border border-border/60">
                v1.0
              </span>
            </div>
            <span className="text-[11px] text-muted-foreground font-normal -mt-0.5">
              Multimodal Speech Emotion Studio
            </span>
          </div>
        </div>

        {/* Right side controls: Live status, Demo button, and Theme toggle */}
        <div className="flex items-center gap-2.5">
          {/* Subtle Live Studio Status Pill */}
          <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-muted/60 border border-border/60 text-muted-foreground text-xs font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500/80" />
            <span>Studio Engine Active</span>
          </div>

          {/* Quick Demo Trigger */}
          {onLoadDemo && (
            <button
              onClick={onLoadDemo}
              type="button"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-foreground bg-secondary hover:bg-secondary/80 border border-border/70 transition-colors"
              title="Load demo audio dialogue sample"
            >
              <Sparkles className="w-3.5 h-3.5 text-muted-foreground" />
              <span>Load Sample</span>
            </button>
          )}

          {/* Reset / New Analysis button if results present */}
          {hasResults && onReset && (
            <button
              onClick={onReset}
              type="button"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-muted-foreground hover:text-foreground bg-muted hover:bg-muted/80 transition-colors"
            >
              <Activity className="w-3.5 h-3.5" />
              <span>New Audio</span>
            </button>
          )}

          {/* Theme Toggle Button */}
          <button
            onClick={toggleTheme}
            type="button"
            aria-label="Toggle theme"
            className="p-2 rounded-lg border border-border/60 bg-card hover:bg-accent hover:text-accent-foreground text-muted-foreground hover:text-foreground transition-colors shadow-2xs"
          >
            <div className="relative w-4 h-4 flex items-center justify-center">
              <Sun
                className={`w-4 h-4 transition-all duration-200 transform ${
                  theme === "dark"
                    ? "rotate-90 scale-0 opacity-0"
                    : "rotate-0 scale-100 opacity-100"
                }`}
              />
              <Moon
                className={`w-4 h-4 absolute transition-all duration-200 transform ${
                  theme === "dark"
                    ? "rotate-0 scale-100 opacity-100"
                    : "-rotate-90 scale-0 opacity-0"
                }`}
              />
            </div>
          </button>
        </div>
      </div>
    </header>
  );
};
