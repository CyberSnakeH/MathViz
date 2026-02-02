/**
 * ManimPreview Component
 *
 * Live preview panel for Manim animations with playback controls,
 * scrubber timeline, quality settings, and export functionality.
 */

import React, { useCallback, useRef, useEffect, useState, memo } from 'react';
import { cn, formatDuration } from '../../utils/helpers';
import {
  usePreviewStore,
  selectIsPreviewReady,
  selectPreviewProgress,
  selectCanPlay,
} from '../../stores/previewStore';
import type { PreviewQuality } from '../../types';

// ============================================================================
// Types
// ============================================================================

interface ManimPreviewProps {
  className?: string;
}

// ============================================================================
// Quality Options
// ============================================================================

const qualityOptions: { value: PreviewQuality; label: string; description: string }[] = [
  { value: 'low', label: 'Low', description: '480p, fast preview' },
  { value: 'medium', label: 'Medium', description: '720p, balanced' },
  { value: 'high', label: 'High', description: '1080p, detailed' },
  { value: 'production', label: 'Production', description: '4K, final render' },
];

// ============================================================================
// Playback Controls Component
// ============================================================================

interface PlaybackControlsProps {
  isPlaying: boolean;
  canPlay: boolean;
  playbackRate: number;
  isLooping: boolean;
  onPlay: () => void;
  onPause: () => void;
  onStop: () => void;
  onSkipBackward: () => void;
  onSkipForward: () => void;
  onSetPlaybackRate: (rate: number) => void;
  onToggleLoop: () => void;
}

const PlaybackControls: React.FC<PlaybackControlsProps> = memo(({
  isPlaying,
  canPlay,
  playbackRate,
  isLooping,
  onPlay,
  onPause,
  onStop,
  onSkipBackward,
  onSkipForward,
  onSetPlaybackRate,
  onToggleLoop,
}) => {
  const [showSpeedMenu, setShowSpeedMenu] = useState(false);

  const speedOptions = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 4];

  return (
    <div className="flex items-center gap-1">
      {/* Skip backward */}
      <button
        className={cn(
          'p-1.5 rounded hover:bg-[var(--mviz-list-hover-background)]',
          !canPlay && 'opacity-50 cursor-not-allowed'
        )}
        onClick={onSkipBackward}
        disabled={!canPlay}
        title="Skip backward 5s"
      >
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
          <path d="M6 6h2v12H6zm3.5 6l8.5 6V6z" />
        </svg>
      </button>

      {/* Play/Pause */}
      <button
        className={cn(
          'p-2 rounded-full bg-[var(--mviz-accent)] text-white',
          'hover:bg-[var(--mviz-button-hover-background)]',
          !canPlay && 'opacity-50 cursor-not-allowed'
        )}
        onClick={isPlaying ? onPause : onPlay}
        disabled={!canPlay}
        title={isPlaying ? 'Pause' : 'Play'}
      >
        {isPlaying ? (
          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
            <path d="M6 4h4v16H6zm8 0h4v16h-4z" />
          </svg>
        ) : (
          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z" />
          </svg>
        )}
      </button>

      {/* Stop */}
      <button
        className={cn(
          'p-1.5 rounded hover:bg-[var(--mviz-list-hover-background)]',
          !canPlay && 'opacity-50 cursor-not-allowed'
        )}
        onClick={onStop}
        disabled={!canPlay}
        title="Stop"
      >
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
          <rect x="6" y="6" width="12" height="12" />
        </svg>
      </button>

      {/* Skip forward */}
      <button
        className={cn(
          'p-1.5 rounded hover:bg-[var(--mviz-list-hover-background)]',
          !canPlay && 'opacity-50 cursor-not-allowed'
        )}
        onClick={onSkipForward}
        disabled={!canPlay}
        title="Skip forward 5s"
      >
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
          <path d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z" />
        </svg>
      </button>

      {/* Separator */}
      <div className="w-px h-6 bg-[var(--mviz-border)] mx-1" />

      {/* Loop toggle */}
      <button
        className={cn(
          'p-1.5 rounded hover:bg-[var(--mviz-list-hover-background)]',
          isLooping && 'text-[var(--mviz-accent)]'
        )}
        onClick={onToggleLoop}
        title="Toggle loop"
      >
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="17 1 21 5 17 9" />
          <path d="M3 11V9a4 4 0 0 1 4-4h14" />
          <polyline points="7 23 3 19 7 15" />
          <path d="M21 13v2a4 4 0 0 1-4 4H3" />
        </svg>
      </button>

      {/* Playback speed */}
      <div className="relative">
        <button
          className={cn(
            'px-2 py-1 text-xs rounded hover:bg-[var(--mviz-list-hover-background)]'
          )}
          onClick={() => setShowSpeedMenu(!showSpeedMenu)}
          title="Playback speed"
        >
          {playbackRate}x
        </button>

        {showSpeedMenu && (
          <div
            className={cn(
              'absolute bottom-full left-0 mb-1 py-1 min-w-[80px]',
              'bg-[var(--mviz-input-background)]',
              'border border-[var(--mviz-border)]',
              'rounded shadow-lg z-50'
            )}
          >
            {speedOptions.map((speed) => (
              <button
                key={speed}
                className={cn(
                  'w-full px-3 py-1 text-xs text-left',
                  'hover:bg-[var(--mviz-list-hover-background)]',
                  playbackRate === speed && 'text-[var(--mviz-accent)]'
                )}
                onClick={() => {
                  onSetPlaybackRate(speed);
                  setShowSpeedMenu(false);
                }}
              >
                {speed}x
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
});

PlaybackControls.displayName = 'PlaybackControls';

// ============================================================================
// Timeline Component
// ============================================================================

interface TimelineProps {
  currentTime: number;
  duration: number;
  onSeek: (time: number) => void;
  disabled: boolean;
}

const Timeline: React.FC<TimelineProps> = memo(({
  currentTime,
  duration,
  onSeek,
  disabled,
}) => {
  const progressRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      if (disabled || !progressRef.current || duration === 0) return;

      const rect = progressRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const percentage = x / rect.width;
      const time = percentage * duration;
      onSeek(Math.max(0, Math.min(duration, time)));
    },
    [disabled, duration, onSeek]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (disabled) return;
      setIsDragging(true);
      handleClick(e);
    },
    [disabled, handleClick]
  );

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!progressRef.current || duration === 0) return;

      const rect = progressRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const percentage = Math.max(0, Math.min(1, x / rect.width));
      const time = percentage * duration;
      onSeek(time);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, duration, onSeek]);

  return (
    <div className="flex items-center gap-3 w-full">
      {/* Current time */}
      <span className="text-xs font-mono w-12 text-right">
        {formatDuration(currentTime)}
      </span>

      {/* Progress bar */}
      <div
        ref={progressRef}
        className={cn(
          'flex-1 h-1.5 bg-[var(--mviz-border)] rounded-full cursor-pointer relative',
          disabled && 'opacity-50 cursor-not-allowed'
        )}
        onClick={handleClick}
        onMouseDown={handleMouseDown}
      >
        {/* Progress fill */}
        <div
          className="absolute inset-y-0 left-0 bg-[var(--mviz-accent)] rounded-full"
          style={{ width: `${progress}%` }}
        />

        {/* Scrubber handle */}
        <div
          className={cn(
            'absolute top-1/2 -translate-y-1/2 w-3 h-3',
            'bg-white rounded-full shadow',
            'transition-transform duration-100',
            isDragging && 'scale-125'
          )}
          style={{ left: `calc(${progress}% - 6px)` }}
        />
      </div>

      {/* Duration */}
      <span className="text-xs font-mono w-12">
        {formatDuration(duration)}
      </span>
    </div>
  );
});

Timeline.displayName = 'Timeline';

// ============================================================================
// Render Progress Component
// ============================================================================

interface RenderProgressProps {
  progress: number;
  phase: string;
  message?: string;
}

const RenderProgress: React.FC<RenderProgressProps> = memo(({
  progress,
  phase,
  message,
}) => {
  return (
    <div className="flex flex-col items-center justify-center gap-4 p-8">
      {/* Spinner */}
      <div className="relative w-16 h-16">
        <svg className="w-full h-full animate-spin" viewBox="0 0 24 24">
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
            fill="none"
          />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          />
        </svg>
        <span className="absolute inset-0 flex items-center justify-center text-sm font-medium">
          {Math.round(progress)}%
        </span>
      </div>

      {/* Phase */}
      <div className="text-center">
        <p className="text-sm font-medium capitalize">{phase}</p>
        {message && <p className="text-xs opacity-70 mt-1">{message}</p>}
      </div>

      {/* Progress bar */}
      <div className="w-full max-w-xs h-2 bg-[var(--mviz-border)] rounded-full overflow-hidden">
        <div
          className="h-full bg-[var(--mviz-accent)] transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
});

RenderProgress.displayName = 'RenderProgress';

// ============================================================================
// ManimPreview Component
// ============================================================================

export const ManimPreview: React.FC<ManimPreviewProps> = ({ className }) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  // Store state
  const status = usePreviewStore((state) => state.status);
  const currentTime = usePreviewStore((state) => state.currentTime);
  const duration = usePreviewStore((state) => state.duration);
  const isPlaying = usePreviewStore((state) => state.isPlaying);
  const playbackRate = usePreviewStore((state) => state.playbackRate);
  const isLooping = usePreviewStore((state) => state.isLooping);
  const quality = usePreviewStore((state) => state.quality);
  const videoUrl = usePreviewStore((state) => state.videoUrl);
  const renderProgress = usePreviewStore((state) => state.renderProgress);
  const error = usePreviewStore((state) => state.error);
  const showGrid = usePreviewStore((state) => state.showGrid);
  const backgroundColor = usePreviewStore((state) => state.backgroundColor);

  // Selectors
  const isReady = usePreviewStore(selectIsPreviewReady);
  const progress = usePreviewStore(selectPreviewProgress);
  const canPlay = usePreviewStore(selectCanPlay);

  // Actions
  const play = usePreviewStore((state) => state.play);
  const pause = usePreviewStore((state) => state.pause);
  const stop = usePreviewStore((state) => state.stop);
  const seek = usePreviewStore((state) => state.seek);
  const seekRelative = usePreviewStore((state) => state.seekRelative);
  const setPlaybackRate = usePreviewStore((state) => state.setPlaybackRate);
  const toggleLoop = usePreviewStore((state) => state.toggleLoop);
  const setQuality = usePreviewStore((state) => state.setQuality);
  const toggleGrid = usePreviewStore((state) => state.toggleGrid);
  const setCurrentTime = usePreviewStore((state) => state.setCurrentTime);
  const setDuration = usePreviewStore((state) => state.setDuration);
  const startRender = usePreviewStore((state) => state.startRender);

  // Sync video element with store
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !videoUrl) return;

    video.src = videoUrl;
    video.load();

    const handleLoadedMetadata = () => {
      setDuration(video.duration);
    };

    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime);
    };

    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    video.addEventListener('timeupdate', handleTimeUpdate);

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('timeupdate', handleTimeUpdate);
    };
  }, [videoUrl, setDuration, setCurrentTime]);

  // Handle play/pause
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (isPlaying) {
      video.play().catch(console.error);
    } else {
      video.pause();
    }
  }, [isPlaying]);

  // Handle playback rate
  useEffect(() => {
    const video = videoRef.current;
    if (video) {
      video.playbackRate = playbackRate;
    }
  }, [playbackRate]);

  // Handle loop
  useEffect(() => {
    const video = videoRef.current;
    if (video) {
      video.loop = isLooping;
    }
  }, [isLooping]);

  // Handle seek
  const handleSeek = useCallback(
    (time: number) => {
      seek(time);
      const video = videoRef.current;
      if (video) {
        video.currentTime = time;
      }
    },
    [seek]
  );

  // Render preview button
  const handleRender = useCallback(() => {
    // In real app, get current file path
    startRender('/path/to/current/file.mviz');
  }, [startRender]);

  return (
    <div className={cn('flex flex-col h-full', className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--mviz-border)]">
        <span className="text-xs font-semibold uppercase tracking-wider opacity-70">
          Preview
        </span>
        <div className="flex items-center gap-2">
          {/* Quality selector */}
          <select
            className={cn(
              'px-2 py-1 text-xs rounded',
              'bg-[var(--mviz-input-background)] text-[var(--mviz-input-foreground)]',
              'border border-[var(--mviz-input-border)]',
              'focus:outline-none focus:border-[var(--mviz-focus-border)]'
            )}
            value={quality}
            onChange={(e) => setQuality(e.target.value as PreviewQuality)}
          >
            {qualityOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>

          {/* Grid toggle */}
          <button
            className={cn(
              'p-1 rounded hover:bg-[var(--mviz-list-hover-background)]',
              showGrid && 'text-[var(--mviz-accent)]'
            )}
            onClick={toggleGrid}
            title="Toggle grid"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="18" height="18" rx="2" />
              <line x1="3" y1="9" x2="21" y2="9" />
              <line x1="3" y1="15" x2="21" y2="15" />
              <line x1="9" y1="3" x2="9" y2="21" />
              <line x1="15" y1="3" x2="15" y2="21" />
            </svg>
          </button>

          {/* Render button */}
          <button
            className={cn(
              'px-3 py-1 text-xs rounded',
              'bg-[var(--mviz-accent)] text-white',
              'hover:bg-[var(--mviz-button-hover-background)]'
            )}
            onClick={handleRender}
          >
            Render
          </button>
        </div>
      </div>

      {/* Preview area */}
      <div
        className="flex-1 flex items-center justify-center overflow-hidden relative"
        style={{ backgroundColor }}
      >
        {/* Grid overlay */}
        {showGrid && (
          <div
            className="absolute inset-0 pointer-events-none"
            style={{
              backgroundImage: `
                linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)
              `,
              backgroundSize: '50px 50px',
            }}
          />
        )}

        {/* Status-based content */}
        {status === 'idle' && (
          <div className="text-center p-8">
            <svg
              className="w-16 h-16 mx-auto mb-4 opacity-30"
              viewBox="0 0 24 24"
              fill="currentColor"
            >
              <path d="M8 5v14l11-7z" />
            </svg>
            <p className="text-sm opacity-70">No preview available</p>
            <p className="text-xs opacity-50 mt-1">
              Click Render to generate preview
            </p>
          </div>
        )}

        {(status === 'compiling' || status === 'rendering') && renderProgress && (
          <RenderProgress
            progress={progress}
            phase={renderProgress.phase}
            message={renderProgress.message}
          />
        )}

        {status === 'error' && (
          <div className="text-center p-8">
            <svg
              className="w-16 h-16 mx-auto mb-4 text-[var(--mviz-error-foreground)]"
              viewBox="0 0 24 24"
              fill="currentColor"
            >
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
            </svg>
            <p className="text-sm font-medium text-[var(--mviz-error-foreground)]">
              Render Failed
            </p>
            <p className="text-xs opacity-70 mt-1 max-w-xs">{error}</p>
          </div>
        )}

        {status === 'ready' && videoUrl && (
          <video
            ref={videoRef}
            className="max-w-full max-h-full"
            playsInline
          />
        )}
      </div>

      {/* Controls */}
      <div className="flex flex-col gap-2 p-3 border-t border-[var(--mviz-border)]">
        {/* Timeline */}
        <Timeline
          currentTime={currentTime}
          duration={duration}
          onSeek={handleSeek}
          disabled={!isReady}
        />

        {/* Playback controls */}
        <div className="flex items-center justify-between">
          <PlaybackControls
            isPlaying={isPlaying}
            canPlay={canPlay}
            playbackRate={playbackRate}
            isLooping={isLooping}
            onPlay={play}
            onPause={pause}
            onStop={stop}
            onSkipBackward={() => seekRelative(-5)}
            onSkipForward={() => seekRelative(5)}
            onSetPlaybackRate={setPlaybackRate}
            onToggleLoop={toggleLoop}
          />

          {/* Export button */}
          <button
            className={cn(
              'px-3 py-1 text-xs rounded',
              'bg-[var(--mviz-button-secondary-background)]',
              'text-[var(--mviz-button-secondary-foreground)]',
              'hover:bg-[var(--mviz-list-hover-background)]',
              !isReady && 'opacity-50 cursor-not-allowed'
            )}
            disabled={!isReady}
            title="Export video"
          >
            Export
          </button>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Export
// ============================================================================

export default ManimPreview;
