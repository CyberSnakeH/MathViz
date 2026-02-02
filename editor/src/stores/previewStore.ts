/**
 * Preview State Management
 *
 * Zustand store for managing Manim preview state including
 * playback, rendering, and export operations.
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import type {
  PreviewQuality,
  PreviewStatus,
  RenderProgress,
  ExportOptions,
} from '../types';

// ============================================================================
// Types
// ============================================================================

interface PreviewStoreState {
  // Preview state
  status: PreviewStatus;
  currentTime: number;
  duration: number;
  isPlaying: boolean;
  playbackRate: number;
  isLooping: boolean;

  // Quality settings
  quality: PreviewQuality;

  // Media
  videoUrl: string | null;
  thumbnailUrl: string | null;
  frames: string[]; // For frame-by-frame preview

  // Rendering
  isRendering: boolean;
  renderProgress: RenderProgress | null;

  // Export
  isExporting: boolean;
  exportProgress: number;

  // Source file
  sourceFilePath: string | null;
  lastRenderTime: Date | null;

  // Error state
  error: string | null;

  // UI state
  showControls: boolean;
  showGrid: boolean;
  showTimeline: boolean;
  backgroundColor: string;
}

interface PreviewActions {
  // Playback control
  play: () => void;
  pause: () => void;
  togglePlayPause: () => void;
  stop: () => void;
  seek: (time: number) => void;
  seekRelative: (delta: number) => void;
  setPlaybackRate: (rate: number) => void;
  toggleLoop: () => void;

  // Quality
  setQuality: (quality: PreviewQuality) => void;

  // Rendering
  startRender: (filePath: string) => Promise<void>;
  cancelRender: () => void;
  setRenderProgress: (progress: RenderProgress) => void;
  setRenderComplete: (videoUrl: string, thumbnailUrl?: string) => void;
  setRenderError: (error: string) => void;

  // Export
  exportVideo: (options: ExportOptions) => Promise<void>;
  cancelExport: () => void;
  setExportProgress: (progress: number) => void;

  // Media
  setVideoUrl: (url: string | null) => void;
  setFrames: (frames: string[]) => void;
  clearMedia: () => void;

  // Time updates
  setCurrentTime: (time: number) => void;
  setDuration: (duration: number) => void;

  // UI
  toggleControls: () => void;
  toggleGrid: () => void;
  toggleTimeline: () => void;
  setBackgroundColor: (color: string) => void;

  // Error handling
  setError: (error: string | null) => void;
  clearError: () => void;

  // Reset
  reset: () => void;
}

type PreviewStore = PreviewStoreState & PreviewActions;

// ============================================================================
// Initial State
// ============================================================================

const initialState: PreviewStoreState = {
  status: 'idle',
  currentTime: 0,
  duration: 0,
  isPlaying: false,
  playbackRate: 1,
  isLooping: false,
  quality: 'medium',
  videoUrl: null,
  thumbnailUrl: null,
  frames: [],
  isRendering: false,
  renderProgress: null,
  isExporting: false,
  exportProgress: 0,
  sourceFilePath: null,
  lastRenderTime: null,
  error: null,
  showControls: true,
  showGrid: false,
  showTimeline: true,
  backgroundColor: '#000000',
};

// ============================================================================
// Store Implementation
// ============================================================================

export const usePreviewStore = create<PreviewStore>()(
  devtools(
    immer((set, get) => ({
      ...initialState,

      // ====================================================================
      // Playback Control
      // ====================================================================

      play: () => {
        set((state) => {
          if (state.videoUrl && state.status === 'ready') {
            state.isPlaying = true;
          }
        });
      },

      pause: () => {
        set((state) => {
          state.isPlaying = false;
        });
      },

      togglePlayPause: () => {
        const state = get();
        if (state.isPlaying) {
          get().pause();
        } else {
          get().play();
        }
      },

      stop: () => {
        set((state) => {
          state.isPlaying = false;
          state.currentTime = 0;
        });
      },

      seek: (time: number) => {
        set((state) => {
          state.currentTime = Math.max(0, Math.min(time, state.duration));
        });
      },

      seekRelative: (delta: number) => {
        const state = get();
        get().seek(state.currentTime + delta);
      },

      setPlaybackRate: (rate: number) => {
        set((state) => {
          state.playbackRate = Math.max(0.25, Math.min(4, rate));
        });
      },

      toggleLoop: () => {
        set((state) => {
          state.isLooping = !state.isLooping;
        });
      },

      // ====================================================================
      // Quality
      // ====================================================================

      setQuality: (quality: PreviewQuality) => {
        set((state) => {
          state.quality = quality;
        });
      },

      // ====================================================================
      // Rendering
      // ====================================================================

      startRender: async (filePath: string) => {
        set((state) => {
          state.status = 'compiling';
          state.isRendering = true;
          state.error = null;
          state.sourceFilePath = filePath;
          state.renderProgress = {
            current: 0,
            total: 100,
            phase: 'parsing',
            message: 'Parsing MathViz source...',
          };
        });

        try {
          // This would call Tauri's render command
          // const result = await invoke('render_preview', {
          //   filePath,
          //   quality: get().quality,
          // });
          // The actual rendering updates would come through events
        } catch (error) {
          set((state) => {
            state.status = 'error';
            state.isRendering = false;
            state.error = error instanceof Error ? error.message : 'Render failed';
          });
        }
      },

      cancelRender: () => {
        set((state) => {
          state.status = 'idle';
          state.isRendering = false;
          state.renderProgress = null;
        });
        // Would also send cancel command to backend
        // invoke('cancel_render');
      },

      setRenderProgress: (progress: RenderProgress) => {
        set((state) => {
          state.renderProgress = progress;

          // Update status based on phase
          switch (progress.phase) {
            case 'parsing':
              state.status = 'compiling';
              break;
            case 'generating':
            case 'rendering':
              state.status = 'rendering';
              break;
            case 'encoding':
              state.status = 'rendering';
              break;
          }
        });
      },

      setRenderComplete: (videoUrl: string, thumbnailUrl?: string) => {
        set((state) => {
          state.status = 'ready';
          state.isRendering = false;
          state.videoUrl = videoUrl;
          state.thumbnailUrl = thumbnailUrl || null;
          state.renderProgress = null;
          state.lastRenderTime = new Date();
          state.currentTime = 0;
          state.isPlaying = false;
        });
      },

      setRenderError: (error: string) => {
        set((state) => {
          state.status = 'error';
          state.isRendering = false;
          state.error = error;
          state.renderProgress = null;
        });
      },

      // ====================================================================
      // Export
      // ====================================================================

      exportVideo: async (_options: ExportOptions) => {
        set((state) => {
          state.isExporting = true;
          state.exportProgress = 0;
          state.error = null;
        });

        try {
          // This would call Tauri's export command
          // await invoke('export_video', { options });
        } catch (error) {
          set((state) => {
            state.isExporting = false;
            state.error = error instanceof Error ? error.message : 'Export failed';
          });
        }
      },

      cancelExport: () => {
        set((state) => {
          state.isExporting = false;
          state.exportProgress = 0;
        });
        // Would also send cancel command to backend
        // invoke('cancel_export');
      },

      setExportProgress: (progress: number) => {
        set((state) => {
          state.exportProgress = Math.max(0, Math.min(100, progress));

          if (progress >= 100) {
            state.isExporting = false;
          }
        });
      },

      // ====================================================================
      // Media
      // ====================================================================

      setVideoUrl: (url: string | null) => {
        set((state) => {
          state.videoUrl = url;
          if (url) {
            state.status = 'ready';
          }
        });
      },

      setFrames: (frames: string[]) => {
        set((state) => {
          state.frames = frames;
        });
      },

      clearMedia: () => {
        set((state) => {
          state.videoUrl = null;
          state.thumbnailUrl = null;
          state.frames = [];
          state.status = 'idle';
          state.currentTime = 0;
          state.duration = 0;
          state.isPlaying = false;
        });
      },

      // ====================================================================
      // Time Updates
      // ====================================================================

      setCurrentTime: (time: number) => {
        set((state) => {
          state.currentTime = time;

          // Handle loop
          if (time >= state.duration && state.isLooping && state.duration > 0) {
            state.currentTime = 0;
          } else if (time >= state.duration) {
            state.isPlaying = false;
          }
        });
      },

      setDuration: (duration: number) => {
        set((state) => {
          state.duration = duration;
        });
      },

      // ====================================================================
      // UI
      // ====================================================================

      toggleControls: () => {
        set((state) => {
          state.showControls = !state.showControls;
        });
      },

      toggleGrid: () => {
        set((state) => {
          state.showGrid = !state.showGrid;
        });
      },

      toggleTimeline: () => {
        set((state) => {
          state.showTimeline = !state.showTimeline;
        });
      },

      setBackgroundColor: (color: string) => {
        set((state) => {
          state.backgroundColor = color;
        });
      },

      // ====================================================================
      // Error Handling
      // ====================================================================

      setError: (error: string | null) => {
        set((state) => {
          state.error = error;
          if (error) {
            state.status = 'error';
          }
        });
      },

      clearError: () => {
        set((state) => {
          state.error = null;
          if (state.status === 'error') {
            state.status = 'idle';
          }
        });
      },

      // ====================================================================
      // Reset
      // ====================================================================

      reset: () => {
        set(() => ({ ...initialState }));
      },
    })),
    { name: 'mathviz-preview-store' }
  )
);

// ============================================================================
// Selectors
// ============================================================================

export const selectIsPreviewReady = (state: PreviewStore): boolean => {
  return state.status === 'ready' && state.videoUrl !== null;
};

export const selectPreviewProgress = (state: PreviewStore): number => {
  if (!state.renderProgress) return 0;
  return (state.renderProgress.current / state.renderProgress.total) * 100;
};

export const selectFormattedTime = (state: PreviewStore): string => {
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return `${formatTime(state.currentTime)} / ${formatTime(state.duration)}`;
};

export const selectCanPlay = (state: PreviewStore): boolean => {
  return state.status === 'ready' && !state.isRendering && !state.isExporting;
};
