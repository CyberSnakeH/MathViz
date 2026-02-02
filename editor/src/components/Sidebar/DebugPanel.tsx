/**
 * Debug Panel Component
 *
 * Panel for compiling and running MathViz files.
 * Features:
 * - Compile button
 * - Run button with preview option
 * - Output console
 * - Breakpoints list
 */

import React, { useCallback, useEffect, useRef } from 'react';
import { cn } from '../../utils/helpers';
import { useCompilerStore } from '../../stores/compilerStore';
import { useEditorStore, selectActiveFile } from '../../stores/editorStore';

// ============================================================================
// Icons
// ============================================================================

const PlayIcon = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
    <polygon points="5 3 19 12 5 21 5 3" />
  </svg>
);

const BuildIcon = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
  </svg>
);

const StopIcon = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
    <rect x="6" y="6" width="12" height="12" rx="1" />
  </svg>
);

const ClearIcon = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6" />
  </svg>
);

const BugIcon = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="8" r="4" />
    <path d="M12 12V21M8 8L4 4M16 8L20 4M4 12h4M16 12h4M6 16l2 2M18 16l-2 2" />
  </svg>
);

// ============================================================================
// Debug Panel Component
// ============================================================================

export const DebugPanel: React.FC = () => {
  const activeFile = useEditorStore(selectActiveFile);
  const status = useCompilerStore((state) => state.status);
  const output = useCompilerStore((state) => state.output);
  const compile = useCompilerStore((state) => state.compile);
  const run = useCompilerStore((state) => state.run);
  const clearOutput = useCompilerStore((state) => state.clearOutput);
  const reset = useCompilerStore((state) => state.reset);

  const outputRef = useRef<HTMLDivElement>(null);

  const isCompiling = status === 'compiling';
  const isRunning = status === 'running';
  const isBusy = isCompiling || isRunning;

  // Auto-scroll to bottom when output changes
  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [output]);

  // Check if file is a MathViz file and exists on disk (not a virtual file like /untitled.mviz)
  const isMathVizFile = activeFile && (activeFile.path.endsWith('.mviz') || activeFile.path.endsWith('.mathviz'));
  const isRealFile = activeFile && !activeFile.path.startsWith('/untitled');
  const canCompile = isMathVizFile && isRealFile;

  const handleCompile = useCallback(async () => {
    if (!canCompile || isBusy || !activeFile) return;
    await compile(activeFile.path);
  }, [canCompile, activeFile, isBusy, compile]);

  const handleRun = useCallback(async () => {
    if (!canCompile || isBusy || !activeFile) return;
    await run(activeFile.path, true, 'm');
  }, [canCompile, activeFile, isBusy, run]);

  const handleStop = useCallback(() => {
    reset();
  }, [reset]);

  const handleClear = useCallback(() => {
    clearOutput();
  }, [clearOutput]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-[var(--mviz-border,#1a1b26)]">
        <span className="text-xs font-semibold uppercase tracking-wider text-[var(--mviz-foreground,#a9b1d6)] opacity-70">
          Run & Debug
        </span>
      </div>

      {/* Actions */}
      <div className="p-3 border-b border-[var(--mviz-border,#1a1b26)]">
        <div className="flex flex-col gap-2">
          {/* Run Button */}
          <button
            className={cn(
              'flex items-center justify-center gap-2 px-4 py-2 rounded-lg',
              'text-sm font-medium',
              'transition-all duration-150',
              canCompile && !isBusy
                ? 'bg-[var(--mviz-git-added,#9ece6a)] text-[#1a1b26] hover:opacity-90'
                : 'bg-[var(--mviz-sidebar-background,#1f2335)] text-[var(--mviz-foreground,#a9b1d6)] opacity-50 cursor-not-allowed'
            )}
            onClick={handleRun}
            disabled={!canCompile || isBusy}
          >
            {isRunning ? (
              <>
                <span className="animate-spin">
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 12a9 9 0 11-6.219-8.56" />
                  </svg>
                </span>
                <span>Running...</span>
              </>
            ) : (
              <>
                <PlayIcon />
                <span>Run</span>
              </>
            )}
          </button>

          {/* Compile Button */}
          <button
            className={cn(
              'flex items-center justify-center gap-2 px-4 py-2 rounded-lg',
              'text-sm font-medium',
              'transition-all duration-150',
              canCompile && !isBusy
                ? 'bg-[var(--mviz-accent,#7aa2f7)] text-[#1a1b26] hover:opacity-90'
                : 'bg-[var(--mviz-sidebar-background,#1f2335)] text-[var(--mviz-foreground,#a9b1d6)] opacity-50 cursor-not-allowed'
            )}
            onClick={handleCompile}
            disabled={!canCompile || isBusy}
          >
            {isCompiling ? (
              <>
                <span className="animate-spin">
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 12a9 9 0 11-6.219-8.56" />
                  </svg>
                </span>
                <span>Compiling...</span>
              </>
            ) : (
              <>
                <BuildIcon />
                <span>Compile</span>
              </>
            )}
          </button>

          {/* Stop Button (when running) */}
          {isBusy && (
            <button
              className={cn(
                'flex items-center justify-center gap-2 px-4 py-2 rounded-lg',
                'text-sm font-medium',
                'bg-[var(--mviz-error-foreground,#f7768e)] text-[#1a1b26]',
                'hover:opacity-90 transition-all duration-150'
              )}
              onClick={handleStop}
            >
              <StopIcon />
              <span>Stop</span>
            </button>
          )}
        </div>

        {/* Status */}
        {status !== 'idle' && (
          <div className={cn(
            'mt-3 px-3 py-2 rounded-lg text-xs',
            status === 'success' && 'bg-[var(--mviz-git-added,#9ece6a)]/10 text-[var(--mviz-git-added,#9ece6a)]',
            status === 'error' && 'bg-[var(--mviz-error-foreground,#f7768e)]/10 text-[var(--mviz-error-foreground,#f7768e)]',
            (status === 'compiling' || status === 'running') && 'bg-[var(--mviz-accent,#7aa2f7)]/10 text-[var(--mviz-accent,#7aa2f7)]'
          )}>
            {status === 'success' && 'Completed successfully'}
            {status === 'error' && 'Failed with errors'}
            {status === 'compiling' && 'Compiling...'}
            {status === 'running' && 'Running...'}
          </div>
        )}
      </div>

      {/* Current File Info */}
      <div className="px-3 py-2 border-b border-[var(--mviz-border,#1a1b26)]">
        <div className="text-xs text-[var(--mviz-foreground,#a9b1d6)] opacity-50 mb-1">Active File</div>
        <div className="text-sm text-[var(--mviz-foreground,#c0caf5)] truncate">
          {activeFile ? activeFile.path.split('/').pop() : 'No file selected'}
        </div>
        {activeFile && !isRealFile && (
          <div className="text-xs text-[var(--mviz-warning-foreground,#e0af68)] mt-1">
            Save file to disk to run
          </div>
        )}
        {activeFile && isRealFile && !isMathVizFile && (
          <div className="text-xs text-[var(--mviz-foreground,#a9b1d6)] opacity-50 mt-1">
            Only .mviz files can be compiled
          </div>
        )}
      </div>

      {/* Output Console */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex items-center justify-between px-3 py-2 border-b border-[var(--mviz-border,#1a1b26)]">
          <span className="text-xs font-semibold uppercase tracking-wider text-[var(--mviz-foreground,#a9b1d6)] opacity-70">
            Output
          </span>
          <button
            className={cn(
              'p-1 rounded',
              'text-[var(--mviz-foreground,#a9b1d6)] opacity-50',
              'hover:opacity-100 hover:bg-[var(--mviz-list-hover-background,#292e42)]',
              'transition-all duration-100'
            )}
            onClick={handleClear}
            title="Clear output"
          >
            <ClearIcon />
          </button>
        </div>

        <div
          ref={outputRef}
          className="flex-1 overflow-auto p-3 font-mono text-xs bg-[var(--mviz-terminal-background,#1a1b26)] leading-relaxed"
        >
          {output.length === 0 ? (
            <div className="text-[var(--mviz-foreground,#a9b1d6)] opacity-30">
              Output will appear here...
            </div>
          ) : (
            output.map((line, index) => {
              // Empty line for spacing
              if (!line.trim()) {
                return <div key={index} className="h-2" />;
              }

              // Determine line type and color
              const isError = line.includes('[Error]') || line.includes('ERROR');
              const isWarning = line.includes('[Warning]') || line.includes('WARNING');
              const isSuccess = line.includes('[Success]');
              const isStatus = line.includes('[Compiling]') || line.includes('[Running]');
              const isStdout = line.startsWith('[stdout]');
              const isInfo = line.startsWith('Manim') || line.startsWith('File ') || line.includes('v0.');

              // Remove prefix for display
              const displayLine = isStdout ? line.replace('[stdout] ', '') : line;

              return (
                <div
                  key={index}
                  className={cn(
                    'py-0.5 whitespace-pre-wrap break-words',
                    isError && 'text-[var(--mviz-error-foreground,#f7768e)]',
                    isWarning && 'text-[var(--mviz-warning-foreground,#e0af68)]',
                    isSuccess && 'text-[var(--mviz-git-added,#9ece6a)] font-medium',
                    isStatus && 'text-[var(--mviz-accent,#7aa2f7)]',
                    isStdout && 'text-[var(--mviz-foreground,#c0caf5)]',
                    isInfo && 'text-[var(--mviz-comment,#565f89)]',
                    !isError && !isWarning && !isSuccess && !isStatus && !isStdout && !isInfo && 'text-[var(--mviz-foreground,#a9b1d6)]'
                  )}
                >
                  {displayLine}
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Debug Controls */}
      <div className="p-3 border-t border-[var(--mviz-border,#1a1b26)]">
        <div className="flex items-center gap-2 text-xs text-[var(--mviz-foreground,#a9b1d6)] opacity-50">
          <BugIcon />
          <span>Debug mode coming soon</span>
        </div>
      </div>
    </div>
  );
};

export default DebugPanel;
