/**
 * StatusBar Component
 *
 * Modern two-tone status bar design with:
 * - Left section with accent color for remote/branch info
 * - Hover tooltips on all items
 * - Click actions for common tasks
 * - Language selector, encoding, line ending indicators
 */

import React, { memo, useState, useCallback } from 'react';
import { cn } from '../../utils/helpers';
import { useEditorStore, selectActiveFile } from '../../stores/editorStore';
import { useGitStore } from '../../stores/gitStore';
import { useLayoutStore } from '../../stores/layoutStore';
import type { LspStatus } from '../../types';

// ============================================================================
// Types
// ============================================================================

interface StatusBarItemProps {
  children: React.ReactNode;
  onClick?: () => void;
  tooltip?: string;
  className?: string;
  accent?: boolean;
}

// ============================================================================
// StatusBarItem Component
// ============================================================================

const StatusBarItem: React.FC<StatusBarItemProps> = memo(({
  children,
  onClick,
  tooltip,
  className,
  accent = false,
}) => {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <div
      className={cn(
        'relative flex items-center gap-1.5 px-2 h-full',
        'text-xs',
        onClick && 'cursor-pointer',
        onClick && (accent
          ? 'hover:bg-black/10'
          : 'hover:bg-white/5'),
        'transition-colors duration-100',
        className
      )}
      onClick={onClick}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      {children}

      {/* Tooltip */}
      {tooltip && showTooltip && (
        <div
          className={cn(
            'absolute bottom-full left-1/2 -translate-x-1/2 mb-2',
            'px-2 py-1 rounded',
            'bg-[var(--mviz-widget-background,#1f2335)]',
            'border border-[var(--mviz-widget-border,#3b4261)]',
            'text-[11px] text-[var(--mviz-foreground,#a9b1d6)]',
            'whitespace-nowrap shadow-lg',
            'animate-fade-in',
            'z-[1000]'
          )}
        >
          {tooltip}
        </div>
      )}
    </div>
  );
});

StatusBarItem.displayName = 'StatusBarItem';

// ============================================================================
// LSP Status Indicator
// ============================================================================

interface LspStatusIndicatorProps {
  status: LspStatus;
  serverName?: string;
}

const LspStatusIndicator: React.FC<LspStatusIndicatorProps> = memo(({ status, serverName }) => {
  const config = {
    stopped: { color: 'bg-[var(--mviz-foreground,#a9b1d6)] opacity-40', label: 'Language Server: Stopped' },
    starting: { color: 'bg-[var(--mviz-warning-foreground,#e0af68)]', label: 'Language Server: Starting...' },
    running: { color: 'bg-[var(--mviz-git-added,#9ece6a)]', label: 'Language Server: Running' },
    error: { color: 'bg-[var(--mviz-error-foreground,#f7768e)]', label: 'Language Server: Error' },
  };

  const { color, label } = config[status];
  const fullLabel = serverName ? `${label} (${serverName})` : label;

  return (
    <StatusBarItem tooltip={fullLabel}>
      <span className={cn('w-2 h-2 rounded-full', color)} />
      {serverName && <span className="hidden sm:inline">{serverName}</span>}
    </StatusBarItem>
  );
});

LspStatusIndicator.displayName = 'LspStatusIndicator';

// ============================================================================
// Git Branch Icon
// ============================================================================

const GitBranchIcon = () => (
  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="6" y1="3" x2="6" y2="15" />
    <circle cx="18" cy="6" r="3" />
    <circle cx="6" cy="18" r="3" />
    <path d="M18 9a9 9 0 0 1-9 9" />
  </svg>
);

// ============================================================================
// StatusBar Component
// ============================================================================

export const StatusBar: React.FC<{ className?: string }> = ({ className }) => {
  // Editor state
  const activeFile = useEditorStore(selectActiveFile);
  const cursorPosition = useEditorStore((state) => state.cursorPosition);
  const selection = useEditorStore((state) => state.selection);
  const diagnosticSummary = useEditorStore((state) => state.diagnosticSummary);

  // Git state
  const currentBranch = useGitStore((state) => state.currentBranch);
  const status = useGitStore((state) => state.status);
  const isRepository = useGitStore((state) => state.isRepository);

  // Layout state
  const toggleBottomPanel = useLayoutStore((state) => state.toggleBottomPanel);
  const setBottomPanelActive = useLayoutStore((state) => state.setBottomPanelActive);

  // Handlers
  const handleProblemsClick = useCallback(() => {
    setBottomPanelActive('problems');
    toggleBottomPanel();
  }, [setBottomPanelActive, toggleBottomPanel]);

  // Language display name
  const languageNames: Record<string, string> = {
    mathviz: 'MathViz',
    python: 'Python',
    javascript: 'JavaScript',
    typescript: 'TypeScript',
    typescriptreact: 'TypeScript React',
    javascriptreact: 'JavaScript React',
    json: 'JSON',
    markdown: 'Markdown',
    plaintext: 'Plain Text',
    toml: 'TOML',
    yaml: 'YAML',
    html: 'HTML',
    css: 'CSS',
    rust: 'Rust',
  };

  const languageName = activeFile
    ? languageNames[activeFile.language] || activeFile.language
    : 'No file';

  // Selection info
  const selectionInfo = selection
    ? ` (${selection.endLineNumber - selection.startLineNumber + 1} lines)`
    : '';

  // Git sync status
  const syncStatus = status
    ? [
        status.ahead > 0 ? `${status.ahead}+` : '',
        status.behind > 0 ? `${status.behind}-` : '',
      ]
        .filter(Boolean)
        .join(' ')
    : '';

  return (
    <div
      className={cn(
        'status-bar flex items-center h-6 select-none',
        'text-[var(--mviz-status-bar-foreground,#a9b1d6)]',
        className
      )}
    >
      {/* Left section - Accent background */}
      <div
        className={cn(
          'flex items-center h-full',
          'bg-[var(--mviz-accent,#7aa2f7)]',
          'text-[var(--mviz-accent-foreground,#1a1b26)]',
          'pr-3'
        )}
        style={{
          clipPath: 'polygon(0 0, calc(100% - 10px) 0, 100% 100%, 0 100%)',
        }}
      >
        {/* Git branch */}
        {isRepository && currentBranch ? (
          <StatusBarItem
            tooltip={`Git Branch: ${currentBranch}${syncStatus ? ` (${syncStatus})` : ''}`}
            accent
          >
            <GitBranchIcon />
            <span className="font-medium">{currentBranch}</span>
            {syncStatus && (
              <span className="opacity-70 text-[10px]">{syncStatus}</span>
            )}
          </StatusBarItem>
        ) : (
          <StatusBarItem tooltip="MathViz Editor" accent>
            <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M5 5h12M5 5l6 7-6 7h12" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span className="font-medium">MathViz</span>
          </StatusBarItem>
        )}
      </div>

      {/* Center section - Main background */}
      <div
        className={cn(
          'flex items-center flex-1 h-full',
          'bg-[var(--mviz-status-bar-background,#1a1b26)]',
          'pl-4'
        )}
      >
        {/* Uncommitted changes indicator */}
        {isRepository && status && !status.isClean && (
          <StatusBarItem tooltip="Uncommitted changes">
            <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 8v4M12 16h.01" />
            </svg>
          </StatusBarItem>
        )}

        {/* Problems (errors & warnings) */}
        <StatusBarItem
          onClick={handleProblemsClick}
          tooltip={`Errors: ${diagnosticSummary.errors}, Warnings: ${diagnosticSummary.warnings}`}
        >
          {/* Errors */}
          <svg
            className={cn(
              'w-3.5 h-3.5',
              diagnosticSummary.errors > 0
                ? 'text-[var(--mviz-error-foreground,#f7768e)]'
                : 'opacity-50'
            )}
            viewBox="0 0 24 24"
            fill="currentColor"
          >
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 11c-.55 0-1-.45-1-1V8c0-.55.45-1 1-1s1 .45 1 1v4c0 .55-.45 1-1 1zm1 4h-2v-2h2v2z" />
          </svg>
          <span className={diagnosticSummary.errors > 0 ? '' : 'opacity-50'}>
            {diagnosticSummary.errors}
          </span>

          {/* Warnings */}
          <svg
            className={cn(
              'w-3.5 h-3.5 ml-1',
              diagnosticSummary.warnings > 0
                ? 'text-[var(--mviz-warning-foreground,#e0af68)]'
                : 'opacity-50'
            )}
            viewBox="0 0 24 24"
            fill="currentColor"
          >
            <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z" />
          </svg>
          <span className={diagnosticSummary.warnings > 0 ? '' : 'opacity-50'}>
            {diagnosticSummary.warnings}
          </span>
        </StatusBarItem>

        {/* Spacer */}
        <div className="flex-1" />
      </div>

      {/* Right section - Main background */}
      <div
        className={cn(
          'flex items-center h-full',
          'bg-[var(--mviz-status-bar-background,#1a1b26)]'
        )}
      >
        {/* Cursor position */}
        {activeFile && (
          <StatusBarItem tooltip="Go to Line (Ctrl+G)">
            <span>
              Ln {cursorPosition.lineNumber}, Col {cursorPosition.column}
            </span>
            {selectionInfo && (
              <span className="opacity-60">{selectionInfo}</span>
            )}
          </StatusBarItem>
        )}

        {/* Indentation */}
        {activeFile && (
          <StatusBarItem tooltip="Select Indentation">
            <span>Spaces: 4</span>
          </StatusBarItem>
        )}

        {/* Encoding */}
        {activeFile && (
          <StatusBarItem tooltip="Select Encoding">
            <span>UTF-8</span>
          </StatusBarItem>
        )}

        {/* Line ending */}
        {activeFile && (
          <StatusBarItem tooltip="Select End of Line Sequence">
            <span>LF</span>
          </StatusBarItem>
        )}

        {/* Language mode */}
        {activeFile && (
          <StatusBarItem tooltip="Select Language Mode (Ctrl+K M)">
            <span>{languageName}</span>
          </StatusBarItem>
        )}

        {/* LSP status */}
        <LspStatusIndicator
          status={activeFile?.language === 'mathviz' ? 'running' : 'stopped'}
          serverName={activeFile?.language === 'mathviz' ? 'mathviz-lsp' : undefined}
        />

        {/* Notifications bell */}
        <StatusBarItem tooltip="No Notifications">
          <svg className="w-3.5 h-3.5 opacity-60" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
            <path d="M13.73 21a2 2 0 0 1-3.46 0" />
          </svg>
        </StatusBarItem>
      </div>
    </div>
  );
};

// ============================================================================
// Export
// ============================================================================

export default StatusBar;
