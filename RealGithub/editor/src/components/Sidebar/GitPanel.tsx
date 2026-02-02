/**
 * GitPanel Component
 *
 * Source control panel with changed files list, staging controls,
 * commit message input, branch selector, and push/pull buttons.
 */

import React, { useCallback, useState, memo } from 'react';
import { cn } from '../../utils/helpers';
import {
  useGitStore,
  selectStagedFiles,
  selectUnstagedFiles,
  selectUntrackedFiles,
  selectConflictedFiles,
  selectHasChanges,
} from '../../stores/gitStore';
import type { GitFile, GitFileStatus } from '../../types';

// ============================================================================
// Constants
// ============================================================================

const statusColors: Record<GitFileStatus, string> = {
  modified: 'text-[var(--mviz-git-modified)]',
  added: 'text-[var(--mviz-git-added)]',
  deleted: 'text-[var(--mviz-git-deleted)]',
  renamed: 'text-[var(--mviz-git-modified)]',
  copied: 'text-[var(--mviz-git-added)]',
  untracked: 'text-[var(--mviz-git-untracked)]',
  ignored: 'text-gray-500',
  conflicted: 'text-[var(--mviz-git-conflicting)]',
};

const statusLabels: Record<GitFileStatus, string> = {
  modified: 'M',
  added: 'A',
  deleted: 'D',
  renamed: 'R',
  copied: 'C',
  untracked: 'U',
  ignored: 'I',
  conflicted: '!',
};

// ============================================================================
// FileItem Component
// ============================================================================

interface FileItemProps {
  file: GitFile;
  onStage?: () => void;
  onUnstage?: () => void;
  onDiscard?: () => void;
  onSelect?: () => void;
  staged: boolean;
}

const FileItem: React.FC<FileItemProps> = memo(({
  file,
  onStage,
  onUnstage,
  onDiscard,
  onSelect,
  staged,
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const fileName = file.path.split('/').pop() || file.path;
  const directory = file.path.split('/').slice(0, -1).join('/');

  return (
    <div
      className={cn(
        'flex items-center gap-2 px-2 py-1 cursor-pointer',
        'hover:bg-[var(--mviz-list-hover-background)]'
      )}
      onClick={onSelect}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Status indicator */}
      <span className={cn('w-4 text-center text-xs font-bold', statusColors[file.status])}>
        {statusLabels[file.status]}
      </span>

      {/* File info */}
      <div className="flex-1 min-w-0">
        <span className="text-sm truncate">{fileName}</span>
        {directory && (
          <span className="text-xs opacity-50 ml-2 truncate">{directory}</span>
        )}
      </div>

      {/* Actions */}
      {isHovered && (
        <div className="flex items-center gap-0.5">
          {staged ? (
            <>
              <button
                className="p-1 rounded hover:bg-[var(--mviz-list-hover-background)]"
                onClick={(e) => {
                  e.stopPropagation();
                  onUnstage?.();
                }}
                title="Unstage"
              >
                <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="5" y1="12" x2="19" y2="12" />
                </svg>
              </button>
            </>
          ) : (
            <>
              <button
                className="p-1 rounded hover:bg-[var(--mviz-list-hover-background)]"
                onClick={(e) => {
                  e.stopPropagation();
                  onDiscard?.();
                }}
                title="Discard changes"
              >
                <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                </svg>
              </button>
              <button
                className="p-1 rounded hover:bg-[var(--mviz-list-hover-background)]"
                onClick={(e) => {
                  e.stopPropagation();
                  onStage?.();
                }}
                title="Stage"
              >
                <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="12" y1="5" x2="12" y2="19" />
                  <line x1="5" y1="12" x2="19" y2="12" />
                </svg>
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
});

FileItem.displayName = 'FileItem';

// ============================================================================
// FileSection Component
// ============================================================================

interface FileSectionProps {
  title: string;
  files: GitFile[];
  staged: boolean;
  onStageAll?: () => void;
  onUnstageAll?: () => void;
  onStageFile: (path: string) => void;
  onUnstageFile: (path: string) => void;
  onDiscardFile: (path: string) => void;
  onSelectFile: (file: GitFile) => void;
}

const FileSection: React.FC<FileSectionProps> = memo(({
  title,
  files,
  staged,
  onStageAll,
  onUnstageAll,
  onStageFile,
  onUnstageFile,
  onDiscardFile,
  onSelectFile,
}) => {
  const [isExpanded, setIsExpanded] = useState(true);

  if (files.length === 0) return null;

  return (
    <div className="mb-2">
      {/* Section header */}
      <div
        className={cn(
          'flex items-center justify-between px-2 py-1 cursor-pointer',
          'hover:bg-[var(--mviz-list-hover-background)]'
        )}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-1">
          <span
            className={cn(
              'w-4 h-4 flex items-center justify-center transition-transform',
              isExpanded && 'rotate-90'
            )}
          >
            <svg className="w-3 h-3" viewBox="0 0 16 16" fill="currentColor">
              <path d="M6 4l4 4-4 4V4z" />
            </svg>
          </span>
          <span className="text-xs font-semibold uppercase">{title}</span>
          <span className="text-xs opacity-50">({files.length})</span>
        </div>

        {/* Section actions */}
        <div className="flex items-center gap-0.5">
          {staged ? (
            <button
              className="p-1 rounded hover:bg-[var(--mviz-list-hover-background)]"
              onClick={(e) => {
                e.stopPropagation();
                onUnstageAll?.();
              }}
              title="Unstage all"
            >
              <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="5" y1="12" x2="19" y2="12" />
              </svg>
            </button>
          ) : (
            <button
              className="p-1 rounded hover:bg-[var(--mviz-list-hover-background)]"
              onClick={(e) => {
                e.stopPropagation();
                onStageAll?.();
              }}
              title="Stage all"
            >
              <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="12" y1="5" x2="12" y2="19" />
                <line x1="5" y1="12" x2="19" y2="12" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* Files */}
      {isExpanded && (
        <div className="ml-2">
          {files.map((file) => (
            <FileItem
              key={file.path}
              file={file}
              staged={staged}
              onStage={() => onStageFile(file.path)}
              onUnstage={() => onUnstageFile(file.path)}
              onDiscard={() => onDiscardFile(file.path)}
              onSelect={() => onSelectFile(file)}
            />
          ))}
        </div>
      )}
    </div>
  );
});

FileSection.displayName = 'FileSection';

// ============================================================================
// GitPanel Component
// ============================================================================

export const GitPanel: React.FC = () => {
  // Store state
  const isRepository = useGitStore((state) => state.isRepository);
  const currentBranch = useGitStore((state) => state.currentBranch);
  const branches = useGitStore((state) => state.branches);
  const commitMessage = useGitStore((state) => state.commitMessage);
  const isLoading = useGitStore((state) => state.isLoading);
  const isCommitting = useGitStore((state) => state.isCommitting);
  const isPushing = useGitStore((state) => state.isPushing);
  const isPulling = useGitStore((state) => state.isPulling);
  const error = useGitStore((state) => state.error);

  // Selectors
  const stagedFiles = useGitStore(selectStagedFiles);
  const unstagedFiles = useGitStore(selectUnstagedFiles);
  const untrackedFiles = useGitStore(selectUntrackedFiles);
  const conflictedFiles = useGitStore(selectConflictedFiles);
  const hasChanges = useGitStore(selectHasChanges);

  // Actions
  const setCommitMessage = useGitStore((state) => state.setCommitMessage);
  const commit = useGitStore((state) => state.commit);
  const stageFile = useGitStore((state) => state.stageFile);
  const unstageFile = useGitStore((state) => state.unstageFile);
  const stageAll = useGitStore((state) => state.stageAll);
  const unstageAll = useGitStore((state) => state.unstageAll);
  const discardChanges = useGitStore((state) => state.discardChanges);
  const push = useGitStore((state) => state.push);
  const pull = useGitStore((state) => state.pull);
  const fetch = useGitStore((state) => state.fetch);
  const setSelectedFile = useGitStore((state) => state.setSelectedFile);
  const refreshStatus = useGitStore((state) => state.refreshStatus);

  // Branch selector state
  const [showBranchMenu, setShowBranchMenu] = useState(false);

  // Handle commit
  const handleCommit = useCallback(() => {
    if (commitMessage.trim() && stagedFiles.length > 0) {
      commit();
    }
  }, [commitMessage, stagedFiles.length, commit]);

  // Handle key press in commit message
  const handleKeyPress = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        handleCommit();
      }
    },
    [handleCommit]
  );

  if (!isRepository) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-4 text-center">
        <svg
          className="w-16 h-16 mb-4 opacity-30"
          viewBox="0 0 24 24"
          fill="currentColor"
        >
          <circle cx="12" cy="6" r="2" />
          <circle cx="12" cy="18" r="2" />
          <circle cx="6" cy="12" r="2" />
          <path d="M12 8v8M8 12h4" fill="none" stroke="currentColor" strokeWidth="2" />
        </svg>
        <p className="text-sm opacity-70">No Git repository</p>
        <p className="text-xs opacity-50 mt-1">
          Initialize a repository to use source control
        </p>
        <button
          className={cn(
            'mt-4 px-4 py-2 text-sm rounded',
            'bg-[var(--mviz-button-background)] text-[var(--mviz-button-foreground)]',
            'hover:bg-[var(--mviz-button-hover-background)]'
          )}
        >
          Initialize Repository
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--mviz-border)]">
        <span className="text-xs font-semibold uppercase tracking-wider opacity-70">
          Source Control
        </span>
        <div className="flex items-center gap-1">
          <button
            className={cn(
              'p-1 rounded hover:bg-[var(--mviz-list-hover-background)]',
              isLoading && 'animate-spin'
            )}
            onClick={() => refreshStatus()}
            title="Refresh"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
            </svg>
          </button>
        </div>
      </div>

      {/* Branch selector */}
      <div className="px-4 py-2 border-b border-[var(--mviz-border)]">
        <div className="relative">
          <button
            className={cn(
              'flex items-center gap-2 w-full px-2 py-1 text-sm rounded',
              'bg-[var(--mviz-input-background)]',
              'border border-[var(--mviz-input-border)]',
              'hover:border-[var(--mviz-focus-border)]'
            )}
            onClick={() => setShowBranchMenu(!showBranchMenu)}
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="6" y1="3" x2="6" y2="15" />
              <circle cx="18" cy="6" r="3" />
              <circle cx="6" cy="18" r="3" />
              <path d="M18 9a9 9 0 0 1-9 9" />
            </svg>
            <span className="flex-1 text-left truncate">{currentBranch || 'No branch'}</span>
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>

          {showBranchMenu && (
            <div
              className={cn(
                'absolute top-full left-0 right-0 mt-1 py-1',
                'bg-[var(--mviz-input-background)]',
                'border border-[var(--mviz-border)]',
                'rounded shadow-lg z-50 max-h-48 overflow-auto'
              )}
            >
              {branches.map((branch) => (
                <button
                  key={branch.name}
                  className={cn(
                    'w-full px-3 py-1.5 text-left text-sm',
                    'hover:bg-[var(--mviz-list-hover-background)]',
                    branch.isCurrent && 'text-[var(--mviz-accent)]'
                  )}
                  onClick={() => {
                    // Would switch branch
                    setShowBranchMenu(false);
                  }}
                >
                  {branch.name}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Commit message */}
      <div className="px-4 py-2 border-b border-[var(--mviz-border)]">
        <textarea
          className={cn(
            'w-full px-2 py-1.5 text-sm rounded resize-none',
            'bg-[var(--mviz-input-background)] text-[var(--mviz-input-foreground)]',
            'border border-[var(--mviz-input-border)]',
            'placeholder:text-[var(--mviz-input-placeholder)]',
            'focus:outline-none focus:border-[var(--mviz-focus-border)]'
          )}
          placeholder="Commit message"
          rows={3}
          value={commitMessage}
          onChange={(e) => setCommitMessage(e.target.value)}
          onKeyDown={handleKeyPress}
        />
        <button
          className={cn(
            'w-full mt-2 px-3 py-1.5 text-sm rounded',
            'bg-[var(--mviz-button-background)] text-[var(--mviz-button-foreground)]',
            'hover:bg-[var(--mviz-button-hover-background)]',
            'disabled:opacity-50 disabled:cursor-not-allowed'
          )}
          disabled={!commitMessage.trim() || stagedFiles.length === 0 || isCommitting}
          onClick={handleCommit}
        >
          {isCommitting ? 'Committing...' : 'Commit'}
        </button>
      </div>

      {/* Error message */}
      {error && (
        <div className="px-4 py-2 text-xs text-[var(--mviz-error-foreground)] bg-[var(--mviz-error-foreground)]/10">
          {error}
        </div>
      )}

      {/* File changes */}
      <div className="flex-1 overflow-auto py-2">
        {conflictedFiles.length > 0 && (
          <FileSection
            title="Merge Conflicts"
            files={conflictedFiles}
            staged={false}
            onStageFile={stageFile}
            onUnstageFile={unstageFile}
            onDiscardFile={discardChanges}
            onSelectFile={setSelectedFile}
          />
        )}

        <FileSection
          title="Staged Changes"
          files={stagedFiles}
          staged={true}
          onUnstageAll={unstageAll}
          onStageFile={stageFile}
          onUnstageFile={unstageFile}
          onDiscardFile={discardChanges}
          onSelectFile={setSelectedFile}
        />

        <FileSection
          title="Changes"
          files={[...unstagedFiles, ...untrackedFiles]}
          staged={false}
          onStageAll={stageAll}
          onStageFile={stageFile}
          onUnstageFile={unstageFile}
          onDiscardFile={discardChanges}
          onSelectFile={setSelectedFile}
        />

        {!hasChanges && (
          <div className="text-center py-8 text-sm opacity-50">
            No changes
          </div>
        )}
      </div>

      {/* Actions bar */}
      <div className="flex items-center gap-2 px-4 py-2 border-t border-[var(--mviz-border)]">
        <button
          className={cn(
            'flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs rounded',
            'bg-[var(--mviz-button-secondary-background)]',
            'text-[var(--mviz-button-secondary-foreground)]',
            'hover:bg-[var(--mviz-list-hover-background)]',
            'disabled:opacity-50 disabled:cursor-not-allowed'
          )}
          disabled={isPulling}
          onClick={() => pull()}
          title="Pull"
        >
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="7 13 12 18 17 13" />
            <line x1="12" y1="18" x2="12" y2="6" />
          </svg>
          {isPulling ? 'Pulling...' : 'Pull'}
        </button>
        <button
          className={cn(
            'flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs rounded',
            'bg-[var(--mviz-button-secondary-background)]',
            'text-[var(--mviz-button-secondary-foreground)]',
            'hover:bg-[var(--mviz-list-hover-background)]',
            'disabled:opacity-50 disabled:cursor-not-allowed'
          )}
          disabled={isPushing}
          onClick={() => push()}
          title="Push"
        >
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="17 11 12 6 7 11" />
            <line x1="12" y1="6" x2="12" y2="18" />
          </svg>
          {isPushing ? 'Pushing...' : 'Push'}
        </button>
        <button
          className={cn(
            'p-1.5 rounded',
            'bg-[var(--mviz-button-secondary-background)]',
            'text-[var(--mviz-button-secondary-foreground)]',
            'hover:bg-[var(--mviz-list-hover-background)]'
          )}
          onClick={() => fetch()}
          title="Fetch"
        >
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
            <path d="M3 3v5h5" />
            <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
            <path d="M16 16h5v5" />
          </svg>
        </button>
      </div>
    </div>
  );
};

// ============================================================================
// Export
// ============================================================================

export default GitPanel;
