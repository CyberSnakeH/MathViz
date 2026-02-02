/**
 * Git State Management
 *
 * Zustand store for managing Git repository state including
 * branches, changes, and commit operations.
 */

import { create } from 'zustand';
import { devtools, subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import type { GitFile, GitBranch, GitStatus, GitCommit, GitRemote } from '../types';

// ============================================================================
// Types
// ============================================================================

interface GitState {
  // Repository info
  isRepository: boolean;
  repositoryPath: string | null;

  // Current state
  currentBranch: string | null;
  status: GitStatus | null;

  // Branches
  branches: GitBranch[];
  remoteBranches: GitBranch[];
  remotes: GitRemote[];

  // Recent commits
  commits: GitCommit[];

  // UI state
  commitMessage: string;
  isLoading: boolean;
  isCommitting: boolean;
  isPushing: boolean;
  isPulling: boolean;
  isFetching: boolean;

  // Errors
  error: string | null;

  // Diff view
  selectedFile: GitFile | null;
  diffContent: string | null;
}

interface GitActions {
  // Repository
  setRepository: (path: string | null) => void;
  setIsRepository: (isRepo: boolean) => void;

  // Status
  setStatus: (status: GitStatus) => void;
  clearStatus: () => void;
  refreshStatus: () => Promise<void>;

  // Branches
  setBranches: (branches: GitBranch[]) => void;
  setRemoteBranches: (branches: GitBranch[]) => void;
  setRemotes: (remotes: GitRemote[]) => void;
  setCurrentBranch: (branch: string) => void;
  switchBranch: (branchName: string) => Promise<void>;
  createBranch: (branchName: string, checkout?: boolean) => Promise<void>;
  deleteBranch: (branchName: string, force?: boolean) => Promise<void>;

  // Staging
  stageFile: (path: string) => Promise<void>;
  unstageFile: (path: string) => Promise<void>;
  stageAll: () => Promise<void>;
  unstageAll: () => Promise<void>;
  discardChanges: (path: string) => Promise<void>;

  // Commits
  setCommitMessage: (message: string) => void;
  commit: (message?: string) => Promise<void>;
  setCommits: (commits: GitCommit[]) => void;

  // Remote operations
  push: (remote?: string, branch?: string, force?: boolean) => Promise<void>;
  pull: (remote?: string, branch?: string) => Promise<void>;
  fetch: (remote?: string) => Promise<void>;

  // Diff
  setSelectedFile: (file: GitFile | null) => void;
  setDiffContent: (content: string | null) => void;
  loadDiff: (path: string, staged?: boolean) => Promise<void>;

  // Loading states
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

type GitStore = GitState & GitActions;

// ============================================================================
// Initial State
// ============================================================================

const initialState: GitState = {
  isRepository: false,
  repositoryPath: null,
  currentBranch: null,
  status: null,
  branches: [],
  remoteBranches: [],
  remotes: [],
  commits: [],
  commitMessage: '',
  isLoading: false,
  isCommitting: false,
  isPushing: false,
  isPulling: false,
  isFetching: false,
  error: null,
  selectedFile: null,
  diffContent: null,
};

// ============================================================================
// Store Implementation
// ============================================================================

export const useGitStore = create<GitStore>()(
  devtools(
    subscribeWithSelector(
      immer((set, get) => ({
        ...initialState,

        // ====================================================================
        // Repository
        // ====================================================================

        setRepository: (path: string | null) => {
          set((state) => {
            state.repositoryPath = path;
            if (!path) {
              Object.assign(state, initialState);
            }
          });
        },

        setIsRepository: (isRepo: boolean) => {
          set((state) => {
            state.isRepository = isRepo;
            if (!isRepo) {
              Object.assign(state, initialState);
            }
          });
        },

        // ====================================================================
        // Status
        // ====================================================================

        setStatus: (status: GitStatus) => {
          set((state) => {
            state.status = status;
            state.currentBranch = status.branch;
          });
        },

        clearStatus: () => {
          set((state) => {
            state.status = null;
          });
        },

        refreshStatus: async () => {
          const state = get();
          if (!state.repositoryPath) return;

          set((s) => {
            s.isLoading = true;
            s.error = null;
          });

          try {
            // This would call Tauri's git commands
            // const status = await invoke('git_status', { path: state.repositoryPath });
            // set((s) => { s.status = status; });
          } catch (error) {
            set((s) => {
              s.error = error instanceof Error ? error.message : 'Failed to get git status';
            });
          } finally {
            set((s) => {
              s.isLoading = false;
            });
          }
        },

        // ====================================================================
        // Branches
        // ====================================================================

        setBranches: (branches: GitBranch[]) => {
          set((state) => {
            state.branches = branches;
          });
        },

        setRemoteBranches: (branches: GitBranch[]) => {
          set((state) => {
            state.remoteBranches = branches;
          });
        },

        setRemotes: (remotes: GitRemote[]) => {
          set((state) => {
            state.remotes = remotes;
          });
        },

        setCurrentBranch: (branch: string) => {
          set((state) => {
            state.currentBranch = branch;
          });
        },

        switchBranch: async (branchName: string) => {
          set((state) => {
            state.isLoading = true;
            state.error = null;
          });

          try {
            // await invoke('git_checkout', { branch: branchName });
            set((state) => {
              state.currentBranch = branchName;
            });
            await get().refreshStatus();
          } catch (error) {
            set((state) => {
              state.error = error instanceof Error ? error.message : 'Failed to switch branch';
            });
          } finally {
            set((state) => {
              state.isLoading = false;
            });
          }
        },

        createBranch: async (branchName: string, checkout = true) => {
          set((state) => {
            state.isLoading = true;
            state.error = null;
          });

          try {
            // await invoke('git_create_branch', { name: branchName, checkout });
            if (checkout) {
              set((state) => {
                state.currentBranch = branchName;
              });
            }
            await get().refreshStatus();
          } catch (error) {
            set((state) => {
              state.error = error instanceof Error ? error.message : 'Failed to create branch';
            });
          } finally {
            set((state) => {
              state.isLoading = false;
            });
          }
        },

        deleteBranch: async (branchName: string, _force = false) => {
          set((state) => {
            state.isLoading = true;
            state.error = null;
          });

          try {
            // await invoke('git_delete_branch', { name: branchName, force });
            set((state) => {
              state.branches = state.branches.filter((b: GitBranch) => b.name !== branchName);
            });
          } catch (error) {
            set((state) => {
              state.error = error instanceof Error ? error.message : 'Failed to delete branch';
            });
          } finally {
            set((state) => {
              state.isLoading = false;
            });
          }
        },

        // ====================================================================
        // Staging
        // ====================================================================

        stageFile: async (_path: string) => {
          try {
            // await invoke('git_add', { paths: [path] });
            await get().refreshStatus();
          } catch (error) {
            set((state) => {
              state.error = error instanceof Error ? error.message : 'Failed to stage file';
            });
          }
        },

        unstageFile: async (_path: string) => {
          try {
            // await invoke('git_reset', { paths: [path] });
            await get().refreshStatus();
          } catch (error) {
            set((state) => {
              state.error = error instanceof Error ? error.message : 'Failed to unstage file';
            });
          }
        },

        stageAll: async () => {
          try {
            // await invoke('git_add_all');
            await get().refreshStatus();
          } catch (error) {
            set((state) => {
              state.error = error instanceof Error ? error.message : 'Failed to stage all files';
            });
          }
        },

        unstageAll: async () => {
          try {
            // await invoke('git_reset_all');
            await get().refreshStatus();
          } catch (error) {
            set((state) => {
              state.error = error instanceof Error ? error.message : 'Failed to unstage all files';
            });
          }
        },

        discardChanges: async (_path: string) => {
          try {
            // await invoke('git_checkout_file', { path });
            await get().refreshStatus();
          } catch (error) {
            set((state) => {
              state.error = error instanceof Error ? error.message : 'Failed to discard changes';
            });
          }
        },

        // ====================================================================
        // Commits
        // ====================================================================

        setCommitMessage: (message: string) => {
          set((state) => {
            state.commitMessage = message;
          });
        },

        commit: async (message?: string) => {
          const state = get();
          const commitMsg = message || state.commitMessage;

          if (!commitMsg.trim()) {
            set((s) => {
              s.error = 'Commit message cannot be empty';
            });
            return;
          }

          set((s) => {
            s.isCommitting = true;
            s.error = null;
          });

          try {
            // await invoke('git_commit', { message: commitMsg });
            set((s) => {
              s.commitMessage = '';
            });
            await get().refreshStatus();
          } catch (error) {
            set((s) => {
              s.error = error instanceof Error ? error.message : 'Failed to commit';
            });
          } finally {
            set((s) => {
              s.isCommitting = false;
            });
          }
        },

        setCommits: (commits: GitCommit[]) => {
          set((state) => {
            state.commits = commits;
          });
        },

        // ====================================================================
        // Remote Operations
        // ====================================================================

        push: async (_remote = 'origin', branch?: string, _force = false) => {
          const state = get();
          const targetBranch = branch || state.currentBranch;

          if (!targetBranch) {
            set((s) => {
              s.error = 'No branch selected';
            });
            return;
          }

          set((s) => {
            s.isPushing = true;
            s.error = null;
          });

          try {
            // await invoke('git_push', { remote, branch: targetBranch, force });
            await get().refreshStatus();
          } catch (error) {
            set((s) => {
              s.error = error instanceof Error ? error.message : 'Failed to push';
            });
          } finally {
            set((s) => {
              s.isPushing = false;
            });
          }
        },

        pull: async (_remote = 'origin', branch?: string) => {
          const state = get();
          const targetBranch = branch || state.currentBranch;

          if (!targetBranch) {
            set((s) => {
              s.error = 'No branch selected';
            });
            return;
          }

          set((s) => {
            s.isPulling = true;
            s.error = null;
          });

          try {
            // await invoke('git_pull', { remote, branch: targetBranch });
            await get().refreshStatus();
          } catch (error) {
            set((s) => {
              s.error = error instanceof Error ? error.message : 'Failed to pull';
            });
          } finally {
            set((s) => {
              s.isPulling = false;
            });
          }
        },

        fetch: async (_remote = 'origin') => {
          set((state) => {
            state.isFetching = true;
            state.error = null;
          });

          try {
            // await invoke('git_fetch', { remote });
            await get().refreshStatus();
          } catch (error) {
            set((state) => {
              state.error = error instanceof Error ? error.message : 'Failed to fetch';
            });
          } finally {
            set((state) => {
              state.isFetching = false;
            });
          }
        },

        // ====================================================================
        // Diff
        // ====================================================================

        setSelectedFile: (file: GitFile | null) => {
          set((state) => {
            state.selectedFile = file;
            if (!file) {
              state.diffContent = null;
            }
          });
        },

        setDiffContent: (content: string | null) => {
          set((state) => {
            state.diffContent = content;
          });
        },

        loadDiff: async (_path: string, _staged = false) => {
          try {
            // const diff = await invoke('git_diff', { path, staged });
            // set((state) => { state.diffContent = diff; });
          } catch (error) {
            set((state) => {
              state.error = error instanceof Error ? error.message : 'Failed to load diff';
            });
          }
        },

        // ====================================================================
        // Loading States
        // ====================================================================

        setLoading: (loading: boolean) => {
          set((state) => {
            state.isLoading = loading;
          });
        },

        setError: (error: string | null) => {
          set((state) => {
            state.error = error;
          });
        },
      }))
    ),
    { name: 'mathviz-git-store' }
  )
);

// ============================================================================
// Selectors
// ============================================================================

export const selectStagedFiles = (state: GitStore): GitFile[] => {
  return state.status?.staged || [];
};

export const selectUnstagedFiles = (state: GitStore): GitFile[] => {
  return state.status?.unstaged || [];
};

export const selectUntrackedFiles = (state: GitStore): GitFile[] => {
  return state.status?.untracked || [];
};

export const selectConflictedFiles = (state: GitStore): GitFile[] => {
  return state.status?.conflicted || [];
};

export const selectHasChanges = (state: GitStore): boolean => {
  if (!state.status) return false;
  return (
    state.status.staged.length > 0 ||
    state.status.unstaged.length > 0 ||
    state.status.untracked.length > 0
  );
};
