/**
 * File System State Management
 *
 * Zustand store for managing file tree state, folder expansion,
 * and file system operations.
 */

import { create } from 'zustand';
import { devtools, subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import type { FileNode, FileExtension } from '../types';

// ============================================================================
// Types
// ============================================================================

interface FileState {
  // Root project
  rootPath: string | null;
  projectName: string | null;

  // File tree
  fileTree: FileNode[];
  expandedFolders: Set<string>;
  selectedPath: string | null;

  // Loading states
  isLoading: boolean;
  isRefreshing: boolean;
  loadingPaths: Set<string>;

  // Search/filter
  searchQuery: string;
  filteredTree: FileNode[] | null;

  // Clipboard
  clipboardPath: string | null;
  clipboardOperation: 'copy' | 'cut' | null;
}

interface FileActions {
  // Root operations
  setRootPath: (path: string, projectName?: string) => void;
  clearRoot: () => void;

  // File tree operations
  setFileTree: (tree: FileNode[]) => void;
  updateNode: (path: string, updates: Partial<FileNode>) => void;
  addNode: (parentPath: string, node: FileNode) => void;
  removeNode: (path: string) => void;
  renameNode: (oldPath: string, newPath: string, newName: string) => void;

  // Folder expansion
  toggleFolder: (path: string) => void;
  expandFolder: (path: string) => void;
  collapseFolder: (path: string) => void;
  expandAll: () => void;
  collapseAll: () => void;

  // Selection
  setSelectedPath: (path: string | null) => void;

  // Loading states
  setLoading: (loading: boolean) => void;
  setRefreshing: (refreshing: boolean) => void;
  setPathLoading: (path: string, loading: boolean) => void;

  // Search/filter
  setSearchQuery: (query: string) => void;
  clearSearch: () => void;

  // Clipboard operations
  copyPath: (path: string) => void;
  cutPath: (path: string) => void;
  clearClipboard: () => void;

  // Refresh
  refreshFiles: () => Promise<void>;
}

type FileStore = FileState & FileActions;

// ============================================================================
// Initial State
// ============================================================================

const initialState: FileState = {
  rootPath: null,
  projectName: null,
  fileTree: [],
  expandedFolders: new Set(),
  selectedPath: null,
  isLoading: false,
  isRefreshing: false,
  loadingPaths: new Set(),
  searchQuery: '',
  filteredTree: null,
  clipboardPath: null,
  clipboardOperation: null,
};

// ============================================================================
// Utility Functions
// ============================================================================

function getExtension(filename: string): FileExtension {
  const ext = filename.split('.').pop()?.toLowerCase();
  const knownExtensions: FileExtension[] = [
    'mviz', 'py', 'json', 'toml', 'yaml', 'yml', 'md', 'txt', 'gitignore'
  ];
  return (knownExtensions.includes(ext as FileExtension) ? ext : 'other') as FileExtension;
}

function findNodeByPath(tree: FileNode[], path: string): FileNode | null {
  for (const node of tree) {
    if (node.path === path) {
      return node;
    }
    if (node.children) {
      const found = findNodeByPath(node.children, path);
      if (found) return found;
    }
  }
  return null;
}

function findParentNode(tree: FileNode[], path: string): FileNode | null {
  const parentPath = path.split('/').slice(0, -1).join('/');
  return findNodeByPath(tree, parentPath);
}

function removeNodeFromTree(tree: FileNode[], path: string): boolean {
  for (let i = 0; i < tree.length; i++) {
    if (tree[i].path === path) {
      tree.splice(i, 1);
      return true;
    }
    if (tree[i].children) {
      if (removeNodeFromTree(tree[i].children!, path)) {
        return true;
      }
    }
  }
  return false;
}

function updateNodeInTree(
  tree: FileNode[],
  path: string,
  updates: Partial<FileNode>
): boolean {
  for (const node of tree) {
    if (node.path === path) {
      Object.assign(node, updates);
      return true;
    }
    if (node.children) {
      if (updateNodeInTree(node.children, path, updates)) {
        return true;
      }
    }
  }
  return false;
}

function collectAllFolderPaths(tree: FileNode[]): string[] {
  const paths: string[] = [];

  function traverse(nodes: FileNode[]) {
    for (const node of nodes) {
      if (node.type === 'directory') {
        paths.push(node.path);
        if (node.children) {
          traverse(node.children);
        }
      }
    }
  }

  traverse(tree);
  return paths;
}

function filterTree(tree: FileNode[], query: string): FileNode[] {
  const lowerQuery = query.toLowerCase();

  function filterNodes(nodes: FileNode[]): FileNode[] {
    const result: FileNode[] = [];

    for (const node of nodes) {
      const nameMatches = node.name.toLowerCase().includes(lowerQuery);

      if (node.type === 'directory' && node.children) {
        const filteredChildren = filterNodes(node.children);
        if (filteredChildren.length > 0 || nameMatches) {
          result.push({
            ...node,
            children: filteredChildren,
            isExpanded: true,
          });
        }
      } else if (nameMatches) {
        result.push(node);
      }
    }

    return result;
  }

  return filterNodes(tree);
}

function sortFileTree(tree: FileNode[]): FileNode[] {
  const sorted = [...tree].sort((a, b) => {
    // Directories first
    if (a.type === 'directory' && b.type !== 'directory') return -1;
    if (a.type !== 'directory' && b.type === 'directory') return 1;
    // Then alphabetically
    return a.name.localeCompare(b.name);
  });

  return sorted.map((node) => {
    if (node.children) {
      return { ...node, children: sortFileTree(node.children) };
    }
    return node;
  });
}

// ============================================================================
// Store Implementation
// ============================================================================

export const useFileStore = create<FileStore>()(
  devtools(
    subscribeWithSelector(
      immer((set, get) => ({
        ...initialState,

        // ====================================================================
        // Root Operations
        // ====================================================================

        setRootPath: (path: string, projectName?: string) => {
          set((state) => {
            state.rootPath = path;
            state.projectName = projectName || path.split('/').pop() || 'Project';
            state.fileTree = [];
            state.expandedFolders.clear();
            state.selectedPath = null;
          });
        },

        clearRoot: () => {
          set((state) => {
            state.rootPath = null;
            state.projectName = null;
            state.fileTree = [];
            state.expandedFolders.clear();
            state.selectedPath = null;
          });
        },

        // ====================================================================
        // File Tree Operations
        // ====================================================================

        setFileTree: (tree: FileNode[]) => {
          set((state) => {
            state.fileTree = sortFileTree(tree);
            state.isLoading = false;
            state.isRefreshing = false;

            // Update filtered tree if search is active
            if (state.searchQuery) {
              state.filteredTree = filterTree(state.fileTree, state.searchQuery);
            }
          });
        },

        updateNode: (path: string, updates: Partial<FileNode>) => {
          set((state) => {
            updateNodeInTree(state.fileTree, path, updates);

            if (state.filteredTree) {
              updateNodeInTree(state.filteredTree, path, updates);
            }
          });
        },

        addNode: (parentPath: string, node: FileNode) => {
          set((state) => {
            const parent = findNodeByPath(state.fileTree, parentPath);
            if (parent && parent.children) {
              parent.children.push(node);
              parent.children = sortFileTree(parent.children);
            } else if (parentPath === state.rootPath || parentPath === '') {
              state.fileTree.push(node);
              state.fileTree = sortFileTree(state.fileTree);
            }

            // Update filtered tree
            if (state.searchQuery) {
              state.filteredTree = filterTree(state.fileTree, state.searchQuery);
            }
          });
        },

        removeNode: (path: string) => {
          set((state) => {
            removeNodeFromTree(state.fileTree, path);
            state.expandedFolders.delete(path);

            if (state.selectedPath === path) {
              state.selectedPath = null;
            }

            // Update filtered tree
            if (state.searchQuery) {
              state.filteredTree = filterTree(state.fileTree, state.searchQuery);
            }
          });
        },

        renameNode: (oldPath: string, newPath: string, newName: string) => {
          set((state) => {
            const node = findNodeByPath(state.fileTree, oldPath);
            if (node) {
              node.path = newPath;
              node.name = newName;
              if (node.type === 'file') {
                node.extension = getExtension(newName);
              }

              // Update expanded folders if directory was renamed
              if (state.expandedFolders.has(oldPath)) {
                state.expandedFolders.delete(oldPath);
                state.expandedFolders.add(newPath);
              }

              // Update selected path if needed
              if (state.selectedPath === oldPath) {
                state.selectedPath = newPath;
              }

              // Re-sort the parent's children
              const parent = findParentNode(state.fileTree, newPath);
              if (parent && parent.children) {
                parent.children = sortFileTree(parent.children);
              } else {
                state.fileTree = sortFileTree(state.fileTree);
              }
            }

            // Update filtered tree
            if (state.searchQuery) {
              state.filteredTree = filterTree(state.fileTree, state.searchQuery);
            }
          });
        },

        // ====================================================================
        // Folder Expansion
        // ====================================================================

        toggleFolder: (path: string) => {
          set((state) => {
            if (state.expandedFolders.has(path)) {
              state.expandedFolders.delete(path);
            } else {
              state.expandedFolders.add(path);
            }
          });
        },

        expandFolder: (path: string) => {
          set((state) => {
            state.expandedFolders.add(path);
          });
        },

        collapseFolder: (path: string) => {
          set((state) => {
            state.expandedFolders.delete(path);
          });
        },

        expandAll: () => {
          set((state) => {
            const allPaths = collectAllFolderPaths(state.fileTree);
            state.expandedFolders = new Set(allPaths);
          });
        },

        collapseAll: () => {
          set((state) => {
            state.expandedFolders.clear();
          });
        },

        // ====================================================================
        // Selection
        // ====================================================================

        setSelectedPath: (path: string | null) => {
          set((state) => {
            state.selectedPath = path;
          });
        },

        // ====================================================================
        // Loading States
        // ====================================================================

        setLoading: (loading: boolean) => {
          set((state) => {
            state.isLoading = loading;
          });
        },

        setRefreshing: (refreshing: boolean) => {
          set((state) => {
            state.isRefreshing = refreshing;
          });
        },

        setPathLoading: (path: string, loading: boolean) => {
          set((state) => {
            if (loading) {
              state.loadingPaths.add(path);
            } else {
              state.loadingPaths.delete(path);
            }
          });
        },

        // ====================================================================
        // Search/Filter
        // ====================================================================

        setSearchQuery: (query: string) => {
          set((state) => {
            state.searchQuery = query;
            if (query) {
              state.filteredTree = filterTree(state.fileTree, query);
            } else {
              state.filteredTree = null;
            }
          });
        },

        clearSearch: () => {
          set((state) => {
            state.searchQuery = '';
            state.filteredTree = null;
          });
        },

        // ====================================================================
        // Clipboard Operations
        // ====================================================================

        copyPath: (path: string) => {
          set((state) => {
            state.clipboardPath = path;
            state.clipboardOperation = 'copy';
          });
        },

        cutPath: (path: string) => {
          set((state) => {
            state.clipboardPath = path;
            state.clipboardOperation = 'cut';
          });
        },

        clearClipboard: () => {
          set((state) => {
            state.clipboardPath = null;
            state.clipboardOperation = null;
          });
        },

        // ====================================================================
        // Refresh
        // ====================================================================

        refreshFiles: async () => {
          const state = get();
          if (!state.rootPath) return;

          set((s) => {
            s.isRefreshing = true;
          });

          try {
            // This would call Tauri's file system API
            // For now, we'll just simulate it
            // const files = await invoke('read_directory', { path: state.rootPath });
            // set((s) => { s.fileTree = files; });
          } catch (error) {
            console.error('Failed to refresh files:', error);
          } finally {
            set((s) => {
              s.isRefreshing = false;
            });
          }
        },
      }))
    ),
    { name: 'mathviz-file-store' }
  )
);

// ============================================================================
// Selectors
// ============================================================================

export const selectVisibleTree = (state: FileStore): FileNode[] => {
  return state.filteredTree ?? state.fileTree;
};

export const selectIsExpanded = (state: FileStore, path: string): boolean => {
  return state.expandedFolders.has(path);
};

export const selectIsSelected = (state: FileStore, path: string): boolean => {
  return state.selectedPath === path;
};

export const selectIsLoading = (state: FileStore, path: string): boolean => {
  return state.loadingPaths.has(path);
};

export const selectNodeByPath = (
  state: FileStore,
  path: string
): FileNode | null => {
  return findNodeByPath(state.fileTree, path);
};

export const selectHasClipboard = (state: FileStore): boolean => {
  return state.clipboardPath !== null;
};
