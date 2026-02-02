/**
 * Editor State Management
 *
 * Zustand store for managing editor state including open files, tabs,
 * cursor positions, and editor settings.
 */

import { create } from 'zustand';
import { devtools, subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import type {
  OpenFile,
  CursorPosition,
  Selection,
  Diagnostic,
  DiagnosticSummary,
  EditorSettings,
} from '../types';

// ============================================================================
// Types
// ============================================================================

interface EditorState {
  // Open files
  openFiles: Map<string, OpenFile>;
  activeFileId: string | null;
  fileOrder: string[]; // Tab order

  // Editor state
  cursorPosition: CursorPosition;
  selection: Selection | null;

  // Diagnostics
  diagnostics: Map<string, Diagnostic[]>;
  diagnosticSummary: DiagnosticSummary;

  // Settings
  settings: EditorSettings;

  // UI state
  isFindReplaceOpen: boolean;
  findQuery: string;
  replaceQuery: string;
}

interface EditorActions {
  // File operations
  openFile: (path: string, content: string, language?: string) => void;
  closeFile: (id: string) => void;
  closeAllFiles: () => void;
  closeOtherFiles: (keepId: string) => void;
  closeSavedFiles: () => void;
  setActiveFile: (id: string | null) => void;
  updateFileContent: (id: string, content: string) => void;
  markFileSaved: (id: string, newContent?: string) => void;
  reorderTabs: (fromIndex: number, toIndex: number) => void;

  // Cursor and selection
  setCursorPosition: (position: CursorPosition) => void;
  setSelection: (selection: Selection | null) => void;

  // Diagnostics
  setDiagnostics: (filePath: string, diagnostics: Diagnostic[]) => void;
  clearDiagnostics: (filePath?: string) => void;

  // Settings
  updateSettings: (settings: Partial<EditorSettings>) => void;

  // Find/Replace
  toggleFindReplace: (open?: boolean) => void;
  setFindQuery: (query: string) => void;
  setReplaceQuery: (query: string) => void;

  // View state
  saveViewState: (id: string, viewState: unknown) => void;
}

type EditorStore = EditorState & EditorActions;

// ============================================================================
// Default Values
// ============================================================================

const defaultSettings: EditorSettings = {
  fontSize: 14,
  fontFamily: "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
  lineHeight: 1.5,
  tabSize: 4,
  insertSpaces: true,
  wordWrap: 'off',
  wordWrapColumn: 80,
  minimap: {
    enabled: true,
    side: 'right',
    maxColumn: 120,
  },
  scrollBeyondLastLine: true,
  renderWhitespace: 'selection',
  cursorStyle: 'line',
  cursorBlinking: 'blink',
  formatOnSave: true,
  formatOnPaste: false,
  autoSave: 'off',
  autoSaveDelay: 1000,
};

const initialState: EditorState = {
  openFiles: new Map(),
  activeFileId: null,
  fileOrder: [],
  cursorPosition: { lineNumber: 1, column: 1 },
  selection: null,
  diagnostics: new Map(),
  diagnosticSummary: { errors: 0, warnings: 0, infos: 0, hints: 0 },
  settings: defaultSettings,
  isFindReplaceOpen: false,
  findQuery: '',
  replaceQuery: '',
};

// ============================================================================
// Utility Functions
// ============================================================================

function getLanguageFromPath(path: string): string {
  const ext = path.split('.').pop()?.toLowerCase();
  const languageMap: Record<string, string> = {
    mviz: 'mathviz',
    py: 'python',
    js: 'javascript',
    ts: 'typescript',
    tsx: 'typescriptreact',
    jsx: 'javascriptreact',
    json: 'json',
    toml: 'toml',
    yaml: 'yaml',
    yml: 'yaml',
    md: 'markdown',
    html: 'html',
    css: 'css',
    scss: 'scss',
    rs: 'rust',
    go: 'go',
    sh: 'shell',
    bash: 'shell',
  };
  return languageMap[ext || ''] || 'plaintext';
}

function getFileNameFromPath(path: string): string {
  return path.split('/').pop() || path;
}

function generateFileId(path: string): string {
  return `file_${path.replace(/[^a-zA-Z0-9]/g, '_')}_${Date.now()}`;
}

function calculateDiagnosticSummary(
  diagnostics: Map<string, Diagnostic[]>
): DiagnosticSummary {
  const summary: DiagnosticSummary = { errors: 0, warnings: 0, infos: 0, hints: 0 };

  for (const fileDiagnostics of diagnostics.values()) {
    for (const diagnostic of fileDiagnostics) {
      switch (diagnostic.severity) {
        case 'error':
          summary.errors++;
          break;
        case 'warning':
          summary.warnings++;
          break;
        case 'info':
          summary.infos++;
          break;
        case 'hint':
          summary.hints++;
          break;
      }
    }
  }

  return summary;
}

// ============================================================================
// Store Implementation
// ============================================================================

export const useEditorStore = create<EditorStore>()(
  devtools(
    subscribeWithSelector(
      immer((set, _get) => ({
        ...initialState,

        // ====================================================================
        // File Operations
        // ====================================================================

        openFile: (path: string, content: string, language?: string) => {
          set((state) => {
            // Check if file is already open
            for (const [id, file] of state.openFiles.entries()) {
              if (file.path === path) {
                state.activeFileId = id;
                return;
              }
            }

            // Create new file entry
            const id = generateFileId(path);
            const newFile: OpenFile = {
              id,
              path,
              name: getFileNameFromPath(path),
              content,
              originalContent: content,
              language: language || getLanguageFromPath(path),
              isDirty: false,
              isReadOnly: false,
              cursorPosition: { lineNumber: 1, column: 1 },
              scrollPosition: { scrollTop: 0, scrollLeft: 0 },
            };

            state.openFiles.set(id, newFile);
            state.fileOrder.push(id);
            state.activeFileId = id;
          });
        },

        closeFile: (id: string) => {
          set((state) => {
            const fileIndex = state.fileOrder.indexOf(id);
            if (fileIndex === -1) return;

            state.openFiles.delete(id);
            state.fileOrder.splice(fileIndex, 1);

            // Update active file if we closed the active one
            if (state.activeFileId === id) {
              if (state.fileOrder.length === 0) {
                state.activeFileId = null;
              } else {
                // Activate the next file, or the previous if we closed the last
                const newIndex = Math.min(fileIndex, state.fileOrder.length - 1);
                state.activeFileId = state.fileOrder[newIndex];
              }
            }
          });
        },

        closeAllFiles: () => {
          set((state) => {
            state.openFiles.clear();
            state.fileOrder = [];
            state.activeFileId = null;
          });
        },

        closeOtherFiles: (keepId: string) => {
          set((state) => {
            const keepFile = state.openFiles.get(keepId);
            if (!keepFile) return;

            state.openFiles.clear();
            state.openFiles.set(keepId, keepFile);
            state.fileOrder = [keepId];
            state.activeFileId = keepId;
          });
        },

        closeSavedFiles: () => {
          set((state) => {
            const toRemove: string[] = [];
            for (const [id, file] of state.openFiles.entries()) {
              if (!file.isDirty) {
                toRemove.push(id);
              }
            }

            for (const id of toRemove) {
              state.openFiles.delete(id);
              const index = state.fileOrder.indexOf(id);
              if (index !== -1) {
                state.fileOrder.splice(index, 1);
              }
            }

            if (state.activeFileId && toRemove.includes(state.activeFileId)) {
              state.activeFileId = state.fileOrder[0] || null;
            }
          });
        },

        setActiveFile: (id: string | null) => {
          set((state) => {
            if (id === null || state.openFiles.has(id)) {
              state.activeFileId = id;
            }
          });
        },

        updateFileContent: (id: string, content: string) => {
          set((state) => {
            const file = state.openFiles.get(id);
            if (file) {
              file.content = content;
              file.isDirty = content !== file.originalContent;
            }
          });
        },

        markFileSaved: (id: string, newContent?: string) => {
          set((state) => {
            const file = state.openFiles.get(id);
            if (file) {
              if (newContent !== undefined) {
                file.content = newContent;
              }
              file.originalContent = file.content;
              file.isDirty = false;
            }
          });
        },

        reorderTabs: (fromIndex: number, toIndex: number) => {
          set((state) => {
            if (
              fromIndex < 0 ||
              fromIndex >= state.fileOrder.length ||
              toIndex < 0 ||
              toIndex >= state.fileOrder.length
            ) {
              return;
            }

            const [removed] = state.fileOrder.splice(fromIndex, 1);
            state.fileOrder.splice(toIndex, 0, removed);
          });
        },

        // ====================================================================
        // Cursor and Selection
        // ====================================================================

        setCursorPosition: (position: CursorPosition) => {
          set((state) => {
            state.cursorPosition = position;

            // Also update the active file's cursor position
            if (state.activeFileId) {
              const file = state.openFiles.get(state.activeFileId);
              if (file) {
                file.cursorPosition = position;
              }
            }
          });
        },

        setSelection: (selection: Selection | null) => {
          set((state) => {
            state.selection = selection;
          });
        },

        // ====================================================================
        // Diagnostics
        // ====================================================================

        setDiagnostics: (filePath: string, diagnostics: Diagnostic[]) => {
          set((state) => {
            state.diagnostics.set(filePath, diagnostics);
            state.diagnosticSummary = calculateDiagnosticSummary(state.diagnostics);
          });
        },

        clearDiagnostics: (filePath?: string) => {
          set((state) => {
            if (filePath) {
              state.diagnostics.delete(filePath);
            } else {
              state.diagnostics.clear();
            }
            state.diagnosticSummary = calculateDiagnosticSummary(state.diagnostics);
          });
        },

        // ====================================================================
        // Settings
        // ====================================================================

        updateSettings: (newSettings: Partial<EditorSettings>) => {
          set((state) => {
            state.settings = { ...state.settings, ...newSettings };
          });
        },

        // ====================================================================
        // Find/Replace
        // ====================================================================

        toggleFindReplace: (open?: boolean) => {
          set((state) => {
            state.isFindReplaceOpen = open ?? !state.isFindReplaceOpen;
          });
        },

        setFindQuery: (query: string) => {
          set((state) => {
            state.findQuery = query;
          });
        },

        setReplaceQuery: (query: string) => {
          set((state) => {
            state.replaceQuery = query;
          });
        },

        // ====================================================================
        // View State
        // ====================================================================

        saveViewState: (id: string, viewState: unknown) => {
          set((state) => {
            const file = state.openFiles.get(id);
            if (file) {
              file.viewState = viewState;
            }
          });
        },
      }))
    ),
    { name: 'mathviz-editor-store' }
  )
);

// ============================================================================
// Selectors
// ============================================================================

export const selectActiveFile = (state: EditorStore): OpenFile | null => {
  if (!state.activeFileId) return null;
  return state.openFiles.get(state.activeFileId) || null;
};

export const selectOpenFilesArray = (state: EditorStore): OpenFile[] => {
  return state.fileOrder
    .map((id) => state.openFiles.get(id))
    .filter((file): file is OpenFile => file !== undefined);
};

export const selectHasUnsavedFiles = (state: EditorStore): boolean => {
  for (const file of state.openFiles.values()) {
    if (file.isDirty) return true;
  }
  return false;
};

export const selectDiagnosticsForFile = (
  state: EditorStore,
  filePath: string
): Diagnostic[] => {
  return state.diagnostics.get(filePath) || [];
};

export const selectFileById = (
  state: EditorStore,
  id: string
): OpenFile | undefined => {
  return state.openFiles.get(id);
};
