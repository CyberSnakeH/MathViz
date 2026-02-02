/**
 * Terminal State Management
 *
 * Zustand store for managing terminal instances,
 * tabs, and terminal output.
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import type { TerminalTab, TerminalOptions, ShellType } from '../types';

// ============================================================================
// Types
// ============================================================================

interface TerminalState {
  // Terminals
  terminals: Map<string, TerminalTab>;
  activeTerminalId: string | null;
  terminalOrder: string[];

  // Default settings
  defaultShell: ShellType;
  defaultCwd: string | null;

  // UI state
  isMaximized: boolean;
}

interface TerminalActions {
  // Terminal management
  createTerminal: (options?: TerminalOptions) => string;
  closeTerminal: (id: string) => void;
  closeAllTerminals: () => void;
  setActiveTerminal: (id: string) => void;
  renameTerminal: (id: string, title: string) => void;
  reorderTerminals: (fromIndex: number, toIndex: number) => void;

  // Terminal state
  setTerminalCwd: (id: string, cwd: string) => void;
  setTerminalPid: (id: string, pid: number) => void;

  // Settings
  setDefaultShell: (shell: ShellType) => void;
  setDefaultCwd: (cwd: string) => void;

  // UI
  toggleMaximize: () => void;
  setMaximized: (maximized: boolean) => void;

  // Kill terminal process
  killTerminal: (id: string) => Promise<void>;
}

type TerminalStore = TerminalState & TerminalActions;

// ============================================================================
// Initial State
// ============================================================================

const initialState: TerminalState = {
  terminals: new Map(),
  activeTerminalId: null,
  terminalOrder: [],
  defaultShell: 'bash',
  defaultCwd: null,
  isMaximized: false,
};

// ============================================================================
// Utility Functions
// ============================================================================

// Use session-based counter to prevent duplicates on hot reload
let terminalCounter = 0;
const sessionId = Date.now();

function generateTerminalId(): string {
  return `terminal_${sessionId}_${++terminalCounter}`;
}

function getDefaultTitle(shell: ShellType, index: number): string {
  const shellNames: Record<ShellType, string> = {
    bash: 'bash',
    zsh: 'zsh',
    fish: 'fish',
    powershell: 'pwsh',
    cmd: 'cmd',
  };
  return `${shellNames[shell]} (${index})`;
}

// ============================================================================
// Store Implementation
// ============================================================================

export const useTerminalStore = create<TerminalStore>()(
  devtools(
    immer((set, get) => ({
      ...initialState,

      // ====================================================================
      // Terminal Management
      // ====================================================================

      createTerminal: (options?: TerminalOptions) => {
        const state = get();
        const id = generateTerminalId();
        const shell = options?.shell as ShellType || state.defaultShell;
        const index = state.terminalOrder.length + 1;

        const terminal: TerminalTab = {
          id,
          title: getDefaultTitle(shell, index),
          cwd: options?.cwd || state.defaultCwd || '/',
          isActive: true,
          shellType: shell,
        };

        set((s) => {
          // Deactivate current active terminal
          if (s.activeTerminalId) {
            const current = s.terminals.get(s.activeTerminalId);
            if (current) {
              current.isActive = false;
            }
          }

          s.terminals.set(id, terminal);
          s.terminalOrder.push(id);
          s.activeTerminalId = id;
        });

        return id;
      },

      closeTerminal: (id: string) => {
        set((state) => {
          const terminalIndex = state.terminalOrder.indexOf(id);
          if (terminalIndex === -1) return;

          state.terminals.delete(id);
          state.terminalOrder.splice(terminalIndex, 1);

          // Update active terminal if we closed the active one
          if (state.activeTerminalId === id) {
            if (state.terminalOrder.length === 0) {
              state.activeTerminalId = null;
            } else {
              // Activate the next terminal, or the previous if we closed the last
              const newIndex = Math.min(terminalIndex, state.terminalOrder.length - 1);
              const newActiveId = state.terminalOrder[newIndex];
              state.activeTerminalId = newActiveId;

              const newActiveTerminal = state.terminals.get(newActiveId);
              if (newActiveTerminal) {
                newActiveTerminal.isActive = true;
              }
            }
          }
        });
      },

      closeAllTerminals: () => {
        set((state) => {
          state.terminals.clear();
          state.terminalOrder = [];
          state.activeTerminalId = null;
        });
      },

      setActiveTerminal: (id: string) => {
        set((state) => {
          if (!state.terminals.has(id)) return;

          // Deactivate current
          if (state.activeTerminalId) {
            const current = state.terminals.get(state.activeTerminalId);
            if (current) {
              current.isActive = false;
            }
          }

          // Activate new
          const newActive = state.terminals.get(id);
          if (newActive) {
            newActive.isActive = true;
          }
          state.activeTerminalId = id;
        });
      },

      renameTerminal: (id: string, title: string) => {
        set((state) => {
          const terminal = state.terminals.get(id);
          if (terminal) {
            terminal.title = title;
          }
        });
      },

      reorderTerminals: (fromIndex: number, toIndex: number) => {
        set((state) => {
          if (
            fromIndex < 0 ||
            fromIndex >= state.terminalOrder.length ||
            toIndex < 0 ||
            toIndex >= state.terminalOrder.length
          ) {
            return;
          }

          const [removed] = state.terminalOrder.splice(fromIndex, 1);
          state.terminalOrder.splice(toIndex, 0, removed);
        });
      },

      // ====================================================================
      // Terminal State
      // ====================================================================

      setTerminalCwd: (id: string, cwd: string) => {
        set((state) => {
          const terminal = state.terminals.get(id);
          if (terminal) {
            terminal.cwd = cwd;
          }
        });
      },

      setTerminalPid: (id: string, pid: number) => {
        set((state) => {
          const terminal = state.terminals.get(id);
          if (terminal) {
            terminal.pid = pid;
          }
        });
      },

      // ====================================================================
      // Settings
      // ====================================================================

      setDefaultShell: (shell: ShellType) => {
        set((state) => {
          state.defaultShell = shell;
        });
      },

      setDefaultCwd: (cwd: string) => {
        set((state) => {
          state.defaultCwd = cwd;
        });
      },

      // ====================================================================
      // UI
      // ====================================================================

      toggleMaximize: () => {
        set((state) => {
          state.isMaximized = !state.isMaximized;
        });
      },

      setMaximized: (maximized: boolean) => {
        set((state) => {
          state.isMaximized = maximized;
        });
      },

      // ====================================================================
      // Kill Terminal
      // ====================================================================

      killTerminal: async (id: string) => {
        const state = get();
        const terminal = state.terminals.get(id);

        if (terminal?.pid) {
          try {
            // await invoke('kill_process', { pid: terminal.pid });
          } catch (error) {
            console.error('Failed to kill terminal process:', error);
          }
        }

        get().closeTerminal(id);
      },
    })),
    { name: 'mathviz-terminal-store' }
  )
);

// ============================================================================
// Selectors
// ============================================================================

export const selectActiveTerminal = (state: TerminalStore): TerminalTab | null => {
  if (!state.activeTerminalId) return null;
  return state.terminals.get(state.activeTerminalId) || null;
};

export const selectTerminalCount = (state: TerminalStore): number => {
  return state.terminalOrder.length;
};

export const selectTerminalsArray = (state: TerminalStore): TerminalTab[] => {
  return state.terminalOrder
    .map((id) => state.terminals.get(id))
    .filter((t): t is TerminalTab => t !== undefined);
};

export const selectTerminalById = (
  state: TerminalStore,
  id: string
): TerminalTab | undefined => {
  return state.terminals.get(id);
};
