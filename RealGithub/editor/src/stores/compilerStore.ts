/**
 * Compiler State Management
 *
 * Zustand store for managing MathViz compilation and execution.
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { invoke, convertFileSrc } from '@tauri-apps/api/core';
import { usePreviewStore } from './previewStore';

// ============================================================================
// Types
// ============================================================================

export interface CompileError {
  line: number;
  column: number;
  message: string;
  severity: string;
  code?: string;
}

export interface CompileWarning {
  line: number;
  column: number;
  message: string;
  code?: string;
}

export interface CompileResult {
  success: boolean;
  outputPath?: string;
  stdout: string;
  stderr: string;
  errors: CompileError[];
  warnings: CompileWarning[];
}

export interface RunResult {
  success: boolean;
  stdout: string;
  stderr: string;
  exitCode?: number;
  videoPath?: string;
}

export interface MathVizInstallation {
  found: boolean;
  method?: string;  // 'PATH' | 'pipx' | 'uv' | 'pip'
  command?: string;
  version?: string;
}

export type CompilerStatus = 'idle' | 'compiling' | 'running' | 'success' | 'error';

interface CompilerState {
  status: CompilerStatus;
  lastResult: CompileResult | null;
  lastRunResult: RunResult | null;
  output: string[];
  isDebugging: boolean;
  breakpoints: Map<string, number[]>; // filePath -> line numbers
  installation: MathVizInstallation | null;
  installationChecked: boolean;
}

interface CompilerActions {
  // Installation check
  checkInstallation: () => Promise<MathVizInstallation>;

  // Compilation
  compile: (sourcePath: string, outputDir?: string) => Promise<CompileResult>;
  run: (sourcePath: string, preview?: boolean, quality?: string) => Promise<RunResult>;
  checkSyntax: (code: string, filePath?: string) => Promise<{ valid: boolean; errors: CompileError[]; warnings: CompileWarning[] }>;
  formatCode: (code: string) => Promise<{ success: boolean; formattedCode?: string; error?: string }>;

  // Output
  appendOutput: (line: string) => void;
  clearOutput: () => void;

  // Debug
  setDebugging: (debugging: boolean) => void;
  toggleBreakpoint: (filePath: string, line: number) => void;
  clearBreakpoints: (filePath?: string) => void;

  // Status
  setStatus: (status: CompilerStatus) => void;
  reset: () => void;
}

type CompilerStore = CompilerState & CompilerActions;

// ============================================================================
// Initial State
// ============================================================================

const initialState: CompilerState = {
  status: 'idle',
  lastResult: null,
  lastRunResult: null,
  output: [],
  isDebugging: false,
  breakpoints: new Map(),
  installation: null,
  installationChecked: false,
};

// ============================================================================
// Store Implementation
// ============================================================================

export const useCompilerStore = create<CompilerStore>()(
  devtools(
    immer((set) => ({
      ...initialState,

      // ====================================================================
      // Installation Check
      // ====================================================================

      checkInstallation: async () => {
        try {
          const result = await invoke<MathVizInstallation>('check_mathviz_installation');
          set((state) => {
            state.installation = result;
            state.installationChecked = true;
          });
          return result;
        } catch (error) {
          const errorResult: MathVizInstallation = {
            found: false,
          };
          set((state) => {
            state.installation = errorResult;
            state.installationChecked = true;
          });
          return errorResult;
        }
      },

      // ====================================================================
      // Compilation
      // ====================================================================

      compile: async (sourcePath: string, outputDir?: string) => {
        set((state) => {
          state.status = 'compiling';
          state.output.push(`[Compiling] ${sourcePath}...`);
        });

        try {
          const result = await invoke<CompileResult>('compile_mathviz', {
            sourcePath,
            outputDir,
            options: null,
          });

          set((state) => {
            state.lastResult = result;
            state.status = result.success ? 'success' : 'error';

            // Show stderr output first
            if (result.stderr && result.stderr.trim()) {
              const lines = result.stderr.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trim().split('\n');
              lines.forEach((line: string) => {
                if (line.trim()) {
                  if (line.includes('ERROR') || line.includes('Error')) {
                    state.output.push(`[Error] ${line.trim()}`);
                  } else {
                    state.output.push(line);
                  }
                }
              });
            }

            // Show stdout output
            if (result.stdout && result.stdout.trim()) {
              const lines = result.stdout.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trim().split('\n');
              lines.forEach((line: string) => {
                if (line.trim()) {
                  state.output.push(line);
                }
              });
            }

            if (result.success) {
              state.output.push('');
              state.output.push(`[Success] Compiled to ${result.outputPath}`);
            } else {
              state.output.push(`[Error] Compilation failed`);
              result.errors.forEach((err) => {
                state.output.push(`  Line ${err.line}:${err.column}: ${err.message}`);
              });
            }

            if (result.warnings.length > 0) {
              result.warnings.forEach((warn) => {
                state.output.push(`  [Warning] Line ${warn.line}:${warn.column}: ${warn.message}`);
              });
            }
          });

          return result;
        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : String(error);
          set((state) => {
            state.status = 'error';
            state.output.push(`[Error] ${errorMsg}`);
          });

          return {
            success: false,
            stdout: '',
            stderr: errorMsg,
            errors: [{ line: 0, column: 0, message: errorMsg, severity: 'error' }],
            warnings: [],
          };
        }
      },

      run: async (sourcePath: string, preview = true, quality = 'm') => {
        set((state) => {
          state.status = 'running';
          state.output.push(`[Running] ${sourcePath}...`);
        });

        try {
          const result = await invoke<RunResult>('run_mathviz', {
            sourcePath,
            preview,
            quality,
          });

          set((state) => {
            state.lastRunResult = result;
            state.status = result.success ? 'success' : 'error';

            // Show stderr output first (contains Manim version info, etc.)
            if (result.stderr && result.stderr.trim()) {
              const text = result.stderr.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trim();
              // Split by newlines and process each line
              text.split('\n').forEach((line: string) => {
                // Color ERROR lines in red
                if (line.includes('ERROR') || line.includes('Error')) {
                  state.output.push(`[Error] ${line}`);
                } else if (line.trim()) {
                  state.output.push(line);
                }
              });
            }

            // Show stdout output
            if (result.stdout && result.stdout.trim()) {
              const text = result.stdout.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trim();
              text.split('\n').forEach((line: string) => {
                if (line.trim()) {
                  state.output.push(`[stdout] ${line}`);
                }
              });
            }

            state.output.push('');
            if (result.success) {
              state.output.push(`[Success] Execution completed`);
              if (result.videoPath) {
                state.output.push(`  Video: ${result.videoPath}`);
              }
            } else {
              state.output.push(`[Error] Execution failed (exit code: ${result.exitCode})`);
            }
          });

          // Update preview store with video if available
          if (result.success && result.videoPath) {
            const videoUrl = convertFileSrc(result.videoPath);
            usePreviewStore.getState().setRenderComplete(videoUrl);
          }

          return result;
        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : String(error);
          set((state) => {
            state.status = 'error';
            state.output.push(`[Error] ${errorMsg}`);
          });

          return {
            success: false,
            stdout: '',
            stderr: errorMsg,
            exitCode: 1,
          };
        }
      },

      checkSyntax: async (code: string, filePath?: string) => {
        try {
          const result = await invoke<{ valid: boolean; errors: CompileError[]; warnings: CompileWarning[] }>('check_syntax', {
            code,
            filePath,
          });
          return result;
        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : String(error);
          return {
            valid: false,
            errors: [{ line: 0, column: 0, message: errorMsg, severity: 'error' }],
            warnings: [],
          };
        }
      },

      formatCode: async (code: string) => {
        try {
          const result = await invoke<{ success: boolean; formattedCode?: string; error?: string }>('format_code', {
            code,
          });
          return result;
        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : String(error);
          return {
            success: false,
            error: errorMsg,
          };
        }
      },

      // ====================================================================
      // Output
      // ====================================================================

      appendOutput: (line: string) => {
        set((state) => {
          state.output.push(line);
          // Keep last 1000 lines
          if (state.output.length > 1000) {
            state.output = state.output.slice(-1000);
          }
        });
      },

      clearOutput: () => {
        set((state) => {
          state.output = [];
        });
      },

      // ====================================================================
      // Debug
      // ====================================================================

      setDebugging: (debugging: boolean) => {
        set((state) => {
          state.isDebugging = debugging;
        });
      },

      toggleBreakpoint: (filePath: string, line: number) => {
        set((state) => {
          const breakpoints = state.breakpoints.get(filePath) || [];
          const index = breakpoints.indexOf(line);

          if (index === -1) {
            breakpoints.push(line);
            breakpoints.sort((a, b) => a - b);
          } else {
            breakpoints.splice(index, 1);
          }

          state.breakpoints.set(filePath, breakpoints);
        });
      },

      clearBreakpoints: (filePath?: string) => {
        set((state) => {
          if (filePath) {
            state.breakpoints.delete(filePath);
          } else {
            state.breakpoints.clear();
          }
        });
      },

      // ====================================================================
      // Status
      // ====================================================================

      setStatus: (status: CompilerStatus) => {
        set((state) => {
          state.status = status;
        });
      },

      reset: () => {
        set((state) => {
          state.status = 'idle';
          state.lastResult = null;
          state.lastRunResult = null;
          state.output = [];
        });
      },
    })),
    { name: 'mathviz-compiler-store' }
  )
);

// ============================================================================
// Selectors
// ============================================================================

export const selectCompilerStatus = (state: CompilerStore) => state.status;
export const selectCompilerOutput = (state: CompilerStore) => state.output;
export const selectIsCompiling = (state: CompilerStore) => state.status === 'compiling';
export const selectIsRunning = (state: CompilerStore) => state.status === 'running';
export const selectBreakpoints = (state: CompilerStore, filePath: string) =>
  state.breakpoints.get(filePath) || [];
export const selectInstallation = (state: CompilerStore) => state.installation;
export const selectInstallationChecked = (state: CompilerStore) => state.installationChecked;
