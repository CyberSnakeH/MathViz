/**
 * useLSP Hook
 *
 * Custom hook for Language Server Protocol integration.
 * Provides completion, hover, diagnostics, and other LSP features.
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import type {
  CompletionItem,
  Diagnostic,
  HoverInfo,
  Location,
  LspStatus,
} from '../types';

interface UseLSPOptions {
  autoStart?: boolean;
  serverPath?: string;
}

export function useLSP(options: UseLSPOptions = {}) {
  const { autoStart: _autoStart = false, serverPath: _serverPath } = options;

  const [status, _setStatus] = useState<LspStatus>('stopped');
  const [error, setError] = useState<string | null>(null);
  const [diagnostics, setDiagnostics] = useState<Map<string, Diagnostic[]>>(new Map());

  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const documentVersionsRef = useRef<Map<string, number>>(new Map());

  // Get completions at position
  const getCompletions = useCallback(
    async (
      code: string,
      line: number,
      column: number,
      filePath?: string
    ): Promise<CompletionItem[]> => {
      try {
        const result = await invoke<
          Array<{
            label: string;
            kind: string;
            detail: string | null;
            documentation: string | null;
            insert_text: string;
          }>
        >('get_completions', {
          code,
          line,
          column,
          filePath,
        });

        return result.map((item: { label: string; kind: string; detail: string | null; documentation: string | null; insert_text: string }) => ({
          label: item.label,
          kind: item.kind as CompletionItem['kind'],
          detail: item.detail || undefined,
          documentation: item.documentation || undefined,
          insertText: item.insert_text,
        }));
      } catch (err) {
        console.error('Failed to get completions:', err);
        return [];
      }
    },
    []
  );

  // Check syntax
  const checkSyntax = useCallback(
    async (code: string, filePath?: string): Promise<Diagnostic[]> => {
      try {
        const result = await invoke<{
          valid: boolean;
          errors: Array<{
            line: number;
            column: number;
            message: string;
            severity: string;
            code: string | null;
          }>;
          warnings: Array<{
            line: number;
            column: number;
            message: string;
            code: string | null;
          }>;
        }>('check_syntax', {
          code,
          filePath,
        });

        type ErrorItem = { line: number; column: number; message: string; severity: string; code: string | null };
        type WarningItem = { line: number; column: number; message: string; code: string | null };

        const diagnosticList: Diagnostic[] = [
          ...result.errors.map((e: ErrorItem, i: number) => ({
            id: `error-${i}`,
            filePath: filePath || 'unknown',
            severity: 'error' as const,
            message: e.message,
            source: 'mathviz',
            code: e.code || undefined,
            range: {
              startLine: e.line,
              startColumn: e.column,
              endLine: e.line,
              endColumn: e.column + 1,
            },
          })),
          ...result.warnings.map((w: WarningItem, i: number) => ({
            id: `warning-${i}`,
            filePath: filePath || 'unknown',
            severity: 'warning' as const,
            message: w.message,
            source: 'mathviz',
            code: w.code || undefined,
            range: {
              startLine: w.line,
              startColumn: w.column,
              endLine: w.line,
              endColumn: w.column + 1,
            },
          })),
        ];

        if (filePath) {
          setDiagnostics((prev) => {
            const next = new Map(prev);
            next.set(filePath, diagnosticList);
            return next;
          });
        }

        return diagnosticList;
      } catch (err) {
        console.error('Failed to check syntax:', err);
        return [];
      }
    },
    []
  );

  // Check syntax with debounce
  const checkSyntaxDebounced = useCallback(
    (code: string, filePath?: string, delay: number = 500) => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }

      debounceTimerRef.current = setTimeout(() => {
        checkSyntax(code, filePath);
      }, delay);
    },
    [checkSyntax]
  );

  // Format code
  const formatCode = useCallback(async (code: string): Promise<string | null> => {
    try {
      const result = await invoke<{
        success: boolean;
        formatted_code: string | null;
        error: string | null;
      }>('format_code', { code });

      if (result.success && result.formatted_code) {
        return result.formatted_code;
      }

      if (result.error) {
        setError(result.error);
      }

      return null;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(`Format failed: ${message}`);
      return null;
    }
  }, []);

  // Get hover information (placeholder - would be implemented with real LSP)
  const getHoverInfo = useCallback(
    async (
      code: string,
      line: number,
      column: number,
      _filePath?: string
    ): Promise<HoverInfo | null> => {
      // For now, return built-in documentation based on word at cursor
      const lines = code.split('\n');
      if (line < 0 || line >= lines.length) return null;

      const lineContent = lines[line];
      if (column < 0 || column >= lineContent.length) return null;

      // Extract word at cursor
      let start = column;
      let end = column;

      while (start > 0 && /\w/.test(lineContent[start - 1])) start--;
      while (end < lineContent.length && /\w/.test(lineContent[end])) end++;

      const word = lineContent.slice(start, end);

      // Built-in documentation
      const docs: Record<string, string> = {
        Scene: '**Scene**\n\nBase class for all MathViz scenes.\n\n```python\nclass MyScene(Scene):\n    def construct(self):\n        ...\n```',
        Circle: '**Circle**\n\nCreate a circular shape.\n\n**Parameters:**\n- `radius`: The radius of the circle (default: 1.0)\n- `color`: The color of the circle',
        Square: '**Square**\n\nCreate a square shape.\n\n**Parameters:**\n- `side_length`: The side length (default: 2.0)\n- `color`: The color of the square',
        Text: '**Text**\n\nCreate a text object.\n\n**Parameters:**\n- `text`: The text content\n- `font_size`: Size of the font',
        MathTex: '**MathTex**\n\nRender LaTeX mathematical expressions.\n\n**Example:**\n```python\nMathTex(r"\\int_0^1 x^2 dx")\n```',
        play: '**play(animation)**\n\nPlay an animation in the scene.',
        wait: '**wait(duration)**\n\nPause the scene for the specified duration in seconds.',
        FadeIn: '**FadeIn(mobject)**\n\nFade in animation for a mobject.',
        FadeOut: '**FadeOut(mobject)**\n\nFade out animation for a mobject.',
        Transform: '**Transform(source, target)**\n\nMorph one mobject into another.',
        Create: '**Create(mobject)**\n\nAnimation that draws the mobject.',
        Write: '**Write(text)**\n\nAnimation for writing text character by character.',
      };

      if (word in docs) {
        return {
          contents: docs[word],
          range: {
            startLine: line,
            startColumn: start,
            endLine: line,
            endColumn: end,
          },
        };
      }

      return null;
    },
    []
  );

  // Go to definition (placeholder)
  const getDefinition = useCallback(
    async (
      _code: string,
      _line: number,
      _column: number,
      _filePath?: string
    ): Promise<Location | null> => {
      // Would be implemented with real LSP
      return null;
    },
    []
  );

  // Find references (placeholder)
  const findReferences = useCallback(
    async (
      _code: string,
      _line: number,
      _column: number,
      _filePath?: string
    ): Promise<Location[]> => {
      // Would be implemented with real LSP
      return [];
    },
    []
  );

  // Get diagnostics for a file
  const getDiagnostics = useCallback(
    (filePath: string): Diagnostic[] => {
      return diagnostics.get(filePath) || [];
    },
    [diagnostics]
  );

  // Clear diagnostics for a file
  const clearDiagnostics = useCallback((filePath: string) => {
    setDiagnostics((prev) => {
      const next = new Map(prev);
      next.delete(filePath);
      return next;
    });
  }, []);

  // Document opened
  const documentOpened = useCallback(
    (filePath: string, content: string) => {
      documentVersionsRef.current.set(filePath, 1);
      checkSyntaxDebounced(content, filePath, 0);
    },
    [checkSyntaxDebounced]
  );

  // Document changed
  const documentChanged = useCallback(
    (filePath: string, content: string) => {
      const version = (documentVersionsRef.current.get(filePath) || 0) + 1;
      documentVersionsRef.current.set(filePath, version);
      checkSyntaxDebounced(content, filePath);
    },
    [checkSyntaxDebounced]
  );

  // Document closed
  const documentClosed = useCallback(
    (filePath: string) => {
      documentVersionsRef.current.delete(filePath);
      clearDiagnostics(filePath);
    },
    [clearDiagnostics]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, []);

  return {
    // State
    status,
    error,
    diagnostics,

    // LSP operations
    getCompletions,
    checkSyntax,
    checkSyntaxDebounced,
    formatCode,
    getHoverInfo,
    getDefinition,
    findReferences,

    // Document operations
    documentOpened,
    documentChanged,
    documentClosed,

    // Diagnostics
    getDiagnostics,
    clearDiagnostics,

    // Utilities
    clearError: () => setError(null),
  };
}

export default useLSP;
