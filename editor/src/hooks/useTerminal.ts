/**
 * useTerminal Hook
 *
 * Custom hook for managing terminal instances and PTY communication.
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import { Terminal as XTerm } from 'xterm';
import { FitAddon } from 'xterm-addon-fit';
import { WebLinksAddon } from 'xterm-addon-web-links';
import { useTerminalStore } from '../stores/terminalStore';
import type { TerminalDimensions } from '../types';

interface TerminalInfo {
  id: string;
  title: string;
  cwd: string;
  cols: number;
  rows: number;
}

interface TerminalOutput {
  id: string;
  data: string;
}

export function useTerminal() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const terminalsRef = useRef<Map<string, { xterm: XTerm; fitAddon: FitAddon }>>(new Map());
  const containerRefsRef = useRef<Map<string, HTMLDivElement>>(new Map());

  const terminals = useTerminalStore((state) => state.terminals);
  const activeTerminalId = useTerminalStore((state) => state.activeTerminalId);
  const createTerminalStore = useTerminalStore((state) => state.createTerminal);
  const closeTerminalStore = useTerminalStore((state) => state.closeTerminal);
  const setActiveTerminal = useTerminalStore((state) => state.setActiveTerminal);

  // Create XTerm instance
  const createXTermInstance = useCallback((): { xterm: XTerm; fitAddon: FitAddon } => {
    const xterm = new XTerm({
      cursorBlink: true,
      cursorStyle: 'bar',
      fontSize: 13,
      fontFamily: 'JetBrains Mono, Fira Code, monospace',
      theme: {
        background: '#1e1e1e',
        foreground: '#cccccc',
        cursor: '#cccccc',
        cursorAccent: '#1e1e1e',
        selectionBackground: 'rgba(124, 58, 237, 0.4)',
        black: '#1e1e1e',
        red: '#f14c4c',
        green: '#4ec9b0',
        yellow: '#dcdcaa',
        blue: '#569cd6',
        magenta: '#c586c0',
        cyan: '#4ec9b0',
        white: '#cccccc',
        brightBlack: '#808080',
        brightRed: '#f14c4c',
        brightGreen: '#4ec9b0',
        brightYellow: '#dcdcaa',
        brightBlue: '#569cd6',
        brightMagenta: '#c586c0',
        brightCyan: '#4ec9b0',
        brightWhite: '#ffffff',
      },
      allowProposedApi: true,
    });

    const fitAddon = new FitAddon();
    xterm.loadAddon(fitAddon);

    const webLinksAddon = new WebLinksAddon();
    xterm.loadAddon(webLinksAddon);

    return { xterm, fitAddon };
  }, []);

  // Create new terminal
  const createTerminal = useCallback(
    async (cwd?: string): Promise<string> => {
      try {
        setIsLoading(true);

        const { xterm, fitAddon } = createXTermInstance();

        // Get initial dimensions
        const cols = 80;
        const rows = 24;

        const info = await invoke<TerminalInfo>('create_terminal', {
          cwd,
          cols,
          rows,
        });

        // Store terminal instance
        terminalsRef.current.set(info.id, { xterm, fitAddon });

        // Create terminal in store using the returned id from Tauri
        createTerminalStore({ cwd: info.cwd });
        setActiveTerminal(info.id);

        // Set up input handler
        xterm.onData((data: string) => {
          writeToTerminal(info.id, data);
        });

        return info.id;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(`Failed to create terminal: ${message}`);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [createXTermInstance, createTerminalStore, setActiveTerminal]
  );

  // Attach terminal to DOM element
  const attachTerminal = useCallback(
    (terminalId: string, container: HTMLDivElement) => {
      const instance = terminalsRef.current.get(terminalId);
      if (!instance) return;

      const { xterm, fitAddon } = instance;

      // Store container reference
      containerRefsRef.current.set(terminalId, container);

      // Open terminal in container
      xterm.open(container);
      fitAddon.fit();

      // Focus terminal
      xterm.focus();
    },
    []
  );

  // Write to terminal
  const writeToTerminal = useCallback(async (terminalId: string, data: string) => {
    try {
      await invoke('write_terminal', { id: terminalId, data });
    } catch (err) {
      console.error('Failed to write to terminal:', err);
    }
  }, []);

  // Resize terminal
  const resizeTerminal = useCallback(
    async (terminalId: string, dimensions?: TerminalDimensions) => {
      const instance = terminalsRef.current.get(terminalId);
      if (!instance) return;

      const { xterm, fitAddon } = instance;

      // Fit to container
      fitAddon.fit();

      const cols = dimensions?.cols ?? xterm.cols;
      const rows = dimensions?.rows ?? xterm.rows;

      try {
        await invoke('resize_terminal', { id: terminalId, cols, rows });
      } catch (err) {
        console.error('Failed to resize terminal:', err);
      }
    },
    []
  );

  // Close terminal
  const closeTerminal = useCallback(
    async (terminalId: string) => {
      try {
        await invoke('close_terminal', { id: terminalId });

        // Dispose XTerm instance
        const instance = terminalsRef.current.get(terminalId);
        if (instance) {
          instance.xterm.dispose();
          terminalsRef.current.delete(terminalId);
        }

        containerRefsRef.current.delete(terminalId);
        closeTerminalStore(terminalId);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(`Failed to close terminal: ${message}`);
      }
    },
    [closeTerminalStore]
  );

  // Clear terminal
  const clearTerminal = useCallback((terminalId: string) => {
    const instance = terminalsRef.current.get(terminalId);
    if (instance) {
      instance.xterm.clear();
    }
  }, []);

  // Focus terminal
  const focusTerminal = useCallback((terminalId: string) => {
    const instance = terminalsRef.current.get(terminalId);
    if (instance) {
      instance.xterm.focus();
    }
    setActiveTerminal(terminalId);
  }, [setActiveTerminal]);

  // Send command to terminal
  const sendCommand = useCallback(
    async (terminalId: string, command: string) => {
      await writeToTerminal(terminalId, command + '\r');
    },
    [writeToTerminal]
  );

  // Get active terminal
  const getActiveTerminal = useCallback(() => {
    if (!activeTerminalId) return null;
    return terminals.get(activeTerminalId) || null;
  }, [terminals, activeTerminalId]);

  // Listen for terminal output
  useEffect(() => {
    let outputUnlisten: UnlistenFn | undefined;
    let closedUnlisten: UnlistenFn | undefined;

    const setupListeners = async () => {
      outputUnlisten = await listen<TerminalOutput>('terminal-output', (event) => {
        const { id, data } = event.payload;
        const instance = terminalsRef.current.get(id);
        if (instance) {
          instance.xterm.write(data);
        }
      });

      closedUnlisten = await listen<{ id: string }>('terminal-closed', (event) => {
        const { id } = event.payload;
        const instance = terminalsRef.current.get(id);
        if (instance) {
          instance.xterm.dispose();
          terminalsRef.current.delete(id);
        }
        containerRefsRef.current.delete(id);
        closeTerminalStore(id);
      });
    };

    setupListeners();

    return () => {
      outputUnlisten?.();
      closedUnlisten?.();
    };
  }, [closeTerminalStore]);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      for (const [terminalId] of terminalsRef.current) {
        resizeTerminal(terminalId);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [resizeTerminal]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      for (const [, instance] of terminalsRef.current) {
        instance.xterm.dispose();
      }
      terminalsRef.current.clear();
    };
  }, []);

  return {
    // State
    terminals,
    activeTerminalId,
    isLoading,
    error,

    // Terminal operations
    createTerminal,
    attachTerminal,
    closeTerminal,
    clearTerminal,
    focusTerminal,
    resizeTerminal,
    sendCommand,
    writeToTerminal,

    // Utilities
    getActiveTerminal,
    clearError: () => setError(null),
  };
}

export default useTerminal;
