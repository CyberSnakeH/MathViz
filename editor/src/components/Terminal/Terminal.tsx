/**
 * Terminal Component
 *
 * xterm.js-based terminal with Tokyo Night theme.
 * Features:
 * - Multiple terminal tabs with pill-style design
 * - Split terminals support
 * - Clear, kill process buttons
 * - Copy on select
 */

import React, { useEffect, useRef, useCallback, memo, useState } from 'react';
import { Terminal as XTerm } from 'xterm';
import { FitAddon } from 'xterm-addon-fit';
import { WebLinksAddon } from 'xterm-addon-web-links';
import { SearchAddon } from 'xterm-addon-search';
import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import { cn } from '../../utils/helpers';
import { useTerminalStore } from '../../stores/terminalStore';
import { generateXtermTheme, tokyoNightTheme } from '../../utils/themes';
import type { TerminalTab } from '../../types';

// ============================================================================
// Tauri Terminal Types
// ============================================================================

interface TerminalInfo {
  id: string;
  title: string;
  cwd: string;
  cols: number;
  rows: number;
}

interface TerminalOutputEvent {
  id: string;
  data: string;
}

interface TerminalClosedEvent {
  id: string;
}

// ============================================================================
// Types
// ============================================================================

interface TerminalProps {
  className?: string;
}

interface TerminalInstanceRef {
  terminal: XTerm;
  fitAddon: FitAddon;
  searchAddon: SearchAddon;
  ptyId: string | null;
  unlistenOutput: UnlistenFn | null;
  unlistenClosed: UnlistenFn | null;
}

// ============================================================================
// Terminal Tab Component - Pill Style
// ============================================================================

interface TerminalTabItemProps {
  tab: TerminalTab;
  isActive: boolean;
  onSelect: () => void;
  onClose: (e: React.MouseEvent) => void;
}

const TerminalTabItem: React.FC<TerminalTabItemProps> = memo(({
  tab,
  isActive,
  onSelect,
  onClose,
}) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      className={cn(
        'bottom-panel-tab',
        'flex items-center gap-2',
        'px-3 py-1',
        'rounded-full',
        'cursor-pointer select-none',
        'transition-all duration-100 ease-out',
        isActive
          ? 'bg-[var(--mviz-accent,#7aa2f7)] text-[var(--mviz-accent-foreground,#1a1b26)]'
          : 'text-[var(--mviz-foreground,#a9b1d6)] opacity-70 hover:opacity-100 hover:bg-[var(--mviz-list-hover-background,#292e42)]'
      )}
      onClick={onSelect}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Terminal icon */}
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <polyline points="4 17 10 11 4 5" />
        <line x1="12" y1="19" x2="20" y2="19" />
      </svg>

      {/* Title */}
      <span className="text-xs font-medium truncate max-w-[80px]">{tab.title}</span>

      {/* Close button */}
      <button
        className={cn(
          'w-4 h-4 flex items-center justify-center rounded-full',
          'transition-all duration-100',
          isActive
            ? 'hover:bg-black/20'
            : 'hover:bg-[var(--mviz-list-hover-background,#292e42)]',
          (isHovered || isActive) ? 'opacity-100' : 'opacity-0'
        )}
        onClick={onClose}
        title="Close terminal"
      >
        <svg className="w-2.5 h-2.5" viewBox="0 0 16 16" fill="currentColor">
          <path d="M8 8.707l3.646 3.647.708-.707L8.707 8l3.647-3.646-.707-.708L8 7.293 4.354 3.646l-.707.708L7.293 8l-3.646 3.646.707.708L8 8.707z" />
        </svg>
      </button>
    </div>
  );
});

TerminalTabItem.displayName = 'TerminalTabItem';

// ============================================================================
// Terminal Instance Component - Connected to Real PTY via Tauri
// ============================================================================

interface TerminalInstanceProps {
  tabId: string;
  isActive: boolean;
}

const TerminalInstance: React.FC<TerminalInstanceProps> = memo(({ tabId, isActive }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const instanceRef = useRef<TerminalInstanceRef | null>(null);
  const renameTerminal = useTerminalStore((state) => state.renameTerminal);
  const closeTerminal = useTerminalStore((state) => state.closeTerminal);

  useEffect(() => {
    if (!containerRef.current) return;

    let isMounted = true;

    // Generate Tokyo Night terminal theme
    const xtermTheme = generateXtermTheme(tokyoNightTheme);

    // Create terminal instance with Tokyo Night theme
    const terminal = new XTerm({
      theme: xtermTheme,
      fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace",
      fontSize: 13,
      lineHeight: 1.4,
      cursorBlink: true,
      cursorStyle: 'bar',
      scrollback: 10000,
      allowProposedApi: true,
    });

    // Create addons
    const fitAddon = new FitAddon();
    const webLinksAddon = new WebLinksAddon();
    const searchAddon = new SearchAddon();

    terminal.loadAddon(fitAddon);
    terminal.loadAddon(webLinksAddon);
    terminal.loadAddon(searchAddon);

    // Open terminal in DOM
    terminal.open(containerRef.current);

    // Initial fit
    fitAddon.fit();

    // Store instance (PTY connection will be set up async)
    instanceRef.current = {
      terminal,
      fitAddon,
      searchAddon,
      ptyId: null,
      unlistenOutput: null,
      unlistenClosed: null,
    };

    // Initialize PTY connection
    const initPty = async () => {
      try {
        const { cols, rows } = terminal;

        // Create PTY via Tauri backend
        const ptyInfo = await invoke<TerminalInfo>('create_terminal', {
          cols,
          rows,
        });

        if (!isMounted || !instanceRef.current) return;

        instanceRef.current.ptyId = ptyInfo.id;

        // Update terminal title in store
        renameTerminal(tabId, ptyInfo.title);

        // Listen for terminal output from Tauri
        const unlistenOutput = await listen<TerminalOutputEvent>('terminal-output', (event) => {
          if (event.payload.id === ptyInfo.id && instanceRef.current) {
            instanceRef.current.terminal.write(event.payload.data);
          }
        });

        if (!isMounted || !instanceRef.current) {
          unlistenOutput();
          return;
        }
        instanceRef.current.unlistenOutput = unlistenOutput;

        // Listen for terminal closed event
        const unlistenClosed = await listen<TerminalClosedEvent>('terminal-closed', (event) => {
          if (event.payload.id === ptyInfo.id) {
            closeTerminal(tabId);
          }
        });

        if (!isMounted || !instanceRef.current) {
          unlistenClosed();
          return;
        }
        instanceRef.current.unlistenClosed = unlistenClosed;

        // Handle user input - send to PTY
        terminal.onData(async (data) => {
          if (instanceRef.current?.ptyId) {
            try {
              await invoke('write_terminal', {
                id: instanceRef.current.ptyId,
                data,
              });
            } catch (error) {
              console.error('Failed to write to terminal:', error);
            }
          }
        });

      } catch (error) {
        console.error('Failed to create PTY:', error);
        // Fallback: show error message in terminal
        if (instanceRef.current) {
          terminal.writeln('\x1b[38;5;210m  Terminal Connection Failed\x1b[0m');
          terminal.writeln('');
          terminal.writeln(`\x1b[38;5;103m  Error: ${error}\x1b[0m`);
          terminal.writeln('');
          terminal.writeln('\x1b[38;5;103m  Running in offline mode.\x1b[0m');
          terminal.writeln('\x1b[38;5;103m  Terminal commands will not execute.\x1b[0m');
          terminal.writeln('');
          terminal.write('\x1b[38;5;110m$ \x1b[0m');
        }
      }
    };

    initPty();

    // Handle resize - notify PTY of new dimensions
    const handleResize = async () => {
      if (!instanceRef.current) return;

      fitAddon.fit();

      const { cols, rows } = terminal;
      const ptyId = instanceRef.current.ptyId;

      if (ptyId) {
        try {
          await invoke('resize_terminal', {
            id: ptyId,
            cols,
            rows,
          });
        } catch (error) {
          console.error('Failed to resize terminal:', error);
        }
      }
    };

    const resizeObserver = new ResizeObserver(() => {
      handleResize();
    });
    resizeObserver.observe(containerRef.current);

    // Cleanup
    return () => {
      isMounted = false;
      resizeObserver.disconnect();

      // Close PTY connection
      const cleanupPty = async () => {
        if (instanceRef.current) {
          if (instanceRef.current.unlistenOutput) {
            instanceRef.current.unlistenOutput();
          }
          if (instanceRef.current.unlistenClosed) {
            instanceRef.current.unlistenClosed();
          }

          if (instanceRef.current.ptyId) {
            try {
              await invoke('close_terminal', { id: instanceRef.current.ptyId });
            } catch (error) {
              console.error('Failed to close terminal:', error);
            }
          }
        }
      };

      cleanupPty();
      terminal.dispose();
      instanceRef.current = null;
    };
  }, [tabId, renameTerminal, closeTerminal]);

  // Fit and focus on visibility change
  useEffect(() => {
    if (isActive && instanceRef.current) {
      instanceRef.current.fitAddon.fit();
      instanceRef.current.terminal.focus();
    }
  }, [isActive]);

  return (
    <div
      ref={containerRef}
      className={cn(
        'w-full h-full',
        isActive ? 'block' : 'hidden'
      )}
    />
  );
});

TerminalInstance.displayName = 'TerminalInstance';

// ============================================================================
// Terminal Panel Component
// ============================================================================

export const Terminal: React.FC<TerminalProps> = ({ className }) => {
  const terminals = useTerminalStore((state) =>
    state.terminalOrder.map((id) => state.terminals.get(id)).filter(Boolean) as TerminalTab[]
  );
  const activeTerminalId = useTerminalStore((state) => state.activeTerminalId);
  const createTerminal = useTerminalStore((state) => state.createTerminal);
  const closeTerminal = useTerminalStore((state) => state.closeTerminal);
  const setActiveTerminal = useTerminalStore((state) => state.setActiveTerminal);
  const isMaximized = useTerminalStore((state) => state.isMaximized);
  const toggleMaximize = useTerminalStore((state) => state.toggleMaximize);

  // Ref to track if initial terminal was created (prevents React Strict Mode double-creation)
  const initialTerminalCreated = useRef(false);

  // Create initial terminal if none exists (only once)
  useEffect(() => {
    if (terminals.length === 0 && !initialTerminalCreated.current) {
      initialTerminalCreated.current = true;
      createTerminal();
    }
  }, [terminals.length, createTerminal]);

  const handleCreateTerminal = useCallback(() => {
    createTerminal();
  }, [createTerminal]);

  const handleCloseTerminal = useCallback(
    (e: React.MouseEvent, id: string) => {
      e.stopPropagation();
      closeTerminal(id);
    },
    [closeTerminal]
  );

  const handleSelectTerminal = useCallback(
    (id: string) => {
      setActiveTerminal(id);
    },
    [setActiveTerminal]
  );

  return (
    <div className={cn('flex flex-col h-full', className)}>
      {/* Tab bar with pill-style tabs */}
      <div
        className={cn(
          'bottom-panel-tabs',
          'flex items-center justify-between',
          'bg-[var(--mviz-sidebar-background,#1f2335)]',
          'border-b border-[var(--mviz-border,#1a1b26)]',
          'px-2 py-1.5'
        )}
      >
        {/* Tabs */}
        <div className="flex items-center gap-1 overflow-x-auto scrollbar-none">
          {terminals.map((tab) => (
            <TerminalTabItem
              key={tab.id}
              tab={tab}
              isActive={tab.id === activeTerminalId}
              onSelect={() => handleSelectTerminal(tab.id)}
              onClose={(e) => handleCloseTerminal(e, tab.id)}
            />
          ))}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1 ml-2">
          {/* New terminal */}
          <button
            className={cn(
              'p-1.5 rounded',
              'text-[var(--mviz-foreground,#a9b1d6)] opacity-60',
              'hover:opacity-100 hover:bg-[var(--mviz-list-hover-background,#292e42)]',
              'transition-all duration-100'
            )}
            onClick={handleCreateTerminal}
            title="New Terminal (Ctrl+Shift+`)"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
          </button>

          {/* Split terminal */}
          <button
            className={cn(
              'p-1.5 rounded',
              'text-[var(--mviz-foreground,#a9b1d6)] opacity-60',
              'hover:opacity-100 hover:bg-[var(--mviz-list-hover-background,#292e42)]',
              'transition-all duration-100'
            )}
            title="Split Terminal"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="18" height="18" rx="2" />
              <line x1="12" y1="3" x2="12" y2="21" />
            </svg>
          </button>

          {/* Clear terminal */}
          <button
            className={cn(
              'p-1.5 rounded',
              'text-[var(--mviz-foreground,#a9b1d6)] opacity-60',
              'hover:opacity-100 hover:bg-[var(--mviz-list-hover-background,#292e42)]',
              'transition-all duration-100'
            )}
            title="Clear Terminal"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6" />
            </svg>
          </button>

          {/* Maximize/restore */}
          <button
            className={cn(
              'p-1.5 rounded',
              'text-[var(--mviz-foreground,#a9b1d6)] opacity-60',
              'hover:opacity-100 hover:bg-[var(--mviz-list-hover-background,#292e42)]',
              'transition-all duration-100'
            )}
            onClick={toggleMaximize}
            title={isMaximized ? 'Restore Panel Size' : 'Maximize Panel'}
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              {isMaximized ? (
                <>
                  <polyline points="4 14 10 14 10 20" />
                  <polyline points="20 10 14 10 14 4" />
                  <line x1="14" y1="10" x2="21" y2="3" />
                  <line x1="3" y1="21" x2="10" y2="14" />
                </>
              ) : (
                <>
                  <polyline points="15 3 21 3 21 9" />
                  <polyline points="9 21 3 21 3 15" />
                  <line x1="21" y1="3" x2="14" y2="10" />
                  <line x1="3" y1="21" x2="10" y2="14" />
                </>
              )}
            </svg>
          </button>
        </div>
      </div>

      {/* Terminal instances */}
      <div
        className={cn(
          'flex-1 overflow-hidden',
          'bg-[var(--mviz-terminal-background,#1a1b26)]'
        )}
      >
        {terminals.map((tab) => (
          <TerminalInstance
            key={tab.id}
            tabId={tab.id}
            isActive={tab.id === activeTerminalId}
          />
        ))}
      </div>
    </div>
  );
};

// ============================================================================
// Export
// ============================================================================

export default Terminal;
