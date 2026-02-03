/**
 * MathViz Editor Application
 *
 * Main application component with:
 * - Tokyo Night theme
 * - Unique UI design (not VS Code clone)
 * - Command palette with fuzzy search
 * - Professional welcome screen
 */

import React, { useEffect, useCallback, useState } from 'react';
import { open } from '@tauri-apps/plugin-dialog';
import { invoke } from '@tauri-apps/api/core';
import { MainLayout } from './components/Layout';
import { Editor } from './components/Editor';
import { FileTree } from './components/FileTree';
import { Terminal } from './components/Terminal';
import { ManimPreview } from './components/Preview';
import { DebugPanel } from './components/Sidebar';
import { StatusBar } from './components/StatusBar';
import { useLayoutStore } from './stores/layoutStore';
import { useFileStore } from './stores/fileStore';
import { useEditorStore, selectActiveFile } from './stores/editorStore';
import { useCompilerStore, selectInstallation, selectInstallationChecked } from './stores/compilerStore';
import { generateCSSVariables, tokyoNightTheme } from './utils/themes';
import { cn } from './utils/helpers';

// ============================================================================
// Types
// ============================================================================

// Directory entry type from backend
interface DirectoryEntry {
  name: string;
  path: string;
  is_directory: boolean;
  is_expanded: boolean;
  children: DirectoryEntry[] | null;
  extension: string | null;
  is_hidden: boolean;
}

// ============================================================================
// Command Palette Component
// ============================================================================

interface CommandItem {
  id: string;
  label: string;
  shortcut?: string;
  icon?: React.ReactNode;
  action: () => void;
}

const CommandPalette: React.FC<{ isOpen: boolean; onClose: () => void }> = ({
  isOpen,
  onClose,
}) => {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);

  const openFile = useEditorStore((state) => state.openFile);
  const toggleLeftSidebar = useLayoutStore((state) => state.toggleLeftSidebar);
  const toggleBottomPanel = useLayoutStore((state) => state.toggleBottomPanel);
  const toggleRightPanel = useLayoutStore((state) => state.toggleRightPanel);
  const setLeftSidebarPanel = useLayoutStore((state) => state.setLeftSidebarPanel);

  const commands: CommandItem[] = [
    {
      id: 'new-file',
      label: 'New File',
      shortcut: 'Ctrl+N',
      icon: (
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="12" y1="18" x2="12" y2="12" />
          <line x1="9" y1="15" x2="15" y2="15" />
        </svg>
      ),
      action: () => {
        openFile('/untitled.mviz', '', 'mathviz');
        onClose();
      },
    },
    {
      id: 'open-file',
      label: 'Open File',
      shortcut: 'Ctrl+Shift+O',
      icon: (
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
        </svg>
      ),
      action: async () => {
        try {
          const selected = await open({
            directory: false,
            multiple: false,
            title: 'Open File',
            filters: [
              { name: 'MathViz', extensions: ['mviz', 'mathviz', 'mvz'] },
              { name: 'All Files', extensions: ['*'] },
            ],
          });
          if (selected && typeof selected === 'string') {
            const content = await invoke<string>('read_file', { path: selected });
            const fileName = selected.split('/').pop() || selected;
            const ext = fileName.split('.').pop() || '';
            const language = ['mviz', 'mathviz', 'mvz'].includes(ext) ? 'mathviz' : ext;
            openFile(selected, content, language);
          }
        } catch (error) {
          console.error('Failed to open file:', error);
        }
        onClose();
      },
    },
    {
      id: 'open-folder',
      label: 'Open Folder',
      shortcut: 'Ctrl+O',
      icon: (
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
        </svg>
      ),
      action: async () => {
        try {
          const selected = await open({
            directory: true,
            multiple: false,
            title: 'Open Folder',
          });
          if (selected && typeof selected === 'string') {
            const folderName = selected.split('/').pop() || selected;
            useFileStore.getState().setRootPath(selected, folderName);
            const entries = await invoke<DirectoryEntry[]>('read_directory', {
              path: selected,
              recursive: false,
            });
            const convertToFileNode = (entry: DirectoryEntry): import('./types').FileNode => ({
              id: entry.path,
              name: entry.name,
              path: entry.path,
              type: entry.is_directory ? 'directory' : 'file',
              extension: entry.extension as import('./types').FileExtension || undefined,
              children: entry.children?.map(convertToFileNode),
            });
            useFileStore.getState().setFileTree(entries.map(convertToFileNode));
          }
        } catch (error) {
          console.error('Failed to open folder:', error);
        }
        onClose();
      },
    },
    {
      id: 'toggle-sidebar',
      label: 'Toggle Sidebar',
      shortcut: 'Ctrl+B',
      icon: (
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <line x1="9" y1="3" x2="9" y2="21" />
        </svg>
      ),
      action: () => {
        toggleLeftSidebar();
        onClose();
      },
    },
    {
      id: 'toggle-terminal',
      label: 'Toggle Terminal',
      shortcut: 'Ctrl+J',
      icon: (
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="4 17 10 11 4 5" />
          <line x1="12" y1="19" x2="20" y2="19" />
        </svg>
      ),
      action: () => {
        toggleBottomPanel();
        onClose();
      },
    },
    {
      id: 'toggle-preview',
      label: 'Toggle Preview',
      shortcut: 'Ctrl+\\',
      icon: (
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polygon points="5 3 19 12 5 21 5 3" />
        </svg>
      ),
      action: () => {
        toggleRightPanel();
        onClose();
      },
    },
    {
      id: 'open-explorer',
      label: 'Show Explorer',
      shortcut: 'Ctrl+Shift+E',
      icon: (
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
        </svg>
      ),
      action: () => {
        setLeftSidebarPanel('explorer');
        onClose();
      },
    },
    {
      id: 'open-search',
      label: 'Search in Files',
      shortcut: 'Ctrl+Shift+F',
      icon: (
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="11" cy="11" r="8" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
      ),
      action: () => {
        setLeftSidebarPanel('search');
        onClose();
      },
    },
    {
      id: 'open-debug',
      label: 'Show Run & Debug',
      shortcut: 'Ctrl+Shift+D',
      icon: (
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="8" r="4" />
          <path d="M12 12V21M8 8L4 4M16 8L20 4M4 12h4M16 12h4" />
        </svg>
      ),
      action: () => {
        setLeftSidebarPanel('debug');
        onClose();
      },
    },
    {
      id: 'run-file',
      label: 'Run with Manim (Scene)',
      shortcut: 'F5',
      icon: (
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
          <polygon points="5 3 19 12 5 21 5 3" />
        </svg>
      ),
      action: () => {
        const activeFile = useEditorStore.getState().openFiles.get(useEditorStore.getState().activeFileId || '');
        const canRun = activeFile &&
          !activeFile.path.startsWith('/untitled') &&
          (activeFile.path.endsWith('.mviz') || activeFile.path.endsWith('.mathviz'));
        if (canRun) {
          useCompilerStore.getState().run(activeFile.path, true, 'm');
        }
        onClose();
      },
    },
    {
      id: 'exec-file',
      label: 'Execute Script',
      shortcut: 'F6',
      icon: (
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="4 17 10 11 4 5" />
          <line x1="12" y1="19" x2="20" y2="19" />
        </svg>
      ),
      action: () => {
        const activeFile = useEditorStore.getState().openFiles.get(useEditorStore.getState().activeFileId || '');
        const canExec = activeFile &&
          !activeFile.path.startsWith('/untitled') &&
          (activeFile.path.endsWith('.mviz') || activeFile.path.endsWith('.mathviz'));
        if (canExec) {
          useCompilerStore.getState().exec(activeFile.path);
        }
        onClose();
      },
    },
    {
      id: 'compile-file',
      label: 'Compile Only',
      shortcut: 'F8',
      icon: (
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
        </svg>
      ),
      action: () => {
        const activeFile = useEditorStore.getState().openFiles.get(useEditorStore.getState().activeFileId || '');
        const canCompile = activeFile &&
          !activeFile.path.startsWith('/untitled') &&
          (activeFile.path.endsWith('.mviz') || activeFile.path.endsWith('.mathviz'));
        if (canCompile) {
          useCompilerStore.getState().compile(activeFile.path);
        }
        onClose();
      },
    },
  ];

  // Filter commands based on query
  const filteredCommands = commands.filter((cmd) =>
    cmd.label.toLowerCase().includes(query.toLowerCase())
  );

  // Handle keyboard navigation
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex((i) => Math.min(i + 1, filteredCommands.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex((i) => Math.max(i - 1, 0));
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (filteredCommands[selectedIndex]) {
          filteredCommands[selectedIndex].action();
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, filteredCommands, selectedIndex, onClose]);

  // Reset on query change
  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="command-palette-overlay fixed inset-0 bg-black/50 backdrop-blur-sm z-[9999]"
        onClick={onClose}
      />

      {/* Palette */}
      <div className="command-palette fixed top-[20%] left-1/2 -translate-x-1/2 w-[560px] max-w-[calc(100vw-32px)] z-[10000] animate-slide-in-top">
        <div className="bg-[var(--mviz-widget-background,#1f2335)] border border-[var(--mviz-widget-border,#3b4261)] rounded-xl shadow-2xl overflow-hidden">
          {/* Input */}
          <div className="p-4 border-b border-[var(--mviz-border,#1a1b26)]">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Type a command..."
              className={cn(
                'w-full bg-transparent text-[15px]',
                'text-[var(--mviz-foreground,#c0caf5)]',
                'placeholder:text-[var(--mviz-input-placeholder,#565f89)]',
                'outline-none'
              )}
              autoFocus
            />
          </div>

          {/* Results */}
          <div className="max-h-[320px] overflow-y-auto p-2">
            {filteredCommands.length === 0 ? (
              <div className="p-4 text-center text-sm text-[var(--mviz-foreground,#a9b1d6)] opacity-50">
                No commands found
              </div>
            ) : (
              filteredCommands.map((cmd, index) => (
                <div
                  key={cmd.id}
                  className={cn(
                    'flex items-center gap-3 px-3 py-2.5 rounded-lg cursor-pointer',
                    'transition-colors duration-100',
                    index === selectedIndex
                      ? 'bg-[var(--mviz-list-active-background,#283457)]'
                      : 'hover:bg-[var(--mviz-list-hover-background,#292e42)]'
                  )}
                  onClick={cmd.action}
                >
                  {cmd.icon && (
                    <span className="text-[var(--mviz-accent,#7aa2f7)]">{cmd.icon}</span>
                  )}
                  <span className="flex-1 text-[14px] text-[var(--mviz-foreground,#c0caf5)]">
                    {cmd.label}
                  </span>
                  {cmd.shortcut && (
                    <span className="text-[12px] text-[var(--mviz-foreground,#a9b1d6)] opacity-50">
                      {cmd.shortcut}
                    </span>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </>
  );
};

// ============================================================================
// Bottom Panel Tabs Component
// ============================================================================

const BottomPanelTabsComponent: React.FC = () => {
  const bottomPanel = useLayoutStore((state) => state.bottomPanel);
  const setBottomPanelActive = useLayoutStore((state) => state.setBottomPanelActive);

  const tabs = [
    { id: 'terminal', label: 'Terminal', icon: (
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <polyline points="4 17 10 11 4 5" />
        <line x1="12" y1="19" x2="20" y2="19" />
      </svg>
    )},
    { id: 'problems', label: 'Problems', icon: (
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <path d="M12 8v4M12 16h.01" />
      </svg>
    )},
    { id: 'output', label: 'Output', icon: (
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M4 19h16M4 15h16M4 11h16M4 7h16" />
      </svg>
    )},
    { id: 'debug-console', label: 'Debug Console', icon: (
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="8" r="4" />
        <path d="M12 12V21M8 8L4 4M16 8L20 4M4 12h4M16 12h4" />
      </svg>
    )},
  ] as const;

  return (
    <div className="bottom-panel-tabs flex items-center gap-1 px-2 py-1.5 bg-[var(--mviz-sidebar-background,#1f2335)] border-b border-[var(--mviz-border,#1a1b26)]">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          className={cn(
            'bottom-panel-tab flex items-center gap-2 px-3 py-1 rounded-full',
            'text-xs font-medium',
            'transition-all duration-100',
            bottomPanel.activePanel === tab.id
              ? 'bg-[var(--mviz-accent,#7aa2f7)] text-[var(--mviz-accent-foreground,#1a1b26)]'
              : 'text-[var(--mviz-foreground,#a9b1d6)] opacity-70 hover:opacity-100 hover:bg-[var(--mviz-list-hover-background,#292e42)]'
          )}
          onClick={() => setBottomPanelActive(tab.id)}
        >
          {tab.icon}
          <span>{tab.label}</span>
        </button>
      ))}
    </div>
  );
};

// ============================================================================
// Bottom Panel Content Component
// ============================================================================

const BottomPanelContent: React.FC = () => {
  const activePanel = useLayoutStore((state) => state.bottomPanel.activePanel);
  const diagnostics = useEditorStore((state) => state.diagnostics);
  const diagnosticSummary = useEditorStore((state) => state.diagnosticSummary);
  const compilerOutput = useCompilerStore((state) => state.output);
  const clearCompilerOutput = useCompilerStore((state) => state.clearOutput);

  return (
    <div className="flex flex-col h-full">
      <BottomPanelTabsComponent />
      <div className="flex-1 overflow-hidden">
        {activePanel === 'terminal' && <Terminal />}
        {activePanel === 'problems' && (
          <div className="p-3 overflow-auto h-full">
            {diagnosticSummary.errors === 0 && diagnosticSummary.warnings === 0 ? (
              <div className="empty-state flex flex-col items-center justify-center h-full">
                <svg className="w-12 h-12 mb-3 text-[var(--mviz-foreground,#a9b1d6)] opacity-20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p className="text-sm text-[var(--mviz-foreground,#a9b1d6)] opacity-50">
                  No problems detected
                </p>
              </div>
            ) : (
              <div className="space-y-1">
                {Array.from(diagnostics.entries()).map(([filePath, fileDiagnostics]) =>
                  fileDiagnostics.map((diagnostic) => (
                    <div
                      key={diagnostic.id}
                      className={cn(
                        'flex items-start gap-3 px-3 py-2 text-sm rounded-lg',
                        'hover:bg-[var(--mviz-list-hover-background,#292e42)] cursor-pointer',
                        'transition-colors duration-100'
                      )}
                    >
                      {diagnostic.severity === 'error' ? (
                        <svg className="w-4 h-4 mt-0.5 text-[var(--mviz-error-foreground,#f7768e)]" viewBox="0 0 24 24" fill="currentColor">
                          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 11c-.55 0-1-.45-1-1V8c0-.55.45-1 1-1s1 .45 1 1v4c0 .55-.45 1-1 1zm1 4h-2v-2h2v2z" />
                        </svg>
                      ) : (
                        <svg className="w-4 h-4 mt-0.5 text-[var(--mviz-warning-foreground,#e0af68)]" viewBox="0 0 24 24" fill="currentColor">
                          <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z" />
                        </svg>
                      )}
                      <div className="flex-1 min-w-0">
                        <p className="text-[var(--mviz-foreground,#c0caf5)]">{diagnostic.message}</p>
                        <p className="text-xs text-[var(--mviz-foreground,#a9b1d6)] opacity-50 mt-0.5">
                          {filePath.split('/').pop()}:{diagnostic.range.startLine}:{diagnostic.range.startColumn}
                        </p>
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        )}
        {activePanel === 'output' && (
          <div className="flex flex-col h-full">
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-[var(--mviz-border,#1a1b26)]">
              <span className="text-xs font-medium text-[var(--mviz-foreground,#a9b1d6)] opacity-70">Compiler Output</span>
              <button
                className="p-1 rounded hover:bg-[var(--mviz-list-hover-background,#292e42)] text-[var(--mviz-foreground,#a9b1d6)] opacity-50 hover:opacity-100"
                onClick={clearCompilerOutput}
                title="Clear output"
              >
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6" />
                </svg>
              </button>
            </div>
            <div className="flex-1 p-3 font-mono text-xs overflow-auto bg-[var(--mviz-terminal-background,#1a1b26)]">
              {compilerOutput.length === 0 ? (
                <div className="text-[var(--mviz-foreground,#a9b1d6)] opacity-30">
                  Run or compile a file to see output...
                </div>
              ) : (
                compilerOutput.map((line, index) => (
                  <div
                    key={index}
                    className={cn(
                      'py-0.5 whitespace-pre-wrap',
                      line.includes('[Error]') && 'text-[var(--mviz-error-foreground,#f7768e)]',
                      line.includes('[Warning]') && 'text-[var(--mviz-warning-foreground,#e0af68)]',
                      line.includes('[Success]') && 'text-[var(--mviz-git-added,#9ece6a)]',
                      line.includes('[Compiling]') && 'text-[var(--mviz-accent,#7aa2f7)]',
                      line.includes('[Running]') && 'text-[var(--mviz-accent,#7aa2f7)]',
                      line.includes('[stderr]') && 'text-[var(--mviz-warning-foreground,#e0af68)]',
                      !line.match(/\[(Error|Warning|Success|Compiling|Running|stderr)\]/) && 'text-[var(--mviz-foreground,#a9b1d6)]'
                    )}
                  >
                    {line}
                  </div>
                ))
              )}
            </div>
          </div>
        )}
        {activePanel === 'debug-console' && (
          <div className="p-3 font-mono text-sm overflow-auto h-full">
            <div className="text-[var(--mviz-foreground,#a9b1d6)] opacity-50">[Debug Console]</div>
          </div>
        )}
      </div>
    </div>
  );
};

// ============================================================================
// Installation Banner Component
// ============================================================================

const InstallationBanner: React.FC<{ onDismiss: () => void }> = ({ onDismiss }) => {
  const installation = useCompilerStore(selectInstallation);

  if (!installation || installation.found) return null;

  return (
    <div className="installation-banner flex items-center justify-between px-4 py-2 bg-[var(--mviz-warning-background,#e0af68)] text-[var(--mviz-warning-foreground,#1a1b26)]">
      <div className="flex items-center gap-3">
        <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
          <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z" />
        </svg>
        <span className="text-sm font-medium">
          MathViz compiler not found. Install it with:{' '}
          <code className="px-1.5 py-0.5 bg-black/10 rounded font-mono text-xs">
            pipx install mathviz
          </code>
          {' '}or{' '}
          <code className="px-1.5 py-0.5 bg-black/10 rounded font-mono text-xs">
            uv tool install mathviz
          </code>
        </span>
      </div>
      <button
        className="p-1 rounded hover:bg-black/10 transition-colors"
        onClick={onDismiss}
        title="Dismiss"
      >
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M18 6L6 18M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
};

// ============================================================================
// Sidebar Content Component
// ============================================================================

const SidebarContent: React.FC = () => {
  const activePanel = useLayoutStore((state) => state.leftSidebar.activePanel);

  return (
    <div className="flex flex-col h-full animate-fade-in">
      {activePanel === 'explorer' && <FileTree />}
      {activePanel === 'search' && (
        <div className="flex flex-col h-full">
          <div className="px-4 py-3 border-b border-[var(--mviz-border,#1a1b26)]">
            <span className="text-xs font-semibold uppercase tracking-wider text-[var(--mviz-foreground,#a9b1d6)] opacity-70">
              Search
            </span>
          </div>
          <div className="p-3">
            <input
              type="text"
              placeholder="Search in files..."
              className="input w-full"
            />
          </div>
          <div className="flex-1 flex items-center justify-center">
            <div className="empty-state">
              <svg className="empty-state-icon w-12 h-12 mb-3 opacity-20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <circle cx="11" cy="11" r="8" />
                <path d="M21 21l-4.35-4.35" />
              </svg>
              <p className="text-sm text-[var(--mviz-foreground,#a9b1d6)] opacity-50">
                Enter a search term
              </p>
            </div>
          </div>
        </div>
      )}
      {activePanel === 'debug' && <DebugPanel />}
    </div>
  );
};

// ============================================================================
// Welcome Screen Component - Professional Design
// ============================================================================

const WelcomeScreen: React.FC = () => {
  const openFile = useEditorStore((state) => state.openFile);
  const setRootPath = useFileStore((state) => state.setRootPath);
  const setFileTree = useFileStore((state) => state.setFileTree);

  const handleNewFile = useCallback(() => {
    openFile('/untitled.mviz', '', 'mathviz');
  }, [openFile]);

  const handleOpenFolder = useCallback(async () => {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: 'Open Folder',
      });

      if (selected && typeof selected === 'string') {
        // Get folder name from path
        const folderName = selected.split('/').pop() || selected;
        setRootPath(selected, folderName);

        // Read directory contents
        const entries = await invoke<DirectoryEntry[]>('read_directory', {
          path: selected,
          recursive: false,
        });

        // Convert backend entries to FileNode format
        const convertToFileNode = (entry: DirectoryEntry): import('./types').FileNode => ({
          id: entry.path,
          name: entry.name,
          path: entry.path,
          type: entry.is_directory ? 'directory' : 'file',
          extension: entry.extension as import('./types').FileExtension || undefined,
          children: entry.children?.map(convertToFileNode),
        });

        setFileTree(entries.map(convertToFileNode));
      }
    } catch (error) {
      console.error('Failed to open folder:', error);
    }
  }, [setRootPath, setFileTree]);

  const handleOpenFile = useCallback(async () => {
    try {
      const selected = await open({
        directory: false,
        multiple: false,
        title: 'Open File',
        filters: [
          { name: 'MathViz', extensions: ['mviz', 'mathviz', 'mvz'] },
          { name: 'All Files', extensions: ['*'] },
        ],
      });

      if (selected && typeof selected === 'string') {
        const content = await invoke<string>('read_file', { path: selected });
        const fileName = selected.split('/').pop() || selected;
        const ext = fileName.split('.').pop() || '';
        const language = ['mviz', 'mathviz', 'mvz'].includes(ext) ? 'mathviz' : ext;
        openFile(selected, content, language);
      }
    } catch (error) {
      console.error('Failed to open file:', error);
    }
  }, [openFile]);

  return (
    <div className="flex flex-col items-center justify-center h-full p-8 bg-[var(--mviz-editor-background,#1a1b26)]">
      {/* Logo and branding */}
      <div className="mb-12 text-center">
        <div className="flex items-center justify-center gap-4 mb-4">
          <svg viewBox="0 0 48 48" className="w-14 h-14" fill="none">
            <path
              d="M10 12h26M10 12l13 12-13 12h26"
              stroke="var(--mviz-accent, #7aa2f7)"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <h1 className="text-4xl font-bold text-[var(--mviz-foreground,#c0caf5)]">
            MathViz
          </h1>
        </div>
        <p className="text-sm text-[var(--mviz-foreground,#a9b1d6)] opacity-60">
          Mathematical Visualization Language
        </p>
      </div>

      {/* Quick actions */}
      <div className="grid grid-cols-2 gap-4 max-w-lg w-full mb-12">
        <button
          className={cn(
            'group flex items-center gap-4 p-5 rounded-xl',
            'bg-[var(--mviz-sidebar-background,#1f2335)]',
            'border border-transparent',
            'hover:border-[var(--mviz-accent,#7aa2f7)] hover:border-opacity-30',
            'transition-all duration-150 text-left'
          )}
          onClick={handleNewFile}
        >
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-[var(--mviz-accent,#7aa2f7)] bg-opacity-10 group-hover:bg-opacity-20 transition-colors">
            <svg className="w-5 h-5 text-[var(--mviz-accent,#7aa2f7)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="12" y1="18" x2="12" y2="12" />
              <line x1="9" y1="15" x2="15" y2="15" />
            </svg>
          </div>
          <div>
            <span className="text-sm font-medium text-[var(--mviz-foreground,#c0caf5)] block">New File</span>
            <span className="text-xs text-[var(--mviz-foreground,#a9b1d6)] opacity-50">Create a new .mviz file</span>
          </div>
        </button>

        <button
          className={cn(
            'group flex items-center gap-4 p-5 rounded-xl',
            'bg-[var(--mviz-sidebar-background,#1f2335)]',
            'border border-transparent',
            'hover:border-[var(--mviz-accent,#7aa2f7)] hover:border-opacity-30',
            'transition-all duration-150 text-left'
          )}
          onClick={handleOpenFile}
        >
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-[var(--mviz-cyan,#7dcfff)] bg-opacity-10 group-hover:bg-opacity-20 transition-colors">
            <svg className="w-5 h-5 text-[var(--mviz-cyan,#7dcfff)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
            </svg>
          </div>
          <div>
            <span className="text-sm font-medium text-[var(--mviz-foreground,#c0caf5)] block">Open File</span>
            <span className="text-xs text-[var(--mviz-foreground,#a9b1d6)] opacity-50">Open a .mviz file</span>
          </div>
        </button>

        <button
          className={cn(
            'group flex items-center gap-4 p-5 rounded-xl',
            'bg-[var(--mviz-sidebar-background,#1f2335)]',
            'border border-transparent',
            'hover:border-[var(--mviz-accent,#7aa2f7)] hover:border-opacity-30',
            'transition-all duration-150 text-left'
          )}
          onClick={handleOpenFolder}
        >
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-[var(--mviz-git-added,#9ece6a)] bg-opacity-10 group-hover:bg-opacity-20 transition-colors">
            <svg className="w-5 h-5 text-[var(--mviz-git-added,#9ece6a)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
            </svg>
          </div>
          <div>
            <span className="text-sm font-medium text-[var(--mviz-foreground,#c0caf5)] block">Open Folder</span>
            <span className="text-xs text-[var(--mviz-foreground,#a9b1d6)] opacity-50">Open existing project</span>
          </div>
        </button>

        <button
          className={cn(
            'group flex items-center gap-4 p-5 rounded-xl',
            'bg-[var(--mviz-sidebar-background,#1f2335)]',
            'border border-transparent',
            'hover:border-[var(--mviz-accent,#7aa2f7)] hover:border-opacity-30',
            'transition-all duration-150 text-left'
          )}
        >
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-[var(--mviz-magenta,#bb9af7)] bg-opacity-10 group-hover:bg-opacity-20 transition-colors">
            <svg className="w-5 h-5 text-[var(--mviz-magenta,#bb9af7)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <polygon points="10 8 16 12 10 16 10 8" />
            </svg>
          </div>
          <div>
            <span className="text-sm font-medium text-[var(--mviz-foreground,#c0caf5)] block">Examples</span>
            <span className="text-xs text-[var(--mviz-foreground,#a9b1d6)] opacity-50">Browse example animations</span>
          </div>
        </button>

        <button
          className={cn(
            'group flex items-center gap-4 p-5 rounded-xl',
            'bg-[var(--mviz-sidebar-background,#1f2335)]',
            'border border-transparent',
            'hover:border-[var(--mviz-accent,#7aa2f7)] hover:border-opacity-30',
            'transition-all duration-150 text-left'
          )}
        >
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-[var(--mviz-cyan,#7dcfff)] bg-opacity-10 group-hover:bg-opacity-20 transition-colors">
            <svg className="w-5 h-5 text-[var(--mviz-cyan,#7dcfff)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
              <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
            </svg>
          </div>
          <div>
            <span className="text-sm font-medium text-[var(--mviz-foreground,#c0caf5)] block">Documentation</span>
            <span className="text-xs text-[var(--mviz-foreground,#a9b1d6)] opacity-50">Learn MathViz syntax</span>
          </div>
        </button>
      </div>

      {/* Keyboard shortcuts */}
      <div className="flex flex-wrap items-center justify-center gap-6 text-xs text-[var(--mviz-foreground,#a9b1d6)] opacity-40">
        <span className="flex items-center gap-2">
          <kbd className="px-2 py-1 bg-[var(--mviz-sidebar-background,#1f2335)] border border-[var(--mviz-border,#1a1b26)] rounded text-[10px] font-mono">F5</kbd>
          <span>Run (Manim)</span>
        </span>
        <span className="flex items-center gap-2">
          <kbd className="px-2 py-1 bg-[var(--mviz-sidebar-background,#1f2335)] border border-[var(--mviz-border,#1a1b26)] rounded text-[10px] font-mono">F6</kbd>
          <span>Execute</span>
        </span>
        <span className="flex items-center gap-2">
          <kbd className="px-2 py-1 bg-[var(--mviz-sidebar-background,#1f2335)] border border-[var(--mviz-border,#1a1b26)] rounded text-[10px] font-mono">F8</kbd>
          <span>Compile</span>
        </span>
        <span className="flex items-center gap-2">
          <kbd className="px-2 py-1 bg-[var(--mviz-sidebar-background,#1f2335)] border border-[var(--mviz-border,#1a1b26)] rounded text-[10px] font-mono">Ctrl+P</kbd>
          <span>Commands</span>
        </span>
      </div>
    </div>
  );
};

// ============================================================================
// Editor Area Component
// ============================================================================

const EditorArea: React.FC = () => {
  const activeFileId = useEditorStore((state) => state.activeFileId);
  const activeFile = useEditorStore(selectActiveFile);
  const markFileSaved = useEditorStore((state) => state.markFileSaved);
  const closeFile = useEditorStore((state) => state.closeFile);
  const openFile = useEditorStore((state) => state.openFile);

  const handleSave = useCallback(
    async (content: string) => {
      if (!activeFileId || !activeFile) return;

      const isUntitled = activeFile.path.startsWith('/untitled');

      if (isUntitled) {
        // Show Save As dialog for untitled files
        try {
          const { save } = await import('@tauri-apps/plugin-dialog');
          const filePath = await save({
            title: 'Save File',
            defaultPath: activeFile.name,
            filters: [
              { name: 'MathViz', extensions: ['mviz', 'mathviz'] },
              { name: 'All Files', extensions: ['*'] },
            ],
          });

          if (filePath) {
            // Write file to disk
            await invoke('write_file', { path: filePath, content });
            // Close the untitled file and open the new one
            closeFile(activeFileId);
            openFile(filePath, content, 'mathviz');
            console.log('File saved as:', filePath);
          }
        } catch (error) {
          console.error('Failed to save file:', error);
        }
      } else {
        // Save directly to existing file
        try {
          await invoke('write_file', { path: activeFile.path, content });
          markFileSaved(activeFileId, content);
          console.log('File saved:', activeFile.path);
        } catch (error) {
          console.error('Failed to save file:', error);
        }
      }
    },
    [activeFileId, activeFile, markFileSaved, closeFile, openFile]
  );

  if (!activeFileId) {
    return <WelcomeScreen />;
  }

  return <Editor onSave={handleSave} />;
};

// ============================================================================
// Main App Component
// ============================================================================

const App: React.FC = () => {
  const rightPanelVisible = useLayoutStore((state) => state.rightPanel.isVisible);
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const [bannerDismissed, setBannerDismissed] = useState(false);
  const installationChecked = useCompilerStore(selectInstallationChecked);
  const checkInstallation = useCompilerStore((state) => state.checkInstallation);

  // Check MathViz installation on mount
  useEffect(() => {
    if (!installationChecked) {
      checkInstallation();
    }
  }, [installationChecked, checkInstallation]);

  // Handle window close with unsaved files
  useEffect(() => {
    let unlisten: (() => void) | undefined;

    const setupCloseHandler = async () => {
      try {
        const { getCurrentWindow } = await import('@tauri-apps/api/window');
        const { confirm } = await import('@tauri-apps/plugin-dialog');
        const currentWindow = getCurrentWindow();

        unlisten = await currentWindow.onCloseRequested(async (event) => {
          const openFiles = useEditorStore.getState().openFiles;
          const hasUnsaved = Array.from(openFiles.values()).some(file => file.isDirty);

          if (hasUnsaved) {
            // Always prevent default first, then handle manually
            event.preventDefault();

            const shouldClose = await confirm(
              'You have unsaved changes. Are you sure you want to quit?',
              {
                title: 'Unsaved Changes',
                kind: 'warning',
                okLabel: 'Quit',
                cancelLabel: 'Cancel',
              }
            );

            if (shouldClose) {
              // Force close the window
              await currentWindow.destroy();
            }
          }
          // If no unsaved files, let it close normally (don't call preventDefault)
        });
      } catch (error) {
        console.error('Failed to setup close handler:', error);
      }
    };

    setupCloseHandler();

    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  // Initialize CSS variables on mount
  useEffect(() => {
    const styleId = 'mathviz-theme-variables';
    let styleEl = document.getElementById(styleId) as HTMLStyleElement | null;

    if (!styleEl) {
      styleEl = document.createElement('style');
      styleEl.id = styleId;
      document.head.appendChild(styleEl);
    }

    styleEl.textContent = generateCSSVariables(tokyoNightTheme);

    return () => {
      styleEl?.remove();
    };
  }, []);



  // Global keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl/Cmd + P: Command palette
      if ((e.ctrlKey || e.metaKey) && e.key === 'p') {
        e.preventDefault();
        setCommandPaletteOpen(true);
      }

      // Ctrl/Cmd + B: Toggle sidebar
      if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
        e.preventDefault();
        useLayoutStore.getState().toggleLeftSidebar();
      }

      // Ctrl/Cmd + J: Toggle bottom panel
      if ((e.ctrlKey || e.metaKey) && e.key === 'j') {
        e.preventDefault();
        useLayoutStore.getState().toggleBottomPanel();
      }

      // Ctrl/Cmd + \: Toggle right panel
      if ((e.ctrlKey || e.metaKey) && e.key === '\\') {
        e.preventDefault();
        useLayoutStore.getState().toggleRightPanel();
      }

      // Ctrl/Cmd + Shift + E: Focus explorer
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'E') {
        e.preventDefault();
        useLayoutStore.getState().setLeftSidebarPanel('explorer');
      }

      // Ctrl/Cmd + Shift + D: Focus debug
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'D') {
        e.preventDefault();
        useLayoutStore.getState().setLeftSidebarPanel('debug');
      }

      // F5: Run with Manim (for scene files)
      if (e.key === 'F5') {
        e.preventDefault();
        const activeFile = useEditorStore.getState().openFiles.get(useEditorStore.getState().activeFileId || '');
        const canRun = activeFile &&
          !activeFile.path.startsWith('/untitled') &&
          (activeFile.path.endsWith('.mviz') || activeFile.path.endsWith('.mathviz'));
        if (canRun) {
          useCompilerStore.getState().run(activeFile.path, true, 'm');
        }
      }

      // F6: Execute script (without Manim, for simple scripts with println)
      if (e.key === 'F6') {
        e.preventDefault();
        const activeFile = useEditorStore.getState().openFiles.get(useEditorStore.getState().activeFileId || '');
        const canExec = activeFile &&
          !activeFile.path.startsWith('/untitled') &&
          (activeFile.path.endsWith('.mviz') || activeFile.path.endsWith('.mathviz'));
        if (canExec) {
          useCompilerStore.getState().exec(activeFile.path);
        }
      }

      // F8: Compile only (no execution)
      if (e.key === 'F8') {
        e.preventDefault();
        const activeFile = useEditorStore.getState().openFiles.get(useEditorStore.getState().activeFileId || '');
        const canCompile = activeFile &&
          !activeFile.path.startsWith('/untitled') &&
          (activeFile.path.endsWith('.mviz') || activeFile.path.endsWith('.mathviz'));
        if (canCompile) {
          useCompilerStore.getState().compile(activeFile.path);
        }
      }

      // Ctrl/Cmd + Shift + B: Compile current file (only for saved .mviz files)
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'B') {
        e.preventDefault();
        const activeFile = useEditorStore.getState().openFiles.get(useEditorStore.getState().activeFileId || '');
        const canCompile = activeFile &&
          !activeFile.path.startsWith('/untitled') &&
          (activeFile.path.endsWith('.mviz') || activeFile.path.endsWith('.mathviz'));
        if (canCompile) {
          useCompilerStore.getState().compile(activeFile.path);
        }
      }

      // Ctrl/Cmd + Shift + F: Focus search
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'F') {
        e.preventDefault();
        useLayoutStore.getState().setLeftSidebarPanel('search');
      }

      // Ctrl/Cmd + Shift + O: Open file
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'O') {
        e.preventDefault();
        (async () => {
          try {
            const { open: openDialog } = await import('@tauri-apps/plugin-dialog');
            const selected = await openDialog({
              directory: false,
              multiple: false,
              title: 'Open File',
              filters: [
                { name: 'MathViz', extensions: ['mviz', 'mathviz', 'mvz'] },
                { name: 'All Files', extensions: ['*'] },
              ],
            });
            if (selected && typeof selected === 'string') {
              const content = await invoke<string>('read_file', { path: selected });
              const fileName = selected.split('/').pop() || selected;
              const ext = fileName.split('.').pop() || '';
              const language = ['mviz', 'mathviz', 'mvz'].includes(ext) ? 'mathviz' : ext;
              useEditorStore.getState().openFile(selected, content, language);
            }
          } catch (error) {
            console.error('Failed to open file:', error);
          }
        })();
      }

      // Escape: Close command palette
      if (e.key === 'Escape') {
        setCommandPaletteOpen(false);
      }

      // Ctrl/Cmd + K Z: Toggle zen mode
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        const handleZ = (e2: KeyboardEvent) => {
          if (e2.key === 'z') {
            e2.preventDefault();
            useLayoutStore.getState().toggleZenMode();
          }
          document.removeEventListener('keydown', handleZ);
        };
        document.addEventListener('keydown', handleZ);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <div className="flex flex-col h-screen">
      {!bannerDismissed && <InstallationBanner onDismiss={() => setBannerDismissed(true)} />}
      <div className="flex-1 overflow-hidden">
        <MainLayout
          sidebar={<SidebarContent />}
          editor={<EditorArea />}
          bottomPanel={<BottomPanelContent />}
          rightPanel={rightPanelVisible ? <ManimPreview /> : undefined}
          statusBar={<StatusBar />}
        />
      </div>
      <CommandPalette
        isOpen={commandPaletteOpen}
        onClose={() => setCommandPaletteOpen(false)}
      />
    </div>
  );
};

// ============================================================================
// Export
// ============================================================================

export default App;
