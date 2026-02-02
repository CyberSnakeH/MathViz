/**
 * TabBar Component
 *
 * Modern rounded tab design with:
 * - Smooth hover transitions
 * - Close button appears on hover
 * - Modified indicator as colored dot
 * - Tab groups with visual separation
 * - Horizontal scroll with wheel support
 */

import React, { useCallback, useRef, useState, memo } from 'react';
import { cn } from '../../utils/helpers';
import { useEditorStore, selectOpenFilesArray } from '../../stores/editorStore';
import type { OpenFile } from '../../types';

// ============================================================================
// File Icon Component
// ============================================================================

interface FileIconProps {
  fileName: string;
  className?: string;
}

const FileIcon: React.FC<FileIconProps> = memo(({ fileName, className }) => {
  const ext = fileName.split('.').pop()?.toLowerCase() || '';

  // Icon configuration with Tokyo Night colors
  const iconConfig: Record<string, { bg: string; fg: string; text: string }> = {
    mviz: { bg: '#bb9af720', fg: '#bb9af7', text: 'M' },
    py: { bg: '#7aa2f720', fg: '#7aa2f7', text: 'Py' },
    js: { bg: '#e0af6820', fg: '#e0af68', text: 'JS' },
    ts: { bg: '#7aa2f720', fg: '#7aa2f7', text: 'TS' },
    tsx: { bg: '#7aa2f720', fg: '#7aa2f7', text: 'Tx' },
    jsx: { bg: '#7dcfff20', fg: '#7dcfff', text: 'Jx' },
    json: { bg: '#e0af6820', fg: '#e0af68', text: '{}' },
    md: { bg: '#7aa2f720', fg: '#7aa2f7', text: 'Md' },
    toml: { bg: '#ff9e6420', fg: '#ff9e64', text: 'T' },
    yaml: { bg: '#f7768e20', fg: '#f7768e', text: 'Y' },
    yml: { bg: '#f7768e20', fg: '#f7768e', text: 'Y' },
    html: { bg: '#f7768e20', fg: '#f7768e', text: 'H' },
    css: { bg: '#bb9af720', fg: '#bb9af7', text: 'C' },
    scss: { bg: '#bb9af720', fg: '#bb9af7', text: 'S' },
    rs: { bg: '#ff9e6420', fg: '#ff9e64', text: 'Rs' },
    go: { bg: '#7dcfff20', fg: '#7dcfff', text: 'Go' },
    sh: { bg: '#9ece6a20', fg: '#9ece6a', text: '$' },
    txt: { bg: '#565f8920', fg: '#565f89', text: 'Tx' },
  };

  const config = iconConfig[ext];

  if (config) {
    return (
      <span
        className={cn(
          'flex items-center justify-center w-4 h-4 rounded text-[10px] font-bold',
          className
        )}
        style={{ backgroundColor: config.bg, color: config.fg }}
      >
        {config.text}
      </span>
    );
  }

  // Default file icon
  return (
    <svg
      className={cn('w-4 h-4 text-[var(--mviz-foreground,#a9b1d6)] opacity-60', className)}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"
      />
    </svg>
  );
});

FileIcon.displayName = 'FileIcon';

// ============================================================================
// Tab Component
// ============================================================================

interface TabProps {
  file: OpenFile;
  isActive: boolean;
  onSelect: () => void;
  onClose: (e: React.MouseEvent) => void;
  onContextMenu: (e: React.MouseEvent) => void;
}

const Tab: React.FC<TabProps> = memo(({ file, isActive, onSelect, onClose, onContextMenu }) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      className={cn(
        'group relative flex items-center gap-2',
        'h-[34px] px-3',
        'rounded-t-lg',
        'cursor-pointer select-none',
        'transition-all duration-150 ease-out',
        isActive
          ? 'bg-[var(--mviz-tab-active-background,#1a1b26)] text-[var(--mviz-tab-active-foreground,#c0caf5)]'
          : 'bg-transparent text-[var(--mviz-tab-inactive-foreground,#565f89)] hover:bg-[var(--mviz-tab-hover-background,#292e42)] hover:text-[var(--mviz-foreground,#a9b1d6)]'
      )}
      onClick={onSelect}
      onContextMenu={onContextMenu}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      title={file.path}
    >
      {/* File icon */}
      <FileIcon fileName={file.name} />

      {/* File name */}
      <span className="truncate text-[13px] max-w-[120px]">{file.name}</span>

      {/* Close button / Modified indicator */}
      <button
        className={cn(
          'flex items-center justify-center w-5 h-5 rounded',
          'transition-all duration-100 ease-out',
          'hover:bg-[var(--mviz-list-hover-background,#292e42)]',
          (isHovered || isActive) ? 'opacity-100' : 'opacity-0',
          file.isDirty && !isHovered && 'opacity-100'
        )}
        onClick={onClose}
        title={file.isDirty ? 'Unsaved changes (Ctrl+S to save)' : 'Close (Ctrl+W)'}
      >
        {file.isDirty && !isHovered ? (
          // Modified dot indicator
          <span className="w-2 h-2 rounded-full bg-[var(--mviz-accent,#7aa2f7)]" />
        ) : (
          // Close X icon
          <svg className="w-3.5 h-3.5" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 8.707l3.646 3.647.708-.707L8.707 8l3.647-3.646-.707-.708L8 7.293 4.354 3.646l-.707.708L7.293 8l-3.646 3.646.707.708L8 8.707z" />
          </svg>
        )}
      </button>

      {/* Active tab indicator line */}
      {isActive && (
        <div
          className={cn(
            'absolute bottom-0 left-2 right-2 h-0.5',
            'bg-[var(--mviz-accent,#7aa2f7)]',
            'rounded-t-full'
          )}
        />
      )}
    </div>
  );
});

Tab.displayName = 'Tab';

// ============================================================================
// Tab Context Menu Component
// ============================================================================

interface TabContextMenuProps {
  file: OpenFile;
  position: { x: number; y: number };
  onClose: () => void;
}

const TabContextMenu: React.FC<TabContextMenuProps> = ({ file, position, onClose }) => {
  const closeFile = useEditorStore((state) => state.closeFile);
  const closeAllFiles = useEditorStore((state) => state.closeAllFiles);
  const closeOtherFiles = useEditorStore((state) => state.closeOtherFiles);
  const closeSavedFiles = useEditorStore((state) => state.closeSavedFiles);

  // Close context menu on click outside
  React.useEffect(() => {
    const handleClick = () => onClose();
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('click', handleClick);
    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('click', handleClick);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [onClose]);

  const menuItems = [
    { label: 'Close', shortcut: 'Ctrl+W', action: () => closeFile(file.id) },
    { label: 'Close Others', action: () => closeOtherFiles(file.id) },
    { label: 'Close All', action: closeAllFiles },
    { label: 'Close Saved', action: closeSavedFiles },
    { separator: true },
    { label: 'Copy Path', action: () => navigator.clipboard.writeText(file.path) },
    { label: 'Copy Name', action: () => navigator.clipboard.writeText(file.name) },
    { separator: true },
    { label: 'Reveal in Explorer', action: () => console.log('Reveal:', file.path) },
  ];

  return (
    <div
      className={cn(
        'context-menu',
        'fixed z-[10000] py-1 min-w-[180px]',
        'bg-[var(--mviz-widget-background,#1f2335)]',
        'border border-[var(--mviz-widget-border,#3b4261)]',
        'rounded-lg shadow-lg',
        'animate-scale-in'
      )}
      style={{ left: position.x, top: position.y }}
      onClick={(e) => e.stopPropagation()}
    >
      {menuItems.map((item, index) =>
        'separator' in item ? (
          <div
            key={index}
            className="my-1 mx-2 border-t border-[var(--mviz-border,#1a1b26)]"
          />
        ) : (
          <button
            key={index}
            className={cn(
              'context-menu-item',
              'w-full flex items-center justify-between px-3 py-1.5',
              'text-[13px] text-left',
              'text-[var(--mviz-foreground,#a9b1d6)]',
              'hover:bg-[var(--mviz-list-hover-background,#292e42)]',
              'rounded mx-1',
              'transition-colors duration-100'
            )}
            onClick={() => {
              item.action();
              onClose();
            }}
          >
            <span>{item.label}</span>
            {item.shortcut && (
              <span className="text-[11px] text-[var(--mviz-foreground,#a9b1d6)] opacity-50 ml-4">
                {item.shortcut}
              </span>
            )}
          </button>
        )
      )}
    </div>
  );
};

// ============================================================================
// TabBar Component
// ============================================================================

export const TabBar: React.FC = () => {
  const openFiles = useEditorStore(selectOpenFilesArray);
  const activeFileId = useEditorStore((state) => state.activeFileId);
  const setActiveFile = useEditorStore((state) => state.setActiveFile);
  const closeFile = useEditorStore((state) => state.closeFile);

  const scrollRef = useRef<HTMLDivElement>(null);
  const [showContextMenu, setShowContextMenu] = useState(false);
  const [contextMenuFile, setContextMenuFile] = useState<OpenFile | null>(null);
  const [contextMenuPosition, setContextMenuPosition] = useState({ x: 0, y: 0 });

  const handleSelect = useCallback(
    (id: string) => {
      setActiveFile(id);
    },
    [setActiveFile]
  );

  const handleClose = useCallback(
    (e: React.MouseEvent, id: string) => {
      e.stopPropagation();
      closeFile(id);
    },
    [closeFile]
  );

  const handleContextMenu = useCallback((e: React.MouseEvent, file: OpenFile) => {
    e.preventDefault();
    setContextMenuFile(file);
    setContextMenuPosition({ x: e.clientX, y: e.clientY });
    setShowContextMenu(true);
  }, []);

  // Handle wheel scroll for horizontal scrolling
  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (scrollRef.current) {
      scrollRef.current.scrollLeft += e.deltaY;
    }
  }, []);

  // Empty state
  if (openFiles.length === 0) {
    return (
      <div
        className={cn(
          'flex items-center h-[38px]',
          'bg-[var(--mviz-tab-inactive-background,#1f2335)]',
          'border-b border-[var(--mviz-border,#1a1b26)]'
        )}
      />
    );
  }

  return (
    <>
      <div
        ref={scrollRef}
        className={cn(
          'tab-bar',
          'flex items-end h-[38px]',
          'bg-[var(--mviz-tab-inactive-background,#1f2335)]',
          'border-b border-[var(--mviz-border,#1a1b26)]',
          'overflow-x-auto overflow-y-hidden',
          'scrollbar-none',
          'gap-0.5 px-1'
        )}
        onWheel={handleWheel}
      >
        {openFiles.map((file) => (
          <Tab
            key={file.id}
            file={file}
            isActive={file.id === activeFileId}
            onSelect={() => handleSelect(file.id)}
            onClose={(e) => handleClose(e, file.id)}
            onContextMenu={(e) => handleContextMenu(e, file)}
          />
        ))}

        {/* New tab button */}
        <button
          className={cn(
            'flex items-center justify-center',
            'w-8 h-8 ml-1',
            'text-[var(--mviz-foreground,#a9b1d6)] opacity-50',
            'hover:opacity-100 hover:bg-[var(--mviz-list-hover-background,#292e42)]',
            'rounded transition-all duration-100'
          )}
          title="New File"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
        </button>
      </div>

      {/* Context Menu */}
      {showContextMenu && contextMenuFile && (
        <TabContextMenu
          file={contextMenuFile}
          position={contextMenuPosition}
          onClose={() => setShowContextMenu(false)}
        />
      )}
    </>
  );
};

// ============================================================================
// Export
// ============================================================================

export default TabBar;
