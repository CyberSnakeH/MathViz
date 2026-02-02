/**
 * FileTree Component
 *
 * Hierarchical file view with expand/collapse, file icons,
 * context menu, and drag-and-drop support.
 */

import React, { useCallback, useState, memo } from 'react';
import { open } from '@tauri-apps/plugin-dialog';
import { invoke } from '@tauri-apps/api/core';
import { cn } from '../../utils/helpers';
import {
  useFileStore,
  selectVisibleTree,
  selectIsExpanded,
  selectIsSelected,
} from '../../stores/fileStore';
import { useEditorStore } from '../../stores/editorStore';
import type { FileNode, ContextMenuItem, FileExtension } from '../../types';

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
// File Icons
// ============================================================================

const FileIcon: React.FC<{ node: FileNode }> = memo(({ node }) => {
  if (node.type === 'directory') {
    const isExpanded = useFileStore((state) => selectIsExpanded(state, node.path));

    // Folder icon
    return (
      <svg
        className="w-4 h-4 text-[#dcb67a]"
        viewBox="0 0 24 24"
        fill="currentColor"
      >
        {isExpanded ? (
          <path d="M19 20H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h6l2 2h6a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2z" />
        ) : (
          <path d="M3 7V17C3 18.1046 3.89543 19 5 19H19C20.1046 19 21 18.1046 21 17V9C21 7.89543 20.1046 7 19 7H13L11 5H5C3.89543 5 3 5.89543 3 7Z" />
        )}
      </svg>
    );
  }

  // File icons based on extension
  const ext = node.name.split('.').pop()?.toLowerCase();

  const iconConfig: Record<string, { color: string; icon: string }> = {
    mviz: { color: '#9b59b6', icon: 'M' },
    py: { color: '#3572a5', icon: 'Py' },
    js: { color: '#f7df1e', icon: 'JS' },
    ts: { color: '#3178c6', icon: 'TS' },
    tsx: { color: '#3178c6', icon: 'Tx' },
    jsx: { color: '#61dafb', icon: 'Jx' },
    json: { color: '#cbcb41', icon: '{}' },
    md: { color: '#083fa1', icon: 'Md' },
    toml: { color: '#9c4221', icon: 'T' },
    yaml: { color: '#cb171e', icon: 'Y' },
    yml: { color: '#cb171e', icon: 'Y' },
    html: { color: '#e34c26', icon: 'H' },
    css: { color: '#563d7c', icon: 'C' },
    scss: { color: '#c6538c', icon: 'S' },
    gitignore: { color: '#f14e32', icon: 'G' },
    lock: { color: '#8b8b8b', icon: 'L' },
  };

  const config = iconConfig[ext || ''];

  if (config) {
    return (
      <span
        className="w-4 h-4 flex items-center justify-center text-[10px] font-bold rounded"
        style={{ color: config.color }}
      >
        {config.icon}
      </span>
    );
  }

  // Default file icon
  return (
    <svg className="w-4 h-4 text-[#8b8b8b]" viewBox="0 0 24 24" fill="currentColor">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zm4 18H6V4h7v5h5v11z" />
    </svg>
  );
});

FileIcon.displayName = 'FileIcon';

// ============================================================================
// TreeItem Component
// ============================================================================

interface TreeItemProps {
  node: FileNode;
  depth: number;
  onSelect: (node: FileNode) => void;
  onToggle: (node: FileNode) => void;
  onContextMenu: (e: React.MouseEvent, node: FileNode) => void;
}

const TreeItem: React.FC<TreeItemProps> = memo(({
  node,
  depth,
  onSelect,
  onToggle,
  onContextMenu,
}) => {
  const isExpanded = useFileStore((state) => selectIsExpanded(state, node.path));
  const isSelected = useFileStore((state) => selectIsSelected(state, node.path));
  const isLoading = useFileStore((state) => state.loadingPaths.has(node.path));

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      if (node.type === 'directory') {
        onToggle(node);
      } else {
        onSelect(node);
      }
    },
    [node, onSelect, onToggle]
  );

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      if (node.type === 'file') {
        onSelect(node);
      }
    },
    [node, onSelect]
  );

  const handleContextMenuClick = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      onContextMenu(e, node);
    },
    [node, onContextMenu]
  );

  return (
    <div>
      <div
        className={cn(
          'flex items-center gap-1 px-2 py-0.5 cursor-pointer',
          'hover:bg-[var(--mviz-list-hover-background)]',
          isSelected && 'bg-[var(--mviz-list-active-background)] text-[var(--mviz-list-active-foreground)]'
        )}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
        onClick={handleClick}
        onDoubleClick={handleDoubleClick}
        onContextMenu={handleContextMenuClick}
      >
        {/* Chevron for directories */}
        {node.type === 'directory' && (
          <span
            className={cn(
              'flex items-center justify-center w-4 h-4 transition-transform duration-100',
              isExpanded && 'rotate-90'
            )}
          >
            <svg className="w-3 h-3" viewBox="0 0 16 16" fill="currentColor">
              <path d="M6 4l4 4-4 4V4z" />
            </svg>
          </span>
        )}

        {/* Spacer for files */}
        {node.type !== 'directory' && <span className="w-4" />}

        {/* Icon */}
        <FileIcon node={node} />

        {/* Name */}
        <span className="text-sm truncate flex-1">{node.name}</span>

        {/* Loading indicator */}
        {isLoading && (
          <span className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin" />
        )}
      </div>

      {/* Children */}
      {node.type === 'directory' && isExpanded && node.children && (
        <div>
          {node.children.map((child) => (
            <TreeItem
              key={child.path}
              node={child}
              depth={depth + 1}
              onSelect={onSelect}
              onToggle={onToggle}
              onContextMenu={onContextMenu}
            />
          ))}
        </div>
      )}
    </div>
  );
});

TreeItem.displayName = 'TreeItem';

// ============================================================================
// FileTree Component
// ============================================================================

export const FileTree: React.FC = () => {
  const fileTree = useFileStore(selectVisibleTree);
  const projectName = useFileStore((state) => state.projectName);
  const rootPath = useFileStore((state) => state.rootPath);
  const toggleFolder = useFileStore((state) => state.toggleFolder);
  const setSelectedPath = useFileStore((state) => state.setSelectedPath);
  const searchQuery = useFileStore((state) => state.searchQuery);
  const setSearchQuery = useFileStore((state) => state.setSearchQuery);
  const expandAll = useFileStore((state) => state.expandAll);
  const collapseAll = useFileStore((state) => state.collapseAll);
  const refreshFiles = useFileStore((state) => state.refreshFiles);
  const setRootPath = useFileStore((state) => state.setRootPath);
  const setFileTree = useFileStore((state) => state.setFileTree);

  const openFile = useEditorStore((state) => state.openFile);

  // Handle open folder
  const handleOpenFolder = useCallback(async () => {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: 'Open Folder',
      });

      if (selected && typeof selected === 'string') {
        const folderName = selected.split('/').pop() || selected;
        setRootPath(selected, folderName);

        const entries = await invoke<DirectoryEntry[]>('read_directory', {
          path: selected,
          recursive: false,
        });

        const convertToFileNode = (entry: DirectoryEntry): FileNode => ({
          id: entry.path,
          name: entry.name,
          path: entry.path,
          type: entry.is_directory ? 'directory' : 'file',
          extension: entry.extension as FileExtension || undefined,
          children: entry.children?.map(convertToFileNode),
        });

        setFileTree(entries.map(convertToFileNode));
      }
    } catch (error) {
      console.error('Failed to open folder:', error);
    }
  }, [setRootPath, setFileTree]);

  const [contextMenu, setContextMenu] = useState<{
    visible: boolean;
    x: number;
    y: number;
    node: FileNode | null;
  }>({ visible: false, x: 0, y: 0, node: null });

  // Handle file selection
  const handleSelect = useCallback(
    (node: FileNode) => {
      setSelectedPath(node.path);
      if (node.type === 'file') {
        // In a real app, we'd read the file content from Tauri
        // For now, we'll just open with placeholder content
        openFile(node.path, `// Content of ${node.name}\n\n// Loading...`, undefined);
      }
    },
    [setSelectedPath, openFile]
  );

  // Handle folder toggle
  const handleToggle = useCallback(
    (node: FileNode) => {
      toggleFolder(node.path);
    },
    [toggleFolder]
  );

  // Handle context menu
  const handleContextMenu = useCallback(
    (e: React.MouseEvent, node: FileNode) => {
      setContextMenu({
        visible: true,
        x: e.clientX,
        y: e.clientY,
        node,
      });
    },
    []
  );

  // Close context menu
  const closeContextMenu = useCallback(() => {
    setContextMenu({ visible: false, x: 0, y: 0, node: null });
  }, []);

  // Context menu actions
  const contextMenuItems: ContextMenuItem[] = contextMenu.node
    ? [
        {
          id: 'new-file',
          label: 'New File',
          action: () => {
            // Would trigger file creation dialog
            console.log('New file in:', contextMenu.node?.path);
          },
        },
        {
          id: 'new-folder',
          label: 'New Folder',
          action: () => {
            console.log('New folder in:', contextMenu.node?.path);
          },
        },
        { id: 'sep1', label: '', separator: true },
        {
          id: 'cut',
          label: 'Cut',
          shortcut: 'Ctrl+X',
          action: () => {
            console.log('Cut:', contextMenu.node?.path);
          },
        },
        {
          id: 'copy',
          label: 'Copy',
          shortcut: 'Ctrl+C',
          action: () => {
            console.log('Copy:', contextMenu.node?.path);
          },
        },
        {
          id: 'paste',
          label: 'Paste',
          shortcut: 'Ctrl+V',
          disabled: true,
          action: () => {},
        },
        { id: 'sep2', label: '', separator: true },
        {
          id: 'rename',
          label: 'Rename',
          shortcut: 'F2',
          action: () => {
            console.log('Rename:', contextMenu.node?.path);
          },
        },
        {
          id: 'delete',
          label: 'Delete',
          shortcut: 'Del',
          action: () => {
            console.log('Delete:', contextMenu.node?.path);
          },
        },
        { id: 'sep3', label: '', separator: true },
        {
          id: 'copy-path',
          label: 'Copy Path',
          action: () => {
            if (contextMenu.node) {
              navigator.clipboard.writeText(contextMenu.node.path);
            }
          },
        },
      ]
    : [];

  // Close context menu on click outside
  React.useEffect(() => {
    if (contextMenu.visible) {
      const handleClick = () => closeContextMenu();
      document.addEventListener('click', handleClick);
      return () => document.removeEventListener('click', handleClick);
    }
  }, [contextMenu.visible, closeContextMenu]);

  if (!rootPath) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-4 text-center">
        <svg
          className="w-16 h-16 mb-4 text-[var(--mviz-foreground)] opacity-30"
          viewBox="0 0 24 24"
          fill="currentColor"
        >
          <path d="M3 7V17C3 18.1046 3.89543 19 5 19H19C20.1046 19 21 18.1046 21 17V9C21 7.89543 20.1046 7 19 7H13L11 5H5C3.89543 5 3 5.89543 3 7Z" />
        </svg>
        <p className="text-sm opacity-70">No folder opened</p>
        <button
          className={cn(
            'mt-4 px-4 py-2 text-sm rounded',
            'bg-[var(--mviz-button-background)] text-[var(--mviz-button-foreground)]',
            'hover:bg-[var(--mviz-button-hover-background)]'
          )}
          onClick={handleOpenFolder}
        >
          Open Folder
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--mviz-border)]">
        <span className="text-xs font-semibold uppercase tracking-wider opacity-70">
          Explorer
        </span>
        <div className="flex items-center gap-1">
          <button
            className="p-1 hover:bg-[var(--mviz-list-hover-background)] rounded"
            onClick={() => refreshFiles()}
            title="Refresh"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M23 4v6h-6" />
              <path d="M1 20v-6h6" />
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
            </svg>
          </button>
          <button
            className="p-1 hover:bg-[var(--mviz-list-hover-background)] rounded"
            onClick={collapseAll}
            title="Collapse All"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 14l1 1 3-3 3 3 1-1-4-4-4 4z" />
              <path d="M4 8l1 1 3-3 3 3 1-1-4-4-4 4z" />
            </svg>
          </button>
        </div>
      </div>

      {/* Search */}
      <div className="px-2 py-1">
        <input
          type="text"
          placeholder="Search files..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className={cn(
            'w-full px-2 py-1 text-sm rounded',
            'bg-[var(--mviz-input-background)] text-[var(--mviz-input-foreground)]',
            'border border-[var(--mviz-input-border)]',
            'placeholder:text-[var(--mviz-input-placeholder)]',
            'focus:outline-none focus:border-[var(--mviz-focus-border)]'
          )}
        />
      </div>

      {/* Project name */}
      <div className="px-2 py-1">
        <div
          className={cn(
            'flex items-center gap-2 px-2 py-1 text-sm font-semibold',
            'cursor-pointer hover:bg-[var(--mviz-list-hover-background)] rounded'
          )}
          onClick={expandAll}
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
            <path d="M3 7V17C3 18.1046 3.89543 19 5 19H19C20.1046 19 21 18.1046 21 17V9C21 7.89543 20.1046 7 19 7H13L11 5H5C3.89543 5 3 5.89543 3 7Z" />
          </svg>
          {projectName}
        </div>
      </div>

      {/* File tree */}
      <div className="flex-1 overflow-auto py-1">
        {fileTree.map((node) => (
          <TreeItem
            key={node.path}
            node={node}
            depth={0}
            onSelect={handleSelect}
            onToggle={handleToggle}
            onContextMenu={handleContextMenu}
          />
        ))}
      </div>

      {/* Context Menu */}
      {contextMenu.visible && (
        <div
          className={cn(
            'fixed z-50 py-1 min-w-[180px]',
            'bg-[var(--mviz-input-background)]',
            'border border-[var(--mviz-border)]',
            'rounded shadow-lg'
          )}
          style={{ left: contextMenu.x, top: contextMenu.y }}
          onClick={(e) => e.stopPropagation()}
        >
          {contextMenuItems.map((item) =>
            item.separator ? (
              <div
                key={item.id}
                className="my-1 border-t border-[var(--mviz-border)]"
              />
            ) : (
              <button
                key={item.id}
                className={cn(
                  'w-full flex items-center justify-between px-3 py-1.5 text-left text-sm',
                  item.disabled
                    ? 'opacity-50 cursor-not-allowed'
                    : 'hover:bg-[var(--mviz-list-hover-background)]',
                  'transition-colors duration-100'
                )}
                onClick={() => {
                  if (!item.disabled) {
                    item.action?.();
                    closeContextMenu();
                  }
                }}
                disabled={item.disabled}
              >
                <span>{item.label}</span>
                {item.shortcut && (
                  <span className="ml-4 text-xs opacity-50">{item.shortcut}</span>
                )}
              </button>
            )
          )}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Export
// ============================================================================

export default FileTree;
