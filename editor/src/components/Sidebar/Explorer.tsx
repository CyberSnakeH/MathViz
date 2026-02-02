/**
 * Explorer Component
 *
 * Modern file explorer with Tokyo Night theme:
 * - Tree lines for visual hierarchy
 * - Search/filter in file tree
 * - Collapsible sections
 * - Context menu support
 * - Drag and drop (future)
 * - File icons based on extension
 */

import React, { useCallback, useState, useMemo, memo } from 'react';
import { cn } from '../../utils/helpers';
import { useFileStore } from '../../stores/fileStore';
import { useEditorStore } from '../../stores/editorStore';
import type { FileNode } from '../../types';

// ============================================================================
// Types
// ============================================================================

interface ExplorerProps {
  className?: string;
}

// ============================================================================
// File Icon Component
// ============================================================================

interface FileIconProps {
  extension?: string;
  isDirectory?: boolean;
  isExpanded?: boolean;
  className?: string;
}

const FileIcon: React.FC<FileIconProps> = memo(({ extension, isDirectory, isExpanded, className }) => {
  if (isDirectory) {
    return isExpanded ? (
      <svg className={cn('w-4 h-4 text-[#e0af68]', className)} viewBox="0 0 24 24" fill="currentColor">
        <path d="M19.906 9c.382 0 .749.057 1.094.162V9a3 3 0 00-3-3h-3.879a.75.75 0 01-.53-.22L11.47 3.66A2.25 2.25 0 009.879 3H6a3 3 0 00-3 3v3.162A3.756 3.756 0 014.094 9h15.812zM4.094 10.5a2.25 2.25 0 00-2.227 2.568l.857 6A2.25 2.25 0 004.951 21H19.05a2.25 2.25 0 002.227-1.932l.857-6a2.25 2.25 0 00-2.227-2.568H4.094z" />
      </svg>
    ) : (
      <svg className={cn('w-4 h-4 text-[#e0af68]', className)} viewBox="0 0 24 24" fill="currentColor">
        <path d="M19.5 21a3 3 0 003-3v-4.5a3 3 0 00-3-3h-15a3 3 0 00-3 3V18a3 3 0 003 3h15zM1.5 10.146V6a3 3 0 013-3h5.379a2.25 2.25 0 011.59.659l2.122 2.121c.14.141.331.22.53.22H19.5a3 3 0 013 3v1.146A4.483 4.483 0 0019.5 9h-15a4.483 4.483 0 00-3 1.146z" />
      </svg>
    );
  }

  // File type icons with Tokyo Night colors
  const iconConfig: Record<string, { color: string; icon: string }> = {
    mviz: { color: '#bb9af7', icon: 'M' },
    py: { color: '#7aa2f7', icon: 'Py' },
    js: { color: '#e0af68', icon: 'JS' },
    ts: { color: '#7aa2f7', icon: 'TS' },
    tsx: { color: '#7aa2f7', icon: 'Tx' },
    jsx: { color: '#7dcfff', icon: 'Jx' },
    json: { color: '#e0af68', icon: '{}' },
    md: { color: '#7aa2f7', icon: 'Md' },
    toml: { color: '#ff9e64', icon: 'T' },
    yaml: { color: '#f7768e', icon: 'Y' },
    yml: { color: '#f7768e', icon: 'Y' },
    html: { color: '#f7768e', icon: 'H' },
    css: { color: '#bb9af7', icon: 'C' },
    scss: { color: '#bb9af7', icon: 'S' },
    rs: { color: '#ff9e64', icon: 'Rs' },
    go: { color: '#7dcfff', icon: 'Go' },
    sh: { color: '#9ece6a', icon: '$' },
    txt: { color: '#565f89', icon: 'Tx' },
  };

  const config = iconConfig[extension || ''];

  if (config) {
    return (
      <span
        className={cn(
          'flex items-center justify-center w-4 h-4 rounded text-[9px] font-bold',
          className
        )}
        style={{ backgroundColor: `${config.color}20`, color: config.color }}
      >
        {config.icon}
      </span>
    );
  }

  // Default file icon
  return (
    <svg className={cn('w-4 h-4 text-[var(--mviz-foreground,#a9b1d6)] opacity-60', className)} viewBox="0 0 24 24" fill="currentColor">
      <path fillRule="evenodd" d="M5.625 1.5c-1.036 0-1.875.84-1.875 1.875v17.25c0 1.035.84 1.875 1.875 1.875h12.75c1.035 0 1.875-.84 1.875-1.875V12.75A3.75 3.75 0 0016.5 9h-1.875a1.875 1.875 0 01-1.875-1.875V5.25A3.75 3.75 0 009 1.5H5.625zM7.5 15a.75.75 0 01.75-.75h7.5a.75.75 0 010 1.5h-7.5A.75.75 0 017.5 15zm.75 2.25a.75.75 0 000 1.5H12a.75.75 0 000-1.5H8.25z" clipRule="evenodd" />
      <path d="M12.971 1.816A5.23 5.23 0 0114.25 5.25v1.875c0 .207.168.375.375.375H16.5a5.23 5.23 0 013.434 1.279 9.768 9.768 0 00-6.963-6.963z" />
    </svg>
  );
});

FileIcon.displayName = 'FileIcon';

// ============================================================================
// Language Detection
// ============================================================================

const getLanguage = (extension?: string): string => {
  const langMap: Record<string, string> = {
    mviz: 'mathviz',
    py: 'python',
    js: 'javascript',
    ts: 'typescript',
    tsx: 'typescriptreact',
    jsx: 'javascriptreact',
    json: 'json',
    md: 'markdown',
    toml: 'toml',
    yaml: 'yaml',
    yml: 'yaml',
    css: 'css',
    html: 'html',
    rs: 'rust',
    go: 'go',
    sh: 'shell',
    bash: 'shell',
  };
  return langMap[extension || ''] || 'plaintext';
};

// ============================================================================
// TreeItem Component
// ============================================================================

interface TreeItemProps {
  node: FileNode;
  depth: number;
  onToggle: (path: string) => void;
  onSelect: (node: FileNode) => void;
  selectedPath?: string;
  searchQuery?: string;
}

const TreeItem: React.FC<TreeItemProps> = memo(({
  node,
  depth,
  onToggle,
  onSelect,
  selectedPath,
  searchQuery,
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const isSelected = node.path === selectedPath;
  const isDirectory = node.type === 'directory';

  // Highlight matching text
  const highlightMatch = (text: string, query: string) => {
    if (!query) return text;
    const index = text.toLowerCase().indexOf(query.toLowerCase());
    if (index === -1) return text;
    return (
      <>
        {text.slice(0, index)}
        <span className="bg-[var(--mviz-accent,#7aa2f7)]/30 text-[var(--mviz-accent,#7aa2f7)]">
          {text.slice(index, index + query.length)}
        </span>
        {text.slice(index + query.length)}
      </>
    );
  };

  const handleClick = useCallback(() => {
    if (isDirectory) {
      onToggle(node.path);
    } else {
      onSelect(node);
    }
  }, [isDirectory, node, onToggle, onSelect]);

  return (
    <div>
      <div
        className={cn(
          'flex items-center gap-1.5 py-0.5 cursor-pointer',
          'transition-colors duration-100',
          isSelected
            ? 'bg-[var(--mviz-list-active-selection-background,#283457)] text-[var(--mviz-list-active-selection-foreground,#c0caf5)]'
            : 'hover:bg-[var(--mviz-list-hover-background,#292e42)]',
          node.isLoading && 'opacity-50'
        )}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
        onClick={handleClick}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        {/* Tree line indicators */}
        {depth > 0 && (
          <div className="absolute left-0" style={{ width: depth * 12 }}>
            {Array.from({ length: depth }).map((_, i) => (
              <div
                key={i}
                className="absolute top-0 bottom-0 w-px bg-[var(--mviz-tree-indent-guide,#292e42)]"
                style={{ left: i * 12 + 16 }}
              />
            ))}
          </div>
        )}

        {/* Expand/collapse icon for directories */}
        {isDirectory ? (
          <span
            className={cn(
              'w-4 h-4 flex items-center justify-center',
              'text-[var(--mviz-foreground,#a9b1d6)] opacity-60',
              'transition-transform duration-100',
              node.isExpanded && 'rotate-90'
            )}
          >
            <svg className="w-3 h-3" viewBox="0 0 16 16" fill="currentColor">
              <path d="M6 4l4 4-4 4V4z" />
            </svg>
          </span>
        ) : (
          <span className="w-4" />
        )}

        {/* File/folder icon */}
        <FileIcon
          extension={node.extension}
          isDirectory={isDirectory}
          isExpanded={node.isExpanded}
        />

        {/* Name */}
        <span className="text-[13px] truncate flex-1">
          {highlightMatch(node.name, searchQuery || '')}
        </span>

        {/* Quick actions on hover */}
        {isHovered && !isDirectory && (
          <div className="flex items-center gap-0.5 mr-2">
            <button
              className={cn(
                'p-0.5 rounded opacity-60 hover:opacity-100',
                'hover:bg-[var(--mviz-list-hover-background,#292e42)]'
              )}
              onClick={(e) => {
                e.stopPropagation();
                // Open in new tab action
              }}
              title="Open to the Side"
            >
              <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <line x1="12" y1="3" x2="12" y2="21" />
              </svg>
            </button>
          </div>
        )}
      </div>

      {/* Children */}
      {isDirectory && node.isExpanded && node.children && (
        <div className="relative">
          {node.children.map((child) => (
            <TreeItem
              key={child.path}
              node={child}
              depth={depth + 1}
              onToggle={onToggle}
              onSelect={onSelect}
              selectedPath={selectedPath}
              searchQuery={searchQuery}
            />
          ))}
        </div>
      )}
    </div>
  );
});

TreeItem.displayName = 'TreeItem';

// ============================================================================
// Explorer Component
// ============================================================================

const Explorer: React.FC<ExplorerProps> = ({ className = '' }) => {
  const [selectedPath, setSelectedPath] = useState<string | undefined>();
  const [searchQuery, setSearchQuery] = useState('');
  const [showSearch, setShowSearch] = useState(false);

  const rootPath = useFileStore((state) => state.rootPath);
  const fileTree = useFileStore((state) => state.fileTree);
  const toggleFolder = useFileStore((state) => state.toggleFolder);
  const openFileAction = useEditorStore((state) => state.openFile);

  // Filter file tree based on search query
  const filteredTree = useMemo(() => {
    if (!searchQuery || !fileTree) return fileTree;

    const filterNodes = (nodes: FileNode[]): FileNode[] => {
      return nodes
        .map((node) => {
          if (node.type === 'directory') {
            const filteredChildren = node.children ? filterNodes(node.children) : [];
            if (filteredChildren.length > 0 || node.name.toLowerCase().includes(searchQuery.toLowerCase())) {
              return { ...node, children: filteredChildren, isExpanded: true };
            }
            return null;
          }
          return node.name.toLowerCase().includes(searchQuery.toLowerCase()) ? node : null;
        })
        .filter(Boolean) as FileNode[];
    };

    return filterNodes(fileTree);
  }, [fileTree, searchQuery]);

  // Handle toggle expand/collapse
  const handleToggle = useCallback(
    (path: string) => {
      toggleFolder(path);
    },
    [toggleFolder]
  );

  // Handle file selection
  const handleSelect = useCallback(
    async (node: FileNode) => {
      setSelectedPath(node.path);

      if (node.type === 'file') {
        try {
          const { invoke } = await import('@tauri-apps/api/core');
          const content = await invoke<string>('read_file', { path: node.path });
          openFileAction(node.path, content, getLanguage(node.extension));
        } catch (error) {
          console.error('Failed to open file:', error);
        }
      }
    },
    [openFileAction]
  );

  // Handle refresh
  const handleRefresh = useCallback(async () => {
    if (rootPath) {
      const { invoke } = await import('@tauri-apps/api/core');
      try {
        await invoke('read_directory', { path: rootPath, recursive: false });
      } catch (error) {
        console.error('Failed to refresh directory:', error);
      }
    }
  }, [rootPath]);

  // Handle collapse all
  const handleCollapseAll = useCallback(() => {
    const collapseRecursive = (node: FileNode) => {
      if (node.type === 'directory' && node.isExpanded) {
        toggleFolder(node.path);
      }
      node.children?.forEach(collapseRecursive);
    };

    if (fileTree) {
      fileTree.forEach(collapseRecursive);
    }
  }, [fileTree, toggleFolder]);

  // Handle new file
  const handleNewFile = useCallback(async () => {
    const pathModule = await import('@tauri-apps/api/path');
    const basePath = selectedPath
      ? await pathModule.dirname(selectedPath)
      : rootPath;

    if (!basePath) return;

    const fileName = prompt('Enter file name:');
    if (!fileName) return;

    try {
      const { invoke } = await import('@tauri-apps/api/core');
      const filePath = `${basePath}/${fileName}`;
      await invoke('create_file', { path: filePath, content: '' });
      handleRefresh();
    } catch (error) {
      console.error('Failed to create file:', error);
    }
  }, [selectedPath, rootPath, handleRefresh]);

  // Handle new folder
  const handleNewFolder = useCallback(async () => {
    const pathModule = await import('@tauri-apps/api/path');
    const basePath = selectedPath
      ? await pathModule.dirname(selectedPath)
      : rootPath;

    if (!basePath) return;

    const folderName = prompt('Enter folder name:');
    if (!folderName) return;

    try {
      const { invoke } = await import('@tauri-apps/api/core');
      await invoke('create_directory', { path: `${basePath}/${folderName}` });
      handleRefresh();
    } catch (error) {
      console.error('Failed to create folder:', error);
    }
  }, [selectedPath, rootPath, handleRefresh]);

  // Empty state
  if (!rootPath || !fileTree || fileTree.length === 0) {
    return (
      <div className={cn('flex flex-col h-full items-center justify-center p-6 text-center', className)}>
        <svg
          className="w-16 h-16 mb-4 opacity-20"
          viewBox="0 0 24 24"
          fill="currentColor"
        >
          <path d="M19.5 21a3 3 0 003-3v-4.5a3 3 0 00-3-3h-15a3 3 0 00-3 3V18a3 3 0 003 3h15zM1.5 10.146V6a3 3 0 013-3h5.379a2.25 2.25 0 011.59.659l2.122 2.121c.14.141.331.22.53.22H19.5a3 3 0 013 3v1.146A4.483 4.483 0 0019.5 9h-15a4.483 4.483 0 00-3 1.146z" />
        </svg>
        <p className="text-sm opacity-70 mb-2">No folder opened</p>
        <button
          className={cn(
            'px-4 py-2 text-sm rounded-lg',
            'bg-[var(--mviz-button-background,#7aa2f7)]',
            'text-[var(--mviz-button-foreground,#1a1b26)]',
            'hover:bg-[var(--mviz-button-hover-background,#89b4fa)]',
            'transition-colors duration-100'
          )}
          onClick={async () => {
            const dialogModule = await import('@tauri-apps/plugin-dialog');
            const selected = await dialogModule.open({ directory: true, multiple: false });
            if (selected && typeof selected === 'string') {
              useFileStore.getState().setRootPath(selected);
            }
          }}
        >
          Open Folder
        </button>
      </div>
    );
  }

  return (
    <div className={cn('flex flex-col h-full', className)}>
      {/* Toolbar */}
      <div
        className={cn(
          'flex items-center justify-between gap-1 px-2 py-1.5',
          'border-b border-[var(--mviz-border,#1a1b26)]'
        )}
      >
        {/* Search toggle or input */}
        {showSearch ? (
          <div className="flex-1 relative">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Filter files..."
              className={cn(
                'w-full pl-7 pr-2 py-1 text-xs rounded',
                'bg-[var(--mviz-input-background,#1a1b26)]',
                'border border-[var(--mviz-input-border,#3b4261)]',
                'text-[var(--mviz-input-foreground,#a9b1d6)]',
                'placeholder:text-[var(--mviz-input-placeholder,#565f89)]',
                'focus:outline-none focus:border-[var(--mviz-focus-border,#7aa2f7)]'
              )}
              autoFocus
              onBlur={() => {
                if (!searchQuery) setShowSearch(false);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Escape') {
                  setSearchQuery('');
                  setShowSearch(false);
                }
              }}
            />
            <svg
              className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 opacity-50"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
          </div>
        ) : (
          <div className="flex-1" />
        )}

        {/* Action buttons */}
        <div className="flex items-center gap-0.5">
          {!showSearch && (
            <button
              className={cn(
                'p-1 rounded opacity-60 hover:opacity-100',
                'hover:bg-[var(--mviz-list-hover-background,#292e42)]',
                'transition-all duration-100'
              )}
              onClick={() => setShowSearch(true)}
              title="Filter (Ctrl+F)"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
            </button>
          )}
          <button
            className={cn(
              'p-1 rounded opacity-60 hover:opacity-100',
              'hover:bg-[var(--mviz-list-hover-background,#292e42)]',
              'transition-all duration-100'
            )}
            onClick={handleNewFile}
            title="New File"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="12" y1="18" x2="12" y2="12" />
              <line x1="9" y1="15" x2="15" y2="15" />
            </svg>
          </button>
          <button
            className={cn(
              'p-1 rounded opacity-60 hover:opacity-100',
              'hover:bg-[var(--mviz-list-hover-background,#292e42)]',
              'transition-all duration-100'
            )}
            onClick={handleNewFolder}
            title="New Folder"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
              <line x1="12" y1="11" x2="12" y2="17" />
              <line x1="9" y1="14" x2="15" y2="14" />
            </svg>
          </button>
          <button
            className={cn(
              'p-1 rounded opacity-60 hover:opacity-100',
              'hover:bg-[var(--mviz-list-hover-background,#292e42)]',
              'transition-all duration-100'
            )}
            onClick={handleRefresh}
            title="Refresh"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="23 4 23 10 17 10" />
              <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
            </svg>
          </button>
          <button
            className={cn(
              'p-1 rounded opacity-60 hover:opacity-100',
              'hover:bg-[var(--mviz-list-hover-background,#292e42)]',
              'transition-all duration-100'
            )}
            onClick={handleCollapseAll}
            title="Collapse All"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="4 14 10 14 10 20" />
              <polyline points="20 10 14 10 14 4" />
              <line x1="14" y1="10" x2="21" y2="3" />
              <line x1="3" y1="21" x2="10" y2="14" />
            </svg>
          </button>
        </div>
      </div>

      {/* File tree */}
      <div className="flex-1 overflow-auto py-1">
        {filteredTree?.map((node) => (
          <TreeItem
            key={node.path}
            node={node}
            depth={0}
            onToggle={handleToggle}
            onSelect={handleSelect}
            selectedPath={selectedPath}
            searchQuery={searchQuery}
          />
        ))}

        {/* No results message */}
        {searchQuery && filteredTree?.length === 0 && (
          <div className="p-4 text-center text-sm opacity-50">
            No files matching "{searchQuery}"
          </div>
        )}
      </div>
    </div>
  );
};

// ============================================================================
// Export
// ============================================================================

export default Explorer;
