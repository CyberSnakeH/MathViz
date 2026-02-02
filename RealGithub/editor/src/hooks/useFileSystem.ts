/**
 * useFileSystem Hook
 *
 * Custom hook for file system operations using Tauri commands.
 */

import { useState, useCallback, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import { useFileStore } from '../stores/fileStore';
import type { FileNode, FileMetadata } from '../types';

interface DirectoryEntry {
  name: string;
  path: string;
  is_directory: boolean;
  is_expanded: boolean;
  children: DirectoryEntry[] | null;
  extension: string | null;
  is_hidden: boolean;
}

interface FileChangeEvent {
  kind: string;
  paths: string[];
}

export function useFileSystem() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const {
    rootPath: workspacePath,
    fileTree,
    setRootPath: setWorkspacePath,
    setFileTree,
    updateNode,
    addNode,
    removeNode,
    toggleFolder: toggleExpanded,
  } = useFileStore();

  // Convert backend DirectoryEntry to FileNode
  const convertToFileNode = useCallback(
    (entry: DirectoryEntry, _parentPath: string = ''): FileNode => {
      const id = entry.path;
      return {
        id,
        name: entry.name,
        path: entry.path,
        type: entry.is_directory ? 'directory' : 'file',
        extension: entry.extension as FileNode['extension'] || undefined,
        children: entry.children?.map((child) => convertToFileNode(child, entry.path)),
        isExpanded: entry.is_expanded,
        isLoading: false,
      };
    },
    []
  );

  // Read file content
  const readFile = useCallback(async (path: string): Promise<string> => {
    try {
      const content = await invoke<string>('read_file', { path });
      return content;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(`Failed to read file: ${message}`);
      throw err;
    }
  }, []);

  // Write file content
  const writeFile = useCallback(async (path: string, content: string): Promise<void> => {
    try {
      await invoke('write_file', { path, content });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(`Failed to write file: ${message}`);
      throw err;
    }
  }, []);

  // Read directory
  const readDirectory = useCallback(
    async (path: string, recursive: boolean = false): Promise<FileNode[]> => {
      try {
        setIsLoading(true);
        const entries = await invoke<DirectoryEntry[]>('read_directory', {
          path,
          recursive,
        });
        return entries.map((entry: DirectoryEntry) => convertToFileNode(entry, path));
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(`Failed to read directory: ${message}`);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [convertToFileNode]
  );

  // Open workspace
  const openWorkspace = useCallback(async (path?: string): Promise<void> => {
    try {
      let workspaceDir = path;

      if (!workspaceDir) {
        // Dynamic import to avoid module resolution issues
        const dialogModule = await import('@tauri-apps/plugin-dialog');
        const selected = await dialogModule.open({
          directory: true,
          multiple: false,
          title: 'Open Workspace',
        });

        if (!selected || Array.isArray(selected)) {
          return;
        }

        workspaceDir = selected;
      }

      setIsLoading(true);
      const pathModule = await import('@tauri-apps/api/path');
      const dirName = await pathModule.basename(workspaceDir);
      setWorkspacePath(workspaceDir, dirName);

      const entries = await readDirectory(workspaceDir, false);
      setFileTree(entries);

      // Start watching for changes
      await invoke('watch_directory', { path: workspaceDir });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(`Failed to open workspace: ${message}`);
    } finally {
      setIsLoading(false);
    }
  }, [readDirectory, setWorkspacePath, setFileTree]);

  // Create new file
  const createFile = useCallback(
    async (parentPath: string, name: string, content: string = ''): Promise<string> => {
      const filePath = `${parentPath}/${name}`;

      try {
        await invoke('create_file', { path: filePath, content });

        const pathModule = await import('@tauri-apps/api/path');
        const ext = await pathModule.extname(filePath);
        const newNode: FileNode = {
          id: filePath,
          name,
          path: filePath,
          type: 'file',
          extension: ext as FileNode['extension'] || undefined,
          isExpanded: false,
          isLoading: false,
        };

        addNode(parentPath, newNode);

        return filePath;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(`Failed to create file: ${message}`);
        throw err;
      }
    },
    [addNode]
  );

  // Create new directory
  const createDirectory = useCallback(
    async (parentPath: string, name: string): Promise<string> => {
      const dirPath = `${parentPath}/${name}`;

      try {
        await invoke('create_directory', { path: dirPath });

        const newNode: FileNode = {
          id: dirPath,
          name,
          path: dirPath,
          type: 'directory',
          children: [],
          isExpanded: false,
          isLoading: false,
        };

        addNode(parentPath, newNode);

        return dirPath;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(`Failed to create directory: ${message}`);
        throw err;
      }
    },
    [addNode]
  );

  // Delete file or directory
  const deletePath = useCallback(
    async (path: string): Promise<void> => {
      const pathModule = await import('@tauri-apps/api/path');
      const dialogModule = await import('@tauri-apps/plugin-dialog');
      const fileName = await pathModule.basename(path);
      const confirmed = await dialogModule.confirm(
        `Are you sure you want to delete "${fileName}"?`,
        { title: 'Confirm Delete', kind: 'warning' }
      );

      if (!confirmed) return;

      try {
        await invoke('delete_path', { path });
        removeNode(path);
      } catch (err) {
        const errMessage = err instanceof Error ? err.message : String(err);
        setError(`Failed to delete: ${errMessage}`);
        throw err;
      }
    },
    [removeNode]
  );

  // Rename file or directory
  const renamePath = useCallback(
    async (oldPath: string, newName: string): Promise<string> => {
      const pathModule = await import('@tauri-apps/api/path');
      const parentDir = await pathModule.dirname(oldPath);
      const newPath = `${parentDir}/${newName}`;

      try {
        await invoke('rename_path', { oldPath, newPath });

        // Update tree by renaming node
        updateNode(oldPath, {
          id: newPath,
          name: newName,
          path: newPath,
        });

        return newPath;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(`Failed to rename: ${message}`);
        throw err;
      }
    },
    [updateNode]
  );

  // Copy file or directory
  const copyPath = useCallback(
    async (source: string, destination: string): Promise<void> => {
      try {
        await invoke('copy_path', { source, destination });
        // Refresh parent directory
        const pathModule = await import('@tauri-apps/api/path');
        const parentDir = await pathModule.dirname(destination);
        await readDirectory(parentDir, false);
        // Update tree with new entries handled by readDirectory
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        setError(`Failed to copy: ${errMsg}`);
        throw err;
      }
    },
    [readDirectory]
  );

  // Get file info
  const getFileInfo = useCallback(async (path: string): Promise<FileMetadata> => {
    try {
      const info = await invoke<{
        size: number;
        modified: number | null;
        created: number | null;
        is_hidden: boolean;
      }>('get_file_info', { path });

      return {
        size: info.size,
        modified: info.modified ? new Date(info.modified * 1000) : undefined,
        created: info.created ? new Date(info.created * 1000) : undefined,
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(`Failed to get file info: ${message}`);
      throw err;
    }
  }, []);

  // Open file dialog
  const openFileDialog = useCallback(async (): Promise<string | null> => {
    const dialogModule = await import('@tauri-apps/plugin-dialog');
    const selected = await dialogModule.open({
      multiple: false,
      filters: [
        {
          name: 'MathViz Files',
          extensions: ['mviz', 'py'],
        },
        {
          name: 'All Files',
          extensions: ['*'],
        },
      ],
    });

    if (!selected || Array.isArray(selected)) {
      return null;
    }

    return selected;
  }, []);

  // Save file dialog
  const saveFileDialog = useCallback(async (defaultPath?: string): Promise<string | null> => {
    const dialogModule = await import('@tauri-apps/plugin-dialog');
    const selected = await dialogModule.save({
      defaultPath,
      filters: [
        {
          name: 'MathViz Files',
          extensions: ['mviz'],
        },
        {
          name: 'Python Files',
          extensions: ['py'],
        },
        {
          name: 'All Files',
          extensions: ['*'],
        },
      ],
    });

    return selected ?? null;
  }, []);

  // Expand directory
  const expandDirectory = useCallback(
    async (path: string): Promise<void> => {
      try {
        // Mark as loading
        updateNode(path, { isLoading: true });

        const entries = await readDirectory(path, false);

        // Update with children
        updateNode(path, {
          children: entries,
          isExpanded: true,
          isLoading: false,
        });
      } catch (_err) {
        updateNode(path, { isLoading: false });
      }
    },
    [readDirectory, updateNode]
  );

  // Collapse directory
  const collapseDirectory = useCallback(
    (path: string): void => {
      toggleExpanded(path);
    },
    [toggleExpanded]
  );

  // Search files
  const searchFiles = useCallback(
    async (
      query: string,
      options?: {
        filePattern?: string;
        caseSensitive?: boolean;
        maxResults?: number;
      }
    ): Promise<
      Array<{
        path: string;
        lineNumber: number;
        lineContent: string;
        matchStart: number;
        matchEnd: number;
      }>
    > => {
      if (!workspacePath) return [];

      try {
        const results = await invoke<
          Array<{
            path: string;
            line_number: number;
            line_content: string;
            match_start: number;
            match_end: number;
          }>
        >('search_files', {
          path: workspacePath,
          query,
          filePattern: options?.filePattern,
          caseSensitive: options?.caseSensitive ?? false,
          maxResults: options?.maxResults ?? 1000,
        });

        return results.map((r: { path: string; line_number: number; line_content: string; match_start: number; match_end: number }) => ({
          path: r.path,
          lineNumber: r.line_number,
          lineContent: r.line_content,
          matchStart: r.match_start,
          matchEnd: r.match_end,
        }));
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(`Search failed: ${message}`);
        return [];
      }
    },
    [workspacePath]
  );

  // Listen for file change events
  useEffect(() => {
    let unlisten: UnlistenFn | undefined;

    const setupListener = async () => {
      unlisten = await listen<FileChangeEvent>('file-change', (event) => {
        const { kind, paths } = event.payload;

        // Handle file changes
        console.log('File change:', kind, paths);

        // Refresh affected directories
        paths.forEach(async (filePath) => {
          const pathModule = await import('@tauri-apps/api/path');
          const parentDir = await pathModule.dirname(filePath);
          if (workspacePath && (parentDir === workspacePath || parentDir.startsWith(workspacePath + '/'))) {
            // Refresh parent directory
            expandDirectory(parentDir);
          }
        });
      });
    };

    if (workspacePath) {
      setupListener();
    }

    return () => {
      unlisten?.();
    };
  }, [workspacePath, expandDirectory]);

  return {
    // State
    isLoading,
    error,
    workspacePath,
    fileTree,

    // File operations
    readFile,
    writeFile,
    createFile,
    createDirectory,
    deletePath,
    renamePath,
    copyPath,
    getFileInfo,

    // Directory operations
    readDirectory,
    openWorkspace,
    expandDirectory,
    collapseDirectory,

    // Dialog operations
    openFileDialog,
    saveFileDialog,

    // Search
    searchFiles,

    // Error handling
    clearError: () => setError(null),
  };
}

export default useFileSystem;
