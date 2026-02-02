/**
 * useEditor Hook
 *
 * Custom hook for managing Monaco editor state and operations.
 */

import { useCallback, useRef } from 'react';
import type * as Monaco from 'monaco-editor';
import { useEditorStore, selectActiveFile } from '../stores/editorStore';
import type { OpenFile } from '../types';

interface UseEditorOptions {
  onSave?: (file: OpenFile) => Promise<void>;
  onContentChange?: (content: string) => void;
}

export function useEditor(options: UseEditorOptions = {}) {
  const { onSave, onContentChange } = options;
  const editorRef = useRef<Monaco.editor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<typeof Monaco | null>(null);

  const openFiles = useEditorStore((state) => state.openFiles);
  const activeFileId = useEditorStore((state) => state.activeFileId);
  const openFile = useEditorStore((state) => state.openFile);
  const closeFile = useEditorStore((state) => state.closeFile);
  const setActiveFile = useEditorStore((state) => state.setActiveFile);
  const updateFileContent = useEditorStore((state) => state.updateFileContent);
  const markFileSaved = useEditorStore((state) => state.markFileSaved);
  const setCursorPosition = useEditorStore((state) => state.setCursorPosition);

  // Get active file
  const activeFile = useEditorStore(selectActiveFile);

  // Handle editor mount
  const handleEditorMount = useCallback(
    (editor: Monaco.editor.IStandaloneCodeEditor, monaco: typeof Monaco) => {
      editorRef.current = editor;
      monacoRef.current = monaco;

      // Set up cursor position tracking
      editor.onDidChangeCursorPosition((e) => {
        setCursorPosition({
          lineNumber: e.position.lineNumber,
          column: e.position.column,
        });
      });

      // Set up content change tracking
      editor.onDidChangeModelContent(() => {
        const content = editor.getValue();
        if (activeFileId) {
          updateFileContent(activeFileId, content);
        }
        onContentChange?.(content);
      });

      // Set up keyboard shortcuts
      editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
        handleSave();
      });

      editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyW, () => {
        if (activeFileId) {
          closeFile(activeFileId);
        }
      });

      // Focus editor
      editor.focus();
    },
    [activeFileId, setCursorPosition, updateFileContent, onContentChange, closeFile]
  );

  // Handle save
  const handleSave = useCallback(async () => {
    if (activeFile && activeFile.isDirty) {
      try {
        await onSave?.(activeFile);
        markFileSaved(activeFile.id);
      } catch (error) {
        console.error('Failed to save file:', error);
      }
    }
  }, [activeFile, onSave, markFileSaved]);

  // Handle file open
  const handleOpenFile = useCallback(
    (file: OpenFile) => {
      openFile(file.path, file.content, file.language);

      // Update editor content
      if (editorRef.current && monacoRef.current) {
        const model = monacoRef.current.editor.createModel(
          file.content,
          file.language,
          monacoRef.current.Uri.file(file.path)
        );
        editorRef.current.setModel(model);
      }
    },
    [openFile]
  );

  // Handle tab switch
  const handleTabSwitch = useCallback(
    (fileId: string) => {
      setActiveFile(fileId);

      const file = openFiles.get(fileId);
      if (file && editorRef.current && monacoRef.current) {
        const uri = monacoRef.current.Uri.file(file.path);
        let model = monacoRef.current.editor.getModel(uri);

        if (!model) {
          model = monacoRef.current.editor.createModel(
            file.content,
            file.language,
            uri
          );
        }

        editorRef.current.setModel(model);
        editorRef.current.focus();

        // Restore cursor position
        if (file.cursorPosition) {
          editorRef.current.setPosition({
            lineNumber: file.cursorPosition.lineNumber,
            column: file.cursorPosition.column,
          });
        }

        // Restore scroll position
        if (file.scrollPosition) {
          editorRef.current.setScrollPosition({
            scrollTop: file.scrollPosition.scrollTop,
            scrollLeft: file.scrollPosition.scrollLeft,
          });
        }
      }
    },
    [setActiveFile, openFiles]
  );

  // Handle tab close
  const handleTabClose = useCallback(
    (fileId: string) => {
      const file = openFiles.get(fileId);

      if (file?.isDirty) {
        // TODO: Show save prompt
        console.log('File has unsaved changes');
      }

      // Clean up model
      if (monacoRef.current) {
        const uri = monacoRef.current.Uri.file(file?.path || '');
        const model = monacoRef.current.editor.getModel(uri);
        model?.dispose();
      }

      closeFile(fileId);
    },
    [openFiles, closeFile]
  );

  // Go to line
  const goToLine = useCallback((lineNumber: number, column: number = 1) => {
    if (editorRef.current) {
      editorRef.current.setPosition({ lineNumber, column });
      editorRef.current.revealLineInCenter(lineNumber);
      editorRef.current.focus();
    }
  }, []);

  // Insert text at cursor
  const insertText = useCallback((text: string) => {
    if (editorRef.current) {
      const position = editorRef.current.getPosition();
      if (position) {
        editorRef.current.executeEdits('insert', [
          {
            range: {
              startLineNumber: position.lineNumber,
              startColumn: position.column,
              endLineNumber: position.lineNumber,
              endColumn: position.column,
            },
            text,
          },
        ]);
      }
    }
  }, []);

  // Format document
  const formatDocument = useCallback(async () => {
    if (editorRef.current) {
      await editorRef.current.getAction('editor.action.formatDocument')?.run();
    }
  }, []);

  // Find and replace
  const openFind = useCallback(() => {
    if (editorRef.current) {
      editorRef.current.getAction('actions.find')?.run();
    }
  }, []);

  const openFindAndReplace = useCallback(() => {
    if (editorRef.current) {
      editorRef.current.getAction('editor.action.startFindReplaceAction')?.run();
    }
  }, []);

  // Undo/Redo
  const undo = useCallback(() => {
    if (editorRef.current) {
      editorRef.current.trigger('keyboard', 'undo', null);
    }
  }, []);

  const redo = useCallback(() => {
    if (editorRef.current) {
      editorRef.current.trigger('keyboard', 'redo', null);
    }
  }, []);

  // Get selected text
  const getSelectedText = useCallback(() => {
    if (editorRef.current) {
      const selection = editorRef.current.getSelection();
      if (selection) {
        return editorRef.current.getModel()?.getValueInRange(selection) || '';
      }
    }
    return '';
  }, []);

  // Set markers (for diagnostics)
  const setMarkers = useCallback(
    (markers: Monaco.editor.IMarkerData[]) => {
      if (monacoRef.current && activeFile) {
        const uri = monacoRef.current.Uri.file(activeFile.path);
        const model = monacoRef.current.editor.getModel(uri);
        if (model) {
          monacoRef.current.editor.setModelMarkers(model, 'mathviz', markers);
        }
      }
    },
    [activeFile]
  );

  return {
    // Refs
    editorRef,
    monacoRef,

    // State
    activeFile,
    openFiles,
    activeFileId,

    // Editor handlers
    handleEditorMount,
    handleOpenFile,
    handleTabSwitch,
    handleTabClose,
    handleSave,

    // Editor actions
    goToLine,
    insertText,
    formatDocument,
    openFind,
    openFindAndReplace,
    undo,
    redo,
    getSelectedText,
    setMarkers,
  };
}

export default useEditor;
