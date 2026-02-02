/**
 * MathViz Editor Stores
 *
 * Re-exports all Zustand stores for convenient access.
 */

export {
  useEditorStore,
  selectActiveFile,
  selectOpenFilesArray,
  selectHasUnsavedFiles,
  selectDiagnosticsForFile,
  selectFileById,
} from './editorStore';

export {
  useFileStore,
  selectVisibleTree,
  selectIsExpanded,
  selectIsSelected,
  selectIsLoading,
  selectNodeByPath,
  selectHasClipboard,
} from './fileStore';

export {
  useLayoutStore,
  selectLeftSidebarWidth,
  selectBottomPanelHeight,
  selectRightPanelWidth,
} from './layoutStore';

export {
  useCompilerStore,
  selectCompilerStatus,
  selectCompilerOutput,
  selectIsCompiling,
  selectIsRunning,
  selectBreakpoints,
} from './compilerStore';

export {
  useTerminalStore,
  selectActiveTerminal,
  selectTerminalCount,
} from './terminalStore';

export {
  usePreviewStore,
  selectIsPreviewReady,
  selectPreviewProgress,
} from './previewStore';
