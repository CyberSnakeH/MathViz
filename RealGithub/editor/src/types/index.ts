/**
 * MathViz Editor Type Definitions
 *
 * Comprehensive type definitions for the MathViz IDE frontend.
 */

// ============================================================================
// File System Types
// ============================================================================

export type FileType =
  | 'file'
  | 'directory'
  | 'symlink';

export type FileExtension =
  | 'mviz'
  | 'py'
  | 'json'
  | 'toml'
  | 'yaml'
  | 'yml'
  | 'md'
  | 'txt'
  | 'gitignore'
  | 'other';

export interface FileNode {
  id: string;
  name: string;
  path: string;
  type: FileType;
  extension?: FileExtension;
  children?: FileNode[];
  isExpanded?: boolean;
  isLoading?: boolean;
  metadata?: FileMetadata;
}

export interface FileMetadata {
  size?: number;
  modified?: Date;
  created?: Date;
  readonly?: boolean;
  permissions?: string;
}

// ============================================================================
// Editor Types
// ============================================================================

export interface OpenFile {
  id: string;
  path: string;
  name: string;
  content: string;
  originalContent: string;
  language: string;
  isDirty: boolean;
  isReadOnly: boolean;
  cursorPosition: CursorPosition;
  scrollPosition: ScrollPosition;
  viewState?: unknown; // Monaco editor view state
}

export interface CursorPosition {
  lineNumber: number;
  column: number;
}

export interface ScrollPosition {
  scrollTop: number;
  scrollLeft: number;
}

export interface Selection {
  startLineNumber: number;
  startColumn: number;
  endLineNumber: number;
  endColumn: number;
}

export interface EditorTab {
  id: string;
  fileId: string;
  title: string;
  isDirty: boolean;
  isPinned: boolean;
  isPreview: boolean;
}

// ============================================================================
// Diagnostic Types
// ============================================================================

export type DiagnosticSeverity = 'error' | 'warning' | 'info' | 'hint';

export interface Diagnostic {
  id: string;
  filePath: string;
  severity: DiagnosticSeverity;
  message: string;
  source?: string;
  code?: string | number;
  range: DiagnosticRange;
  relatedInformation?: RelatedInformation[];
}

export interface DiagnosticRange {
  startLine: number;
  startColumn: number;
  endLine: number;
  endColumn: number;
}

export interface RelatedInformation {
  filePath: string;
  range: DiagnosticRange;
  message: string;
}

export interface DiagnosticSummary {
  errors: number;
  warnings: number;
  infos: number;
  hints: number;
}

// ============================================================================
// Git Types
// ============================================================================

export type GitFileStatus =
  | 'modified'
  | 'added'
  | 'deleted'
  | 'renamed'
  | 'copied'
  | 'untracked'
  | 'ignored'
  | 'conflicted';

export interface GitFile {
  path: string;
  status: GitFileStatus;
  staged: boolean;
  originalPath?: string; // For renamed files
}

export interface GitBranch {
  name: string;
  isRemote: boolean;
  isCurrent: boolean;
  upstream?: string;
  ahead?: number;
  behind?: number;
}

export interface GitStatus {
  branch: string;
  isClean: boolean;
  ahead: number;
  behind: number;
  files: GitFile[];
  staged: GitFile[];
  unstaged: GitFile[];
  untracked: GitFile[];
  conflicted: GitFile[];
}

export interface GitCommit {
  hash: string;
  shortHash: string;
  message: string;
  author: string;
  authorEmail: string;
  date: Date;
  parents: string[];
}

export interface GitRemote {
  name: string;
  fetchUrl: string;
  pushUrl: string;
}

// ============================================================================
// Terminal Types
// ============================================================================

export interface TerminalTab {
  id: string;
  title: string;
  cwd: string;
  isActive: boolean;
  pid?: number;
  shellType?: ShellType;
}

export type ShellType = 'bash' | 'zsh' | 'fish' | 'powershell' | 'cmd';

export interface TerminalDimensions {
  cols: number;
  rows: number;
}

export interface TerminalOptions {
  cwd?: string;
  env?: Record<string, string>;
  shell?: string;
  args?: string[];
}

// ============================================================================
// Preview Types
// ============================================================================

export type PreviewQuality = 'low' | 'medium' | 'high' | 'production';

export type PreviewStatus =
  | 'idle'
  | 'compiling'
  | 'rendering'
  | 'ready'
  | 'error';

export interface PreviewState {
  status: PreviewStatus;
  currentTime: number;
  duration: number;
  isPlaying: boolean;
  quality: PreviewQuality;
  error?: string;
  videoUrl?: string;
  thumbnailUrl?: string;
}

export interface RenderProgress {
  current: number;
  total: number;
  phase: 'parsing' | 'generating' | 'rendering' | 'encoding';
  message?: string;
}

export interface ExportOptions {
  format: 'mp4' | 'gif' | 'webm' | 'png_sequence';
  quality: PreviewQuality;
  fps: number;
  width: number;
  height: number;
  outputPath: string;
}

// ============================================================================
// LSP Types
// ============================================================================

export type LspStatus = 'stopped' | 'starting' | 'running' | 'error';

export interface LspState {
  status: LspStatus;
  serverName?: string;
  serverVersion?: string;
  error?: string;
  capabilities?: LspCapabilities;
}

export interface LspCapabilities {
  completionProvider: boolean;
  hoverProvider: boolean;
  definitionProvider: boolean;
  referencesProvider: boolean;
  renameProvider: boolean;
  documentFormattingProvider: boolean;
  codeActionProvider: boolean;
}

export interface CompletionItem {
  label: string;
  kind: CompletionItemKind;
  detail?: string;
  documentation?: string;
  insertText: string;
  sortText?: string;
  filterText?: string;
}

export type CompletionItemKind =
  | 'text'
  | 'method'
  | 'function'
  | 'constructor'
  | 'field'
  | 'variable'
  | 'class'
  | 'interface'
  | 'module'
  | 'property'
  | 'unit'
  | 'value'
  | 'enum'
  | 'keyword'
  | 'snippet'
  | 'color'
  | 'file'
  | 'reference'
  | 'folder'
  | 'enumMember'
  | 'constant'
  | 'struct'
  | 'event'
  | 'operator'
  | 'typeParameter';

export interface HoverInfo {
  contents: string;
  range?: DiagnosticRange;
}

export interface Location {
  uri: string;
  range: DiagnosticRange;
}

// ============================================================================
// UI Types
// ============================================================================

export type PanelPosition = 'left' | 'right' | 'bottom';

export type SidebarPanel =
  | 'explorer'
  | 'search'
  | 'git'
  | 'debug';

export type BottomPanel =
  | 'terminal'
  | 'problems'
  | 'output'
  | 'debug-console';

export interface PanelState {
  isVisible: boolean;
  size: number;
  minSize: number;
  maxSize: number;
}

export interface LayoutState {
  leftSidebar: PanelState & { activePanel: SidebarPanel };
  rightPanel: PanelState;
  bottomPanel: PanelState & { activePanel: BottomPanel };
}

export interface WindowState {
  isMaximized: boolean;
  isFullscreen: boolean;
  width: number;
  height: number;
  x: number;
  y: number;
}

// ============================================================================
// Context Menu Types
// ============================================================================

export interface ContextMenuItem {
  id: string;
  label: string;
  icon?: string;
  shortcut?: string;
  disabled?: boolean;
  separator?: boolean;
  submenu?: ContextMenuItem[];
  action?: () => void;
}

export interface ContextMenuPosition {
  x: number;
  y: number;
}

// ============================================================================
// Search Types
// ============================================================================

export interface SearchOptions {
  query: string;
  isRegex: boolean;
  isCaseSensitive: boolean;
  isWholeWord: boolean;
  includePattern?: string;
  excludePattern?: string;
}

export interface SearchResult {
  filePath: string;
  matches: SearchMatch[];
}

export interface SearchMatch {
  lineNumber: number;
  column: number;
  length: number;
  lineContent: string;
  previewBefore?: string;
  previewAfter?: string;
}

export interface ReplaceOptions extends SearchOptions {
  replacement: string;
  preserveCase: boolean;
}

// ============================================================================
// Keybinding Types
// ============================================================================

export interface Keybinding {
  key: string;
  command: string;
  when?: string;
  args?: unknown;
}

export interface Command {
  id: string;
  title: string;
  category?: string;
  handler: (...args: unknown[]) => void | Promise<void>;
}

// ============================================================================
// Settings Types
// ============================================================================

export interface EditorSettings {
  fontSize: number;
  fontFamily: string;
  lineHeight: number;
  tabSize: number;
  insertSpaces: boolean;
  wordWrap: 'off' | 'on' | 'wordWrapColumn' | 'bounded';
  wordWrapColumn: number;
  minimap: {
    enabled: boolean;
    side: 'left' | 'right';
    maxColumn: number;
  };
  scrollBeyondLastLine: boolean;
  renderWhitespace: 'none' | 'boundary' | 'selection' | 'all';
  cursorStyle: 'line' | 'block' | 'underline' | 'line-thin' | 'block-outline' | 'underline-thin';
  cursorBlinking: 'blink' | 'smooth' | 'phase' | 'expand' | 'solid';
  formatOnSave: boolean;
  formatOnPaste: boolean;
  autoSave: 'off' | 'afterDelay' | 'onFocusChange' | 'onWindowChange';
  autoSaveDelay: number;
}

export interface TerminalSettings {
  fontSize: number;
  fontFamily: string;
  lineHeight: number;
  cursorStyle: 'block' | 'underline' | 'bar';
  cursorBlink: boolean;
  scrollback: number;
  copyOnSelection: boolean;
}

export interface PreviewSettings {
  defaultQuality: PreviewQuality;
  autoRefresh: boolean;
  refreshDelay: number;
  showGrid: boolean;
  backgroundColor: string;
}

export interface AppSettings {
  theme: 'dark' | 'light' | 'system';
  editor: EditorSettings;
  terminal: TerminalSettings;
  preview: PreviewSettings;
}

// ============================================================================
// Event Types
// ============================================================================

export type EditorEvent =
  | { type: 'file-opened'; payload: { path: string } }
  | { type: 'file-closed'; payload: { path: string } }
  | { type: 'file-saved'; payload: { path: string } }
  | { type: 'file-changed'; payload: { path: string; content: string } }
  | { type: 'cursor-moved'; payload: CursorPosition }
  | { type: 'selection-changed'; payload: Selection };

export type FileSystemEvent =
  | { type: 'created'; payload: { path: string; isDirectory: boolean } }
  | { type: 'deleted'; payload: { path: string } }
  | { type: 'renamed'; payload: { oldPath: string; newPath: string } }
  | { type: 'modified'; payload: { path: string } };

// ============================================================================
// Utility Types
// ============================================================================

export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type Nullable<T> = T | null;

export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;
