/**
 * MathViz Editor Themes
 *
 * Tokyo Night color scheme - the popular VS Code theme
 * with distinctive purple/blue palette and excellent readability.
 */

import type * as monaco from 'monaco-editor';

// ============================================================================
// Theme Types
// ============================================================================

export interface ThemeColors {
  // Base colors
  background: string;
  foreground: string;
  border: string;

  // Editor
  editorBackground: string;
  editorForeground: string;
  editorLineHighlight: string;
  editorLineNumber: string;
  editorLineNumberActive: string;
  editorSelectionBackground: string;
  editorCursorForeground: string;
  editorWhitespace: string;
  editorIndentGuide: string;
  editorIndentGuideActive: string;

  // Sidebar
  sidebarBackground: string;
  sidebarForeground: string;
  sidebarBorder: string;

  // Activity bar
  activityBarBackground: string;
  activityBarForeground: string;
  activityBarInactiveForeground: string;
  activityBarBadge: string;
  activityBarBadgeForeground: string;

  // Title bar
  titleBarBackground: string;
  titleBarForeground: string;
  titleBarInactiveForeground: string;

  // Status bar
  statusBarBackground: string;
  statusBarForeground: string;
  statusBarRemoteBackground: string;
  statusBarRemoteForeground: string;

  // Panel
  panelBackground: string;
  panelForeground: string;
  panelBorder: string;

  // Tab bar
  tabActiveBackground: string;
  tabActiveForeground: string;
  tabActiveBorder: string;
  tabInactiveBackground: string;
  tabInactiveForeground: string;
  tabBorder: string;
  tabHoverBackground: string;

  // Input
  inputBackground: string;
  inputForeground: string;
  inputBorder: string;
  inputPlaceholder: string;
  inputActiveBackground: string;

  // Button
  buttonBackground: string;
  buttonForeground: string;
  buttonHoverBackground: string;
  buttonSecondaryBackground: string;
  buttonSecondaryForeground: string;
  buttonSecondaryHoverBackground: string;

  // List
  listActiveBackground: string;
  listActiveForeground: string;
  listHoverBackground: string;
  listFocusBackground: string;
  listFocusForeground: string;

  // Scrollbar
  scrollbarBackground: string;
  scrollbarThumb: string;
  scrollbarThumbHover: string;

  // Minimap
  minimapBackground: string;
  minimapSliderBackground: string;

  // Diagnostic colors
  errorForeground: string;
  errorBackground: string;
  warningForeground: string;
  warningBackground: string;
  infoForeground: string;
  hintForeground: string;

  // Git colors
  gitModified: string;
  gitAdded: string;
  gitDeleted: string;
  gitUntracked: string;
  gitConflicting: string;
  gitIgnored: string;

  // Diff colors
  diffAddedBackground: string;
  diffAddedForeground: string;
  diffRemovedBackground: string;
  diffRemovedForeground: string;
  diffModifiedBackground: string;

  // Accent colors
  accent: string;
  accentForeground: string;
  link: string;
  linkHover: string;

  // Focus
  focusBorder: string;

  // Badges
  badgeBackground: string;
  badgeForeground: string;

  // Breadcrumb
  breadcrumbForeground: string;
  breadcrumbFocusForeground: string;

  // Widget (hover, autocomplete, etc.)
  widgetBackground: string;
  widgetBorder: string;
  widgetShadow: string;

  // Dropdown
  dropdownBackground: string;
  dropdownBorder: string;
  dropdownForeground: string;

  // Notification
  notificationBackground: string;
  notificationForeground: string;
  notificationBorder: string;

  // Progress bar
  progressBarBackground: string;

  // Terminal colors
  terminalBackground: string;
  terminalForeground: string;
  terminalCursor: string;
  terminalBlack: string;
  terminalRed: string;
  terminalGreen: string;
  terminalYellow: string;
  terminalBlue: string;
  terminalMagenta: string;
  terminalCyan: string;
  terminalWhite: string;
  terminalBrightBlack: string;
  terminalBrightRed: string;
  terminalBrightGreen: string;
  terminalBrightYellow: string;
  terminalBrightBlue: string;
  terminalBrightMagenta: string;
  terminalBrightCyan: string;
  terminalBrightWhite: string;
}

export interface SyntaxColors {
  comment: string;
  keyword: string;
  keywordControl: string;
  string: string;
  stringEscape: string;
  number: string;
  operator: string;
  type: string;
  typeParameter: string;
  function: string;
  functionCall: string;
  variable: string;
  variableProperty: string;
  constant: string;
  parameter: string;
  property: string;
  punctuation: string;
  punctuationBracket: string;
  decorator: string;
  namespace: string;
  tag: string;
  tagBracket: string;
  attribute: string;
  regexp: string;
  escape: string;
  invalid: string;
  markup: string;
  markupHeading: string;
  markupBold: string;
  markupItalic: string;
  markupLink: string;
  markupCode: string;
}

export interface Theme {
  name: string;
  type: 'dark' | 'light';
  colors: ThemeColors;
  syntax: SyntaxColors;
}

// ============================================================================
// Tokyo Night Storm Theme (Main Theme)
// ============================================================================

export const tokyoNightTheme: Theme = {
  name: 'Tokyo Night',
  type: 'dark',
  colors: {
    // Base colors - Tokyo Night Storm
    background: '#1a1b26',
    foreground: '#a9b1d6',
    border: '#1a1b26',

    // Editor
    editorBackground: '#1a1b26',
    editorForeground: '#a9b1d6',
    editorLineHighlight: '#292e42',
    editorLineNumber: '#3b4261',
    editorLineNumberActive: '#737aa2',
    editorSelectionBackground: '#283457',
    editorCursorForeground: '#c0caf5',
    editorWhitespace: '#3b4261',
    editorIndentGuide: '#292e42',
    editorIndentGuideActive: '#3b4261',

    // Sidebar - Slightly lighter for contrast
    sidebarBackground: '#1f2335',
    sidebarForeground: '#a9b1d6',
    sidebarBorder: '#1a1b26',

    // Activity bar - Darker for depth
    activityBarBackground: '#1a1b26',
    activityBarForeground: '#a9b1d6',
    activityBarInactiveForeground: '#565f89',
    activityBarBadge: '#7aa2f7',
    activityBarBadgeForeground: '#1a1b26',

    // Title bar
    titleBarBackground: '#1a1b26',
    titleBarForeground: '#a9b1d6',
    titleBarInactiveForeground: '#565f89',

    // Status bar - Purple accent like Tokyo Night
    statusBarBackground: '#1a1b26',
    statusBarForeground: '#a9b1d6',
    statusBarRemoteBackground: '#7aa2f7',
    statusBarRemoteForeground: '#1a1b26',

    // Panel
    panelBackground: '#1a1b26',
    panelForeground: '#a9b1d6',
    panelBorder: '#1f2335',

    // Tab bar - Tokyo Night style
    tabActiveBackground: '#1a1b26',
    tabActiveForeground: '#c0caf5',
    tabActiveBorder: '#7aa2f7',
    tabInactiveBackground: '#1f2335',
    tabInactiveForeground: '#565f89',
    tabBorder: '#1a1b26',
    tabHoverBackground: '#292e42',

    // Input
    inputBackground: '#1a1b26',
    inputForeground: '#c0caf5',
    inputBorder: '#3b4261',
    inputPlaceholder: '#565f89',
    inputActiveBackground: '#24283b',

    // Button - Blue accent
    buttonBackground: '#7aa2f7',
    buttonForeground: '#1a1b26',
    buttonHoverBackground: '#89b4fa',
    buttonSecondaryBackground: '#292e42',
    buttonSecondaryForeground: '#c0caf5',
    buttonSecondaryHoverBackground: '#3b4261',

    // List
    listActiveBackground: '#283457',
    listActiveForeground: '#c0caf5',
    listHoverBackground: '#292e42',
    listFocusBackground: '#283457',
    listFocusForeground: '#c0caf5',

    // Scrollbar
    scrollbarBackground: 'transparent',
    scrollbarThumb: '#3b426180',
    scrollbarThumbHover: '#565f89',

    // Minimap
    minimapBackground: '#1a1b26',
    minimapSliderBackground: '#3b426140',

    // Diagnostic colors - Tokyo Night palette
    errorForeground: '#f7768e',
    errorBackground: '#f7768e20',
    warningForeground: '#e0af68',
    warningBackground: '#e0af6820',
    infoForeground: '#7aa2f7',
    hintForeground: '#9ece6a',

    // Git colors
    gitModified: '#e0af68',
    gitAdded: '#9ece6a',
    gitDeleted: '#f7768e',
    gitUntracked: '#73daca',
    gitConflicting: '#ff9e64',
    gitIgnored: '#565f89',

    // Diff colors
    diffAddedBackground: '#9ece6a20',
    diffAddedForeground: '#9ece6a',
    diffRemovedBackground: '#f7768e20',
    diffRemovedForeground: '#f7768e',
    diffModifiedBackground: '#e0af6820',

    // Accent colors - Tokyo Night blue
    accent: '#7aa2f7',
    accentForeground: '#1a1b26',
    link: '#7aa2f7',
    linkHover: '#89b4fa',

    // Focus
    focusBorder: '#7aa2f7',

    // Badges
    badgeBackground: '#7aa2f7',
    badgeForeground: '#1a1b26',

    // Breadcrumb
    breadcrumbForeground: '#565f89',
    breadcrumbFocusForeground: '#c0caf5',

    // Widget
    widgetBackground: '#1f2335',
    widgetBorder: '#3b4261',
    widgetShadow: '#00000080',

    // Dropdown
    dropdownBackground: '#1f2335',
    dropdownBorder: '#3b4261',
    dropdownForeground: '#a9b1d6',

    // Notification
    notificationBackground: '#1f2335',
    notificationForeground: '#a9b1d6',
    notificationBorder: '#3b4261',

    // Progress bar
    progressBarBackground: '#7aa2f7',

    // Terminal colors - Tokyo Night palette
    terminalBackground: '#1a1b26',
    terminalForeground: '#a9b1d6',
    terminalCursor: '#c0caf5',
    terminalBlack: '#1a1b26',
    terminalRed: '#f7768e',
    terminalGreen: '#9ece6a',
    terminalYellow: '#e0af68',
    terminalBlue: '#7aa2f7',
    terminalMagenta: '#bb9af7',
    terminalCyan: '#7dcfff',
    terminalWhite: '#a9b1d6',
    terminalBrightBlack: '#565f89',
    terminalBrightRed: '#f7768e',
    terminalBrightGreen: '#9ece6a',
    terminalBrightYellow: '#e0af68',
    terminalBrightBlue: '#7aa2f7',
    terminalBrightMagenta: '#bb9af7',
    terminalBrightCyan: '#7dcfff',
    terminalBrightWhite: '#c0caf5',
  },
  syntax: {
    // Tokyo Night syntax colors
    comment: '#565f89',
    keyword: '#bb9af7',           // Soft purple
    keywordControl: '#bb9af7',
    string: '#9ece6a',            // Green
    stringEscape: '#73daca',
    number: '#ff9e64',            // Orange
    operator: '#89ddff',          // Cyan
    type: '#2ac3de',              // Cyan (lighter)
    typeParameter: '#2ac3de',
    function: '#7aa2f7',          // Blue
    functionCall: '#7aa2f7',
    variable: '#c0caf5',          // Light foreground
    variableProperty: '#7dcfff',
    constant: '#ff9e64',          // Orange
    parameter: '#e0af68',         // Yellow
    property: '#7dcfff',          // Light cyan
    punctuation: '#89ddff',
    punctuationBracket: '#a9b1d6',
    decorator: '#bb9af7',
    namespace: '#2ac3de',
    tag: '#f7768e',               // Red/Pink
    tagBracket: '#a9b1d6',
    attribute: '#bb9af7',
    regexp: '#b4f9f8',
    escape: '#73daca',
    invalid: '#f7768e',
    markup: '#c0caf5',
    markupHeading: '#7aa2f7',
    markupBold: '#c0caf5',
    markupItalic: '#c0caf5',
    markupLink: '#7aa2f7',
    markupCode: '#9ece6a',
  },
};

// ============================================================================
// MathViz Dark Theme (Legacy - for backwards compatibility)
// Now uses Tokyo Night as the base
// ============================================================================

export const mathvizDarkTheme = tokyoNightTheme;

// ============================================================================
// Tokyo Night Light Theme (Optional)
// ============================================================================

export const tokyoNightLightTheme: Theme = {
  name: 'Tokyo Night Light',
  type: 'light',
  colors: {
    // Base colors
    background: '#d5d6db',
    foreground: '#343b58',
    border: '#d5d6db',

    // Editor
    editorBackground: '#d5d6db',
    editorForeground: '#343b58',
    editorLineHighlight: '#c4c8da',
    editorLineNumber: '#9699a3',
    editorLineNumberActive: '#343b58',
    editorSelectionBackground: '#b6bfe2',
    editorCursorForeground: '#343b58',
    editorWhitespace: '#9699a3',
    editorIndentGuide: '#c4c8da',
    editorIndentGuideActive: '#9699a3',

    // Sidebar
    sidebarBackground: '#d5d6db',
    sidebarForeground: '#343b58',
    sidebarBorder: '#c4c8da',

    // Activity bar
    activityBarBackground: '#c4c8da',
    activityBarForeground: '#343b58',
    activityBarInactiveForeground: '#9699a3',
    activityBarBadge: '#34548a',
    activityBarBadgeForeground: '#ffffff',

    // Title bar
    titleBarBackground: '#c4c8da',
    titleBarForeground: '#343b58',
    titleBarInactiveForeground: '#9699a3',

    // Status bar
    statusBarBackground: '#c4c8da',
    statusBarForeground: '#343b58',
    statusBarRemoteBackground: '#34548a',
    statusBarRemoteForeground: '#ffffff',

    // Panel
    panelBackground: '#d5d6db',
    panelForeground: '#343b58',
    panelBorder: '#c4c8da',

    // Tab bar
    tabActiveBackground: '#d5d6db',
    tabActiveForeground: '#343b58',
    tabActiveBorder: '#34548a',
    tabInactiveBackground: '#c4c8da',
    tabInactiveForeground: '#9699a3',
    tabBorder: '#d5d6db',
    tabHoverBackground: '#b6bfe2',

    // Input
    inputBackground: '#ffffff',
    inputForeground: '#343b58',
    inputBorder: '#9699a3',
    inputPlaceholder: '#9699a3',
    inputActiveBackground: '#ffffff',

    // Button
    buttonBackground: '#34548a',
    buttonForeground: '#ffffff',
    buttonHoverBackground: '#485e99',
    buttonSecondaryBackground: '#c4c8da',
    buttonSecondaryForeground: '#343b58',
    buttonSecondaryHoverBackground: '#b6bfe2',

    // List
    listActiveBackground: '#b6bfe2',
    listActiveForeground: '#343b58',
    listHoverBackground: '#c4c8da',
    listFocusBackground: '#b6bfe2',
    listFocusForeground: '#343b58',

    // Scrollbar
    scrollbarBackground: 'transparent',
    scrollbarThumb: '#9699a380',
    scrollbarThumbHover: '#9699a3',

    // Minimap
    minimapBackground: '#d5d6db',
    minimapSliderBackground: '#9699a340',

    // Diagnostic colors
    errorForeground: '#8c4351',
    errorBackground: '#8c435120',
    warningForeground: '#8f5e15',
    warningBackground: '#8f5e1520',
    infoForeground: '#34548a',
    hintForeground: '#33635c',

    // Git colors
    gitModified: '#8f5e15',
    gitAdded: '#33635c',
    gitDeleted: '#8c4351',
    gitUntracked: '#33635c',
    gitConflicting: '#965027',
    gitIgnored: '#9699a3',

    // Diff colors
    diffAddedBackground: '#33635c20',
    diffAddedForeground: '#33635c',
    diffRemovedBackground: '#8c435120',
    diffRemovedForeground: '#8c4351',
    diffModifiedBackground: '#8f5e1520',

    // Accent colors
    accent: '#34548a',
    accentForeground: '#ffffff',
    link: '#34548a',
    linkHover: '#485e99',

    // Focus
    focusBorder: '#34548a',

    // Badges
    badgeBackground: '#34548a',
    badgeForeground: '#ffffff',

    // Breadcrumb
    breadcrumbForeground: '#9699a3',
    breadcrumbFocusForeground: '#343b58',

    // Widget
    widgetBackground: '#d5d6db',
    widgetBorder: '#c4c8da',
    widgetShadow: '#00000020',

    // Dropdown
    dropdownBackground: '#d5d6db',
    dropdownBorder: '#c4c8da',
    dropdownForeground: '#343b58',

    // Notification
    notificationBackground: '#d5d6db',
    notificationForeground: '#343b58',
    notificationBorder: '#c4c8da',

    // Progress bar
    progressBarBackground: '#34548a',

    // Terminal colors
    terminalBackground: '#d5d6db',
    terminalForeground: '#343b58',
    terminalCursor: '#343b58',
    terminalBlack: '#0f0f14',
    terminalRed: '#8c4351',
    terminalGreen: '#33635c',
    terminalYellow: '#8f5e15',
    terminalBlue: '#34548a',
    terminalMagenta: '#5a4a78',
    terminalCyan: '#0f4b6e',
    terminalWhite: '#343b58',
    terminalBrightBlack: '#9699a3',
    terminalBrightRed: '#8c4351',
    terminalBrightGreen: '#33635c',
    terminalBrightYellow: '#8f5e15',
    terminalBrightBlue: '#34548a',
    terminalBrightMagenta: '#5a4a78',
    terminalBrightCyan: '#0f4b6e',
    terminalBrightWhite: '#343b58',
  },
  syntax: {
    comment: '#9699a3',
    keyword: '#5a4a78',
    keywordControl: '#5a4a78',
    string: '#33635c',
    stringEscape: '#33635c',
    number: '#965027',
    operator: '#0f4b6e',
    type: '#166775',
    typeParameter: '#166775',
    function: '#34548a',
    functionCall: '#34548a',
    variable: '#343b58',
    variableProperty: '#0f4b6e',
    constant: '#965027',
    parameter: '#8f5e15',
    property: '#0f4b6e',
    punctuation: '#0f4b6e',
    punctuationBracket: '#343b58',
    decorator: '#5a4a78',
    namespace: '#166775',
    tag: '#8c4351',
    tagBracket: '#343b58',
    attribute: '#5a4a78',
    regexp: '#166775',
    escape: '#33635c',
    invalid: '#8c4351',
    markup: '#343b58',
    markupHeading: '#34548a',
    markupBold: '#343b58',
    markupItalic: '#343b58',
    markupLink: '#34548a',
    markupCode: '#33635c',
  },
};

export const mathvizLightTheme = tokyoNightLightTheme;

// ============================================================================
// Monaco Theme Conversion
// ============================================================================

export function createMonacoTheme(theme: Theme): monaco.editor.IStandaloneThemeData {
  return {
    base: theme.type === 'dark' ? 'vs-dark' : 'vs',
    inherit: false,
    rules: [
      // Comments
      { token: 'comment', foreground: theme.syntax.comment.slice(1), fontStyle: 'italic' },
      { token: 'comment.doc', foreground: theme.syntax.comment.slice(1), fontStyle: 'italic' },
      { token: 'comment.block', foreground: theme.syntax.comment.slice(1), fontStyle: 'italic' },
      { token: 'comment.line', foreground: theme.syntax.comment.slice(1), fontStyle: 'italic' },

      // Keywords
      { token: 'keyword', foreground: theme.syntax.keyword.slice(1) },
      { token: 'keyword.control', foreground: theme.syntax.keywordControl.slice(1) },
      { token: 'keyword.operator', foreground: theme.syntax.operator.slice(1) },

      // Strings
      { token: 'string', foreground: theme.syntax.string.slice(1) },
      { token: 'string.quote', foreground: theme.syntax.string.slice(1) },
      { token: 'string.escape', foreground: theme.syntax.stringEscape.slice(1) },
      { token: 'string.invalid', foreground: theme.syntax.invalid.slice(1) },
      { token: 'string.regexp', foreground: theme.syntax.regexp.slice(1) },

      // Numbers
      { token: 'number', foreground: theme.syntax.number.slice(1) },
      { token: 'number.float', foreground: theme.syntax.number.slice(1) },
      { token: 'number.hex', foreground: theme.syntax.number.slice(1) },
      { token: 'number.octal', foreground: theme.syntax.number.slice(1) },
      { token: 'number.binary', foreground: theme.syntax.number.slice(1) },

      // Types
      { token: 'type', foreground: theme.syntax.type.slice(1) },
      { token: 'type.identifier', foreground: theme.syntax.type.slice(1) },
      { token: 'support.type', foreground: theme.syntax.type.slice(1) },

      // Functions
      { token: 'function', foreground: theme.syntax.function.slice(1) },
      { token: 'support.function', foreground: theme.syntax.function.slice(1) },
      { token: 'entity.name.function', foreground: theme.syntax.function.slice(1) },
      { token: 'predefined', foreground: theme.syntax.function.slice(1) },

      // Variables
      { token: 'variable', foreground: theme.syntax.variable.slice(1) },
      { token: 'variable.parameter', foreground: theme.syntax.parameter.slice(1) },
      { token: 'variable.other', foreground: theme.syntax.variable.slice(1) },
      { token: 'identifier', foreground: theme.syntax.variable.slice(1) },

      // Constants
      { token: 'constant', foreground: theme.syntax.constant.slice(1) },
      { token: 'constant.language', foreground: theme.syntax.constant.slice(1) },
      { token: 'constant.numeric', foreground: theme.syntax.number.slice(1) },

      // Operators
      { token: 'operator', foreground: theme.syntax.operator.slice(1) },
      { token: 'delimiter', foreground: theme.syntax.punctuation.slice(1) },
      { token: 'delimiter.bracket', foreground: theme.syntax.punctuationBracket.slice(1) },
      { token: 'delimiter.parenthesis', foreground: theme.syntax.punctuationBracket.slice(1) },
      { token: 'delimiter.curly', foreground: theme.syntax.punctuationBracket.slice(1) },
      { token: 'delimiter.square', foreground: theme.syntax.punctuationBracket.slice(1) },
      { token: 'delimiter.angle', foreground: theme.syntax.punctuationBracket.slice(1) },

      // Tags (HTML/XML)
      { token: 'tag', foreground: theme.syntax.tag.slice(1) },
      { token: 'tag.attribute.name', foreground: theme.syntax.attribute.slice(1) },

      // Decorators
      { token: 'decorator', foreground: theme.syntax.decorator.slice(1) },
      { token: 'annotation', foreground: theme.syntax.decorator.slice(1) },

      // Namespace
      { token: 'namespace', foreground: theme.syntax.namespace.slice(1) },

      // Property
      { token: 'property', foreground: theme.syntax.property.slice(1) },
      { token: 'attribute', foreground: theme.syntax.attribute.slice(1) },

      // Invalid
      { token: 'invalid', foreground: theme.syntax.invalid.slice(1) },

      // Markup
      { token: 'markup.heading', foreground: theme.syntax.markupHeading.slice(1), fontStyle: 'bold' },
      { token: 'markup.bold', foreground: theme.syntax.markupBold.slice(1), fontStyle: 'bold' },
      { token: 'markup.italic', foreground: theme.syntax.markupItalic.slice(1), fontStyle: 'italic' },
      { token: 'markup.underline', foreground: theme.syntax.markup.slice(1), fontStyle: 'underline' },
    ],
    colors: {
      // Editor
      'editor.background': theme.colors.editorBackground,
      'editor.foreground': theme.colors.editorForeground,
      'editor.lineHighlightBackground': theme.colors.editorLineHighlight,
      'editor.lineHighlightBorder': '#00000000',
      'editorLineNumber.foreground': theme.colors.editorLineNumber,
      'editorLineNumber.activeForeground': theme.colors.editorLineNumberActive,
      'editor.selectionBackground': theme.colors.editorSelectionBackground,
      'editor.selectionHighlightBackground': theme.colors.editorSelectionBackground + '40',
      'editorCursor.foreground': theme.colors.editorCursorForeground,
      'editorWhitespace.foreground': theme.colors.editorWhitespace,
      'editorIndentGuide.background': theme.colors.editorIndentGuide,
      'editorIndentGuide.activeBackground': theme.colors.editorIndentGuideActive,

      // Minimap
      'minimap.background': theme.colors.minimapBackground,
      'minimapSlider.background': theme.colors.minimapSliderBackground,
      'minimapSlider.hoverBackground': theme.colors.minimapSliderBackground,
      'minimapSlider.activeBackground': theme.colors.minimapSliderBackground,

      // Scrollbar
      'scrollbar.shadow': theme.colors.widgetShadow,
      'scrollbarSlider.background': theme.colors.scrollbarThumb,
      'scrollbarSlider.hoverBackground': theme.colors.scrollbarThumbHover,
      'scrollbarSlider.activeBackground': theme.colors.scrollbarThumbHover,

      // Editor errors
      'editorError.foreground': theme.colors.errorForeground,
      'editorWarning.foreground': theme.colors.warningForeground,
      'editorInfo.foreground': theme.colors.infoForeground,
      'editorHint.foreground': theme.colors.hintForeground,

      // Focus border
      focusBorder: theme.colors.focusBorder,

      // Widget
      'editorWidget.background': theme.colors.widgetBackground,
      'editorWidget.border': theme.colors.widgetBorder,

      // Hover widget
      'editorHoverWidget.background': theme.colors.widgetBackground,
      'editorHoverWidget.border': theme.colors.widgetBorder,

      // Suggest widget (autocomplete)
      'editorSuggestWidget.background': theme.colors.widgetBackground,
      'editorSuggestWidget.border': theme.colors.widgetBorder,
      'editorSuggestWidget.foreground': theme.colors.foreground,
      'editorSuggestWidget.selectedBackground': theme.colors.listActiveBackground,

      // Peek view
      'peekView.border': theme.colors.accent,
      'peekViewTitle.background': theme.colors.widgetBackground,
      'peekViewTitleLabel.foreground': theme.colors.foreground,
      'peekViewEditor.background': theme.colors.editorBackground,
      'peekViewResult.background': theme.colors.sidebarBackground,

      // Diff
      'diffEditor.insertedTextBackground': theme.colors.diffAddedBackground,
      'diffEditor.removedTextBackground': theme.colors.diffRemovedBackground,

      // Bracket matching
      'editorBracketMatch.background': theme.colors.editorSelectionBackground,
      'editorBracketMatch.border': theme.colors.accent,
    },
  };
}

// ============================================================================
// CSS Variable Generation
// ============================================================================

export function generateCSSVariables(theme: Theme): string {
  const lines: string[] = [':root {'];

  // Colors
  for (const [key, value] of Object.entries(theme.colors)) {
    const cssKey = key.replace(/([A-Z])/g, '-$1').toLowerCase();
    lines.push(`  --mviz-${cssKey}: ${value};`);
  }

  // Syntax colors
  for (const [key, value] of Object.entries(theme.syntax)) {
    const cssKey = key.replace(/([A-Z])/g, '-$1').toLowerCase();
    lines.push(`  --mviz-syntax-${cssKey}: ${value};`);
  }

  lines.push('}');
  return lines.join('\n');
}

// ============================================================================
// XTerm.js Theme Generation
// ============================================================================

export function generateXtermTheme(theme: Theme): {
  background: string;
  foreground: string;
  cursor: string;
  cursorAccent: string;
  selectionBackground: string;
  black: string;
  red: string;
  green: string;
  yellow: string;
  blue: string;
  magenta: string;
  cyan: string;
  white: string;
  brightBlack: string;
  brightRed: string;
  brightGreen: string;
  brightYellow: string;
  brightBlue: string;
  brightMagenta: string;
  brightCyan: string;
  brightWhite: string;
} {
  return {
    background: theme.colors.terminalBackground,
    foreground: theme.colors.terminalForeground,
    cursor: theme.colors.terminalCursor,
    cursorAccent: theme.colors.terminalBackground,
    selectionBackground: theme.colors.editorSelectionBackground,
    black: theme.colors.terminalBlack,
    red: theme.colors.terminalRed,
    green: theme.colors.terminalGreen,
    yellow: theme.colors.terminalYellow,
    blue: theme.colors.terminalBlue,
    magenta: theme.colors.terminalMagenta,
    cyan: theme.colors.terminalCyan,
    white: theme.colors.terminalWhite,
    brightBlack: theme.colors.terminalBrightBlack,
    brightRed: theme.colors.terminalBrightRed,
    brightGreen: theme.colors.terminalBrightGreen,
    brightYellow: theme.colors.terminalBrightYellow,
    brightBlue: theme.colors.terminalBrightBlue,
    brightMagenta: theme.colors.terminalBrightMagenta,
    brightCyan: theme.colors.terminalBrightCyan,
    brightWhite: theme.colors.terminalBrightWhite,
  };
}

// ============================================================================
// Theme Registration
// ============================================================================

export function registerThemes(monacoInstance: typeof monaco): void {
  monacoInstance.editor.defineTheme('tokyo-night', createMonacoTheme(tokyoNightTheme));
  monacoInstance.editor.defineTheme('tokyo-night-light', createMonacoTheme(tokyoNightLightTheme));
  // Keep legacy names for compatibility
  monacoInstance.editor.defineTheme('mathviz-dark', createMonacoTheme(tokyoNightTheme));
  monacoInstance.editor.defineTheme('mathviz-light', createMonacoTheme(tokyoNightLightTheme));
}

// ============================================================================
// Exports
// ============================================================================

export const themes = {
  tokyoNight: tokyoNightTheme,
  tokyoNightLight: tokyoNightLightTheme,
  dark: tokyoNightTheme,
  light: tokyoNightLightTheme,
};

export function getTheme(type: 'dark' | 'light' | 'tokyo-night' | 'tokyo-night-light'): Theme {
  switch (type) {
    case 'tokyo-night':
    case 'dark':
      return tokyoNightTheme;
    case 'tokyo-night-light':
    case 'light':
      return tokyoNightLightTheme;
    default:
      return tokyoNightTheme;
  }
}

// Current active theme (can be changed at runtime)
export let currentTheme: Theme = tokyoNightTheme;

export function setCurrentTheme(theme: Theme): void {
  currentTheme = theme;
}
