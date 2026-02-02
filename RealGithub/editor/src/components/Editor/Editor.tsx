/**
 * Editor Component
 *
 * Monaco editor with full Tokyo Night theme integration:
 * - Custom MathViz language support
 * - Syntax highlighting with Tokyo Night colors
 * - Auto-completion and parameter hints
 * - Error highlighting with Tokyo Night diagnostic colors
 * - Minimap with theme colors
 * - Bracket pair colorization
 */

import React, { useCallback, useRef, memo } from 'react';
import MonacoEditor, { OnMount, BeforeMount } from '@monaco-editor/react';
import type * as Monaco from 'monaco-editor';
import { cn } from '../../utils/helpers';
import { useEditorStore, selectActiveFile } from '../../stores/editorStore';
import { createMonacoTheme, tokyoNightTheme } from '../../utils/themes';

// ============================================================================
// Types
// ============================================================================

interface EditorProps {
  className?: string;
  onSave?: (content: string) => void;
}

// ============================================================================
// MathViz Language Definition
// ============================================================================

function registerMathVizLanguage(monaco: typeof Monaco): void {
  // Check if already registered
  const languages = monaco.languages.getLanguages();
  if (languages.some((lang) => lang.id === 'mathviz')) {
    return;
  }

  // Register the language
  monaco.languages.register({
    id: 'mathviz',
    extensions: ['.mviz', '.mathviz'],
    aliases: ['MathViz', 'mathviz'],
  });

  // Set the language configuration
  monaco.languages.setLanguageConfiguration('mathviz', {
    comments: {
      lineComment: '//',
      blockComment: ['/*', '*/'],
    },
    brackets: [
      ['{', '}'],
      ['[', ']'],
      ['(', ')'],
    ],
    autoClosingPairs: [
      { open: '{', close: '}' },
      { open: '[', close: ']' },
      { open: '(', close: ')' },
      { open: '"', close: '"', notIn: ['string'] },
      { open: "'", close: "'", notIn: ['string'] },
    ],
    surroundingPairs: [
      { open: '{', close: '}' },
      { open: '[', close: ']' },
      { open: '(', close: ')' },
      { open: '"', close: '"' },
      { open: "'", close: "'" },
    ],
    indentationRules: {
      increaseIndentPattern: /^\s*(scene|fn|if|else|for|while|match|struct|impl)\b.*[:{]\s*$/,
      decreaseIndentPattern: /^\s*[}\]]\s*$/,
    },
  });

  // Set the tokenizer (monarch syntax highlighting)
  monaco.languages.setMonarchTokensProvider('mathviz', {
    defaultToken: 'invalid',
    tokenPostfix: '.mviz',

    keywords: [
      'scene', 'fn', 'let', 'const', 'mut', 'if', 'else', 'for', 'while',
      'return', 'break', 'continue', 'match', 'struct', 'impl', 'self',
      'true', 'false', 'null', 'use', 'as', 'in', 'pub', 'static',
    ],

    typeKeywords: [
      'int', 'float', 'bool', 'str', 'void', 'Vec', 'Point', 'Color',
      'Mobject', 'Circle', 'Square', 'Rectangle', 'Triangle', 'Line',
      'Arrow', 'Arc', 'Polygon', 'Text', 'MathTex', 'NumberLine',
      'Axes', 'Graph', 'ParametricFunction', 'Surface', 'VGroup',
      'Animation', 'Scene',
    ],

    builtins: [
      'play', 'wait', 'create', 'uncreate', 'write', 'unwrite',
      'fade_in', 'fade_out', 'transform', 'move_to', 'shift',
      'rotate', 'scale', 'set_color', 'set_fill', 'set_stroke',
      'UP', 'DOWN', 'LEFT', 'RIGHT', 'ORIGIN', 'PI', 'TAU', 'E',
      'RED', 'GREEN', 'BLUE', 'YELLOW', 'WHITE', 'BLACK', 'ORANGE',
      'PURPLE', 'PINK', 'CYAN', 'GRAY', 'GREY',
    ],

    operators: [
      '=', '>', '<', '!', '~', '?', ':',
      '==', '<=', '>=', '!=', '&&', '||',
      '+', '-', '*', '/', '%', '**',
      '+=', '-=', '*=', '/=', '%=',
      '->', '=>', '..',
    ],

    symbols: /[=><!~?:&|+\-*\/\^%]+/,

    escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

    tokenizer: {
      root: [
        // Identifiers and keywords
        [/[a-z_$][\w$]*/, {
          cases: {
            '@keywords': 'keyword',
            '@typeKeywords': 'type',
            '@builtins': 'predefined',
            '@default': 'identifier',
          },
        }],
        [/[A-Z][\w$]*/, 'type.identifier'],

        // Whitespace
        { include: '@whitespace' },

        // Delimiters and operators
        [/[{}()\[\]]/, '@brackets'],
        [/[<>](?!@symbols)/, '@brackets'],
        [/@symbols/, {
          cases: {
            '@operators': 'operator',
            '@default': '',
          },
        }],

        // Numbers
        [/\d*\.\d+([eE][\-+]?\d+)?/, 'number.float'],
        [/0[xX][0-9a-fA-F]+/, 'number.hex'],
        [/\d+/, 'number'],

        // Delimiter
        [/[;,.]/, 'delimiter'],

        // Strings
        [/"([^"\\]|\\.)*$/, 'string.invalid'],
        [/"/, { token: 'string.quote', bracket: '@open', next: '@string' }],
      ],

      comment: [
        [/[^\/*]+/, 'comment'],
        [/\/\*/, 'comment', '@push'],
        ['\\*/', 'comment', '@pop'],
        [/[\/*]/, 'comment'],
      ],

      string: [
        [/[^\\"]+/, 'string'],
        [/@escapes/, 'string.escape'],
        [/\\./, 'string.escape.invalid'],
        [/"/, { token: 'string.quote', bracket: '@close', next: '@pop' }],
      ],

      whitespace: [
        [/[ \t\r\n]+/, 'white'],
        [/\/\*/, 'comment', '@comment'],
        [/\/\/.*$/, 'comment'],
      ],
    },
  });

  // Register completion provider
  monaco.languages.registerCompletionItemProvider('mathviz', {
    provideCompletionItems: (model, position) => {
      const word = model.getWordUntilPosition(position);
      const range = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn,
      };

      const suggestions: Monaco.languages.CompletionItem[] = [
        // Keywords
        { label: 'scene', kind: monaco.languages.CompletionItemKind.Keyword, insertText: 'scene ${1:Name} {\n\tfn construct(self) {\n\t\t$0\n\t}\n}', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
        { label: 'fn', kind: monaco.languages.CompletionItemKind.Keyword, insertText: 'fn ${1:name}(${2:params}) {\n\t$0\n}', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
        { label: 'let', kind: monaco.languages.CompletionItemKind.Keyword, insertText: 'let ${1:name} = $0', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
        { label: 'if', kind: monaco.languages.CompletionItemKind.Keyword, insertText: 'if ${1:condition} {\n\t$0\n}', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
        { label: 'for', kind: monaco.languages.CompletionItemKind.Keyword, insertText: 'for ${1:item} in ${2:items} {\n\t$0\n}', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },

        // Types
        { label: 'Circle', kind: monaco.languages.CompletionItemKind.Class, insertText: 'Circle(radius: ${1:1.0}, color: ${2:BLUE})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
        { label: 'Square', kind: monaco.languages.CompletionItemKind.Class, insertText: 'Square(side_length: ${1:2.0}, color: ${2:GREEN})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
        { label: 'Text', kind: monaco.languages.CompletionItemKind.Class, insertText: 'Text("${1:text}")', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
        { label: 'Arrow', kind: monaco.languages.CompletionItemKind.Class, insertText: 'Arrow(start: ${1:LEFT}, end: ${2:RIGHT})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },

        // Functions
        { label: 'play', kind: monaco.languages.CompletionItemKind.Function, insertText: 'play(${1:animation})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
        { label: 'wait', kind: monaco.languages.CompletionItemKind.Function, insertText: 'wait(${1:1.0})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
        { label: 'create', kind: monaco.languages.CompletionItemKind.Function, insertText: 'create(${1:mobject})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },
        { label: 'transform', kind: monaco.languages.CompletionItemKind.Function, insertText: 'transform(${1:source}, ${2:target})', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, range },

        // Constants
        { label: 'UP', kind: monaco.languages.CompletionItemKind.Constant, insertText: 'UP', range },
        { label: 'DOWN', kind: monaco.languages.CompletionItemKind.Constant, insertText: 'DOWN', range },
        { label: 'LEFT', kind: monaco.languages.CompletionItemKind.Constant, insertText: 'LEFT', range },
        { label: 'RIGHT', kind: monaco.languages.CompletionItemKind.Constant, insertText: 'RIGHT', range },
        { label: 'BLUE', kind: monaco.languages.CompletionItemKind.Constant, insertText: 'BLUE', range },
        { label: 'RED', kind: monaco.languages.CompletionItemKind.Constant, insertText: 'RED', range },
        { label: 'GREEN', kind: monaco.languages.CompletionItemKind.Constant, insertText: 'GREEN', range },
      ];

      return { suggestions };
    },
  });
}

// ============================================================================
// Editor Component
// ============================================================================

export const Editor: React.FC<EditorProps> = memo(({ className, onSave }) => {
  const editorRef = useRef<Monaco.editor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<typeof Monaco | null>(null);

  // Store state
  const activeFile = useEditorStore(selectActiveFile);
  const updateFileContent = useEditorStore((state) => state.updateFileContent);
  const setCursorPosition = useEditorStore((state) => state.setCursorPosition);
  const setSelection = useEditorStore((state) => state.setSelection);
  const settings = useEditorStore((state) => state.settings);

  // Before mount - register themes and language
  const handleBeforeMount: BeforeMount = useCallback((monaco) => {
    monacoRef.current = monaco;

    // Register Tokyo Night theme
    monaco.editor.defineTheme('tokyo-night', createMonacoTheme(tokyoNightTheme));

    // Register MathViz language
    registerMathVizLanguage(monaco);
  }, []);

  // On mount - setup editor
  const handleMount: OnMount = useCallback((editor, monaco) => {
    editorRef.current = editor;
    monacoRef.current = monaco;

    // Set theme
    monaco.editor.setTheme('tokyo-night');

    // Handle content changes
    editor.onDidChangeModelContent(() => {
      if (activeFile) {
        const content = editor.getValue();
        updateFileContent(activeFile.id, content);
      }
    });

    // Handle cursor position changes
    editor.onDidChangeCursorPosition((e) => {
      setCursorPosition({
        lineNumber: e.position.lineNumber,
        column: e.position.column,
      });
    });

    // Handle selection changes
    editor.onDidChangeCursorSelection((e) => {
      const selection = e.selection;
      if (
        selection.startLineNumber === selection.endLineNumber &&
        selection.startColumn === selection.endColumn
      ) {
        setSelection(null);
      } else {
        setSelection({
          startLineNumber: selection.startLineNumber,
          startColumn: selection.startColumn,
          endLineNumber: selection.endLineNumber,
          endColumn: selection.endColumn,
        });
      }
    });

    // Keyboard shortcuts
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
      if (onSave) {
        onSave(editor.getValue());
      }
    });

    // Focus the editor
    editor.focus();
  }, [activeFile, updateFileContent, setCursorPosition, setSelection, onSave]);

  // Handle content changes from active file
  const handleChange = useCallback((value: string | undefined) => {
    if (activeFile && value !== undefined) {
      updateFileContent(activeFile.id, value);
    }
  }, [activeFile, updateFileContent]);

  // Determine language from file extension
  const getLanguage = () => {
    if (!activeFile) return 'plaintext';
    const ext = activeFile.path.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'mviz':
      case 'mathviz':
        return 'mathviz';
      case 'js':
        return 'javascript';
      case 'ts':
        return 'typescript';
      case 'py':
        return 'python';
      case 'rs':
        return 'rust';
      case 'json':
        return 'json';
      case 'md':
        return 'markdown';
      case 'toml':
        return 'toml';
      case 'yaml':
      case 'yml':
        return 'yaml';
      default:
        return 'plaintext';
    }
  };

  return (
    <div
      className={cn(
        'w-full h-full',
        'bg-[var(--mviz-editor-background,#1a1b26)]',
        className
      )}
      dir="ltr"
      style={{ direction: 'ltr' }}
    >
      <MonacoEditor
        height="100%"
        language={getLanguage()}
        value={activeFile?.content || ''}
        theme="tokyo-night"
        beforeMount={handleBeforeMount}
        onMount={handleMount}
        onChange={handleChange}
        options={{
          // Font settings
          fontSize: settings.fontSize,
          fontFamily: settings.fontFamily || "'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace",
          fontLigatures: true,
          lineHeight: settings.lineHeight,
          letterSpacing: 0.5,

          // Editor settings
          tabSize: settings.tabSize,
          insertSpaces: settings.insertSpaces,
          wordWrap: settings.wordWrap,
          wordWrapColumn: settings.wordWrapColumn,
          scrollBeyondLastLine: settings.scrollBeyondLastLine,
          renderWhitespace: settings.renderWhitespace,
          cursorStyle: settings.cursorStyle,
          cursorBlinking: settings.cursorBlinking,
          cursorWidth: 2,

          // Minimap
          minimap: {
            enabled: settings.minimap.enabled,
            side: settings.minimap.side,
            maxColumn: settings.minimap.maxColumn,
            renderCharacters: false,
            showSlider: 'mouseover',
          },

          // Line numbers and gutter
          lineNumbers: 'on',
          lineNumbersMinChars: 4,
          glyphMargin: true,
          folding: true,
          foldingStrategy: 'indentation',
          showFoldingControls: 'mouseover',
          lineDecorationsWidth: 10,

          // Bracket matching
          matchBrackets: 'always',
          bracketPairColorization: {
            enabled: true,
            independentColorPoolPerBracketType: true,
          },

          // Guides
          guides: {
            bracketPairs: true,
            bracketPairsHorizontal: true,
            indentation: true,
            highlightActiveIndentation: true,
          },

          // Auto features
          autoClosingBrackets: 'always',
          autoClosingQuotes: 'always',
          autoClosingDelete: 'always',
          autoIndent: 'advanced',
          autoSurround: 'languageDefined',
          formatOnPaste: settings.formatOnPaste,
          formatOnType: true,

          // Suggestions
          quickSuggestions: {
            other: true,
            comments: false,
            strings: true,
          },
          suggestOnTriggerCharacters: true,
          acceptSuggestionOnEnter: 'smart',
          tabCompletion: 'on',
          snippetSuggestions: 'inline',
          wordBasedSuggestions: 'matchingDocuments',
          suggestSelection: 'first',

          // Suggest widget
          suggest: {
            insertMode: 'replace',
            filterGraceful: true,
            showWords: true,
            showSnippets: true,
            showIcons: true,
          },

          // Find widget
          find: {
            addExtraSpaceOnTop: false,
            autoFindInSelection: 'multiline',
            seedSearchStringFromSelection: 'selection',
          },

          // Scrollbar
          scrollbar: {
            vertical: 'auto',
            horizontal: 'auto',
            useShadows: false,
            verticalScrollbarSize: 10,
            horizontalScrollbarSize: 10,
            arrowSize: 0,
          },

          // Hover
          hover: {
            enabled: true,
            delay: 300,
            sticky: true,
          },

          // Parameter hints
          parameterHints: {
            enabled: true,
            cycle: true,
          },

          // Other features
          contextmenu: true,
          mouseWheelZoom: true,
          smoothScrolling: true,
          links: true,
          colorDecorators: true,
          renderLineHighlight: 'all',
          renderLineHighlightOnlyWhenFocus: false,
          selectOnLineNumbers: true,
          roundedSelection: true,
          overviewRulerBorder: false,

          // Padding
          padding: {
            top: 16,
            bottom: 16,
          },

          // Accessibility
          accessibilitySupport: 'auto',
          ariaLabel: 'MathViz Code Editor',

          // Performance
          fastScrollSensitivity: 5,
          multiCursorModifier: 'ctrlCmd',
          dragAndDrop: true,
          copyWithSyntaxHighlighting: true,

          // Auto layout
          automaticLayout: true,
        }}
      />
    </div>
  );
});

Editor.displayName = 'Editor';

// ============================================================================
// Export
// ============================================================================

export default Editor;
