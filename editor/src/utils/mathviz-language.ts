/**
 * MathViz Language Definition for Monaco Editor
 *
 * Provides syntax highlighting, auto-completion, and language
 * configuration for the MathViz programming language.
 */

import type * as monaco from 'monaco-editor';

// ============================================================================
// Language Configuration
// ============================================================================

export const mathvizLanguageConfiguration: monaco.languages.LanguageConfiguration = {
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
    { open: "'", close: "'", notIn: ['string', 'comment'] },
    { open: '`', close: '`', notIn: ['string', 'comment'] },
    { open: '/*', close: '*/', notIn: ['string'] },
  ],
  surroundingPairs: [
    { open: '{', close: '}' },
    { open: '[', close: ']' },
    { open: '(', close: ')' },
    { open: '"', close: '"' },
    { open: "'", close: "'" },
    { open: '`', close: '`' },
  ],
  folding: {
    markers: {
      start: /^\s*\/\/\s*#?region\b/,
      end: /^\s*\/\/\s*#?endregion\b/,
    },
  },
  wordPattern: /(-?\d*\.\d\w*)|([^\`\~\!\@\#\%\^\&\*\(\)\-\=\+\[\{\]\}\\\|\;\:\'\"\,\.\<\>\/\?\s]+)/g,
  indentationRules: {
    increaseIndentPattern: /^.*\{[^}"']*$|^.*\([^)"']*$|^.*\[[^\]"']*$/,
    decreaseIndentPattern: /^\s*[\}\]\)]/,
  },
};

// ============================================================================
// Monarch Language Definition (Syntax Highlighting)
// ============================================================================

export const mathvizLanguageDefinition: monaco.languages.IMonarchLanguage = {
  defaultToken: 'invalid',
  tokenPostfix: '.mviz',

  // Keywords
  keywords: [
    'fn',
    'let',
    'const',
    'mut',
    'if',
    'else',
    'elif',
    'for',
    'while',
    'loop',
    'break',
    'continue',
    'return',
    'match',
    'struct',
    'trait',
    'enum',
    'impl',
    'type',
    'pub',
    'priv',
    'static',
    'async',
    'await',
    'yield',
    'import',
    'from',
    'as',
    'use',
    'mod',
    'self',
    'super',
    'where',
    'defer',
    'try',
    'catch',
    'throw',
    'with',
    'lambda',
  ],

  // Type keywords
  typeKeywords: [
    'Int',
    'Float',
    'Bool',
    'String',
    'Char',
    'List',
    'Set',
    'Dict',
    'Map',
    'Tuple',
    'Optional',
    'Result',
    'None',
    'Some',
    'Ok',
    'Err',
    'Any',
    'Never',
    'Void',
    'Self',
    // Math types
    'Vec2',
    'Vec3',
    'Vec4',
    'Matrix',
    'Complex',
    'Fraction',
    'Point',
    'Line',
    'Circle',
    'Polygon',
    'Curve',
    // Animation types
    'Scene',
    'Animation',
    'Mobject',
    'Transform',
    'Tex',
    'Text',
    'Graph',
    'Axes',
    'NumberPlane',
    'Camera',
  ],

  // Constants
  constants: [
    'true',
    'false',
    'null',
    'PI',
    'TAU',
    'E',
    'PHI',
    'INF',
    'NAN',
    'ORIGIN',
    'UP',
    'DOWN',
    'LEFT',
    'RIGHT',
    'IN',
    'OUT',
    'UL',
    'UR',
    'DL',
    'DR',
  ],

  // Built-in functions
  builtins: [
    // Core functions
    'print',
    'println',
    'input',
    'assert',
    'panic',
    'todo',
    'unreachable',
    'dbg',
    'typeof',
    'sizeof',
    'len',
    'range',
    'enumerate',
    'zip',
    'map',
    'filter',
    'reduce',
    'fold',
    'sort',
    'reverse',
    'clone',
    'copy',
    // Math functions
    'abs',
    'min',
    'max',
    'clamp',
    'floor',
    'ceil',
    'round',
    'sqrt',
    'pow',
    'exp',
    'log',
    'log2',
    'log10',
    'sin',
    'cos',
    'tan',
    'asin',
    'acos',
    'atan',
    'atan2',
    'sinh',
    'cosh',
    'tanh',
    'degrees',
    'radians',
    // Animation functions
    'play',
    'wait',
    'create',
    'write',
    'fade_in',
    'fade_out',
    'grow',
    'shrink',
    'transform',
    'move_to',
    'shift',
    'scale',
    'rotate',
    'animate',
    // Color functions
    'rgb',
    'rgba',
    'hsl',
    'hsla',
    'hex',
    'interpolate_color',
  ],

  // Operators
  operators: [
    '=',
    '>',
    '<',
    '!',
    '~',
    '?',
    ':',
    '==',
    '<=',
    '>=',
    '!=',
    '&&',
    '||',
    '++',
    '--',
    '+',
    '-',
    '*',
    '/',
    '&',
    '|',
    '^',
    '%',
    '<<',
    '>>',
    '>>>',
    '+=',
    '-=',
    '*=',
    '/=',
    '&=',
    '|=',
    '^=',
    '%=',
    '<<=',
    '>>=',
    '=>',
    '->',
    '|>',
    '<|',
    '::',
    '..',
    '..=',
    '**',
  ],

  // Symbols
  symbols: /[=><!~?:&|+\-*\/\^%]+/,
  escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,
  digits: /\d+(_+\d+)*/,
  octaldigits: /[0-7]+(_+[0-7]+)*/,
  binarydigits: /[0-1]+(_+[0-1]+)*/,
  hexdigits: /[[0-9a-fA-F]+(_+[0-9a-fA-F]+)*/,

  // Tokenizer
  tokenizer: {
    root: [
      // Identifiers and keywords
      [
        /[a-zA-Z_]\w*/,
        {
          cases: {
            '@keywords': 'keyword',
            '@typeKeywords': 'type.identifier',
            '@constants': 'constant',
            '@builtins': 'predefined',
            '@default': 'identifier',
          },
        },
      ],

      // Whitespace
      { include: '@whitespace' },

      // Delimiters and operators
      [/[{}()\[\]]/, '@brackets'],
      [/[<>](?!@symbols)/, '@brackets'],
      [
        /@symbols/,
        {
          cases: {
            '@operators': 'operator',
            '@default': '',
          },
        },
      ],

      // Numbers
      [/(@digits)[eE]([\-+]?(@digits))?[fFdD]?/, 'number.float'],
      [/(@digits)\.(@digits)([eE][\-+]?(@digits))?[fFdD]?/, 'number.float'],
      [/0[xX](@hexdigits)[Ll]?/, 'number.hex'],
      [/0[oO]?(@octaldigits)[Ll]?/, 'number.octal'],
      [/0[bB](@binarydigits)[Ll]?/, 'number.binary'],
      [/(@digits)[fFdD]/, 'number.float'],
      [/(@digits)[lL]?/, 'number'],

      // Delimiter: comma, semicolon, dot
      [/[;,.]/, 'delimiter'],

      // Strings
      [/f"/, { token: 'string.quote', bracket: '@open', next: '@fstring' }],
      [/"([^"\\]|\\.)*$/, 'string.invalid'], // non-terminated string
      [/"/, { token: 'string.quote', bracket: '@open', next: '@string' }],

      // Raw strings
      [/r"/, { token: 'string.quote', bracket: '@open', next: '@rawstring' }],

      // Characters
      [/'[^\\']'/, 'string'],
      [/(')(@escapes)(')/, ['string', 'string.escape', 'string']],
      [/'/, 'string.invalid'],
    ],

    // Comments
    comment: [
      [/[^\/*]+/, 'comment'],
      [/\/\*/, 'comment', '@push'], // nested comment
      ['\\*/', 'comment', '@pop'],
      [/[\/*]/, 'comment'],
    ],

    // Strings
    string: [
      [/[^\\"]+/, 'string'],
      [/@escapes/, 'string.escape'],
      [/\\./, 'string.escape.invalid'],
      [/"/, { token: 'string.quote', bracket: '@close', next: '@pop' }],
    ],

    // F-strings (interpolated strings)
    fstring: [
      [/\{/, { token: 'delimiter.bracket', next: '@fstringExpr' }],
      [/[^\\"{]+/, 'string'],
      [/@escapes/, 'string.escape'],
      [/\\./, 'string.escape.invalid'],
      [/"/, { token: 'string.quote', bracket: '@close', next: '@pop' }],
    ],

    fstringExpr: [
      [/\}/, { token: 'delimiter.bracket', next: '@pop' }],
      { include: 'root' },
    ],

    // Raw strings
    rawstring: [
      [/[^"]+/, 'string'],
      [/"/, { token: 'string.quote', bracket: '@close', next: '@pop' }],
    ],

    // Whitespace and comments
    whitespace: [
      [/[ \t\r\n]+/, 'white'],
      [/\/\*/, 'comment', '@comment'],
      [/\/\/\/.*$/, 'comment.doc'],
      [/\/\/.*$/, 'comment'],
    ],
  },
};

// ============================================================================
// Completions
// ============================================================================

export const mathvizCompletionItemProvider: monaco.languages.CompletionItemProvider = {
  triggerCharacters: ['.', ':', '@'],

  provideCompletionItems: (
    model: monaco.editor.ITextModel,
    position: monaco.Position,
    _context: monaco.languages.CompletionContext,
    _token: monaco.CancellationToken
  ): monaco.languages.ProviderResult<monaco.languages.CompletionList> => {
    const word = model.getWordUntilPosition(position);
    const range: monaco.IRange = {
      startLineNumber: position.lineNumber,
      endLineNumber: position.lineNumber,
      startColumn: word.startColumn,
      endColumn: word.endColumn,
    };

    const suggestions: monaco.languages.CompletionItem[] = [];

    // Keywords
    const keywords = [
      'fn', 'let', 'const', 'mut', 'if', 'else', 'elif', 'for', 'while',
      'loop', 'break', 'continue', 'return', 'match', 'struct', 'trait',
      'enum', 'impl', 'type', 'pub', 'import', 'from', 'as', 'use',
    ];

    for (const kw of keywords) {
      suggestions.push({
        label: kw,
        kind: 17, // monaco.languages.CompletionItemKind.Keyword
        insertText: kw,
        range,
        detail: 'keyword',
      });
    }

    // Types
    const types = [
      'Int', 'Float', 'Bool', 'String', 'List', 'Set', 'Dict',
      'Optional', 'Result', 'Vec2', 'Vec3', 'Matrix', 'Scene',
      'Animation', 'Mobject', 'Transform', 'Tex', 'Text',
    ];

    for (const type of types) {
      suggestions.push({
        label: type,
        kind: 6, // monaco.languages.CompletionItemKind.Class
        insertText: type,
        range,
        detail: 'type',
      });
    }

    // Built-in functions
    const builtins = [
      { name: 'print', snippet: 'print($1)$0', doc: 'Print to stdout' },
      { name: 'println', snippet: 'println($1)$0', doc: 'Print with newline' },
      { name: 'len', snippet: 'len($1)$0', doc: 'Get length of collection' },
      { name: 'range', snippet: 'range($1, $2)$0', doc: 'Create a range iterator' },
      { name: 'map', snippet: 'map($1, |$2| $3)$0', doc: 'Map over collection' },
      { name: 'filter', snippet: 'filter($1, |$2| $3)$0', doc: 'Filter collection' },
      { name: 'abs', snippet: 'abs($1)$0', doc: 'Absolute value' },
      { name: 'sqrt', snippet: 'sqrt($1)$0', doc: 'Square root' },
      { name: 'sin', snippet: 'sin($1)$0', doc: 'Sine function' },
      { name: 'cos', snippet: 'cos($1)$0', doc: 'Cosine function' },
    ];

    for (const fn of builtins) {
      suggestions.push({
        label: fn.name,
        kind: 1, // monaco.languages.CompletionItemKind.Function
        insertText: fn.snippet,
        insertTextRules: 4, // monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet
        range,
        detail: fn.doc,
        documentation: fn.doc,
      });
    }

    // Animation functions
    const animationFunctions = [
      { name: 'play', snippet: 'play($1)$0', doc: 'Play an animation' },
      { name: 'wait', snippet: 'wait($1)$0', doc: 'Wait for duration' },
      { name: 'create', snippet: 'create($1)$0', doc: 'Create animation' },
      { name: 'write', snippet: 'write($1)$0', doc: 'Write text animation' },
      { name: 'fade_in', snippet: 'fade_in($1)$0', doc: 'Fade in animation' },
      { name: 'fade_out', snippet: 'fade_out($1)$0', doc: 'Fade out animation' },
      { name: 'transform', snippet: 'transform($1, $2)$0', doc: 'Transform between objects' },
      { name: 'move_to', snippet: 'move_to($1)$0', doc: 'Move to position' },
      { name: 'shift', snippet: 'shift($1)$0', doc: 'Shift by vector' },
      { name: 'scale', snippet: 'scale($1)$0', doc: 'Scale by factor' },
      { name: 'rotate', snippet: 'rotate($1)$0', doc: 'Rotate by angle' },
    ];

    for (const fn of animationFunctions) {
      suggestions.push({
        label: fn.name,
        kind: 1, // Function
        insertText: fn.snippet,
        insertTextRules: 4,
        range,
        detail: `Animation: ${fn.doc}`,
        documentation: fn.doc,
      });
    }

    // Snippets
    const snippets = [
      {
        label: 'fn',
        snippet: 'fn ${1:name}(${2:params}) -> ${3:ReturnType} {\n\t$0\n}',
        doc: 'Function definition',
      },
      {
        label: 'struct',
        snippet: 'struct ${1:Name} {\n\t${2:field}: ${3:Type},\n}',
        doc: 'Struct definition',
      },
      {
        label: 'impl',
        snippet: 'impl ${1:Type} {\n\t$0\n}',
        doc: 'Implementation block',
      },
      {
        label: 'trait',
        snippet: 'trait ${1:Name} {\n\t$0\n}',
        doc: 'Trait definition',
      },
      {
        label: 'enum',
        snippet: 'enum ${1:Name} {\n\t${2:Variant},\n}',
        doc: 'Enum definition',
      },
      {
        label: 'match',
        snippet: 'match ${1:expr} {\n\t${2:pattern} => $3,\n}',
        doc: 'Match expression',
      },
      {
        label: 'for',
        snippet: 'for ${1:item} in ${2:iter} {\n\t$0\n}',
        doc: 'For loop',
      },
      {
        label: 'while',
        snippet: 'while ${1:condition} {\n\t$0\n}',
        doc: 'While loop',
      },
      {
        label: 'if',
        snippet: 'if ${1:condition} {\n\t$0\n}',
        doc: 'If statement',
      },
      {
        label: 'ifelse',
        snippet: 'if ${1:condition} {\n\t$2\n} else {\n\t$0\n}',
        doc: 'If-else statement',
      },
      {
        label: 'scene',
        snippet: 'scene ${1:SceneName} {\n\tfn construct(self) {\n\t\t$0\n\t}\n}',
        doc: 'Scene definition for animations',
      },
    ];

    for (const snippet of snippets) {
      suggestions.push({
        label: snippet.label,
        kind: 27, // monaco.languages.CompletionItemKind.Snippet
        insertText: snippet.snippet,
        insertTextRules: 4,
        range,
        detail: snippet.doc,
        documentation: snippet.doc,
      });
    }

    return { suggestions };
  },
};

// ============================================================================
// Hover Provider
// ============================================================================

export const mathvizHoverProvider: monaco.languages.HoverProvider = {
  provideHover: (
    model: monaco.editor.ITextModel,
    position: monaco.Position,
    _token: monaco.CancellationToken
  ): monaco.languages.ProviderResult<monaco.languages.Hover> => {
    const word = model.getWordAtPosition(position);
    if (!word) return null;

    const docs: Record<string, { signature: string; description: string }> = {
      // Functions
      print: { signature: 'fn print(value: Any) -> Void', description: 'Print a value to stdout.' },
      println: { signature: 'fn println(value: Any) -> Void', description: 'Print a value with newline.' },
      len: { signature: 'fn len<T>(collection: T) -> Int', description: 'Returns the length of a collection.' },
      range: { signature: 'fn range(start: Int, end: Int, step?: Int) -> Iterator<Int>', description: 'Creates an iterator over a range of integers.' },
      sqrt: { signature: 'fn sqrt(x: Float) -> Float', description: 'Computes the square root of x.' },
      sin: { signature: 'fn sin(x: Float) -> Float', description: 'Computes the sine of x (in radians).' },
      cos: { signature: 'fn cos(x: Float) -> Float', description: 'Computes the cosine of x (in radians).' },
      // Animation
      play: { signature: 'fn play(animation: Animation) -> Void', description: 'Play an animation in the current scene.' },
      wait: { signature: 'fn wait(duration: Float) -> Void', description: 'Pause for the specified duration in seconds.' },
      transform: { signature: 'fn transform(source: Mobject, target: Mobject) -> Animation', description: 'Create a transformation animation between two objects.' },
      // Types
      Vec2: { signature: 'struct Vec2 { x: Float, y: Float }', description: 'A 2D vector type.' },
      Vec3: { signature: 'struct Vec3 { x: Float, y: Float, z: Float }', description: 'A 3D vector type.' },
      Scene: { signature: 'trait Scene', description: 'Base trait for animation scenes.' },
      Mobject: { signature: 'trait Mobject', description: 'Base trait for mathematical objects.' },
    };

    const info = docs[word.word];
    if (!info) return null;

    return {
      range: {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn,
      },
      contents: [
        { value: `\`\`\`mathviz\n${info.signature}\n\`\`\`` },
        { value: info.description },
      ],
    };
  },
};

// ============================================================================
// Registration Function
// ============================================================================

export function registerMathVizLanguage(monacoInstance: typeof monaco): void {
  // Register the language
  monacoInstance.languages.register({ id: 'mathviz', extensions: ['.mviz'] });

  // Set the language configuration
  monacoInstance.languages.setLanguageConfiguration('mathviz', mathvizLanguageConfiguration);

  // Set the token provider
  monacoInstance.languages.setMonarchTokensProvider('mathviz', mathvizLanguageDefinition);

  // Register the completion provider
  monacoInstance.languages.registerCompletionItemProvider('mathviz', mathvizCompletionItemProvider);

  // Register the hover provider
  monacoInstance.languages.registerHoverProvider('mathviz', mathvizHoverProvider);
}
