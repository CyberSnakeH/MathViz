// MathViz language definition for Prism.js
Prism.languages.mviz = {
  'comment': [
    { pattern: /\/\/.*/, greedy: true },
    { pattern: /\/\*[\s\S]*?\*\//, greedy: true },
    { pattern: /#.*/, greedy: true }
  ],
  'string': {
    pattern: /f?"(?:[^"\\]|\\.)*"|f?'(?:[^'\\]|\\.)*'/,
    greedy: true
  },
  'keyword': /\b(?:let|const|fn|class|scene|if|else|elif|for|while|return|import|from|as|in|true|false|True|False|None|and|or|not|break|continue|pass|match|where|use|mod|pub|struct|impl|trait|enum|self|async|await|try|catch|throw)\b/,
  'builtin': /\b(?:print|println|play|wait|animate|len|range|map|filter|Some|Ok|Err|FadeIn|FadeOut|Transform|Create|Write|Circle|Square|Rectangle|Line|Arrow|Text|MathTex)\b/,
  'class-name': /\b(?:Int|Float|Bool|String|List|Set|Dict|Tuple|Optional|Result|Vec|Mat|Array|Scene|Mobject)\b/,
  'decorator': { pattern: /@\w+/, alias: 'annotation' },
  'function': /\b[a-z_]\w*(?=\s*\()/i,
  'number': /\b\d+(?:\.\d+)?\b/,
  'operator': /->|=>|[+\-*\/%=<>!&|^~?:]+|∪|∩|∈|⊂|⊃/,
  'punctuation': /[{}[\]();,.]/
};

// Alias
Prism.languages.mathviz = Prism.languages.mviz;

// Re-highlight all code blocks
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', function() {
    Prism.highlightAll();
  });
} else {
  Prism.highlightAll();
}
