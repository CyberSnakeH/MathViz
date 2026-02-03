/**
 * MathViz Documentation Site
 */

// ============================================
// Prism.js MathViz Language Definition
// ============================================
if (typeof Prism !== 'undefined') {
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
    'builtin': /\b(?:print|println|play|wait|animate|len|range|map|filter|Some|Ok|Err)\b/,
    'type': /\b(?:Int|Float|Bool|String|List|Set|Dict|Tuple|Optional|Result|Vec|Mat|Array|Scene|Circle|Square|Rectangle|Line|Arrow|Text|MathTex|Mobject)\b/,
    'decorator': /@\w+/,
    'function': /\b[a-z_]\w*(?=\s*\()/i,
    'number': /\b\d+(?:\.\d+)?\b/,
    'operator': /->|=>|[+\-*/%=<>!&|^~?:]+|∪|∩|∈|⊂|⊃/,
    'punctuation': /[{}[\]();,.]/
  };

  Prism.languages.mathviz = Prism.languages.mviz;
}

// ============================================
// Helper: Create SVG icon
// ============================================
function createIcon(type) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', '16');
  svg.setAttribute('height', '16');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '2');

  if (type === 'copy') {
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', '9');
    rect.setAttribute('y', '9');
    rect.setAttribute('width', '13');
    rect.setAttribute('height', '13');
    rect.setAttribute('rx', '2');
    svg.appendChild(rect);

    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', 'M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1');
    svg.appendChild(path);
  } else if (type === 'check') {
    const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    polyline.setAttribute('points', '20 6 9 17 4 12');
    svg.appendChild(polyline);
  }

  return svg;
}

// ============================================
// Copy Buttons
// ============================================
function initCopyButtons() {
  document.querySelectorAll('pre[class*="language-"]').forEach(pre => {
    if (pre.querySelector('.copy-btn')) return;

    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.appendChild(createIcon('copy'));
    btn.title = 'Copy code';

    pre.style.position = 'relative';
    pre.appendChild(btn);

    btn.addEventListener('click', async () => {
      const code = pre.querySelector('code');
      const text = code ? code.textContent : pre.textContent;

      try {
        await navigator.clipboard.writeText(text);
        btn.textContent = '';
        btn.appendChild(createIcon('check'));
        btn.classList.add('copied');

        setTimeout(() => {
          btn.textContent = '';
          btn.appendChild(createIcon('copy'));
          btn.classList.remove('copied');
        }, 2000);
      } catch (e) {
        console.error('Copy failed:', e);
      }
    });
  });
}

// ============================================
// Search
// ============================================
const Search = {
  pages: [
    { title: 'Getting Started', url: 'docs/getting-started.html', tags: 'install setup' },
    { title: 'Language Reference', url: 'docs/language-reference.html', tags: 'syntax types' },
    { title: 'Scenes & Animations', url: 'docs/scenes.html', tags: 'manim animation' },
    { title: 'Module System', url: 'docs/modules.html', tags: 'use mod pub import' },
    { title: 'CLI Reference', url: 'docs/cli.html', tags: 'compile run fmt' },
    { title: 'Editor Guide', url: 'docs/editor.html', tags: 'tauri desktop' },
  ],

  init() {
    const modal = document.getElementById('search-modal');
    if (!modal) return;

    const input = modal.querySelector('.search-input');
    const results = modal.querySelector('.search-results');

    document.addEventListener('keydown', e => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        modal.classList.toggle('open');
        if (modal.classList.contains('open')) {
          input.focus();
          input.value = '';
          this.render(results, '');
        }
      }
      if (e.key === 'Escape') modal.classList.remove('open');
    });

    modal.addEventListener('click', e => {
      if (e.target === modal) modal.classList.remove('open');
    });

    input.addEventListener('input', () => this.render(results, input.value));

    document.querySelectorAll('[data-search]').forEach(btn => {
      btn.addEventListener('click', () => {
        modal.classList.add('open');
        input.focus();
      });
    });
  },

  render(container, query) {
    const q = query.toLowerCase().trim();
    while (container.firstChild) container.removeChild(container.firstChild);

    if (!q) {
      const hint = document.createElement('div');
      hint.className = 'search-hint';
      hint.textContent = 'Type to search...';
      container.appendChild(hint);
      return;
    }

    const matches = this.pages.filter(p =>
      (p.title + ' ' + p.tags).toLowerCase().includes(q)
    );

    if (!matches.length) {
      const hint = document.createElement('div');
      hint.className = 'search-hint';
      hint.textContent = 'No results';
      container.appendChild(hint);
      return;
    }

    matches.forEach(p => {
      const a = document.createElement('a');
      a.href = p.url;
      a.className = 'search-result';
      a.textContent = p.title;
      container.appendChild(a);
    });
  }
};

// ============================================
// Reveal Animations
// ============================================
function initReveal() {
  // Mark that JS is loaded for CSS animation
  document.documentElement.classList.add('js-loaded');

  const reveals = document.querySelectorAll('.reveal');

  // Immediately show elements that are already in viewport
  reveals.forEach(el => {
    const rect = el.getBoundingClientRect();
    if (rect.top < window.innerHeight && rect.bottom > 0) {
      el.classList.add('visible');
    }
  });

  // Observer for elements that scroll into view
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '50px' });

  reveals.forEach(el => {
    if (!el.classList.contains('visible')) {
      observer.observe(el);
    }
  });
}

// ============================================
// Mobile Nav
// ============================================
function initMobileNav() {
  const toggle = document.querySelector('.mobile-nav-toggle');
  const sidebar = document.querySelector('.sidebar');

  if (toggle && sidebar) {
    toggle.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });
  }
}

// ============================================
// Active Link Highlight
// ============================================
function initActiveLinks() {
  const path = location.pathname;
  document.querySelectorAll('.sidebar a, .nav a').forEach(link => {
    if (link.getAttribute('href') && path.includes(link.getAttribute('href'))) {
      link.classList.add('active');
    }
  });
}

// ============================================
// TOC Scroll Spy
// ============================================
function initTocSpy() {
  const toc = document.querySelector('.toc');
  if (!toc) return;

  const headings = document.querySelectorAll('h2[id], h3[id]');
  const links = toc.querySelectorAll('a');

  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        links.forEach(link => {
          link.classList.toggle('active', link.hash === '#' + entry.target.id);
        });
      }
    });
  }, { rootMargin: '-100px 0px -70% 0px' });

  headings.forEach(h => observer.observe(h));
}

// ============================================
// Init
// ============================================
document.addEventListener('DOMContentLoaded', () => {
  initCopyButtons();
  Search.init();
  initReveal();
  initMobileNav();
  initActiveLinks();
  initTocSpy();

  if (typeof Prism !== 'undefined') {
    Prism.highlightAll();
  }
});
