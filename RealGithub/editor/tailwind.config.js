/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // Tokyo Night Storm Base Colors
        'tn-bg': '#1a1b26',
        'tn-bg-dark': '#16161e',
        'tn-bg-highlight': '#292e42',
        'tn-fg': '#a9b1d6',
        'tn-fg-dark': '#565f89',
        'tn-fg-gutter': '#3b4261',

        // Tokyo Night Accent Colors
        'tn-blue': '#7aa2f7',
        'tn-cyan': '#7dcfff',
        'tn-blue1': '#2ac3de',
        'tn-magenta': '#bb9af7',
        'tn-purple': '#9d7cd8',
        'tn-orange': '#ff9e64',
        'tn-yellow': '#e0af68',
        'tn-green': '#9ece6a',
        'tn-green1': '#73daca',
        'tn-red': '#f7768e',

        // Editor-specific colors (Tokyo Night mapping)
        'editor-bg': '#1a1b26',
        'editor-sidebar': '#1f2335',
        'editor-panel': '#1a1b26',
        'editor-border': '#1a1b26',
        'editor-active': '#292e42',
        'editor-hover': '#292e42',
        'editor-text': '#a9b1d6',
        'editor-text-muted': '#565f89',
        'editor-accent': '#7aa2f7',
        'editor-success': '#9ece6a',
        'editor-warning': '#e0af68',
        'editor-error': '#f7768e',
        'editor-info': '#7aa2f7',

        // MathViz brand colors
        'mathviz-primary': '#7aa2f7',
        'mathviz-secondary': '#bb9af7',
      },
      fontFamily: {
        sans: [
          'Inter',
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'Roboto',
          'sans-serif',
        ],
        mono: [
          'JetBrains Mono',
          'Fira Code',
          'Cascadia Code',
          'Consolas',
          'Monaco',
          'monospace',
        ],
      },
      fontSize: {
        'editor-xs': '11px',
        'editor-sm': '12px',
        'editor-base': '13px',
        'editor-lg': '14px',
      },
      borderRadius: {
        'sm': '4px',
        'md': '6px',
        'lg': '8px',
        'xl': '12px',
      },
      transitionDuration: {
        'fast': '100ms',
        'normal': '150ms',
        'slow': '250ms',
      },
      transitionTimingFunction: {
        'out': 'ease-out',
      },
      animation: {
        'fade-in': 'fadeIn 150ms ease-out',
        'fade-out': 'fadeOut 150ms ease-out',
        'slide-in-left': 'slideInFromLeft 150ms ease-out',
        'slide-in-right': 'slideInFromRight 150ms ease-out',
        'slide-in-top': 'slideInFromTop 150ms ease-out',
        'slide-in-bottom': 'slideInFromBottom 150ms ease-out',
        'scale-in': 'scaleIn 150ms ease-out',
        'bounce-in': 'bounceIn 250ms ease-out',
        'glow': 'glow 2s ease-in-out infinite',
        'pulse-slow': 'pulse 3s ease-in-out infinite',
        'spin-slow': 'spin 2s linear infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        fadeOut: {
          '0%': { opacity: '1' },
          '100%': { opacity: '0' },
        },
        slideInFromLeft: {
          '0%': { transform: 'translateX(-12px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        slideInFromRight: {
          '0%': { transform: 'translateX(12px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        slideInFromTop: {
          '0%': { transform: 'translateY(-12px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideInFromBottom: {
          '0%': { transform: 'translateY(12px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        scaleIn: {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        bounceIn: {
          '0%': { transform: 'scale(0.9)', opacity: '0' },
          '50%': { transform: 'scale(1.02)' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        glow: {
          '0%, 100%': { boxShadow: '0 0 5px #7aa2f7' },
          '50%': { boxShadow: '0 0 20px #7aa2f7' },
        },
      },
      boxShadow: {
        'sm': '0 1px 2px rgba(0, 0, 0, 0.3)',
        'md': '0 4px 6px rgba(0, 0, 0, 0.4)',
        'lg': '0 10px 15px rgba(0, 0, 0, 0.5)',
        'glow': '0 0 20px rgba(122, 162, 247, 0.15)',
        'glow-lg': '0 0 40px rgba(122, 162, 247, 0.25)',
      },
      backdropBlur: {
        'xs': '4px',
        'sm': '8px',
        'md': '12px',
        'lg': '16px',
      },
    },
  },
  plugins: [],
};
