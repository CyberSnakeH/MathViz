/**
 * useKeyboardShortcut Hook
 *
 * Custom hook for handling keyboard shortcuts in the editor.
 */

import { useEffect, useRef } from 'react';

// ============================================================================
// Types
// ============================================================================

interface KeyboardShortcutOptions {
  ctrl?: boolean;
  meta?: boolean;
  shift?: boolean;
  alt?: boolean;
  key: string;
  callback: (e: KeyboardEvent) => void;
  enabled?: boolean;
  preventDefault?: boolean;
}

interface ShortcutKey {
  key: string;
  ctrl?: boolean;
  meta?: boolean;
  shift?: boolean;
  alt?: boolean;
}

// ============================================================================
// Hook Implementation
// ============================================================================

export function useKeyboardShortcut(options: KeyboardShortcutOptions): void {
  const {
    ctrl = false,
    meta = false,
    shift = false,
    alt = false,
    key,
    callback,
    enabled = true,
    preventDefault = true,
  } = options;

  const callbackRef = useRef(callback);
  callbackRef.current = callback;

  useEffect(() => {
    if (!enabled) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      const ctrlMatch = ctrl ? e.ctrlKey : !e.ctrlKey;
      const metaMatch = meta ? e.metaKey : !e.metaKey;
      const shiftMatch = shift ? e.shiftKey : !e.shiftKey;
      const altMatch = alt ? e.altKey : !e.altKey;
      const keyMatch = e.key.toLowerCase() === key.toLowerCase();

      // For Ctrl/Cmd shortcuts, allow either
      const modifierMatch = (ctrl || meta)
        ? (e.ctrlKey || e.metaKey) && shiftMatch && altMatch
        : ctrlMatch && metaMatch && shiftMatch && altMatch;

      if (modifierMatch && keyMatch) {
        if (preventDefault) {
          e.preventDefault();
        }
        callbackRef.current(e);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [ctrl, meta, shift, alt, key, enabled, preventDefault]);
}

// ============================================================================
// Multiple Shortcuts Hook
// ============================================================================

export function useKeyboardShortcuts(
  shortcuts: KeyboardShortcutOptions[]
): void {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      for (const shortcut of shortcuts) {
        if (shortcut.enabled === false) continue;

        const {
          ctrl = false,
          meta = false,
          shift = false,
          alt = false,
          key,
          callback,
          preventDefault = true,
        } = shortcut;

        const ctrlMatch = ctrl ? e.ctrlKey : !e.ctrlKey;
        const metaMatch = meta ? e.metaKey : !e.metaKey;
        const shiftMatch = shift ? e.shiftKey : !e.shiftKey;
        const altMatch = alt ? e.altKey : !e.altKey;
        const keyMatch = e.key.toLowerCase() === key.toLowerCase();

        // For Ctrl/Cmd shortcuts, allow either
        const modifierMatch = (ctrl || meta)
          ? (e.ctrlKey || e.metaKey) && shiftMatch && altMatch
          : ctrlMatch && metaMatch && shiftMatch && altMatch;

        if (modifierMatch && keyMatch) {
          if (preventDefault) {
            e.preventDefault();
          }
          callback(e);
          break;
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts]);
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Parse a shortcut string like "Ctrl+Shift+S" into components
 */
export function parseShortcut(shortcutString: string): ShortcutKey {
  const parts = shortcutString.toLowerCase().split('+');
  const key = parts.pop() || '';

  return {
    key,
    ctrl: parts.includes('ctrl') || parts.includes('control'),
    meta: parts.includes('cmd') || parts.includes('meta') || parts.includes('command'),
    shift: parts.includes('shift'),
    alt: parts.includes('alt') || parts.includes('option'),
  };
}

/**
 * Format a shortcut for display based on platform
 */
export function formatShortcut(shortcut: ShortcutKey, isMac = false): string {
  const parts: string[] = [];

  if (shortcut.ctrl || shortcut.meta) {
    parts.push(isMac ? 'Cmd' : 'Ctrl');
  }
  if (shortcut.shift) {
    parts.push('Shift');
  }
  if (shortcut.alt) {
    parts.push(isMac ? 'Option' : 'Alt');
  }

  // Capitalize key
  const key = shortcut.key.length === 1
    ? shortcut.key.toUpperCase()
    : shortcut.key.charAt(0).toUpperCase() + shortcut.key.slice(1);

  parts.push(key);

  return parts.join(isMac ? '' : '+');
}

// ============================================================================
// Export
// ============================================================================

export default useKeyboardShortcut;
