/**
 * MathViz Editor Helper Functions
 *
 * Common utility functions used throughout the editor.
 */

import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// ============================================================================
// Class Name Utilities
// ============================================================================

/**
 * Merge class names with Tailwind CSS support.
 * Combines clsx for conditional classes and tailwind-merge for deduplication.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

// ============================================================================
// Formatting Utilities
// ============================================================================

/**
 * Format bytes to human-readable string.
 */
export function formatBytes(bytes: number, decimals = 2): string {
  if (bytes === 0) return '0 B';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

/**
 * Format date to relative time string.
 */
export function formatDate(date: Date | string | number): string {
  const d = new Date(date);
  const now = new Date();
  const diff = now.getTime() - d.getTime();

  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (seconds < 60) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;

  return d.toLocaleDateString();
}

/**
 * Format duration in seconds to mm:ss or hh:mm:ss.
 */
export function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);

  if (h > 0) {
    return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  }
  return `${m}:${s.toString().padStart(2, '0')}`;
}

// ============================================================================
// Function Utilities
// ============================================================================

/**
 * Debounce a function.
 */
export function debounce<T extends (...args: unknown[]) => unknown>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: ReturnType<typeof setTimeout> | null = null;

  return function (this: unknown, ...args: Parameters<T>) {
    if (timeout) {
      clearTimeout(timeout);
    }

    timeout = setTimeout(() => {
      func.apply(this, args);
    }, wait);
  };
}

/**
 * Throttle a function.
 */
export function throttle<T extends (...args: unknown[]) => unknown>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle = false;

  return function (this: unknown, ...args: Parameters<T>) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => {
        inThrottle = false;
      }, limit);
    }
  };
}

// ============================================================================
// Path Utilities
// ============================================================================

/**
 * Get file name from path.
 */
export function getFileName(path: string): string {
  return path.split('/').pop() || path;
}

/**
 * Get file extension from path.
 */
export function getFileExtension(path: string): string {
  const name = getFileName(path);
  const lastDot = name.lastIndexOf('.');
  return lastDot === -1 ? '' : name.slice(lastDot + 1).toLowerCase();
}

/**
 * Get directory path from file path.
 */
export function getDirectory(path: string): string {
  const lastSlash = path.lastIndexOf('/');
  return lastSlash === -1 ? '' : path.slice(0, lastSlash);
}

/**
 * Join path segments.
 */
export function joinPath(...segments: string[]): string {
  return segments
    .map((s, i) => {
      if (i === 0) return s.replace(/\/+$/, '');
      return s.replace(/^\/+|\/+$/g, '');
    })
    .filter(Boolean)
    .join('/');
}

// ============================================================================
// Platform Utilities
// ============================================================================

/**
 * Check if running on macOS.
 */
export function isMac(): boolean {
  return typeof navigator !== 'undefined' && /Mac/i.test(navigator.platform);
}

/**
 * Check if running on Windows.
 */
export function isWindows(): boolean {
  return typeof navigator !== 'undefined' && /Win/i.test(navigator.platform);
}

/**
 * Check if running on Linux.
 */
export function isLinux(): boolean {
  return typeof navigator !== 'undefined' && /Linux/i.test(navigator.platform);
}

/**
 * Get keyboard modifier key based on platform.
 */
export function getModKey(): string {
  return isMac() ? 'Cmd' : 'Ctrl';
}

// ============================================================================
// Async Utilities
// ============================================================================

/**
 * Sleep for a given number of milliseconds.
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Retry an async function with exponential backoff.
 */
export async function retry<T>(
  fn: () => Promise<T>,
  options: { maxAttempts?: number; delay?: number; backoff?: number } = {}
): Promise<T> {
  const { maxAttempts = 3, delay = 1000, backoff = 2 } = options;

  let lastError: Error | undefined;
  let currentDelay = delay;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      if (attempt < maxAttempts) {
        await sleep(currentDelay);
        currentDelay *= backoff;
      }
    }
  }

  throw lastError;
}

// ============================================================================
// Validation Utilities
// ============================================================================

/**
 * Check if a string is a valid file name.
 */
export function isValidFileName(name: string): boolean {
  if (!name || name.length === 0 || name.length > 255) return false;
  // Disallow: / \ : * ? " < > | and control characters
  const invalidChars = /[/\\:*?"<>|\x00-\x1f]/;
  return !invalidChars.test(name);
}

/**
 * Sanitize a file name by removing invalid characters.
 */
export function sanitizeFileName(name: string): string {
  return name
    .replace(/[/\\:*?"<>|\x00-\x1f]/g, '_')
    .trim()
    .slice(0, 255);
}

// ============================================================================
// Color Utilities
// ============================================================================

/**
 * Convert hex color to RGB.
 */
export function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
      }
    : null;
}

/**
 * Convert RGB to hex color.
 */
export function rgbToHex(r: number, g: number, b: number): string {
  return '#' + [r, g, b].map((x) => x.toString(16).padStart(2, '0')).join('');
}

/**
 * Lighten a hex color by a percentage.
 */
export function lightenColor(hex: string, percent: number): string {
  const rgb = hexToRgb(hex);
  if (!rgb) return hex;

  const { r, g, b } = rgb;
  const amt = Math.round(2.55 * percent);

  return rgbToHex(
    Math.min(255, r + amt),
    Math.min(255, g + amt),
    Math.min(255, b + amt)
  );
}

/**
 * Darken a hex color by a percentage.
 */
export function darkenColor(hex: string, percent: number): string {
  const rgb = hexToRgb(hex);
  if (!rgb) return hex;

  const { r, g, b } = rgb;
  const amt = Math.round(2.55 * percent);

  return rgbToHex(
    Math.max(0, r - amt),
    Math.max(0, g - amt),
    Math.max(0, b - amt)
  );
}

// ============================================================================
// Unique ID Generation
// ============================================================================

let idCounter = 0;

/**
 * Generate a unique ID.
 */
export function generateId(prefix = 'id'): string {
  return `${prefix}_${++idCounter}_${Date.now().toString(36)}`;
}

/**
 * Generate a UUID v4.
 */
export function uuid(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}
