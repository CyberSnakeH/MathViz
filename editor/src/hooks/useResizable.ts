/**
 * useResizable Hook
 *
 * Custom hook for creating resizable panels with mouse drag support.
 */

import { useState, useCallback, useRef, useEffect } from 'react';

// ============================================================================
// Types
// ============================================================================

interface UseResizableOptions {
  direction: 'horizontal' | 'vertical';
  initialSize: number;
  minSize?: number;
  maxSize?: number;
  onResize?: (size: number) => void;
  onResizeStart?: () => void;
  onResizeEnd?: (size: number) => void;
}

interface UseResizableReturn {
  size: number;
  isResizing: boolean;
  handleMouseDown: (e: React.MouseEvent) => void;
  setSize: (size: number) => void;
}

// ============================================================================
// Hook Implementation
// ============================================================================

export function useResizable(options: UseResizableOptions): UseResizableReturn {
  const {
    direction,
    initialSize,
    minSize = 0,
    maxSize = Infinity,
    onResize,
    onResizeStart,
    onResizeEnd,
  } = options;

  const [size, setSize] = useState(initialSize);
  const [isResizing, setIsResizing] = useState(false);

  const startPos = useRef(0);
  const startSize = useRef(initialSize);

  const clampSize = useCallback(
    (value: number) => Math.max(minSize, Math.min(maxSize, value)),
    [minSize, maxSize]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      setIsResizing(true);
      startPos.current = direction === 'horizontal' ? e.clientX : e.clientY;
      startSize.current = size;
      onResizeStart?.();
    },
    [direction, size, onResizeStart]
  );

  useEffect(() => {
    if (!isResizing) return;

    const handleMouseMove = (e: MouseEvent) => {
      const currentPos = direction === 'horizontal' ? e.clientX : e.clientY;
      const delta = currentPos - startPos.current;
      const newSize = clampSize(startSize.current + delta);

      setSize(newSize);
      onResize?.(newSize);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      onResizeEnd?.(size);
    };

    // Set cursor style on body during resize
    document.body.style.cursor =
      direction === 'horizontal' ? 'col-resize' : 'row-resize';
    document.body.style.userSelect = 'none';

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing, direction, clampSize, onResize, onResizeEnd, size]);

  const setSizeExternal = useCallback(
    (newSize: number) => {
      const clamped = clampSize(newSize);
      setSize(clamped);
      onResize?.(clamped);
    },
    [clampSize, onResize]
  );

  return {
    size,
    isResizing,
    handleMouseDown,
    setSize: setSizeExternal,
  };
}

// ============================================================================
// Export
// ============================================================================

export default useResizable;
