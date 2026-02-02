/**
 * Resizer Component
 *
 * Smooth draggable resizer for panel sizing with:
 * - Subtle visual indicator on hover
 * - Accent color highlight when active
 * - Smooth cursor transitions
 * - Double-click to reset to default
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { cn } from '../../utils/helpers';

// ============================================================================
// Types
// ============================================================================

interface ResizerProps {
  direction: 'horizontal' | 'vertical';
  onResize: (delta: number) => void;
  onResizeStart?: () => void;
  onResizeEnd?: () => void;
  onDoubleClick?: () => void;
  className?: string;
}

// ============================================================================
// Resizer Component
// ============================================================================

export const Resizer: React.FC<ResizerProps> = ({
  direction,
  onResize,
  onResizeStart,
  onResizeEnd,
  onDoubleClick,
  className,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const lastPosition = useRef<number>(0);
  const resizerRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(true);
      lastPosition.current = direction === 'horizontal' ? e.clientX : e.clientY;
      onResizeStart?.();
    },
    [direction, onResizeStart]
  );

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging) return;

      const currentPosition = direction === 'horizontal' ? e.clientX : e.clientY;
      const delta = currentPosition - lastPosition.current;

      // Only trigger resize if delta is significant (reduces jitter)
      if (Math.abs(delta) >= 1) {
        lastPosition.current = currentPosition;
        onResize(delta);
      }
    },
    [isDragging, direction, onResize]
  );

  const handleMouseUp = useCallback(() => {
    if (isDragging) {
      setIsDragging(false);
      onResizeEnd?.();
    }
  }, [isDragging, onResizeEnd]);

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      onDoubleClick?.();
    },
    [onDoubleClick]
  );

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);

      // Add cursor style to body during drag
      document.body.style.cursor = direction === 'horizontal' ? 'col-resize' : 'row-resize';
      document.body.style.userSelect = 'none';

      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp, direction]);

  // Prevent text selection during drag
  useEffect(() => {
    const preventSelect = (e: Event) => {
      if (isDragging) {
        e.preventDefault();
      }
    };

    document.addEventListener('selectstart', preventSelect);
    return () => {
      document.removeEventListener('selectstart', preventSelect);
    };
  }, [isDragging]);

  const isHorizontal = direction === 'horizontal';

  return (
    <div
      ref={resizerRef}
      className={cn(
        'resizer group relative flex-shrink-0',
        'transition-all duration-100 ease-out',
        isHorizontal
          ? 'w-2 cursor-col-resize'
          : 'h-2 cursor-row-resize',
        className
      )}
      onMouseDown={handleMouseDown}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onDoubleClick={handleDoubleClick}
    >
      {/* Hit area - larger than visual for easier grabbing */}
      <div
        className={cn(
          'absolute z-10',
          isHorizontal
            ? 'inset-y-0 -left-1 -right-1'
            : 'inset-x-0 -top-1 -bottom-1'
        )}
      />

      {/* Visual indicator line */}
      <div
        className={cn(
          'absolute transition-all duration-150 ease-out rounded-full',
          isHorizontal
            ? 'top-0 bottom-0 left-1/2 -translate-x-1/2 w-[2px]'
            : 'left-0 right-0 top-1/2 -translate-y-1/2 h-[2px]',
          isDragging
            ? 'bg-[var(--mviz-accent,#7aa2f7)]'
            : isHovered
              ? 'bg-[var(--mviz-accent,#7aa2f7)]/60'
              : 'bg-transparent'
        )}
      />

      {/* Grip handle (only visible on hover/drag) */}
      <div
        className={cn(
          'absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2',
          'flex transition-opacity duration-100',
          isHorizontal ? 'flex-col gap-0.5' : 'flex-row gap-0.5',
          (isHovered || isDragging) ? 'opacity-60' : 'opacity-0'
        )}
      >
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className={cn(
              'rounded-full bg-[var(--mviz-foreground,#a9b1d6)]',
              isHorizontal ? 'w-0.5 h-1.5' : 'w-1.5 h-0.5'
            )}
          />
        ))}
      </div>

      {/* Glow effect when dragging */}
      {isDragging && (
        <div
          className={cn(
            'absolute pointer-events-none',
            isHorizontal
              ? 'inset-y-0 left-1/2 -translate-x-1/2 w-1'
              : 'inset-x-0 top-1/2 -translate-y-1/2 h-1',
            'bg-[var(--mviz-accent,#7aa2f7)]',
            'blur-sm opacity-50'
          )}
        />
      )}
    </div>
  );
};

// ============================================================================
// Export
// ============================================================================

export default Resizer;
