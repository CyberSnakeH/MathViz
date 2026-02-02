/**
 * MainLayout Component
 *
 * Unique MathViz layout with:
 * - Floating panels with rounded corners (8px radius)
 * - Subtle gap between panels for visual separation
 * - Smooth resize animations (150ms)
 * - Glassmorphism effects on panels
 * - Tokyo Night color scheme throughout
 */

import React, { useCallback, useRef, useState } from 'react';
import { cn } from '../../utils/helpers';
import { useLayoutStore } from '../../stores/layoutStore';
import { ActivityBar } from './ActivityBar';
import { TabBar } from './TabBar';
import { Resizer } from './Resizer';

// ============================================================================
// Types
// ============================================================================

interface MainLayoutProps {
  sidebar?: React.ReactNode;
  editor?: React.ReactNode;
  bottomPanel?: React.ReactNode;
  rightPanel?: React.ReactNode;
  statusBar?: React.ReactNode;
}

// ============================================================================
// Breadcrumbs Component
// ============================================================================

const Breadcrumbs: React.FC<{ path?: string }> = ({ path }) => {
  if (!path) return null;

  const parts = path.split('/').filter(Boolean);
  const fileName = parts.pop();

  return (
    <div
      className={cn(
        'flex items-center gap-1 px-4 py-1.5',
        'bg-[var(--mviz-editor-background,#1a1b26)]',
        'border-b border-[var(--mviz-border,#1a1b26)]',
        'text-xs text-[var(--mviz-foreground,#a9b1d6)]'
      )}
    >
      {/* Home icon */}
      <svg
        className="w-3.5 h-3.5 opacity-50"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
      >
        <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
        <polyline points="9 22 9 12 15 12 15 22" />
      </svg>

      {/* Path parts */}
      {parts.map((part, index) => (
        <React.Fragment key={index}>
          <svg className="w-3 h-3 opacity-30" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8.59 16.59L13.17 12 8.59 7.41 10 6l6 6-6 6-1.41-1.41z" />
          </svg>
          <span className="opacity-50 hover:opacity-100 cursor-pointer transition-opacity">
            {part}
          </span>
        </React.Fragment>
      ))}

      {/* File name */}
      {fileName && (
        <>
          <svg className="w-3 h-3 opacity-30" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8.59 16.59L13.17 12 8.59 7.41 10 6l6 6-6 6-1.41-1.41z" />
          </svg>
          <span className="font-medium">{fileName}</span>
        </>
      )}
    </div>
  );
};

// ============================================================================
// FloatingPanel Wrapper
// ============================================================================

interface FloatingPanelProps {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
  rounded?: 'all' | 'left' | 'right' | 'top' | 'bottom' | 'none';
}

const FloatingPanel: React.FC<FloatingPanelProps> = ({
  children,
  className,
  style,
  rounded = 'all',
}) => {
  const roundedClasses = {
    all: 'rounded-lg',
    left: 'rounded-l-lg',
    right: 'rounded-r-lg',
    top: 'rounded-t-lg',
    bottom: 'rounded-b-lg',
    none: '',
  };

  return (
    <div
      className={cn(
        'overflow-hidden',
        roundedClasses[rounded],
        'shadow-lg shadow-black/20',
        className
      )}
      style={style}
    >
      {children}
    </div>
  );
};

// ============================================================================
// MainLayout Component
// ============================================================================

export const MainLayout: React.FC<MainLayoutProps> = ({
  sidebar,
  editor,
  bottomPanel,
  rightPanel,
  statusBar,
}) => {
  const {
    leftSidebar,
    rightPanel: rightPanelState,
    bottomPanel: bottomPanelState,
    activityBar,
    setLeftSidebarWidth,
    setRightPanelWidth,
    setBottomPanelHeight,
  } = useLayoutStore();

  // Refs for resize calculations
  const containerRef = useRef<HTMLDivElement>(null);

  // Track resize state for animations
  const [isResizing, setIsResizing] = useState(false);

  // Handle left sidebar resize
  const handleLeftSidebarResize = useCallback(
    (delta: number) => {
      setLeftSidebarWidth(leftSidebar.width + delta);
    },
    [leftSidebar.width, setLeftSidebarWidth]
  );

  // Handle right panel resize
  const handleRightPanelResize = useCallback(
    (delta: number) => {
      setRightPanelWidth(rightPanelState.width - delta);
    },
    [rightPanelState.width, setRightPanelWidth]
  );

  // Handle bottom panel resize
  const handleBottomPanelResize = useCallback(
    (delta: number) => {
      setBottomPanelHeight(bottomPanelState.height - delta);
    },
    [bottomPanelState.height, setBottomPanelHeight]
  );

  // Resize callbacks
  const handleResizeStart = useCallback(() => setIsResizing(true), []);
  const handleResizeEnd = useCallback(() => setIsResizing(false), []);

  return (
    <div
      ref={containerRef}
      className={cn(
        'flex flex-col h-screen w-screen overflow-hidden',
        'bg-[var(--mviz-background,#1a1b26)]',
        'text-[var(--mviz-foreground,#a9b1d6)]'
      )}
    >
      {/* Main content area with subtle padding for floating effect */}
      <div className="flex flex-1 overflow-hidden p-1 gap-1">
        {/* Activity Bar (far left) - No padding, flush with edge */}
        {activityBar.isVisible && (
          <FloatingPanel rounded="right" className="flex-shrink-0">
            <ActivityBar />
          </FloatingPanel>
        )}

        {/* Left Sidebar */}
        {leftSidebar.isVisible && (
          <>
            <FloatingPanel
              className={cn(
                'flex flex-col overflow-hidden',
                'bg-[var(--mviz-sidebar-background,#1f2335)]',
                !isResizing && 'transition-[width] duration-150 ease-out'
              )}
              style={{ width: leftSidebar.width }}
              rounded="all"
            >
              {sidebar}
            </FloatingPanel>
            <Resizer
              direction="horizontal"
              onResize={handleLeftSidebarResize}
              onResizeStart={handleResizeStart}
              onResizeEnd={handleResizeEnd}
            />
          </>
        )}

        {/* Center area (Editor + Bottom Panel) */}
        <div className="flex flex-1 flex-col overflow-hidden gap-1">
          {/* Editor Area */}
          <FloatingPanel
            className="flex flex-1 flex-col overflow-hidden bg-[var(--mviz-editor-background,#1a1b26)]"
            rounded="all"
          >
            {/* Tab Bar */}
            <TabBar />

            {/* Breadcrumbs */}
            <Breadcrumbs />

            {/* Editor Content */}
            <div className="flex-1 overflow-hidden">
              {editor}
            </div>
          </FloatingPanel>

          {/* Bottom Panel */}
          {bottomPanelState.isVisible && (
            <>
              <Resizer
                direction="vertical"
                onResize={handleBottomPanelResize}
                onResizeStart={handleResizeStart}
                onResizeEnd={handleResizeEnd}
              />
              <FloatingPanel
                className={cn(
                  'overflow-hidden',
                  'bg-[var(--mviz-panel-background,#1f2335)]',
                  !isResizing && 'transition-[height] duration-150 ease-out'
                )}
                style={{ height: bottomPanelState.height }}
                rounded="all"
              >
                {bottomPanel}
              </FloatingPanel>
            </>
          )}
        </div>

        {/* Right Panel (Preview) */}
        {rightPanelState.isVisible && (
          <>
            <Resizer
              direction="horizontal"
              onResize={handleRightPanelResize}
              onResizeStart={handleResizeStart}
              onResizeEnd={handleResizeEnd}
            />
            <FloatingPanel
              className={cn(
                'flex flex-col overflow-hidden',
                'bg-[var(--mviz-panel-background,#1f2335)]',
                !isResizing && 'transition-[width] duration-150 ease-out'
              )}
              style={{ width: rightPanelState.width }}
              rounded="all"
            >
              {rightPanel}
            </FloatingPanel>
          </>
        )}
      </div>

      {/* Status Bar - Full width at bottom */}
      {statusBar}
    </div>
  );
};

// ============================================================================
// Export
// ============================================================================

export default MainLayout;
