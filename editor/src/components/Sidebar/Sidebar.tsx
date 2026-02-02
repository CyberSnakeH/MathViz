/**
 * Sidebar Component
 *
 * Modern sidebar container with:
 * - Tokyo Night themed header
 * - Smooth panel transitions
 * - Collapsible section headers
 * - Action buttons in header
 */

import React, { memo } from 'react';
import { cn } from '../../utils/helpers';
import { useLayoutStore } from '../../stores/layoutStore';
import type { SidebarPanel } from '../../types';
import Explorer from './Explorer';
import Search from './Search';
import GitPanel from './GitPanel';

// ============================================================================
// Types
// ============================================================================

interface SidebarProps {
  className?: string;
}

// ============================================================================
// Panel Configuration
// ============================================================================

const panelConfig: Record<SidebarPanel, { title: string; icon: React.ReactNode }> = {
  explorer: {
    title: 'Explorer',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 9.776c.112-.017.227-.026.344-.026h15.812c.117 0 .232.009.344.026m-16.5 0a2.25 2.25 0 00-1.883 2.542l.857 6a2.25 2.25 0 002.227 1.932H19.05a2.25 2.25 0 002.227-1.932l.857-6a2.25 2.25 0 00-1.883-2.542m-16.5 0V6A2.25 2.25 0 016 3.75h3.879a1.5 1.5 0 011.06.44l2.122 2.12a1.5 1.5 0 001.06.44H18A2.25 2.25 0 0120.25 9v.776" />
      </svg>
    ),
  },
  search: {
    title: 'Search',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4">
        <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
      </svg>
    ),
  },
  git: {
    title: 'Source Control',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4">
        <circle cx="12" cy="6" r="2" />
        <circle cx="12" cy="18" r="2" />
        <circle cx="6" cy="12" r="2" />
        <path strokeLinecap="round" d="M12 8v8M8 12h4" />
      </svg>
    ),
  },
  debug: {
    title: 'Run & Debug',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 12V21M8 8L4 4M16 8L20 4M4 12h4M16 12h4" />
        <circle cx="12" cy="8" r="4" />
      </svg>
    ),
  },
};

// ============================================================================
// Debug Panel Placeholder
// ============================================================================

const DebugPanel: React.FC = memo(() => (
  <div className="flex flex-col items-center justify-center h-full p-6 text-center">
    <svg
      className="w-16 h-16 mb-4 opacity-20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 12V21M8 8L4 4M16 8L20 4M4 12h4M16 12h4" />
      <circle cx="12" cy="8" r="4" />
    </svg>
    <p className="text-sm opacity-70 mb-2">Run and Debug</p>
    <p className="text-xs opacity-50">
      Open a file with a run configuration to start debugging
    </p>
    <button
      className={cn(
        'mt-4 px-4 py-2 text-sm rounded-lg',
        'bg-[var(--mviz-button-background,#7aa2f7)]',
        'text-[var(--mviz-button-foreground,#1a1b26)]',
        'hover:bg-[var(--mviz-button-hover-background,#89b4fa)]',
        'transition-colors duration-100'
      )}
    >
      Create Configuration
    </button>
  </div>
));

DebugPanel.displayName = 'DebugPanel';

// ============================================================================
// Sidebar Component
// ============================================================================

const Sidebar: React.FC<SidebarProps> = ({ className = '' }) => {
  const activePanel = useLayoutStore((state) => state.leftSidebar.activePanel);

  const renderPanel = () => {
    switch (activePanel) {
      case 'explorer':
        return <Explorer />;
      case 'search':
        return <Search />;
      case 'git':
        return <GitPanel />;
      case 'debug':
        return <DebugPanel />;
      default:
        return <Explorer />;
    }
  };

  const config = panelConfig[activePanel];

  return (
    <div
      className={cn(
        'flex flex-col h-full',
        'bg-[var(--mviz-sidebar-background,#1f2335)]',
        'text-[var(--mviz-sidebar-foreground,#a9b1d6)]',
        className
      )}
    >
      {/* Panel header */}
      <div
        className={cn(
          'flex items-center justify-between px-4 py-2.5',
          'border-b border-[var(--mviz-border,#1a1b26)]'
        )}
      >
        <div className="flex items-center gap-2">
          <span className="opacity-60">{config.icon}</span>
          <span className="text-xs font-semibold uppercase tracking-wider">
            {config.title}
          </span>
        </div>

        {/* Header actions - More options */}
        <button
          className={cn(
            'p-1 rounded',
            'opacity-50 hover:opacity-100',
            'hover:bg-[var(--mviz-list-hover-background,#292e42)]',
            'transition-all duration-100'
          )}
          title="More Actions"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
            <circle cx="12" cy="5" r="1.5" />
            <circle cx="12" cy="12" r="1.5" />
            <circle cx="12" cy="19" r="1.5" />
          </svg>
        </button>
      </div>

      {/* Panel content */}
      <div className="flex-1 overflow-hidden animate-fade-in">
        {renderPanel()}
      </div>
    </div>
  );
};

// ============================================================================
// Export
// ============================================================================

export default Sidebar;
