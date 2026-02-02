/**
 * ActivityBar Component
 *
 * Unique activity bar design with:
 * - Vertical text labels that appear on hover
 * - Icons with subtle hover animations
 * - Rounded pill-style active indicator
 * - 52px width for comfortable touch targets
 */

import React, { useState, useCallback } from 'react';
import { cn } from '../../utils/helpers';
import { useLayoutStore } from '../../stores/layoutStore';
import type { SidebarPanel } from '../../types';

// ============================================================================
// Types
// ============================================================================

type ActivitySidebarPanel = Exclude<SidebarPanel, 'extensions'>;

interface ActivityItem {
  id: ActivitySidebarPanel | 'settings';
  label: string;
  icon: React.ReactNode;
  badge?: number;
}

// ============================================================================
// Icons - Custom SVG icons for unique look
// ============================================================================

const ExplorerIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.5" stroke="currentColor" className="w-6 h-6">
    <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 9.776c.112-.017.227-.026.344-.026h15.812c.117 0 .232.009.344.026m-16.5 0a2.25 2.25 0 00-1.883 2.542l.857 6a2.25 2.25 0 002.227 1.932H19.05a2.25 2.25 0 002.227-1.932l.857-6a2.25 2.25 0 00-1.883-2.542m-16.5 0V6A2.25 2.25 0 016 3.75h3.879a1.5 1.5 0 011.06.44l2.122 2.12a1.5 1.5 0 001.06.44H18A2.25 2.25 0 0120.25 9v.776" />
  </svg>
);

const SearchIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.5" stroke="currentColor" className="w-6 h-6">
    <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
  </svg>
);

const DebugIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.5" stroke="currentColor" className="w-6 h-6">
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 12V21M8 8L4 4M16 8L20 4M4 12h4M16 12h4" />
    <circle cx="12" cy="8" r="4" />
  </svg>
);

const SettingsIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.5" stroke="currentColor" className="w-6 h-6">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 .255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.932 6.932 0 010-.255c.007-.378-.138-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
);

// ============================================================================
// MathViz Logo Component
// ============================================================================

const MathVizLogo: React.FC = () => (
  <div className="flex items-center justify-center w-10 h-10">
    <svg viewBox="0 0 32 32" className="w-6 h-6" fill="none">
      <path
        d="M6 8h18M6 8l9 8-9 8h18"
        stroke="var(--mviz-accent, #7aa2f7)"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  </div>
);

// ============================================================================
// Activity Items Configuration
// ============================================================================

const activityItems: ActivityItem[] = [
  { id: 'explorer', label: 'Explorer', icon: <ExplorerIcon /> },
  { id: 'search', label: 'Search', icon: <SearchIcon /> },
  { id: 'debug', label: 'Run & Debug', icon: <DebugIcon /> },
];

// ============================================================================
// ActivityBarItem Component
// ============================================================================

interface ActivityBarItemProps {
  item: ActivityItem;
  isActive: boolean;
  onClick: () => void;
}

const ActivityBarItem: React.FC<ActivityBarItemProps> = ({ item, isActive, onClick }) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <button
      className={cn(
        'activity-bar-item group relative',
        'w-[52px] h-12',
        'flex items-center justify-center',
        'transition-all duration-150 ease-out',
        isActive
          ? 'text-[var(--mviz-activity-bar-foreground,#a9b1d6)]'
          : 'text-[var(--mviz-activity-bar-inactive-foreground,#565f89)] hover:text-[var(--mviz-activity-bar-foreground,#a9b1d6)]'
      )}
      onClick={onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      aria-label={item.label}
    >
      {/* Icon with hover animation */}
      <div
        className={cn(
          'relative z-10 transition-transform duration-150 ease-out',
          isHovered && 'scale-110'
        )}
      >
        {item.icon}
      </div>

      {/* Active indicator - Rounded pill on left */}
      {isActive && (
        <div
          className={cn(
            'absolute left-0 top-1/2 -translate-y-1/2',
            'w-[3px] h-6',
            'bg-[var(--mviz-accent,#7aa2f7)]',
            'rounded-r-full',
            'transition-all duration-150 ease-out'
          )}
        />
      )}

      {/* Badge */}
      {item.badge !== undefined && item.badge > 0 && (
        <div
          className={cn(
            'absolute top-2 right-2',
            'min-w-[16px] h-4 px-1',
            'flex items-center justify-center',
            'text-[10px] font-semibold',
            'bg-[var(--mviz-badge-background,#7aa2f7)]',
            'text-[var(--mviz-badge-foreground,#1a1b26)]',
            'rounded-full'
          )}
        >
          {item.badge > 99 ? '99+' : item.badge}
        </div>
      )}

      {/* Hover tooltip with vertical label */}
      <div
        className={cn(
          'absolute left-full ml-2 px-3 py-1.5',
          'bg-[var(--mviz-widget-background,#1f2335)]',
          'border border-[var(--mviz-widget-border,#3b4261)]',
          'rounded-md shadow-lg',
          'text-xs font-medium whitespace-nowrap',
          'text-[var(--mviz-foreground,#a9b1d6)]',
          'pointer-events-none',
          'z-[1000]',
          'transition-all duration-100 ease-out',
          isHovered ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-1'
        )}
      >
        {item.label}
      </div>
    </button>
  );
};

// ============================================================================
// ActivityBar Component
// ============================================================================

export const ActivityBar: React.FC = () => {
  const { leftSidebar, setLeftSidebarPanel } = useLayoutStore();

  const handleItemClick = useCallback(
    (panel: ActivitySidebarPanel) => {
      setLeftSidebarPanel(panel);
    },
    [setLeftSidebarPanel]
  );

  return (
    <div
      className={cn(
        'flex flex-col items-center',
        'w-[52px] min-w-[52px]',
        'bg-[var(--mviz-activity-bar-background,#1a1b26)]',
        'border-r border-[var(--mviz-border,#1a1b26)]'
      )}
    >
      {/* Logo at top */}
      <div className="py-3 border-b border-[var(--mviz-border,#1a1b26)] w-full flex justify-center">
        <MathVizLogo />
      </div>

      {/* Main navigation items */}
      <nav className="flex flex-col items-center py-2 gap-1">
        {activityItems.map((item) => (
          <ActivityBarItem
            key={item.id}
            item={item}
            isActive={leftSidebar.isVisible && leftSidebar.activePanel === item.id}
            onClick={() => handleItemClick(item.id as ActivitySidebarPanel)}
          />
        ))}
      </nav>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Bottom section - Settings */}
      <div className="flex flex-col items-center py-2">
        <ActivityBarItem
          item={{ id: 'settings', label: 'Settings', icon: <SettingsIcon /> }}
          isActive={false}
          onClick={() => {
            // TODO: Open settings panel
            console.log('Settings clicked');
          }}
        />
      </div>
    </div>
  );
};

// ============================================================================
// Export
// ============================================================================

export default ActivityBar;
