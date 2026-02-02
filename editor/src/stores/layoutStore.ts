/**
 * Layout State Management
 *
 * Zustand store for managing UI layout state including panels,
 * sidebars, and window dimensions.
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import type { SidebarPanel, BottomPanel } from '../types';

// ============================================================================
// Types
// ============================================================================

interface LayoutState {
  // Left sidebar
  leftSidebar: {
    isVisible: boolean;
    width: number;
    minWidth: number;
    maxWidth: number;
    activePanel: SidebarPanel;
  };

  // Right panel (preview)
  rightPanel: {
    isVisible: boolean;
    width: number;
    minWidth: number;
    maxWidth: number;
  };

  // Bottom panel (terminal, problems, output)
  bottomPanel: {
    isVisible: boolean;
    height: number;
    minHeight: number;
    maxHeight: number;
    activePanel: BottomPanel;
  };

  // Activity bar
  activityBar: {
    isVisible: boolean;
    position: 'left' | 'right';
  };

  // Menu bar
  menuBar: {
    isVisible: boolean;
  };

  // Zen mode
  isZenMode: boolean;

  // Full screen
  isFullscreen: boolean;
}

interface LayoutActions {
  // Left sidebar
  toggleLeftSidebar: () => void;
  setLeftSidebarVisible: (visible: boolean) => void;
  setLeftSidebarWidth: (width: number) => void;
  setLeftSidebarPanel: (panel: SidebarPanel) => void;

  // Right panel
  toggleRightPanel: () => void;
  setRightPanelVisible: (visible: boolean) => void;
  setRightPanelWidth: (width: number) => void;

  // Bottom panel
  toggleBottomPanel: () => void;
  setBottomPanelVisible: (visible: boolean) => void;
  setBottomPanelHeight: (height: number) => void;
  setBottomPanelActive: (panel: BottomPanel) => void;

  // Activity bar
  toggleActivityBar: () => void;
  setActivityBarPosition: (position: 'left' | 'right') => void;

  // Menu bar
  toggleMenuBar: () => void;

  // Zen mode
  toggleZenMode: () => void;
  exitZenMode: () => void;

  // Full screen
  setFullscreen: (fullscreen: boolean) => void;

  // Reset
  resetLayout: () => void;
}

type LayoutStore = LayoutState & LayoutActions;

// ============================================================================
// Default Values
// ============================================================================

const defaultState: LayoutState = {
  leftSidebar: {
    isVisible: true,
    width: 260,
    minWidth: 170,
    maxWidth: 500,
    activePanel: 'explorer',
  },
  rightPanel: {
    isVisible: false,
    width: 400,
    minWidth: 200,
    maxWidth: 800,
  },
  bottomPanel: {
    isVisible: true,
    height: 250,
    minHeight: 100,
    maxHeight: 600,
    activePanel: 'terminal',
  },
  activityBar: {
    isVisible: true,
    position: 'left',
  },
  menuBar: {
    isVisible: true,
  },
  isZenMode: false,
  isFullscreen: false,
};

// ============================================================================
// Store Implementation
// ============================================================================

export const useLayoutStore = create<LayoutStore>()(
  devtools(
    persist(
      immer((set) => ({
        ...defaultState,

        // ====================================================================
        // Left Sidebar
        // ====================================================================

        toggleLeftSidebar: () => {
          set((state) => {
            state.leftSidebar.isVisible = !state.leftSidebar.isVisible;
          });
        },

        setLeftSidebarVisible: (visible: boolean) => {
          set((state) => {
            state.leftSidebar.isVisible = visible;
          });
        },

        setLeftSidebarWidth: (width: number) => {
          set((state) => {
            const { minWidth, maxWidth } = state.leftSidebar;
            state.leftSidebar.width = Math.max(minWidth, Math.min(maxWidth, width));
          });
        },

        setLeftSidebarPanel: (panel: SidebarPanel) => {
          set((state) => {
            // If clicking the same panel, toggle visibility
            if (state.leftSidebar.activePanel === panel && state.leftSidebar.isVisible) {
              state.leftSidebar.isVisible = false;
            } else {
              state.leftSidebar.activePanel = panel;
              state.leftSidebar.isVisible = true;
            }
          });
        },

        // ====================================================================
        // Right Panel
        // ====================================================================

        toggleRightPanel: () => {
          set((state) => {
            state.rightPanel.isVisible = !state.rightPanel.isVisible;
          });
        },

        setRightPanelVisible: (visible: boolean) => {
          set((state) => {
            state.rightPanel.isVisible = visible;
          });
        },

        setRightPanelWidth: (width: number) => {
          set((state) => {
            const { minWidth, maxWidth } = state.rightPanel;
            state.rightPanel.width = Math.max(minWidth, Math.min(maxWidth, width));
          });
        },

        // ====================================================================
        // Bottom Panel
        // ====================================================================

        toggleBottomPanel: () => {
          set((state) => {
            state.bottomPanel.isVisible = !state.bottomPanel.isVisible;
          });
        },

        setBottomPanelVisible: (visible: boolean) => {
          set((state) => {
            state.bottomPanel.isVisible = visible;
          });
        },

        setBottomPanelHeight: (height: number) => {
          set((state) => {
            const { minHeight, maxHeight } = state.bottomPanel;
            state.bottomPanel.height = Math.max(minHeight, Math.min(maxHeight, height));
          });
        },

        setBottomPanelActive: (panel: BottomPanel) => {
          set((state) => {
            // If clicking the same panel, toggle visibility
            if (state.bottomPanel.activePanel === panel && state.bottomPanel.isVisible) {
              state.bottomPanel.isVisible = false;
            } else {
              state.bottomPanel.activePanel = panel;
              state.bottomPanel.isVisible = true;
            }
          });
        },

        // ====================================================================
        // Activity Bar
        // ====================================================================

        toggleActivityBar: () => {
          set((state) => {
            state.activityBar.isVisible = !state.activityBar.isVisible;
          });
        },

        setActivityBarPosition: (position: 'left' | 'right') => {
          set((state) => {
            state.activityBar.position = position;
          });
        },

        // ====================================================================
        // Menu Bar
        // ====================================================================

        toggleMenuBar: () => {
          set((state) => {
            state.menuBar.isVisible = !state.menuBar.isVisible;
          });
        },

        // ====================================================================
        // Zen Mode
        // ====================================================================

        toggleZenMode: () => {
          set((state) => {
            if (state.isZenMode) {
              // Exit zen mode - restore panels
              state.isZenMode = false;
              state.leftSidebar.isVisible = true;
              state.bottomPanel.isVisible = true;
              state.activityBar.isVisible = true;
              state.menuBar.isVisible = true;
            } else {
              // Enter zen mode - hide panels
              state.isZenMode = true;
              state.leftSidebar.isVisible = false;
              state.rightPanel.isVisible = false;
              state.bottomPanel.isVisible = false;
              state.activityBar.isVisible = false;
            }
          });
        },

        exitZenMode: () => {
          set((state) => {
            if (state.isZenMode) {
              state.isZenMode = false;
              state.leftSidebar.isVisible = true;
              state.bottomPanel.isVisible = true;
              state.activityBar.isVisible = true;
              state.menuBar.isVisible = true;
            }
          });
        },

        // ====================================================================
        // Full Screen
        // ====================================================================

        setFullscreen: (fullscreen: boolean) => {
          set((state) => {
            state.isFullscreen = fullscreen;
          });
        },

        // ====================================================================
        // Reset
        // ====================================================================

        resetLayout: () => {
          set(() => ({ ...defaultState }));
        },
      })),
      {
        name: 'mathviz-layout-store',
        partialize: (state) => ({
          leftSidebar: state.leftSidebar,
          rightPanel: state.rightPanel,
          bottomPanel: state.bottomPanel,
          activityBar: state.activityBar,
          menuBar: state.menuBar,
        }),
      }
    ),
    { name: 'mathviz-layout-store' }
  )
);

// ============================================================================
// Selectors
// ============================================================================

export const selectLeftSidebarWidth = (state: LayoutStore): number => {
  return state.leftSidebar.isVisible ? state.leftSidebar.width : 0;
};

export const selectBottomPanelHeight = (state: LayoutStore): number => {
  return state.bottomPanel.isVisible ? state.bottomPanel.height : 0;
};

export const selectRightPanelWidth = (state: LayoutStore): number => {
  return state.rightPanel.isVisible ? state.rightPanel.width : 0;
};
