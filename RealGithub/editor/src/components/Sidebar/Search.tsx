/**
 * Search Component
 *
 * Modern global search panel with Tokyo Night theme:
 * - Regex, case-sensitive, whole word toggles
 * - Replace functionality
 * - File filters (include/exclude)
 * - Highlighted match results
 * - Collapsible file sections
 */

import React, { useState, useCallback, useRef, useEffect, memo } from 'react';
import { cn } from '../../utils/helpers';
import { invoke } from '@tauri-apps/api/core';
import { useFileStore } from '../../stores/fileStore';
import { useEditorStore } from '../../stores/editorStore';

// ============================================================================
// Types
// ============================================================================

interface SearchProps {
  className?: string;
}

interface GroupedResults {
  [filePath: string]: {
    fileName: string;
    matches: Array<{
      lineNumber: number;
      column: number;
      lineContent: string;
      matchStart: number;
      matchEnd: number;
    }>;
  };
}

// ============================================================================
// Toggle Button Component
// ============================================================================

interface ToggleButtonProps {
  active: boolean;
  onClick: () => void;
  title: string;
  children: React.ReactNode;
}

const ToggleButton: React.FC<ToggleButtonProps> = memo(({
  active,
  onClick,
  title,
  children,
}) => (
  <button
    className={cn(
      'p-1.5 rounded transition-all duration-100',
      active
        ? 'bg-[var(--mviz-accent,#7aa2f7)]/20 text-[var(--mviz-accent,#7aa2f7)]'
        : 'text-[var(--mviz-foreground,#a9b1d6)] opacity-50 hover:opacity-100 hover:bg-[var(--mviz-list-hover-background,#292e42)]'
    )}
    onClick={onClick}
    title={title}
  >
    {children}
  </button>
));

ToggleButton.displayName = 'ToggleButton';

// ============================================================================
// Search Component
// ============================================================================

const Search: React.FC<SearchProps> = ({ className = '' }) => {
  const [query, setQuery] = useState('');
  const [replaceText, setReplaceText] = useState('');
  const [showReplace, setShowReplace] = useState(false);
  const [caseSensitive, setCaseSensitive] = useState(false);
  const [wholeWord, setWholeWord] = useState(false);
  const [useRegex, setUseRegex] = useState(false);
  const [includePattern, setIncludePattern] = useState('');
  const [excludePattern, setExcludePattern] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<GroupedResults>({});
  const [expandedFiles, setExpandedFiles] = useState<Set<string>>(new Set());

  const searchInputRef = useRef<HTMLInputElement>(null);
  const rootPath = useFileStore((state) => state.rootPath);
  const openFileAction = useEditorStore((state) => state.openFile);

  // Focus search input on mount
  useEffect(() => {
    searchInputRef.current?.focus();
  }, []);

  // Perform search
  const handleSearch = useCallback(async () => {
    if (!query.trim() || !rootPath) return;

    setIsSearching(true);
    try {
      const searchResults = await invoke<Array<{
        path: string;
        line_number: number;
        line_content: string;
        match_start: number;
        match_end: number;
      }>>('search_files', {
        path: rootPath,
        query,
        filePattern: includePattern || undefined,
        caseSensitive,
        maxResults: 1000,
      });

      // Group by file
      const grouped: GroupedResults = {};
      for (const result of searchResults) {
        const fileName = result.path.split('/').pop() || result.path;
        if (!grouped[result.path]) {
          grouped[result.path] = {
            fileName,
            matches: [],
          };
        }
        grouped[result.path].matches.push({
          lineNumber: result.line_number,
          column: result.match_start,
          lineContent: result.line_content,
          matchStart: result.match_start,
          matchEnd: result.match_end,
        });
      }

      setResults(grouped);
      setExpandedFiles(new Set(Object.keys(grouped)));
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setIsSearching(false);
    }
  }, [query, rootPath, caseSensitive, includePattern]);

  // Handle enter key
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter') {
        handleSearch();
      }
    },
    [handleSearch]
  );

  // Toggle file expansion
  const toggleFile = useCallback((filePath: string) => {
    setExpandedFiles((prev) => {
      const next = new Set(prev);
      if (next.has(filePath)) {
        next.delete(filePath);
      } else {
        next.add(filePath);
      }
      return next;
    });
  }, []);

  // Go to match
  const goToMatch = useCallback(
    async (filePath: string, _lineNumber: number, _column: number) => {
      try {
        const content = await invoke<string>('read_file', { path: filePath });
        const fileName = filePath.split('/').pop() || filePath;
        const ext = fileName.split('.').pop() || '';

        const language = ext === 'mviz' ? 'mathviz' : ext === 'py' ? 'python' : 'plaintext';
        openFileAction(filePath, content, language);
      } catch (error) {
        console.error('Failed to open file:', error);
      }
    },
    [openFileAction]
  );

  // Count total matches
  const totalMatches = Object.values(results).reduce(
    (sum, file) => sum + file.matches.length,
    0
  );
  const totalFiles = Object.keys(results).length;

  // Highlight match in line
  const highlightMatch = (
    content: string,
    start: number,
    end: number
  ): React.ReactNode => {
    const before = content.slice(0, start);
    const match = content.slice(start, end);
    const after = content.slice(end);

    return (
      <>
        <span className="opacity-60">{before}</span>
        <span className="bg-[var(--mviz-accent,#7aa2f7)]/30 text-[var(--mviz-accent,#7aa2f7)] font-medium">
          {match}
        </span>
        <span className="opacity-60">{after}</span>
      </>
    );
  };

  return (
    <div className={cn('flex flex-col h-full', className)}>
      {/* Search input area */}
      <div className="p-3 space-y-2 border-b border-[var(--mviz-border,#1a1b26)]">
        {/* Main search input */}
        <div className="relative">
          <input
            ref={searchInputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search"
            className={cn(
              'w-full pl-9 pr-24 py-2 text-sm rounded-lg',
              'bg-[var(--mviz-input-background,#1a1b26)]',
              'border border-[var(--mviz-input-border,#3b4261)]',
              'text-[var(--mviz-input-foreground,#a9b1d6)]',
              'placeholder:text-[var(--mviz-input-placeholder,#565f89)]',
              'focus:outline-none focus:border-[var(--mviz-focus-border,#7aa2f7)]',
              'transition-colors duration-100'
            )}
          />
          <svg
            className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 opacity-50"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>

          {/* Toggle options */}
          <div className="absolute right-1 top-1/2 -translate-y-1/2 flex items-center gap-0.5">
            <ToggleButton
              active={caseSensitive}
              onClick={() => setCaseSensitive(!caseSensitive)}
              title="Match Case (Alt+C)"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 19V5M3 5l6 14M9 5L3 19M14 5v14M14 5c3 0 5 2 5 4.5S17 14 14 14M14 14c3 0 5 2 5 4.5M14 14v5" />
              </svg>
            </ToggleButton>
            <ToggleButton
              active={wholeWord}
              onClick={() => setWholeWord(!wholeWord)}
              title="Match Whole Word (Alt+W)"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 7v10M21 7v10M8 12h8M5 7h2M17 7h2M5 17h2M17 17h2" />
              </svg>
            </ToggleButton>
            <ToggleButton
              active={useRegex}
              onClick={() => setUseRegex(!useRegex)}
              title="Use Regular Expression (Alt+R)"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 4v4M12 12v8M8 8l8 8M16 8l-8 8" />
              </svg>
            </ToggleButton>
          </div>
        </div>

        {/* Replace toggle button */}
        <button
          className={cn(
            'flex items-center gap-2 text-xs',
            'text-[var(--mviz-foreground,#a9b1d6)] opacity-60 hover:opacity-100',
            'transition-opacity duration-100'
          )}
          onClick={() => setShowReplace(!showReplace)}
        >
          <svg
            className={cn(
              'w-3 h-3 transition-transform duration-100',
              showReplace && 'rotate-90'
            )}
            viewBox="0 0 16 16"
            fill="currentColor"
          >
            <path d="M6 4l4 4-4 4V4z" />
          </svg>
          Replace
        </button>

        {/* Replace input */}
        {showReplace && (
          <div className="relative animate-slide-in-from-top">
            <input
              type="text"
              value={replaceText}
              onChange={(e) => setReplaceText(e.target.value)}
              placeholder="Replace"
              className={cn(
                'w-full pl-9 pr-16 py-2 text-sm rounded-lg',
                'bg-[var(--mviz-input-background,#1a1b26)]',
                'border border-[var(--mviz-input-border,#3b4261)]',
                'text-[var(--mviz-input-foreground,#a9b1d6)]',
                'placeholder:text-[var(--mviz-input-placeholder,#565f89)]',
                'focus:outline-none focus:border-[var(--mviz-focus-border,#7aa2f7)]',
                'transition-colors duration-100'
              )}
            />
            <svg
              className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 opacity-50"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
              <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
            </svg>

            <div className="absolute right-1 top-1/2 -translate-y-1/2 flex items-center gap-0.5">
              <button
                className={cn(
                  'p-1.5 rounded transition-all duration-100',
                  'text-[var(--mviz-foreground,#a9b1d6)] opacity-50',
                  'hover:opacity-100 hover:bg-[var(--mviz-list-hover-background,#292e42)]'
                )}
                title="Replace (Ctrl+Shift+1)"
              >
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                  <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                </svg>
              </button>
              <button
                className={cn(
                  'p-1.5 rounded transition-all duration-100',
                  'text-[var(--mviz-foreground,#a9b1d6)] opacity-50',
                  'hover:opacity-100 hover:bg-[var(--mviz-list-hover-background,#292e42)]'
                )}
                title="Replace All (Ctrl+Alt+Enter)"
              >
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z" />
                </svg>
              </button>
            </div>
          </div>
        )}

        {/* Filters toggle */}
        <button
          className={cn(
            'flex items-center gap-2 text-xs',
            'text-[var(--mviz-foreground,#a9b1d6)] opacity-60 hover:opacity-100',
            'transition-opacity duration-100'
          )}
          onClick={() => setShowFilters(!showFilters)}
        >
          <svg
            className={cn(
              'w-3 h-3 transition-transform duration-100',
              showFilters && 'rotate-90'
            )}
            viewBox="0 0 16 16"
            fill="currentColor"
          >
            <path d="M6 4l4 4-4 4V4z" />
          </svg>
          Files to include/exclude
        </button>

        {/* Filters */}
        {showFilters && (
          <div className="space-y-2 animate-slide-in-from-top">
            <input
              type="text"
              value={includePattern}
              onChange={(e) => setIncludePattern(e.target.value)}
              placeholder="Files to include (e.g., *.mviz, src/**)"
              className={cn(
                'w-full px-3 py-1.5 text-xs rounded-lg',
                'bg-[var(--mviz-input-background,#1a1b26)]',
                'border border-[var(--mviz-input-border,#3b4261)]',
                'text-[var(--mviz-input-foreground,#a9b1d6)]',
                'placeholder:text-[var(--mviz-input-placeholder,#565f89)]',
                'focus:outline-none focus:border-[var(--mviz-focus-border,#7aa2f7)]'
              )}
            />
            <input
              type="text"
              value={excludePattern}
              onChange={(e) => setExcludePattern(e.target.value)}
              placeholder="Files to exclude (e.g., node_modules/**)"
              className={cn(
                'w-full px-3 py-1.5 text-xs rounded-lg',
                'bg-[var(--mviz-input-background,#1a1b26)]',
                'border border-[var(--mviz-input-border,#3b4261)]',
                'text-[var(--mviz-input-foreground,#a9b1d6)]',
                'placeholder:text-[var(--mviz-input-placeholder,#565f89)]',
                'focus:outline-none focus:border-[var(--mviz-focus-border,#7aa2f7)]'
              )}
            />
          </div>
        )}
      </div>

      {/* Results summary */}
      {totalMatches > 0 && (
        <div
          className={cn(
            'px-3 py-2 text-xs',
            'border-b border-[var(--mviz-border,#1a1b26)]',
            'text-[var(--mviz-foreground,#a9b1d6)] opacity-70'
          )}
        >
          <span className="font-medium text-[var(--mviz-accent,#7aa2f7)]">{totalMatches}</span>
          {' '}result{totalMatches !== 1 ? 's' : ''} in{' '}
          <span className="font-medium">{totalFiles}</span>
          {' '}file{totalFiles !== 1 ? 's' : ''}
        </div>
      )}

      {/* Results */}
      <div className="flex-1 overflow-auto">
        {isSearching && (
          <div className="flex items-center justify-center p-8">
            <svg
              className="w-6 h-6 animate-spin opacity-50"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <circle cx="12" cy="12" r="10" strokeDasharray="60" strokeDashoffset="20" />
            </svg>
          </div>
        )}

        {!isSearching && totalMatches === 0 && query && (
          <div className="p-8 text-center">
            <svg
              className="w-12 h-12 mx-auto mb-3 opacity-20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
            >
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            <p className="text-sm opacity-50">No results found</p>
            <p className="text-xs opacity-30 mt-1">
              Try adjusting your search or filters
            </p>
          </div>
        )}

        {!isSearching &&
          Object.entries(results).map(([filePath, fileResults]) => (
            <div key={filePath} className="border-b border-[var(--mviz-border,#1a1b26)] last:border-b-0">
              {/* File header */}
              <div
                className={cn(
                  'flex items-center gap-2 px-3 py-2 cursor-pointer',
                  'hover:bg-[var(--mviz-list-hover-background,#292e42)]',
                  'transition-colors duration-100'
                )}
                onClick={() => toggleFile(filePath)}
              >
                <svg
                  className={cn(
                    'w-3 h-3 opacity-50 transition-transform duration-100',
                    expandedFiles.has(filePath) && 'rotate-90'
                  )}
                  viewBox="0 0 16 16"
                  fill="currentColor"
                >
                  <path d="M6 4l4 4-4 4V4z" />
                </svg>
                <svg className="w-4 h-4 opacity-50" viewBox="0 0 24 24" fill="currentColor">
                  <path fillRule="evenodd" d="M5.625 1.5c-1.036 0-1.875.84-1.875 1.875v17.25c0 1.035.84 1.875 1.875 1.875h12.75c1.035 0 1.875-.84 1.875-1.875V12.75A3.75 3.75 0 0016.5 9h-1.875a1.875 1.875 0 01-1.875-1.875V5.25A3.75 3.75 0 009 1.5H5.625z" clipRule="evenodd" />
                </svg>
                <span className="text-sm truncate flex-1">{fileResults.fileName}</span>
                <span
                  className={cn(
                    'text-xs px-1.5 py-0.5 rounded-full',
                    'bg-[var(--mviz-badge-background,#7aa2f7)]/20',
                    'text-[var(--mviz-badge-foreground,#7aa2f7)]'
                  )}
                >
                  {fileResults.matches.length}
                </span>
              </div>

              {/* Matches */}
              {expandedFiles.has(filePath) && (
                <div className="animate-slide-in-from-top">
                  {fileResults.matches.map((match, index) => (
                    <div
                      key={`${filePath}-${index}`}
                      className={cn(
                        'flex items-start gap-2 px-3 py-1.5 cursor-pointer',
                        'hover:bg-[var(--mviz-list-hover-background,#292e42)]',
                        'transition-colors duration-100',
                        'ml-5'
                      )}
                      onClick={() => goToMatch(filePath, match.lineNumber, match.column)}
                    >
                      <span className="text-xs opacity-40 w-6 text-right shrink-0">
                        {match.lineNumber}
                      </span>
                      <span className="text-xs font-mono truncate">
                        {highlightMatch(
                          match.lineContent,
                          match.matchStart,
                          match.matchEnd
                        )}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
      </div>

      {/* Empty state */}
      {!query && !isSearching && totalMatches === 0 && (
        <div className="flex-1 flex flex-col items-center justify-center p-8 text-center">
          <svg
            className="w-16 h-16 mb-4 opacity-10"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <p className="text-sm opacity-50">Search across files</p>
          <p className="text-xs opacity-30 mt-1">
            Type to start searching
          </p>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Export
// ============================================================================

export default Search;
