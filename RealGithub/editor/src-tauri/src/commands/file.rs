//! File System Operations
//!
//! This module provides Tauri commands for file system operations including:
//! - Reading and writing files
//! - Directory operations (create, read, delete)
//! - File watching for live updates
//! - File search functionality

use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use tauri::{AppHandle, Emitter, Manager};
use walkdir::WalkDir;

/// File information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub name: String,
    pub path: String,
    pub is_directory: bool,
    pub is_file: bool,
    pub size: u64,
    pub modified: Option<u64>,
    pub created: Option<u64>,
    pub extension: Option<String>,
    pub is_hidden: bool,
}

/// Directory entry for tree view
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryEntry {
    pub name: String,
    pub path: String,
    pub is_directory: bool,
    pub is_expanded: bool,
    pub children: Option<Vec<DirectoryEntry>>,
    pub extension: Option<String>,
    pub is_hidden: bool,
}

/// Search result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub path: String,
    pub line_number: usize,
    pub line_content: String,
    pub match_start: usize,
    pub match_end: usize,
}

/// State for managing file watchers
pub struct WatcherState {
    watchers: HashMap<String, RecommendedWatcher>,
}

impl Default for WatcherState {
    fn default() -> Self {
        Self {
            watchers: HashMap::new(),
        }
    }
}

/// Convert system time to Unix timestamp
fn system_time_to_unix(time: SystemTime) -> Option<u64> {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
}

/// Check if a file is hidden (starts with dot on Unix)
fn is_hidden(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.starts_with('.'))
        .unwrap_or(false)
}

/// Read file contents as string
#[tauri::command]
pub async fn read_file(path: String) -> Result<String, String> {
    fs::read_to_string(&path).map_err(|e| format!("Failed to read file: {}", e))
}

/// Write content to a file
#[tauri::command]
pub async fn write_file(path: String, content: String) -> Result<(), String> {
    // Ensure parent directory exists
    if let Some(parent) = Path::new(&path).parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
    }
    fs::write(&path, content).map_err(|e| format!("Failed to write file: {}", e))
}

/// Read directory contents (non-recursive helper)
fn read_directory_sync(path: &Path) -> Result<Vec<DirectoryEntry>, String> {
    if !path.exists() {
        return Err("Directory does not exist".to_string());
    }

    if !path.is_dir() {
        return Err("Path is not a directory".to_string());
    }

    let mut entries: Vec<DirectoryEntry> = Vec::new();

    let read_result = fs::read_dir(path);
    let dir_entries = read_result.map_err(|e| format!("Failed to read directory: {}", e))?;

    for entry in dir_entries.flatten() {
        let entry_path = entry.path();
        let metadata = entry.metadata().ok();

        let is_dir = metadata.as_ref().map(|m| m.is_dir()).unwrap_or(false);
        let name = entry
            .file_name()
            .to_str()
            .unwrap_or_default()
            .to_string();

        let extension = if is_dir {
            None
        } else {
            entry_path
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_string())
        };

        entries.push(DirectoryEntry {
            name,
            path: entry_path.to_string_lossy().to_string(),
            is_directory: is_dir,
            is_expanded: false,
            children: None,
            extension,
            is_hidden: is_hidden(&entry_path),
        });
    }

    // Sort: directories first, then alphabetically
    entries.sort_by(|a, b| {
        match (a.is_directory, b.is_directory) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
        }
    });

    Ok(entries)
}

/// Read directory contents
#[tauri::command]
pub async fn read_directory(path: String, _recursive: bool) -> Result<Vec<DirectoryEntry>, String> {
    let path = Path::new(&path);
    read_directory_sync(path)
}

/// Create a new file
#[tauri::command]
pub async fn create_file(path: String, content: Option<String>) -> Result<(), String> {
    let file_path = Path::new(&path);

    if file_path.exists() {
        return Err("File already exists".to_string());
    }

    // Ensure parent directory exists
    if let Some(parent) = file_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
    }

    fs::write(&path, content.unwrap_or_default())
        .map_err(|e| format!("Failed to create file: {}", e))
}

/// Create a new directory
#[tauri::command]
pub async fn create_directory(path: String) -> Result<(), String> {
    fs::create_dir_all(&path).map_err(|e| format!("Failed to create directory: {}", e))
}

/// Delete a file or directory
#[tauri::command]
pub async fn delete_path(path: String) -> Result<(), String> {
    let target = Path::new(&path);

    if !target.exists() {
        return Err("Path does not exist".to_string());
    }

    if target.is_dir() {
        fs::remove_dir_all(&path).map_err(|e| format!("Failed to delete directory: {}", e))
    } else {
        fs::remove_file(&path).map_err(|e| format!("Failed to delete file: {}", e))
    }
}

/// Rename a file or directory
#[tauri::command]
pub async fn rename_path(old_path: String, new_path: String) -> Result<(), String> {
    fs::rename(&old_path, &new_path).map_err(|e| format!("Failed to rename: {}", e))
}

/// Copy a file or directory
#[tauri::command]
pub async fn copy_path(source: String, destination: String) -> Result<(), String> {
    let src = Path::new(&source);
    let dest = Path::new(&destination);

    if src.is_dir() {
        copy_dir_recursive(src, dest)
    } else {
        fs::copy(&source, &destination)
            .map(|_| ())
            .map_err(|e| format!("Failed to copy: {}", e))
    }
}

/// Recursively copy a directory
fn copy_dir_recursive(src: &Path, dest: &Path) -> Result<(), String> {
    fs::create_dir_all(dest).map_err(|e| format!("Failed to create directory: {}", e))?;

    for entry in fs::read_dir(src).map_err(|e| format!("Failed to read directory: {}", e))? {
        let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
        let entry_path = entry.path();
        let dest_path = dest.join(entry.file_name());

        if entry_path.is_dir() {
            copy_dir_recursive(&entry_path, &dest_path)?;
        } else {
            fs::copy(&entry_path, &dest_path).map_err(|e| format!("Failed to copy file: {}", e))?;
        }
    }

    Ok(())
}

/// Get detailed file information
#[tauri::command]
pub async fn get_file_info(path: String) -> Result<FileInfo, String> {
    let file_path = Path::new(&path);

    if !file_path.exists() {
        return Err("File does not exist".to_string());
    }

    let metadata = fs::metadata(&path).map_err(|e| format!("Failed to get metadata: {}", e))?;

    Ok(FileInfo {
        name: file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string(),
        path: path.clone(),
        is_directory: metadata.is_dir(),
        is_file: metadata.is_file(),
        size: metadata.len(),
        modified: metadata.modified().ok().and_then(system_time_to_unix),
        created: metadata.created().ok().and_then(system_time_to_unix),
        extension: file_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_string()),
        is_hidden: is_hidden(file_path),
    })
}

/// Start watching a directory for changes
#[tauri::command]
pub async fn watch_directory(
    app: AppHandle,
    path: String,
) -> Result<(), String> {
    let watch_path = PathBuf::from(&path);

    if !watch_path.exists() {
        return Err("Directory does not exist".to_string());
    }

    let app_handle = app.clone();
    let path_clone = path.clone();

    let mut watcher = RecommendedWatcher::new(
        move |result: Result<notify::Event, notify::Error>| {
            if let Ok(event) = result {
                let _ = app_handle.emit(
                    "file-change",
                    serde_json::json!({
                        "kind": format!("{:?}", event.kind),
                        "paths": event.paths.iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>(),
                    }),
                );
            }
        },
        Config::default(),
    )
    .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher
        .watch(&watch_path, RecursiveMode::Recursive)
        .map_err(|e| format!("Failed to watch directory: {}", e))?;

    // Store watcher in app state
    app.state::<Arc<Mutex<WatcherState>>>()
        .lock()
        .map_err(|_| "Failed to lock watcher state")?
        .watchers
        .insert(path_clone, watcher);

    Ok(())
}

/// Stop watching a directory
#[tauri::command]
pub async fn unwatch_directory(
    app: AppHandle,
    path: String,
) -> Result<(), String> {
    app.state::<Arc<Mutex<WatcherState>>>()
        .lock()
        .map_err(|_| "Failed to lock watcher state")?
        .watchers
        .remove(&path);

    Ok(())
}

/// Search for text in files
#[tauri::command]
pub async fn search_files(
    path: String,
    query: String,
    file_pattern: Option<String>,
    case_sensitive: bool,
    max_results: Option<usize>,
) -> Result<Vec<SearchResult>, String> {
    let search_path = Path::new(&path);
    let max = max_results.unwrap_or(1000);

    if !search_path.exists() {
        return Err("Directory does not exist".to_string());
    }

    let pattern = if case_sensitive {
        query.clone()
    } else {
        query.to_lowercase()
    };

    let regex_pattern = file_pattern.as_deref().unwrap_or("*");

    let mut results = Vec::new();

    for entry in WalkDir::new(search_path)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if results.len() >= max {
            break;
        }

        let entry_path = entry.path();

        // Skip directories and hidden files
        if entry_path.is_dir() || is_hidden(entry_path) {
            continue;
        }

        // Check file pattern match
        if regex_pattern != "*" {
            if let Some(ext) = entry_path.extension().and_then(|e| e.to_str()) {
                let pattern_ext = regex_pattern.trim_start_matches("*.");
                if ext != pattern_ext {
                    continue;
                }
            } else {
                continue;
            }
        }

        // Read and search file
        if let Ok(content) = fs::read_to_string(entry_path) {
            for (line_num, line) in content.lines().enumerate() {
                if results.len() >= max {
                    break;
                }

                let search_line = if case_sensitive {
                    line.to_string()
                } else {
                    line.to_lowercase()
                };

                if let Some(pos) = search_line.find(&pattern) {
                    results.push(SearchResult {
                        path: entry_path.to_string_lossy().to_string(),
                        line_number: line_num + 1,
                        line_content: line.to_string(),
                        match_start: pos,
                        match_end: pos + pattern.len(),
                    });
                }
            }
        }
    }

    Ok(results)
}
