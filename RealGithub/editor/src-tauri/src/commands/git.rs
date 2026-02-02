//! Git Integration
//!
//! This module provides Tauri commands for Git operations including:
//! - Status, diff, staging, and committing
//! - Branch management
//! - Push and pull operations
//! - Repository initialization

use git2::{BranchType, Commit, DiffOptions, Repository, StatusOptions};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Git status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitStatus {
    pub is_repo: bool,
    pub branch: Option<String>,
    pub ahead: usize,
    pub behind: usize,
    pub staged: Vec<FileStatus>,
    pub unstaged: Vec<FileStatus>,
    pub untracked: Vec<String>,
    pub conflicted: Vec<String>,
}

/// File status in Git
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileStatus {
    pub path: String,
    pub status: String,
}

/// Diff information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffResult {
    pub files: Vec<FileDiff>,
    pub stats: DiffStats,
}

/// File diff information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDiff {
    pub path: String,
    pub old_path: Option<String>,
    pub status: String,
    pub hunks: Vec<DiffHunk>,
    pub additions: usize,
    pub deletions: usize,
}

/// Diff hunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffHunk {
    pub old_start: u32,
    pub old_lines: u32,
    pub new_start: u32,
    pub new_lines: u32,
    pub lines: Vec<DiffLine>,
}

/// Diff line
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffLine {
    pub origin: char,
    pub content: String,
    pub old_lineno: Option<u32>,
    pub new_lineno: Option<u32>,
}

/// Diff statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffStats {
    pub files_changed: usize,
    pub insertions: usize,
    pub deletions: usize,
}

/// Branch information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchInfo {
    pub name: String,
    pub is_current: bool,
    pub is_remote: bool,
    pub upstream: Option<String>,
    pub last_commit: Option<String>,
}

/// Commit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitInfo {
    pub hash: String,
    pub short_hash: String,
    pub message: String,
    pub author: String,
    pub email: String,
    pub date: String,
}

/// Get repository status
#[tauri::command]
pub async fn git_status(path: String) -> Result<GitStatus, String> {
    let repo = match Repository::discover(&path) {
        Ok(r) => r,
        Err(_) => {
            return Ok(GitStatus {
                is_repo: false,
                branch: None,
                ahead: 0,
                behind: 0,
                staged: Vec::new(),
                unstaged: Vec::new(),
                untracked: Vec::new(),
                conflicted: Vec::new(),
            });
        }
    };

    let mut opts = StatusOptions::new();
    opts.include_untracked(true)
        .recurse_untracked_dirs(true)
        .include_ignored(false);

    let statuses = repo
        .statuses(Some(&mut opts))
        .map_err(|e| format!("Failed to get status: {}", e))?;

    let mut staged = Vec::new();
    let mut unstaged = Vec::new();
    let mut untracked = Vec::new();
    let mut conflicted = Vec::new();

    for entry in statuses.iter() {
        let status = entry.status();
        let path = entry.path().unwrap_or_default().to_string();

        if status.is_conflicted() {
            conflicted.push(path.clone());
        }

        if status.is_index_new() {
            staged.push(FileStatus {
                path: path.clone(),
                status: "added".to_string(),
            });
        } else if status.is_index_modified() {
            staged.push(FileStatus {
                path: path.clone(),
                status: "modified".to_string(),
            });
        } else if status.is_index_deleted() {
            staged.push(FileStatus {
                path: path.clone(),
                status: "deleted".to_string(),
            });
        } else if status.is_index_renamed() {
            staged.push(FileStatus {
                path: path.clone(),
                status: "renamed".to_string(),
            });
        }

        if status.is_wt_modified() {
            unstaged.push(FileStatus {
                path: path.clone(),
                status: "modified".to_string(),
            });
        } else if status.is_wt_deleted() {
            unstaged.push(FileStatus {
                path: path.clone(),
                status: "deleted".to_string(),
            });
        } else if status.is_wt_renamed() {
            unstaged.push(FileStatus {
                path: path.clone(),
                status: "renamed".to_string(),
            });
        } else if status.is_wt_new() {
            untracked.push(path);
        }
    }

    // Get current branch
    let branch = repo
        .head()
        .ok()
        .and_then(|h| h.shorthand().map(|s| s.to_string()));

    // Get ahead/behind counts
    let (ahead, behind) = get_ahead_behind(&repo).unwrap_or((0, 0));

    Ok(GitStatus {
        is_repo: true,
        branch,
        ahead,
        behind,
        staged,
        unstaged,
        untracked,
        conflicted,
    })
}

/// Get ahead/behind counts relative to upstream
fn get_ahead_behind(repo: &Repository) -> Option<(usize, usize)> {
    let head = repo.head().ok()?;
    let local = head.target()?;

    let branch = repo.find_branch(head.shorthand()?, BranchType::Local).ok()?;
    let upstream = branch.upstream().ok()?;
    let remote = upstream.get().target()?;

    repo.graph_ahead_behind(local, remote).ok()
}

/// Get diff of changes
#[tauri::command]
pub async fn git_diff(path: String, staged: bool) -> Result<DiffResult, String> {
    let repo = Repository::discover(&path).map_err(|e| format!("Not a git repository: {}", e))?;

    let mut opts = DiffOptions::new();
    opts.include_untracked(true);

    let diff = if staged {
        let head = repo
            .head()
            .and_then(|h| h.peel_to_tree())
            .map_err(|e| format!("Failed to get HEAD: {}", e))?;

        repo.diff_tree_to_index(Some(&head), None, Some(&mut opts))
    } else {
        repo.diff_index_to_workdir(None, Some(&mut opts))
    }
    .map_err(|e| format!("Failed to get diff: {}", e))?;

    let mut files = Vec::new();
    let stats = diff.stats().map_err(|e| format!("Failed to get stats: {}", e))?;

    for delta_idx in 0..diff.deltas().len() {
        let delta = diff.get_delta(delta_idx).unwrap();
        let new_file = delta.new_file();
        let old_file = delta.old_file();

        let path = new_file
            .path()
            .or_else(|| old_file.path())
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();

        let old_path = if delta.status() == git2::Delta::Renamed {
            old_file.path().map(|p| p.to_string_lossy().to_string())
        } else {
            None
        };

        let status = match delta.status() {
            git2::Delta::Added => "added",
            git2::Delta::Deleted => "deleted",
            git2::Delta::Modified => "modified",
            git2::Delta::Renamed => "renamed",
            git2::Delta::Copied => "copied",
            git2::Delta::Typechange => "typechange",
            _ => "unknown",
        }
        .to_string();

        let mut hunks = Vec::new();
        let mut additions = 0;
        let mut deletions = 0;

        if let Ok(patch) = git2::Patch::from_diff(&diff, delta_idx) {
            if let Some(patch) = patch {
                for hunk_idx in 0..patch.num_hunks() {
                    if let Ok((hunk, _)) = patch.hunk(hunk_idx) {
                        let mut lines = Vec::new();

                        for line_idx in 0..patch.num_lines_in_hunk(hunk_idx).unwrap_or(0) {
                            if let Ok(line) = patch.line_in_hunk(hunk_idx, line_idx) {
                                let origin = line.origin();
                                let content = String::from_utf8_lossy(line.content()).to_string();

                                if origin == '+' {
                                    additions += 1;
                                } else if origin == '-' {
                                    deletions += 1;
                                }

                                lines.push(DiffLine {
                                    origin,
                                    content,
                                    old_lineno: line.old_lineno(),
                                    new_lineno: line.new_lineno(),
                                });
                            }
                        }

                        hunks.push(DiffHunk {
                            old_start: hunk.old_start(),
                            old_lines: hunk.old_lines(),
                            new_start: hunk.new_start(),
                            new_lines: hunk.new_lines(),
                            lines,
                        });
                    }
                }
            }
        }

        files.push(FileDiff {
            path,
            old_path,
            status,
            hunks,
            additions,
            deletions,
        });
    }

    Ok(DiffResult {
        files,
        stats: DiffStats {
            files_changed: stats.files_changed(),
            insertions: stats.insertions(),
            deletions: stats.deletions(),
        },
    })
}

/// Stage files
#[tauri::command]
pub async fn git_stage(path: String, files: Vec<String>) -> Result<(), String> {
    let repo = Repository::discover(&path).map_err(|e| format!("Not a git repository: {}", e))?;

    let mut index = repo
        .index()
        .map_err(|e| format!("Failed to get index: {}", e))?;

    for file in files {
        let file_path = Path::new(&file);
        if file_path.exists() {
            index
                .add_path(file_path)
                .map_err(|e| format!("Failed to stage {}: {}", file, e))?;
        } else {
            index
                .remove_path(file_path)
                .map_err(|e| format!("Failed to stage deletion {}: {}", file, e))?;
        }
    }

    index
        .write()
        .map_err(|e| format!("Failed to write index: {}", e))?;

    Ok(())
}

/// Unstage files
#[tauri::command]
pub async fn git_unstage(path: String, files: Vec<String>) -> Result<(), String> {
    let repo = Repository::discover(&path).map_err(|e| format!("Not a git repository: {}", e))?;

    let head = repo
        .head()
        .and_then(|h| h.peel_to_commit())
        .map_err(|e| format!("Failed to get HEAD: {}", e))?;

    let mut index = repo
        .index()
        .map_err(|e| format!("Failed to get index: {}", e))?;

    let head_object = head.as_object();
    for file in files {
        let file_path = Path::new(&file);
        repo.reset_default(Some(head_object), [file_path])
            .map_err(|e| format!("Failed to unstage {}: {}", file, e))?;
    }

    index
        .write()
        .map_err(|e| format!("Failed to write index: {}", e))?;

    Ok(())
}

/// Create a commit
#[tauri::command]
pub async fn git_commit(path: String, message: String) -> Result<String, String> {
    let repo = Repository::discover(&path).map_err(|e| format!("Not a git repository: {}", e))?;

    let sig = repo
        .signature()
        .map_err(|e| format!("Failed to get signature: {}", e))?;

    let mut index = repo
        .index()
        .map_err(|e| format!("Failed to get index: {}", e))?;

    let tree_id = index
        .write_tree()
        .map_err(|e| format!("Failed to write tree: {}", e))?;

    let tree = repo
        .find_tree(tree_id)
        .map_err(|e| format!("Failed to find tree: {}", e))?;

    let parent = repo
        .head()
        .and_then(|h| h.peel_to_commit())
        .ok();

    let parents: Vec<&Commit> = parent.iter().collect();

    let commit_id = repo
        .commit(Some("HEAD"), &sig, &sig, &message, &tree, &parents)
        .map_err(|e| format!("Failed to create commit: {}", e))?;

    Ok(commit_id.to_string())
}

/// Push to remote
#[tauri::command]
pub async fn git_push(
    path: String,
    remote: Option<String>,
    branch: Option<String>,
) -> Result<(), String> {
    let repo = Repository::discover(&path).map_err(|e| format!("Not a git repository: {}", e))?;

    let remote_name = remote.unwrap_or_else(|| "origin".to_string());
    let mut remote = repo
        .find_remote(&remote_name)
        .map_err(|e| format!("Failed to find remote: {}", e))?;

    let branch_name = branch.unwrap_or_else(|| {
        repo.head()
            .ok()
            .and_then(|h| h.shorthand().map(|s| s.to_string()))
            .unwrap_or_else(|| "main".to_string())
    });

    let refspec = format!("refs/heads/{}:refs/heads/{}", branch_name, branch_name);

    remote
        .push(&[&refspec], None)
        .map_err(|e| format!("Failed to push: {}", e))?;

    Ok(())
}

/// Pull from remote
#[tauri::command]
pub async fn git_pull(
    path: String,
    remote: Option<String>,
    branch: Option<String>,
) -> Result<(), String> {
    let repo = Repository::discover(&path).map_err(|e| format!("Not a git repository: {}", e))?;

    let remote_name = remote.unwrap_or_else(|| "origin".to_string());
    let mut remote = repo
        .find_remote(&remote_name)
        .map_err(|e| format!("Failed to find remote: {}", e))?;

    let branch_name = branch.unwrap_or_else(|| {
        repo.head()
            .ok()
            .and_then(|h| h.shorthand().map(|s| s.to_string()))
            .unwrap_or_else(|| "main".to_string())
    });

    // Fetch
    remote
        .fetch(&[&branch_name], None, None)
        .map_err(|e| format!("Failed to fetch: {}", e))?;

    // Get the fetch head
    let fetch_head = repo
        .find_reference("FETCH_HEAD")
        .map_err(|e| format!("Failed to find FETCH_HEAD: {}", e))?;

    let fetch_commit = repo
        .reference_to_annotated_commit(&fetch_head)
        .map_err(|e| format!("Failed to get annotated commit: {}", e))?;

    // Merge
    let (analysis, _) = repo
        .merge_analysis(&[&fetch_commit])
        .map_err(|e| format!("Failed to analyze merge: {}", e))?;

    if analysis.is_fast_forward() {
        let refname = format!("refs/heads/{}", branch_name);
        let mut reference = repo
            .find_reference(&refname)
            .map_err(|e| format!("Failed to find reference: {}", e))?;

        reference
            .set_target(fetch_commit.id(), "Fast-forward")
            .map_err(|e| format!("Failed to fast-forward: {}", e))?;

        repo.set_head(&refname)
            .map_err(|e| format!("Failed to set HEAD: {}", e))?;

        repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))
            .map_err(|e| format!("Failed to checkout: {}", e))?;
    } else if analysis.is_normal() {
        return Err("Merge required - please handle manually".to_string());
    }

    Ok(())
}

/// List branches
#[tauri::command]
pub async fn git_branch_list(path: String) -> Result<Vec<BranchInfo>, String> {
    let repo = Repository::discover(&path).map_err(|e| format!("Not a git repository: {}", e))?;

    let mut branches = Vec::new();

    let current_branch = repo
        .head()
        .ok()
        .and_then(|h| h.shorthand().map(|s| s.to_string()));

    // Local branches
    for branch in repo
        .branches(Some(BranchType::Local))
        .map_err(|e| format!("Failed to list branches: {}", e))?
    {
        let (branch, _) = branch.map_err(|e| format!("Failed to get branch: {}", e))?;
        let name = branch
            .name()
            .map_err(|e| format!("Failed to get branch name: {}", e))?
            .unwrap_or_default()
            .to_string();

        let upstream = branch
            .upstream()
            .ok()
            .and_then(|u| u.name().ok().flatten().map(|s| s.to_string()));

        let last_commit = branch
            .get()
            .peel_to_commit()
            .ok()
            .map(|c| c.message().unwrap_or_default().lines().next().unwrap_or_default().to_string());

        branches.push(BranchInfo {
            name: name.clone(),
            is_current: current_branch.as_ref() == Some(&name),
            is_remote: false,
            upstream,
            last_commit,
        });
    }

    // Remote branches
    for branch in repo
        .branches(Some(BranchType::Remote))
        .map_err(|e| format!("Failed to list remote branches: {}", e))?
    {
        let (branch, _) = branch.map_err(|e| format!("Failed to get branch: {}", e))?;
        let name = branch
            .name()
            .map_err(|e| format!("Failed to get branch name: {}", e))?
            .unwrap_or_default()
            .to_string();

        let last_commit = branch
            .get()
            .peel_to_commit()
            .ok()
            .map(|c| c.message().unwrap_or_default().lines().next().unwrap_or_default().to_string());

        branches.push(BranchInfo {
            name,
            is_current: false,
            is_remote: true,
            upstream: None,
            last_commit,
        });
    }

    Ok(branches)
}

/// Checkout a branch
#[tauri::command]
pub async fn git_checkout(path: String, branch: String) -> Result<(), String> {
    let repo = Repository::discover(&path).map_err(|e| format!("Not a git repository: {}", e))?;

    let obj = repo
        .revparse_single(&format!("refs/heads/{}", branch))
        .or_else(|_| repo.revparse_single(&format!("refs/remotes/origin/{}", branch)))
        .map_err(|e| format!("Failed to find branch: {}", e))?;

    repo.checkout_tree(&obj, None)
        .map_err(|e| format!("Failed to checkout: {}", e))?;

    repo.set_head(&format!("refs/heads/{}", branch))
        .map_err(|e| format!("Failed to set HEAD: {}", e))?;

    Ok(())
}

/// Create a new branch
#[tauri::command]
pub async fn git_create_branch(
    path: String,
    name: String,
    checkout: bool,
) -> Result<(), String> {
    // Create branch in a block to drop git2 objects before any await
    {
        let repo = Repository::discover(&path).map_err(|e| format!("Not a git repository: {}", e))?;

        let head = repo
            .head()
            .and_then(|h| h.peel_to_commit())
            .map_err(|e| format!("Failed to get HEAD: {}", e))?;

        repo.branch(&name, &head, false)
            .map_err(|e| format!("Failed to create branch: {}", e))?;
    }

    if checkout {
        git_checkout(path, name).await?;
    }

    Ok(())
}

/// Get commit log
#[tauri::command]
pub async fn git_log(path: String, limit: Option<usize>) -> Result<Vec<CommitInfo>, String> {
    let repo = Repository::discover(&path).map_err(|e| format!("Not a git repository: {}", e))?;

    let mut revwalk = repo
        .revwalk()
        .map_err(|e| format!("Failed to create revwalk: {}", e))?;

    revwalk
        .push_head()
        .map_err(|e| format!("Failed to push HEAD: {}", e))?;

    let limit = limit.unwrap_or(100);
    let mut commits = Vec::new();

    for oid in revwalk.take(limit) {
        let oid = oid.map_err(|e| format!("Failed to get oid: {}", e))?;
        let commit = repo
            .find_commit(oid)
            .map_err(|e| format!("Failed to find commit: {}", e))?;

        let author = commit.author();

        commits.push(CommitInfo {
            hash: oid.to_string(),
            short_hash: oid.to_string()[..7].to_string(),
            message: commit
                .message()
                .unwrap_or_default()
                .to_string(),
            author: author.name().unwrap_or_default().to_string(),
            email: author.email().unwrap_or_default().to_string(),
            date: chrono_from_git_time(author.when()),
        });
    }

    Ok(commits)
}

/// Initialize a new Git repository
#[tauri::command]
pub async fn git_init(path: String) -> Result<(), String> {
    Repository::init(&path).map_err(|e| format!("Failed to init repository: {}", e))?;
    Ok(())
}

/// Convert git time to ISO string
fn chrono_from_git_time(time: git2::Time) -> String {
    use std::time::{Duration, UNIX_EPOCH};

    let secs = time.seconds() as u64;
    let datetime = UNIX_EPOCH + Duration::from_secs(secs);

    format!("{:?}", datetime)
}
