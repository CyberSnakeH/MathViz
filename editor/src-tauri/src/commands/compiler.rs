//! MathViz Compiler Integration
//!
//! This module provides Tauri commands for interacting with the MathViz compiler,
//! including compilation, syntax checking, code formatting, and autocompletion.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Stdio;
use tokio::process::Command;

/// Compilation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileResult {
    pub success: bool,
    pub output_path: Option<String>,
    pub stdout: String,
    pub stderr: String,
    pub errors: Vec<CompileError>,
    pub warnings: Vec<CompileWarning>,
}

/// Compile error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileError {
    pub line: usize,
    pub column: usize,
    pub message: String,
    pub severity: String,
    pub code: Option<String>,
}

/// Compile warning information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileWarning {
    pub line: usize,
    pub column: usize,
    pub message: String,
    pub code: Option<String>,
}

/// Syntax check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntaxCheckResult {
    pub valid: bool,
    pub errors: Vec<CompileError>,
    pub warnings: Vec<CompileWarning>,
}

/// Completion item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionItem {
    pub label: String,
    pub kind: String,
    pub detail: Option<String>,
    pub documentation: Option<String>,
    pub insert_text: String,
}

/// Format result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatResult {
    pub success: bool,
    pub formatted_code: Option<String>,
    pub error: Option<String>,
}

/// Run result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub video_path: Option<String>,
}

/// Installation method for MathViz
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathVizInstallation {
    pub found: bool,
    pub method: Option<String>,
    pub command: Option<String>,
    pub version: Option<String>,
}

/// Check MathViz installation status (called at startup)
#[tauri::command]
pub async fn check_mathviz_installation() -> Result<MathVizInstallation, String> {
    use std::process::Command as StdCommand;

    // 1. Check for mathviz executable directly in PATH
    if let Ok(output) = StdCommand::new("which").arg("mathviz").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                // Get version
                let version = get_mathviz_version(&path);
                return Ok(MathVizInstallation {
                    found: true,
                    method: Some("PATH".to_string()),
                    command: Some(path),
                    version,
                });
            }
        }
    }

    // 2. Check for pipx installation
    if let Ok(output) = StdCommand::new("pipx").args(["list", "--short"]).output() {
        if output.status.success() {
            let list = String::from_utf8_lossy(&output.stdout);
            if list.contains("mathviz") {
                // pipx installs to ~/.local/bin/mathviz
                let home = std::env::var("HOME").unwrap_or_default();
                let pipx_path = format!("{}/.local/bin/mathviz", home);
                if std::path::Path::new(&pipx_path).exists() {
                    let version = get_mathviz_version(&pipx_path);
                    return Ok(MathVizInstallation {
                        found: true,
                        method: Some("pipx".to_string()),
                        command: Some(pipx_path),
                        version,
                    });
                }
                // Fallback to just "mathviz" command
                let version = get_mathviz_version("mathviz");
                return Ok(MathVizInstallation {
                    found: true,
                    method: Some("pipx".to_string()),
                    command: Some("mathviz".to_string()),
                    version,
                });
            }
        }
    }

    // 3. Check for uv installation (project-local)
    if let Ok(output) = StdCommand::new("uv")
        .args(["run", "python", "-c", "import mathviz; print(mathviz.__version__)"])
        .output()
    {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            return Ok(MathVizInstallation {
                found: true,
                method: Some("uv".to_string()),
                command: Some("uv run python -m mathviz".to_string()),
                version: if version.is_empty() { None } else { Some(version) },
            });
        }
    }

    // 4. Check for pip installation via python module
    for python in &["python3", "python"] {
        if let Ok(output) = StdCommand::new(python)
            .args(["-c", "import mathviz; print(mathviz.__version__)"])
            .output()
        {
            if output.status.success() {
                let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
                return Ok(MathVizInstallation {
                    found: true,
                    method: Some("pip".to_string()),
                    command: Some(format!("{} -m mathviz", python)),
                    version: if version.is_empty() { None } else { Some(version) },
                });
            }
        }
    }

    // Not found
    Ok(MathVizInstallation {
        found: false,
        method: None,
        command: None,
        version: None,
    })
}

/// Get MathViz version from a command
fn get_mathviz_version(cmd: &str) -> Option<String> {
    use std::process::Command as StdCommand;

    let output = if cmd.contains(' ') {
        // Command with args like "uv run python -m mathviz"
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        if parts.is_empty() {
            return None;
        }
        StdCommand::new(parts[0])
            .args(&parts[1..])
            .arg("--version")
            .output()
            .ok()?
    } else {
        StdCommand::new(cmd).arg("--version").output().ok()?
    };

    if output.status.success() {
        let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
        // Extract version number (e.g., "mathviz 0.1.0" -> "0.1.0")
        if let Some(v) = version.split_whitespace().last() {
            return Some(v.to_string());
        }
        if !version.is_empty() {
            return Some(version);
        }
    }
    None
}

/// Find the MathViz compiler executable
fn find_mathviz_compiler() -> Option<String> {
    use std::process::Command as StdCommand;

    // 1. First, try to find the mathviz executable directly in PATH
    if let Ok(output) = StdCommand::new("which").arg("mathviz").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(path);
            }
        }
    }

    // 2. Check for pipx installation (~/.local/bin/mathviz)
    let home = std::env::var("HOME").unwrap_or_default();
    let pipx_path = format!("{}/.local/bin/mathviz", home);
    if std::path::Path::new(&pipx_path).exists() {
        return Some(pipx_path);
    }

    // 3. Try uv run (for uv-managed projects)
    if let Ok(output) = StdCommand::new("uv")
        .args(["run", "python", "-c", "import mathviz"])
        .output()
    {
        if output.status.success() {
            return Some("uv run python -m mathviz".to_string());
        }
    }

    // 4. Try python3 first, then python (pip installation)
    for python in &["python3", "python"] {
        if let Ok(output) = StdCommand::new("which").arg(python).output() {
            if output.status.success() {
                // Check if mathviz module is available
                if let Ok(check) = StdCommand::new(python)
                    .args(["-c", "import mathviz"])
                    .output()
                {
                    if check.status.success() {
                        return Some(format!("{} -m mathviz", python));
                    }
                }
            }
        }
    }

    // Not found
    None
}

/// Compile a MathViz source file
#[tauri::command]
pub async fn compile_mathviz(
    source_path: String,
    output_dir: Option<String>,
    options: Option<Vec<String>>,
) -> Result<CompileResult, String> {
    let compiler = find_mathviz_compiler().ok_or("MathViz compiler not found")?;

    let output = output_dir.unwrap_or_else(|| {
        Path::new(&source_path)
            .parent()
            .map(|p| p.join("output").to_string_lossy().to_string())
            .unwrap_or_else(|| "./output".to_string())
    });

    let mut cmd_parts: Vec<&str> = compiler.split_whitespace().collect();
    let program = cmd_parts.remove(0);

    let mut command = Command::new(program);
    command.args(&cmd_parts);
    command.arg("compile");
    command.arg(&source_path);
    command.arg("--output");
    command.arg(&output);

    if let Some(opts) = options {
        command.args(opts);
    }

    // Force unbuffered Python output and disable Rich terminal formatting
    command.env("PYTHONUNBUFFERED", "1");
    command.env("NO_COLOR", "1");
    command.env("TERM", "dumb");
    command.env("FORCE_COLOR", "0");

    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());

    let result = command
        .output()
        .await
        .map_err(|e| format!("Failed to run compiler: {}", e))?;

    let stdout = String::from_utf8_lossy(&result.stdout).to_string();
    let stderr = String::from_utf8_lossy(&result.stderr).to_string();

    // Parse errors and warnings from output
    let (errors, warnings) = parse_compiler_output(&stderr);

    Ok(CompileResult {
        success: result.status.success() && errors.is_empty(),
        output_path: Some(output),
        stdout,
        stderr,
        errors,
        warnings,
    })
}

/// Run a MathViz file and generate animation
#[tauri::command]
pub async fn run_mathviz(
    source_path: String,
    preview: bool,
    quality: Option<String>,
) -> Result<RunResult, String> {
    let compiler = find_mathviz_compiler().ok_or("MathViz compiler not found")?;

    let mut cmd_parts: Vec<&str> = compiler.split_whitespace().collect();
    let program = cmd_parts.remove(0);

    let mut command = Command::new(program);
    command.args(&cmd_parts);
    command.arg("run");
    command.arg(&source_path);

    if preview {
        command.arg("--preview");
    }

    if let Some(q) = quality {
        command.arg("--quality");
        command.arg(q);
    }

    // Force unbuffered Python output and disable Rich terminal formatting
    command.env("PYTHONUNBUFFERED", "1");
    command.env("NO_COLOR", "1");
    command.env("TERM", "dumb");
    command.env("FORCE_COLOR", "0");

    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());

    let result = command
        .output()
        .await
        .map_err(|e| format!("Failed to run MathViz: {}", e))?;

    let stdout = String::from_utf8_lossy(&result.stdout).to_string();
    let stderr = String::from_utf8_lossy(&result.stderr).to_string();

    // Try to extract video path from output (check both stdout and stderr)
    // Manim writes INFO messages (including video path) to stderr
    let video_path = extract_video_path(&stderr)
        .or_else(|| extract_video_path(&stdout));

    Ok(RunResult {
        success: result.status.success(),
        stdout,
        stderr,
        exit_code: result.status.code(),
        video_path,
    })
}

/// Check syntax of MathViz code
#[tauri::command]
pub async fn check_syntax(code: String, file_path: Option<String>) -> Result<SyntaxCheckResult, String> {
    let compiler = find_mathviz_compiler().ok_or("MathViz compiler not found")?;

    let mut cmd_parts: Vec<&str> = compiler.split_whitespace().collect();
    let program = cmd_parts.remove(0);

    let mut command = Command::new(program);
    command.args(&cmd_parts);
    command.arg("check");
    command.arg("--stdin");

    if let Some(path) = file_path {
        command.arg("--file");
        command.arg(path);
    }

    command.stdin(Stdio::piped());
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|e| format!("Failed to spawn checker: {}", e))?;

    // Write code to stdin
    if let Some(mut stdin) = child.stdin.take() {
        use tokio::io::AsyncWriteExt;
        stdin
            .write_all(code.as_bytes())
            .await
            .map_err(|e| format!("Failed to write to stdin: {}", e))?;
    }

    let result = child
        .wait_with_output()
        .await
        .map_err(|e| format!("Failed to wait for checker: {}", e))?;

    let stderr = String::from_utf8_lossy(&result.stderr).to_string();
    let (errors, warnings) = parse_compiler_output(&stderr);

    Ok(SyntaxCheckResult {
        valid: result.status.success() && errors.is_empty(),
        errors,
        warnings,
    })
}

/// Get code completions at a position
#[tauri::command]
pub async fn get_completions(
    _code: String,
    _line: usize,
    _column: usize,
    _file_path: Option<String>,
) -> Result<Vec<CompletionItem>, String> {
    // For now, return built-in completions
    // In the future, this will call the LSP server
    let mut completions = Vec::new();

    // MathViz built-in functions and types
    let builtins = vec![
        ("Scene", "class", "Base scene class for animations"),
        ("Circle", "class", "Create a circle shape"),
        ("Square", "class", "Create a square shape"),
        ("Rectangle", "class", "Create a rectangle shape"),
        ("Line", "class", "Create a line"),
        ("Arrow", "class", "Create an arrow"),
        ("Text", "class", "Create text object"),
        ("MathTex", "class", "Create LaTeX math expression"),
        ("play", "method", "Play an animation"),
        ("wait", "method", "Wait for a duration"),
        ("FadeIn", "function", "Fade in animation"),
        ("FadeOut", "function", "Fade out animation"),
        ("Transform", "function", "Transform one object to another"),
        ("Create", "function", "Create animation"),
        ("Write", "function", "Write text animation"),
        ("move_to", "method", "Move object to position"),
        ("scale", "method", "Scale object"),
        ("rotate", "method", "Rotate object"),
        ("shift", "method", "Shift object by vector"),
        ("animate", "property", "Access animation builder"),
    ];

    for (label, kind, detail) in builtins {
        completions.push(CompletionItem {
            label: label.to_string(),
            kind: kind.to_string(),
            detail: Some(detail.to_string()),
            documentation: None,
            insert_text: label.to_string(),
        });
    }

    Ok(completions)
}

/// Format MathViz code
#[tauri::command]
pub async fn format_code(code: String) -> Result<FormatResult, String> {
    let compiler = find_mathviz_compiler().ok_or("MathViz compiler not found")?;

    let mut cmd_parts: Vec<&str> = compiler.split_whitespace().collect();
    let program = cmd_parts.remove(0);

    let mut command = Command::new(program);
    command.args(&cmd_parts);
    command.arg("format");
    command.arg("--stdin");

    command.stdin(Stdio::piped());
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|e| format!("Failed to spawn formatter: {}", e))?;

    // Write code to stdin
    if let Some(mut stdin) = child.stdin.take() {
        use tokio::io::AsyncWriteExt;
        stdin
            .write_all(code.as_bytes())
            .await
            .map_err(|e| format!("Failed to write to stdin: {}", e))?;
    }

    let result = child
        .wait_with_output()
        .await
        .map_err(|e| format!("Failed to wait for formatter: {}", e))?;

    if result.status.success() {
        Ok(FormatResult {
            success: true,
            formatted_code: Some(String::from_utf8_lossy(&result.stdout).to_string()),
            error: None,
        })
    } else {
        Ok(FormatResult {
            success: false,
            formatted_code: None,
            error: Some(String::from_utf8_lossy(&result.stderr).to_string()),
        })
    }
}

/// Parse compiler output to extract errors and warnings
fn parse_compiler_output(output: &str) -> (Vec<CompileError>, Vec<CompileWarning>) {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    for line in output.lines() {
        // Parse error format: "file:line:column: error: message"
        if line.contains("error:") || line.contains("Error:") {
            if let Some((location, message)) = line.split_once(": error:") {
                let parts: Vec<&str> = location.split(':').collect();
                if parts.len() >= 3 {
                    if let (Ok(line_num), Ok(col)) =
                        (parts[1].parse::<usize>(), parts[2].parse::<usize>())
                    {
                        errors.push(CompileError {
                            line: line_num,
                            column: col,
                            message: message.trim().to_string(),
                            severity: "error".to_string(),
                            code: None,
                        });
                    }
                }
            } else if let Some((location, message)) = line.split_once(": Error:") {
                let parts: Vec<&str> = location.split(':').collect();
                if parts.len() >= 3 {
                    if let (Ok(line_num), Ok(col)) =
                        (parts[1].parse::<usize>(), parts[2].parse::<usize>())
                    {
                        errors.push(CompileError {
                            line: line_num,
                            column: col,
                            message: message.trim().to_string(),
                            severity: "error".to_string(),
                            code: None,
                        });
                    }
                }
            }
        } else if line.contains("warning:") || line.contains("Warning:") {
            if let Some((location, message)) = line.split_once(": warning:") {
                let parts: Vec<&str> = location.split(':').collect();
                if parts.len() >= 3 {
                    if let (Ok(line_num), Ok(col)) =
                        (parts[1].parse::<usize>(), parts[2].parse::<usize>())
                    {
                        warnings.push(CompileWarning {
                            line: line_num,
                            column: col,
                            message: message.trim().to_string(),
                            code: None,
                        });
                    }
                }
            }
        }
    }

    (errors, warnings)
}

/// Extract video path from compiler output
fn extract_video_path(output: &str) -> Option<String> {
    // Rich terminal output can split paths across multiple lines
    // Join all lines and look for paths

    // First, try to find "File ready at" pattern which indicates final video
    let normalized = output.replace('\n', " ").replace('\r', " ");

    // Look for path patterns with video extensions
    // Pattern: 'path/to/video.mp4' or /path/to/video.mp4
    let extensions = [".mp4", ".webm", ".gif"];

    for ext in extensions {
        // Find all occurrences of the extension
        let mut search_start = 0;
        while let Some(ext_pos) = normalized[search_start..].find(ext) {
            let abs_pos = search_start + ext_pos + ext.len();

            // Look backwards from the extension to find the start of the path
            let before_ext = &normalized[..search_start + ext_pos];

            // Find the start of the path (either ' or / that starts an absolute path)
            if let Some(path_start) = before_ext.rfind('\'') {
                let path = &normalized[path_start + 1..abs_pos];
                // Clean up any whitespace that Rich might have added
                let clean_path: String = path.chars().filter(|c| !c.is_whitespace()).collect();
                if clean_path.starts_with('/') && std::path::Path::new(&clean_path).exists() {
                    return Some(clean_path);
                }
            }

            // Also try finding paths without quotes (just starting with /)
            if before_ext.contains('/') {
                // Find the true start of the path (first / after whitespace or start)
                let path_region = &normalized[..search_start + ext_pos + ext.len()];
                // Look for pattern like /home/... or /tmp/...
                for start_pattern in ["/home/", "/tmp/", "/var/", "/media/"] {
                    if let Some(start_pos) = path_region.rfind(start_pattern) {
                        let path = &normalized[start_pos..abs_pos];
                        let clean_path: String = path.chars().filter(|c| !c.is_whitespace()).collect();
                        if std::path::Path::new(&clean_path).exists() {
                            return Some(clean_path);
                        }
                    }
                }
            }

            search_start = abs_pos;
        }
    }

    // Fallback: simple line-by-line search
    for line in output.lines() {
        if line.contains(".mp4") || line.contains(".webm") || line.contains(".gif") {
            let words: Vec<&str> = line.split_whitespace().collect();
            for word in words {
                if word.ends_with(".mp4") || word.ends_with(".webm") || word.ends_with(".gif") {
                    let path = word.trim_matches(|c| c == '"' || c == '\'');
                    if std::path::Path::new(path).exists() {
                        return Some(path.to_string());
                    }
                }
            }
        }
    }

    None
}
