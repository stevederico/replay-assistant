//! Frames from the `ffmpeg` command line, piped as raw RGB.
//!
//! `ffprobe` (same package) reports the frame size, rate and rotation; `ffmpeg`
//! then decodes to `rgb24` on stdout. Input is always a file, never a pipe,
//! because MOV/MP4 files often keep their index at the end and cannot be
//! decoded from a stream. Images (JPEG, PNG) go through the same path and yield
//! one frame.

use std::io::Read;
use std::path::Path;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::thread::JoinHandle;

use crate::json::{self, Value};

/// Bytes of ffmpeg stderr kept for error messages.
const STDERR_KEEP: usize = 4096;
/// Frame rate reported when the container does not say (also the old service's fallback).
const DEFAULT_FPS: f64 = 30.0;

/// Why a video could not be read.
#[derive(Debug, Clone, PartialEq)]
pub enum VideoError {
    /// `ffmpeg` or `ffprobe` could not be started (not installed, not executable).
    Tool(String),
    /// The input is not something ffmpeg can decode into frames.
    Invalid(String),
}

impl std::fmt::Display for VideoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VideoError::Tool(m) | VideoError::Invalid(m) => f.write_str(m),
        }
    }
}

/// What `ffprobe` says about the first video stream.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoInfo {
    /// Width of decoded frames, after any rotation is applied.
    pub width: usize,
    /// Height of decoded frames, after any rotation is applied.
    pub height: usize,
    /// Average frames per second, [`DEFAULT_FPS`] when unknown.
    pub fps: f64,
    /// Frame count when the container or duration says so.
    pub total_frames: Option<u64>,
}

/// Parse a rate such as `30000/1001` or `25`. `None` for `0/0` and junk.
pub fn parse_rate(s: &str) -> Option<f64> {
    let (num, den) = match s.split_once('/') {
        Some((n, d)) => (n.trim().parse::<f64>().ok()?, d.trim().parse::<f64>().ok()?),
        None => (s.trim().parse::<f64>().ok()?, 1.0),
    };
    let rate = num / den;
    (rate.is_finite() && rate > 0.0).then_some(rate)
}

/// Read a JSON field that ffprobe may print as a number or a string.
fn number_field(v: &Value, key: &str) -> Option<f64> {
    match v.get(key)? {
        Value::Num(n) => Some(*n),
        Value::Str(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// Rotation in degrees from display-matrix side data or the legacy `rotate` tag.
fn rotation_of(stream: &Value) -> f64 {
    let from_side_data = stream
        .get("side_data_list")
        .and_then(Value::as_array)
        .and_then(|list| list.iter().find_map(|item| number_field(item, "rotation")));
    let from_tag = stream.get("tags").and_then(|t| number_field(t, "rotate"));
    from_side_data.or(from_tag).unwrap_or(0.0)
}

/// Interpret `ffprobe -of json` output.
///
/// ffmpeg autorotates, so a stream tagged 90 or 270 degrees decodes to frames
/// with width and height swapped; this reports the decoded size.
///
/// # Errors
/// [`VideoError::Invalid`] when there is no video stream or its size is missing.
pub fn parse_probe(text: &str) -> Result<VideoInfo, VideoError> {
    let root = json::parse(text).map_err(|e| VideoError::Invalid(format!("unreadable ffprobe output: {e}")))?;
    let stream = root
        .get("streams")
        .and_then(Value::as_array)
        .and_then(|s| s.first())
        .ok_or_else(|| VideoError::Invalid("no video stream found in the upload".to_string()))?;
    let dim = |key: &str| number_field(stream, key).filter(|n| *n >= 1.0).map(|n| n as usize);
    let (Some(mut width), Some(mut height)) = (dim("width"), dim("height")) else {
        return Err(VideoError::Invalid("video stream has no frame size".to_string()));
    };
    let quarter_turns = (rotation_of(stream) / 90.0).round() as i64;
    if quarter_turns.rem_euclid(2) == 1 {
        std::mem::swap(&mut width, &mut height);
    }
    let rate = |key: &str| stream.get(key).and_then(Value::as_str).and_then(parse_rate);
    let fps = rate("avg_frame_rate").or_else(|| rate("r_frame_rate")).unwrap_or(DEFAULT_FPS);
    let total_frames = number_field(stream, "nb_frames")
        .filter(|n| *n >= 1.0)
        .or_else(|| number_field(stream, "duration").map(|d| (d * fps).round()).filter(|n| *n >= 1.0))
        .map(|n| n as u64);
    Ok(VideoInfo { width, height, fps, total_frames })
}

/// Run `ffprobe` on `path`.
///
/// # Errors
/// [`VideoError::Tool`] if ffprobe cannot run, [`VideoError::Invalid`] if the
/// file has no readable video stream.
pub fn probe(ffprobe: &str, path: &Path) -> Result<VideoInfo, VideoError> {
    let output = Command::new(ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:stream_tags=rotate:stream_side_data=rotation",
            "-of",
            "json",
        ])
        .arg(path)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| VideoError::Tool(format!("cannot run {ffprobe}: {e}")))?;
    if !output.status.success() {
        let why = String::from_utf8_lossy(&output.stderr);
        return Err(VideoError::Invalid(format!("not a video or image ffmpeg can read: {}", why.trim())));
    }
    parse_probe(&String::from_utf8_lossy(&output.stdout))
}

/// A running `ffmpeg` decoding one file into raw RGB frames.
pub struct FrameReader {
    child: Child,
    stdout: ChildStdout,
    stderr: Option<JoinHandle<String>>,
    frame_len: usize,
}

impl FrameReader {
    /// Start decoding `path`, keeping every `every_n`-th frame and at most `max_frames`.
    ///
    /// # Errors
    /// [`VideoError::Tool`] if ffmpeg cannot be started.
    pub fn spawn(ffmpeg: &str, path: &Path, info: &VideoInfo, every_n: u32, max_frames: usize) -> Result<FrameReader, VideoError> {
        let mut cmd = Command::new(ffmpeg);
        cmd.args(["-nostdin", "-v", "error", "-i"]).arg(path).args(["-an", "-sn", "-dn"]);
        if every_n > 1 {
            // The backslash escapes the comma inside the filter graph.
            cmd.arg("-vf").arg(format!("select=not(mod(n\\,{every_n}))"));
        }
        cmd.args(["-fps_mode", "passthrough", "-frames:v"])
            .arg(max_frames.to_string())
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| VideoError::Tool(format!("cannot run {ffmpeg}: {e}")))?;
        let stdout = child.stdout.take().ok_or_else(|| VideoError::Tool("ffmpeg stdout unavailable".to_string()))?;
        let stderr = child.stderr.take().map(|mut pipe| {
            std::thread::spawn(move || {
                let mut kept = Vec::new();
                let mut buf = [0u8; 1024];
                // Keep draining after the cap so ffmpeg never blocks on a full pipe.
                while let Ok(n) = pipe.read(&mut buf) {
                    if n == 0 {
                        break;
                    }
                    if kept.len() < STDERR_KEEP {
                        kept.extend_from_slice(&buf[..n.min(STDERR_KEEP - kept.len())]);
                    }
                }
                String::from_utf8_lossy(&kept).trim().to_string()
            })
        });
        Ok(FrameReader { child, stdout, stderr, frame_len: info.width * info.height * 3 })
    }

    /// Fill `buf` (exactly one frame) with the next frame.
    ///
    /// Returns `Ok(false)` at a clean end of stream.
    ///
    /// # Errors
    /// [`VideoError::Invalid`] if the stream stops in the middle of a frame.
    pub fn read_frame(&mut self, buf: &mut [u8]) -> Result<bool, VideoError> {
        debug_assert_eq!(buf.len(), self.frame_len);
        let mut filled = 0;
        while filled < buf.len() {
            match self.stdout.read(&mut buf[filled..]) {
                Ok(0) if filled == 0 => return Ok(false),
                Ok(0) => return Err(VideoError::Invalid("video ended in the middle of a frame".to_string())),
                Ok(n) => filled += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(VideoError::Invalid(format!("reading frames failed: {e}"))),
            }
        }
        Ok(true)
    }

    /// Wait for ffmpeg after the stream ended and return its stderr if it failed.
    pub fn finish(mut self) -> Result<(), String> {
        let status = self.child.wait().map_err(|e| e.to_string())?;
        let stderr = self.stderr.take().and_then(|h| h.join().ok()).unwrap_or_default();
        if status.success() {
            Ok(())
        } else {
            Err(if stderr.is_empty() { format!("ffmpeg exited with {status}") } else { stderr })
        }
    }
}

impl Drop for FrameReader {
    fn drop(&mut self) {
        // Idempotent: harmless after `finish` already reaped the child.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rate_handles_fractions_integers_and_junk() {
        assert!((parse_rate("30000/1001").unwrap() - 29.97).abs() < 0.001);
        assert_eq!(parse_rate("25"), Some(25.0));
        assert_eq!(parse_rate("0/0"), None);
        assert_eq!(parse_rate("N/A"), None);
        assert_eq!(parse_rate(""), None);
    }

    #[test]
    fn parse_probe_reads_size_rate_and_frame_count() {
        let info = parse_probe(
            r#"{"streams":[{"width":1280,"height":720,"r_frame_rate":"30000/1001","avg_frame_rate":"30000/1001","duration":"30.030000","nb_frames":"900"}]}"#,
        )
        .unwrap();
        assert_eq!((info.width, info.height), (1280, 720));
        assert!((info.fps - 29.97).abs() < 0.001);
        assert_eq!(info.total_frames, Some(900));
    }

    #[test]
    fn parse_probe_swaps_size_for_a_quarter_turn_display_matrix() {
        let info = parse_probe(
            r#"{"streams":[{"width":1920,"height":1080,"avg_frame_rate":"30/1","side_data_list":[{"rotation":-90}]}]}"#,
        )
        .unwrap();
        assert_eq!((info.width, info.height), (1080, 1920));
    }

    #[test]
    fn parse_probe_swaps_size_for_the_legacy_rotate_tag() {
        let info = parse_probe(r#"{"streams":[{"width":1920,"height":1080,"avg_frame_rate":"30/1","tags":{"rotate":"270"}}]}"#).unwrap();
        assert_eq!((info.width, info.height), (1080, 1920));
    }

    #[test]
    fn parse_probe_keeps_size_for_a_half_turn() {
        let info = parse_probe(r#"{"streams":[{"width":1920,"height":1080,"side_data_list":[{"rotation":180}]}]}"#).unwrap();
        assert_eq!((info.width, info.height), (1920, 1080));
    }

    #[test]
    fn parse_probe_estimates_frames_from_duration_when_count_is_missing() {
        let info = parse_probe(r#"{"streams":[{"width":640,"height":480,"avg_frame_rate":"25/1","duration":"2.0"}]}"#).unwrap();
        assert_eq!(info.total_frames, Some(50));
    }

    #[test]
    fn parse_probe_falls_back_to_default_fps_and_unknown_total() {
        let info = parse_probe(r#"{"streams":[{"width":640,"height":480,"avg_frame_rate":"0/0"}]}"#).unwrap();
        assert_eq!(info.fps, DEFAULT_FPS);
        assert_eq!(info.total_frames, None);
    }

    #[test]
    fn parse_probe_rejects_files_without_a_video_stream() {
        assert!(matches!(parse_probe(r#"{"streams":[]}"#), Err(VideoError::Invalid(_))));
        assert!(matches!(parse_probe("{}"), Err(VideoError::Invalid(_))));
        assert!(matches!(parse_probe("not json"), Err(VideoError::Invalid(_))));
    }

    #[test]
    fn parse_probe_rejects_a_stream_without_a_frame_size() {
        assert!(matches!(parse_probe(r#"{"streams":[{"width":0,"height":0}]}"#), Err(VideoError::Invalid(_))));
    }

    #[test]
    fn probe_reports_a_missing_tool_as_a_tool_error() {
        let err = probe("definitely-not-ffprobe", Path::new("/nonexistent")).unwrap_err();
        assert!(matches!(err, VideoError::Tool(_)));
    }
}
