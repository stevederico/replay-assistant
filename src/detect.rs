//! The detection pipeline: ffprobe, ffmpeg frames, letterbox, model, decode, JSON.
//!
//! The server talks to a [`Detector`] so its HTTP behaviour can be tested with a
//! fake; [`Yolo`] is the real one (onnxruntime + ffmpeg).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::classes::{class_name, NUM_CLASSES};
use crate::json::Writer;
use crate::ort::{Session, Tensor};
use crate::video::{self, FrameReader, VideoError};
use crate::yolo::{self, Detection, Letterbox, Params, Protos, INPUT_SIZE, MASK_COEFFS};

/// Model name reported in responses and `/health`.
pub const MODEL_NAME: &str = "yolo26n-seg";
/// Largest decoded frame accepted, in pixels (a bit over 4K UHD). Bounds the frame buffer.
pub const MAX_FRAME_PIXELS: usize = 4096 * 2304;
/// Model input tensor shape `[1, 3, 640, 640]`.
const INPUT_SHAPE: [i64; 4] = [1, 3, INPUT_SIZE as i64, INPUT_SIZE as i64];
/// Detection ids remembered for `GET /progress/{id}`.
const PROGRESS_ENTRIES: usize = 64;
/// Longest accepted detection id.
const MAX_DETECTION_ID_LEN: usize = 64;

/// Progress of one detection, shaped like the old service's `/detection_progress`.
#[derive(Debug, Clone, PartialEq)]
pub struct ProgressEntry {
    /// Source frame index most recently processed.
    pub frame: u64,
    /// Total frames in the clip, 0 when unknown.
    pub total: u64,
    /// `processing`, `completed` or `failed`.
    pub status: &'static str,
}

/// Bounded in-memory map of detection id to progress. Oldest ids fall out first.
#[derive(Debug, Default)]
pub struct Progress {
    entries: Mutex<VecDeque<(String, ProgressEntry)>>,
}

impl Progress {
    /// Record `entry` for `id`, evicting the oldest id when full.
    pub fn set(&self, id: &str, entry: ProgressEntry) {
        let Ok(mut entries) = self.entries.lock() else { return };
        if let Some(slot) = entries.iter_mut().find(|(k, _)| k == id) {
            slot.1 = entry;
            return;
        }
        if entries.len() >= PROGRESS_ENTRIES {
            entries.pop_front();
        }
        entries.push_back((id.to_string(), entry));
    }

    /// Latest entry for `id`.
    pub fn get(&self, id: &str) -> Option<ProgressEntry> {
        let entries = self.entries.lock().ok()?;
        entries.iter().find(|(k, _)| k == id).map(|(_, e)| e.clone())
    }
}

/// True for ids made of letters, digits, `_` and `-`, at most [`MAX_DETECTION_ID_LEN`] long.
pub fn valid_detection_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= MAX_DETECTION_ID_LEN && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// What to detect, parsed and validated from the query string.
#[derive(Debug, Clone)]
pub struct DetectRequest {
    /// Keep every N-th frame (1 keeps all).
    pub every_n: u32,
    /// Decode settings.
    pub params: Params,
    /// Most sampled frames this request may produce.
    pub max_frames: usize,
    /// Optional id for `GET /progress/{id}`.
    pub detection_id: Option<String>,
}

/// One unit of work handed to the inference thread.
#[derive(Debug)]
pub struct Job {
    /// Uploaded video or image on disk.
    pub input: PathBuf,
    /// What to run.
    pub request: DetectRequest,
    /// Give up at this time.
    pub deadline: Instant,
    /// Set by the HTTP thread when nobody is waiting any more.
    pub cancel: Arc<AtomicBool>,
}

/// Why a detection did not finish.
#[derive(Debug, Clone, PartialEq)]
pub enum DetectError {
    /// The upload is not decodable or is out of bounds (HTTP 422).
    Invalid(String),
    /// The result would exceed a frame or size limit (HTTP 422).
    TooMany(String),
    /// The deadline passed (HTTP 504).
    TimedOut,
    /// The caller went away.
    Cancelled,
    /// Something on our side failed (HTTP 500).
    Internal(String),
}

impl From<VideoError> for DetectError {
    fn from(e: VideoError) -> Self {
        match e {
            VideoError::Invalid(m) => DetectError::Invalid(m),
            VideoError::Tool(m) => DetectError::Internal(m),
        }
    }
}

/// Something that can turn a [`Job`] into a JSON response body.
pub trait Detector: Send {
    /// Run `job`, updating `progress` as frames complete.
    ///
    /// # Errors
    /// A [`DetectError`] describing why no result was produced.
    fn detect(&mut self, job: &Job, progress: &Progress) -> Result<String, DetectError>;
}

/// Settings for the real detector.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Path to the `.onnx` file.
    pub model_path: PathBuf,
    /// onnxruntime intra-op threads.
    pub threads: usize,
    /// `ffmpeg` executable.
    pub ffmpeg: String,
    /// `ffprobe` executable.
    pub ffprobe: String,
    /// Largest response body built in memory.
    pub max_response_bytes: usize,
}

/// onnxruntime + ffmpeg detector. Owns reusable frame and tensor buffers.
pub struct Yolo {
    session: Session,
    cfg: EngineConfig,
    input: Vec<f32>,
    frame: Vec<u8>,
}

impl Yolo {
    /// Load the model and run it once on a blank input.
    ///
    /// The warm-up pays the runtime's first-run allocation cost at boot, and
    /// proves the outputs have the expected YOLO-seg shapes.
    ///
    /// # Errors
    /// A message when the model cannot be loaded or its outputs are not
    /// `[1, 116, N]` and `[1, 32, H, W]`.
    pub fn load(cfg: EngineConfig) -> Result<Yolo, String> {
        let session = Session::load(&cfg.model_path, cfg.threads)
            .map_err(|e| format!("cannot load model {}: {e}", cfg.model_path.display()))?;
        let input = vec![0.0f32; 3 * INPUT_SIZE * INPUT_SIZE];
        session.run(&input, &INPUT_SHAPE, check_output_shapes)??;
        Ok(Yolo { session, cfg, input, frame: Vec::new() })
    }
}

/// Confirm the outputs look like a YOLO-seg export with the 80 COCO classes.
fn check_output_shapes(outs: &[Tensor<'_>]) -> Result<(), String> {
    let expected_rows = (4 + NUM_CLASSES + MASK_COEFFS) as i64;
    match outs {
        [out0, protos, ..] => {
            if !matches!(out0.shape, [1, rows, _] if *rows == expected_rows) {
                return Err(format!("output0 shape {:?} is not [1, {expected_rows}, N]", out0.shape));
            }
            if !matches!(protos.shape, [1, c, _, _] if *c == MASK_COEFFS as i64) {
                return Err(format!("output1 shape {:?} is not [1, {MASK_COEFFS}, H, W]", protos.shape));
            }
            Ok(())
        }
        _ => Err("model must have two outputs (detections and mask prototypes)".to_string()),
    }
}

/// Decode one frame's tensors into detections in source pixels.
fn decode_outputs(outs: &[Tensor<'_>], lb: &Letterbox, src_w: usize, src_h: usize, params: &Params) -> Result<Vec<Detection>, String> {
    let out0 = outs.first().ok_or_else(|| "model returned no outputs".to_string())?;
    let protos = match outs.get(1).map(|t| (t.shape, t.data)) {
        Some(([1, c, h, w], data)) if *c > 0 && *h > 0 && *w > 0 && data.len() >= (*c * *h * *w) as usize => {
            Some(Protos { data, channels: *c as usize, height: *h as usize, width: *w as usize })
        }
        Some(_) => return Err("mask prototype output is malformed".to_string()),
        None => None,
    };
    yolo::postprocess(out0.data, out0.shape, protos, lb, src_w, src_h, params)
}

/// Stop early when the caller left or time ran out.
fn check_stop(job: &Job) -> Result<(), DetectError> {
    if job.cancel.load(Ordering::Relaxed) {
        Err(DetectError::Cancelled)
    } else if Instant::now() >= job.deadline {
        Err(DetectError::TimedOut)
    } else {
        Ok(())
    }
}

/// Write one sampled frame's detections.
fn write_frame(w: &mut Writer, source_frame: u64, dets: &[Detection], with_polygons: bool) {
    w.begin_object();
    w.key("frame");
    w.int(source_frame as i64);
    w.key("detections");
    w.begin_array();
    for d in dets {
        w.begin_object();
        w.key("class");
        w.int(d.class as i64);
        w.key("name");
        w.string(class_name(d.class));
        w.key("score");
        w.number(d.score as f64, 4);
        w.key("box");
        w.begin_array();
        for v in d.bbox {
            w.number(v as f64, 2);
        }
        w.end_array();
        if with_polygons {
            w.key("polygon");
            w.begin_array();
            for p in &d.polygon {
                w.begin_array();
                w.number(p[0] as f64, 2);
                w.number(p[1] as f64, 2);
                w.end_array();
            }
            w.end_array();
        }
        w.end_object();
    }
    w.end_array();
    w.end_object();
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

impl Detector for Yolo {
    fn detect(&mut self, job: &Job, progress: &Progress) -> Result<String, DetectError> {
        let result = self.run(job, progress);
        if let Some(id) = &job.request.detection_id {
            let last = progress.get(id);
            let total = last.as_ref().map_or(0, |p| p.total);
            let entry = match &result {
                Ok(_) => ProgressEntry { frame: total, total, status: "completed" },
                Err(_) => ProgressEntry { frame: last.map_or(0, |p| p.frame), total, status: "failed" },
            };
            progress.set(id, entry);
        }
        result
    }
}

impl Yolo {
    /// The pipeline body; [`Detector::detect`] wraps it with progress bookkeeping.
    fn run(&mut self, job: &Job, progress: &Progress) -> Result<String, DetectError> {
        let started = Instant::now();
        let req = &job.request;
        let info = video::probe(&self.cfg.ffprobe, &job.input)?;
        if info.width * info.height > MAX_FRAME_PIXELS {
            return Err(DetectError::Invalid(format!(
                "frames are {}x{}; the largest supported size is about 4K ({MAX_FRAME_PIXELS} pixels)",
                info.width, info.height
            )));
        }
        let every_n = req.every_n.max(1) as u64;
        if let Some(expected) = info.total_frames.map(|t| t.div_ceil(every_n)) {
            if expected > req.max_frames as u64 {
                return Err(DetectError::TooMany(format!(
                    "this clip has about {expected} frames after sampling but the limit is {}; raise every_n",
                    req.max_frames
                )));
            }
        }
        let total = info.total_frames.unwrap_or(0);
        let mut reader = FrameReader::spawn(&self.cfg.ffmpeg, &job.input, &info, req.every_n, req.max_frames + 1)?;
        self.frame.resize(info.width * info.height * 3, 0);

        let mut w = Writer::new();
        w.begin_object();
        w.key("model");
        w.string(MODEL_NAME);
        w.key("width");
        w.int(info.width as i64);
        w.key("height");
        w.int(info.height as i64);
        w.key("fps");
        w.number(info.fps, 3);
        w.key("totalFrames");
        match info.total_frames {
            Some(t) => w.int(t as i64),
            None => w.null(),
        }
        w.key("everyN");
        w.int(every_n as i64);
        w.key("conf");
        w.number(req.params.conf as f64, 4);
        w.key("iou");
        w.number(req.params.iou as f64, 4);
        w.key("frames");
        w.begin_array();

        let (mut sampled, mut letterbox_time, mut infer_time, mut post_time) = (0usize, Duration::ZERO, Duration::ZERO, Duration::ZERO);
        loop {
            check_stop(job)?;
            if !reader.read_frame(&mut self.frame)? {
                break;
            }
            if sampled >= req.max_frames {
                return Err(DetectError::TooMany(format!("more than {} frames after sampling; raise every_n", req.max_frames)));
            }
            let source_frame = sampled as u64 * every_n;

            let t = Instant::now();
            let lb = yolo::letterbox_into(&self.frame, info.width, info.height, &mut self.input);
            letterbox_time += t.elapsed();

            let t = Instant::now();
            let mut post = Duration::ZERO;
            let dets = self
                .session
                .run(&self.input, &INPUT_SHAPE, |outs| {
                    let t = Instant::now();
                    let dets = decode_outputs(outs, &lb, info.width, info.height, &req.params);
                    post = t.elapsed();
                    dets
                })
                .and_then(|r| r)
                .map_err(DetectError::Internal)?;
            infer_time += t.elapsed().saturating_sub(post);
            post_time += post;

            write_frame(&mut w, source_frame, &dets, req.params.masks);
            if w.len() > self.cfg.max_response_bytes {
                return Err(DetectError::TooMany(format!(
                    "the result is larger than {} MB; raise conf or every_n, or turn masks off",
                    self.cfg.max_response_bytes / (1024 * 1024)
                )));
            }
            sampled += 1;
            if let Some(id) = &req.detection_id {
                progress.set(id, ProgressEntry { frame: source_frame, total, status: "processing" });
            }
        }
        if let Err(why) = reader.finish() {
            if sampled == 0 {
                return Err(DetectError::Invalid(format!("ffmpeg could not decode the upload: {why}")));
            }
        }
        if sampled == 0 {
            return Err(DetectError::Invalid("no frames could be decoded from the upload".to_string()));
        }
        w.end_array();
        let total_time = started.elapsed();
        w.key("sampled");
        w.int(sampled as i64);
        w.key("timing");
        w.begin_object();
        w.key("totalMs");
        w.number(ms(total_time), 1);
        w.key("letterboxMs");
        w.number(ms(letterbox_time), 1);
        w.key("inferMs");
        w.number(ms(infer_time), 1);
        w.key("postMs");
        w.number(ms(post_time), 1);
        w.key("msPerFrame");
        w.number(ms(total_time) / sampled as f64, 2);
        w.key("inferMsPerFrame");
        w.number(ms(infer_time) / sampled as f64, 2);
        w.end_object();
        w.end_object();
        Ok(w.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(frame: u64) -> ProgressEntry {
        ProgressEntry { frame, total: 100, status: "processing" }
    }

    #[test]
    fn progress_returns_the_latest_entry_for_an_id() {
        let p = Progress::default();
        p.set("a", entry(1));
        p.set("a", entry(2));
        assert_eq!(p.get("a"), Some(entry(2)));
        assert_eq!(p.get("missing"), None);
    }

    #[test]
    fn progress_is_bounded_and_evicts_the_oldest_id() {
        let p = Progress::default();
        for i in 0..PROGRESS_ENTRIES + 5 {
            p.set(&format!("id{i}"), entry(i as u64));
        }
        assert_eq!(p.get("id0"), None, "oldest evicted");
        assert!(p.get(&format!("id{}", PROGRESS_ENTRIES + 4)).is_some());
    }

    #[test]
    fn detection_ids_are_limited_to_safe_characters() {
        assert!(valid_detection_id("det_1700000000_abc-9"));
        assert!(!valid_detection_id(""));
        assert!(!valid_detection_id("../etc"));
        assert!(!valid_detection_id("a b"));
        assert!(!valid_detection_id(&"x".repeat(MAX_DETECTION_ID_LEN + 1)));
    }

    #[test]
    fn video_errors_map_to_the_right_detect_errors() {
        assert!(matches!(DetectError::from(VideoError::Invalid("x".into())), DetectError::Invalid(_)));
        assert!(matches!(DetectError::from(VideoError::Tool("x".into())), DetectError::Internal(_)));
    }

    #[test]
    fn output_shapes_must_be_yolo_seg() {
        let data = [0.0f32; 1];
        let good0 = Tensor { shape: &[1, 116, 8400], data: &data };
        let good1 = Tensor { shape: &[1, 32, 160, 160], data: &data };
        assert!(check_output_shapes(&[good0, good1]).is_ok());
        let bad0 = Tensor { shape: &[1, 84, 8400], data: &data };
        assert!(check_output_shapes(&[bad0, good1]).is_err());
        assert!(check_output_shapes(&[good0]).is_err());
    }

    #[test]
    fn write_frame_emits_boxes_and_optional_polygons() {
        let dets = vec![Detection { class: 0, score: 0.5, bbox: [1.0, 2.0, 3.0, 4.0], polygon: vec![[1.0, 2.0], [3.0, 4.0]] }];
        let mut w = Writer::new();
        write_frame(&mut w, 6, &dets, true);
        assert_eq!(
            w.finish(),
            r#"{"frame":6,"detections":[{"class":0,"name":"person","score":0.5,"box":[1,2,3,4],"polygon":[[1,2],[3,4]]}]}"#
        );
        let mut w = Writer::new();
        write_frame(&mut w, 0, &dets, false);
        assert!(!w.finish().contains("polygon"));
    }

    #[test]
    fn decode_outputs_rejects_malformed_prototypes() {
        let out0 = vec![0.0f32; 116];
        let bad_protos = [0.0f32; 4];
        let outs = [
            Tensor { shape: &[1, 116, 1], data: &out0 },
            Tensor { shape: &[1, 32, 160, 160], data: &bad_protos },
        ];
        let lb = Letterbox::new(640, 640);
        assert!(decode_outputs(&outs, &lb, 640, 640, &Params::default()).is_err());
    }

    #[test]
    fn check_stop_reports_cancel_before_timeout() {
        let job = Job {
            input: PathBuf::new(),
            request: DetectRequest { every_n: 1, params: Params::default(), max_frames: 1, detection_id: None },
            deadline: Instant::now() - Duration::from_secs(1),
            cancel: Arc::new(AtomicBool::new(true)),
        };
        assert_eq!(check_stop(&job), Err(DetectError::Cancelled));
        job.cancel.store(false, Ordering::Relaxed);
        assert_eq!(check_stop(&job), Err(DetectError::TimedOut));
    }
}
