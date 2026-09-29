//! Runs real images and clips through the real model.
//!
//! Needs libonnxruntime at build and run time (`scripts/test.sh` sets
//! `LIBRARY_PATH` and `LD_LIBRARY_PATH`) and `ffmpeg`/`ffprobe` on PATH. Without
//! the `onnxruntime` feature the whole file compiles down to one test that
//! prints why it is skipped; without ffmpeg each test prints `SKIP:` and passes.
//!
//! The ultralytics comparison reads `tests/fixtures/ultralytics_expected.json`,
//! produced by `scripts/export_model.py` from the same ONNX file on the same
//! ffmpeg-decoded pixels. The images are not shipped, so those tests skip until
//! you add your own `frame14.jpg`, `frame20.jpg` and regenerate the reference.

#[cfg(not(feature = "onnxruntime"))]
#[test]
fn model_tests_are_skipped_without_onnxruntime() {
    eprintln!("SKIP: built with --no-default-features, so libonnxruntime is not linked and the model tests cannot run.");
    eprintln!("      Install it with scripts/install-onnxruntime.sh and run scripts/test.sh.");
}

#[cfg(feature = "onnxruntime")]
mod with_model {
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use yolo_server::detect::{DetectError, DetectRequest, Detector, EngineConfig, Job, Progress, Yolo};
    use yolo_server::json::{self, Value};
    use yolo_server::server::{Config, Server};
    use yolo_server::yolo::Params;

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn fixture(name: &str) -> PathBuf {
        root().join("tests/fixtures").join(name)
    }

    /// The comparison images are not shipped (they were broadcast frames). Tests that need them skip.
    fn have_fixtures() -> bool {
        fixture("frame14.jpg").exists() && fixture("frame20.jpg").exists() && fixture("ultralytics_expected.json").exists()
    }

    fn have(tool: &str) -> bool {
        Command::new(tool).arg("-version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success())
    }

    fn engine_config() -> EngineConfig {
        EngineConfig {
            model_path: root().join("models/yolo26n-seg.onnx"),
            threads: 4,
            ffmpeg: "ffmpeg".to_string(),
            ffprobe: "ffprobe".to_string(),
            max_response_bytes: 64 * 1024 * 1024,
        }
    }

    /// The detector, or `None` (after saying why) when ffmpeg is missing.
    fn engine() -> Option<Yolo> {
        if !have("ffmpeg") || !have("ffprobe") {
            eprintln!("SKIP: ffmpeg and ffprobe are required for the model tests");
            return None;
        }
        Some(Yolo::load(engine_config()).expect("the model should load"))
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("yolo-server-model-{}-{name}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn request(every_n: u32, params: Params, max_frames: usize) -> DetectRequest {
        DetectRequest { every_n, params, max_frames, detection_id: None }
    }

    fn detect(yolo: &mut Yolo, input: &Path, req: DetectRequest) -> Result<Value, DetectError> {
        let job = Job {
            input: input.to_path_buf(),
            request: req,
            deadline: Instant::now() + Duration::from_secs(120),
            cancel: Arc::new(AtomicBool::new(false)),
        };
        let body = yolo.detect(&job, &Progress::default())?;
        Ok(json::parse(&body).expect("the response is valid JSON"))
    }

    /// One detection as the tests compare it.
    struct Det {
        class: usize,
        score: f64,
        bbox: [f64; 4],
        polygon: Vec<[f64; 2]>,
    }

    fn nums(v: &Value) -> Vec<f64> {
        v.as_array().unwrap().iter().map(|n| n.as_f64().unwrap()).collect()
    }

    fn dets(list: &Value) -> Vec<Det> {
        list.as_array()
            .unwrap()
            .iter()
            .map(|d| {
                let b = nums(d.get("box").unwrap());
                Det {
                    class: d.get("class").and_then(Value::as_f64).unwrap() as usize,
                    score: d.get("score").and_then(Value::as_f64).unwrap(),
                    bbox: [b[0], b[1], b[2], b[3]],
                    polygon: d
                        .get("polygon")
                        .and_then(Value::as_array)
                        .map(|p| p.iter().map(|pt| {
                            let xy = nums(pt);
                            [xy[0], xy[1]]
                        }).collect())
                        .unwrap_or_default(),
                }
            })
            .collect()
    }

    fn first_frame_dets(result: &Value) -> Vec<Det> {
        let frames = result.get("frames").and_then(Value::as_array).unwrap();
        dets(frames[0].get("detections").unwrap())
    }

    fn polygon_area(p: &[[f64; 2]]) -> f64 {
        let n = p.len();
        (0..n).map(|i| p[i][0] * p[(i + 1) % n][1] - p[(i + 1) % n][0] * p[i][1]).sum::<f64>().abs() / 2.0
    }

    fn max_box_diff(a: &[f64; 4], b: &[f64; 4]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    /// Pair every reference detection with the closest unused Rust detection of the same class.
    fn pair<'a>(reference: &'a [Det], got: &'a [Det]) -> Vec<(&'a Det, &'a Det)> {
        let mut used = vec![false; got.len()];
        reference
            .iter()
            .map(|r| {
                let (i, g) = got
                    .iter()
                    .enumerate()
                    .filter(|(i, g)| !used[*i] && g.class == r.class)
                    .min_by(|a, b| max_box_diff(&a.1.bbox, &r.bbox).total_cmp(&max_box_diff(&b.1.bbox, &r.bbox)))
                    .unwrap_or_else(|| panic!("no detection of class {} near {:?}", r.class, r.bbox));
                used[i] = true;
                (r, g)
            })
            .collect()
    }

    fn reference() -> Value {
        let text = fs::read_to_string(fixture("ultralytics_expected.json")).expect("reference dump is committed");
        json::parse(&text).unwrap()
    }

    /// Boxes within this many source pixels of ultralytics (OpenCV's fixed-point resize vs ours).
    const BOX_TOLERANCE_PX: f64 = 1.5;
    /// Scores within this of ultralytics.
    const SCORE_TOLERANCE: f64 = 0.02;

    #[test]
    fn decoded_boxes_scores_and_polygons_match_ultralytics_on_the_fixtures() {
        if !have_fixtures() {
            eprintln!("SKIP: tests/fixtures is not shipped");
            return;
        }
        let Some(mut yolo) = engine() else { return };
        let reference = reference();
        let images = reference.get("images").and_then(Value::as_object).unwrap();
        assert!(!images.is_empty());
        for (name, expected) in images {
            let want = dets(expected.get("detections").unwrap());
            let params = Params { simplify: 0.0, ..Params::default() };
            let result = detect(&mut yolo, &fixture(name), request(1, params, 10)).unwrap();
            assert_eq!(result.get("width").and_then(Value::as_f64), expected.get("width").and_then(Value::as_f64), "{name}");
            assert_eq!(result.get("height").and_then(Value::as_f64), expected.get("height").and_then(Value::as_f64), "{name}");
            assert_eq!(result.get("sampled").and_then(Value::as_f64), Some(1.0));
            let got = first_frame_dets(&result);
            assert_eq!(got.len(), want.len(), "{name}: detection count");
            for (r, g) in pair(&want, &got) {
                let diff = max_box_diff(&r.bbox, &g.bbox);
                assert!(diff <= BOX_TOLERANCE_PX, "{name}: box off by {diff:.2}px: want {:?} got {:?}", r.bbox, g.bbox);
                assert!((r.score - g.score).abs() <= SCORE_TOLERANCE, "{name}: score {} vs {}", r.score, g.score);
                // A marginal mask (score under 0.3) has many logits near zero, so a few
                // pixels flip on float rounding; confident masks agree much more closely.
                let tolerance = if r.score >= 0.3 { 0.03 } else { 0.08 };
                let (ra, ga) = (polygon_area(&r.polygon), polygon_area(&g.polygon));
                assert!((ra - ga).abs() <= ra * tolerance, "{name}: polygon area {ra:.0} vs {ga:.0} (score {})", r.score);
            }
        }
    }

    #[test]
    fn simplified_polygons_keep_their_area_with_far_fewer_points() {
        if !have_fixtures() {
            eprintln!("SKIP: tests/fixtures is not shipped");
            return;
        }
        let Some(mut yolo) = engine() else { return };
        let exact = detect(&mut yolo, &fixture("frame20.jpg"), request(1, Params { simplify: 0.0, ..Params::default() }, 10)).unwrap();
        let simple = detect(&mut yolo, &fixture("frame20.jpg"), request(1, Params::default(), 10)).unwrap();
        let (e, s) = (&first_frame_dets(&exact)[0], &first_frame_dets(&simple)[0]);
        assert!(s.polygon.len() * 3 < e.polygon.len(), "{} vs {} points", s.polygon.len(), e.polygon.len());
        let (ea, sa) = (polygon_area(&e.polygon), polygon_area(&s.polygon));
        assert!((ea - sa).abs() <= ea * 0.03, "area {ea:.0} vs {sa:.0}");
    }

    #[test]
    fn masks_off_omits_polygons_and_keeps_boxes() {
        if !have_fixtures() {
            eprintln!("SKIP: tests/fixtures is not shipped");
            return;
        }
        let Some(mut yolo) = engine() else { return };
        let params = Params { masks: false, ..Params::default() };
        let result = detect(&mut yolo, &fixture("frame14.jpg"), request(1, params, 10)).unwrap();
        let got = first_frame_dets(&result);
        assert_eq!(got.len(), 6);
        assert!(got.iter().all(|d| d.polygon.is_empty()));
    }

    #[test]
    fn class_filter_and_confidence_change_what_comes_back() {
        if !have_fixtures() {
            eprintln!("SKIP: tests/fixtures is not shipped");
            return;
        }
        let Some(mut yolo) = engine() else { return };
        let mut ball_only = vec![false; 80];
        ball_only[32] = true;
        let none = detect(&mut yolo, &fixture("frame14.jpg"), request(1, Params { allowed: Some(ball_only), ..Params::default() }, 10)).unwrap();
        assert!(first_frame_dets(&none).is_empty(), "no sports balls in the fixture");
        let strict = detect(&mut yolo, &fixture("frame14.jpg"), request(1, Params { conf: 0.6, ..Params::default() }, 10)).unwrap();
        let strict = first_frame_dets(&strict);
        assert_eq!(strict.len(), 1, "only the one person scoring above 0.6");
        assert!(strict.iter().all(|d| d.score > 0.6));
    }

    /// A short synthetic clip so the frame-sampling tests do not depend on repo assets.
    fn synthetic_clip(dir: &Path, extra_input_args: &[&str]) -> Option<PathBuf> {
        let plain = dir.join("plain.mp4");
        let status = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=10:duration=3", "-pix_fmt", "yuv420p"])
            .arg(&plain)
            .status()
            .ok()?;
        if !status.success() {
            return None;
        }
        if extra_input_args.is_empty() {
            return Some(plain);
        }
        let out = dir.join("rotated.mp4");
        let status = Command::new("ffmpeg")
            .args(["-v", "error", "-y"])
            .args(extra_input_args)
            .arg("-i")
            .arg(&plain)
            .args(["-c", "copy"])
            .arg(&out)
            .status()
            .ok()?;
        status.success().then_some(out)
    }

    #[test]
    fn every_nth_frame_is_sampled_with_source_frame_numbers() {
        let Some(mut yolo) = engine() else { return };
        let dir = scratch("sample");
        let Some(clip) = synthetic_clip(&dir, &[]) else {
            eprintln!("SKIP: could not generate a test clip with ffmpeg");
            return;
        };
        let result = detect(&mut yolo, &clip, request(4, Params::default(), 100)).unwrap();
        let frames = result.get("frames").and_then(Value::as_array).unwrap();
        let numbers: Vec<f64> = frames.iter().map(|f| f.get("frame").and_then(Value::as_f64).unwrap()).collect();
        assert_eq!(numbers, vec![0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0], "30 frames sampled every 4th");
        assert_eq!(result.get("sampled").and_then(Value::as_f64), Some(8.0));
        assert_eq!(result.get("totalFrames").and_then(Value::as_f64), Some(30.0));
        assert_eq!(result.get("fps").and_then(Value::as_f64), Some(10.0));
        assert_eq!(result.get("everyN").and_then(Value::as_f64), Some(4.0));
        let timing = result.get("timing").unwrap();
        assert!(timing.get("msPerFrame").and_then(Value::as_f64).unwrap() > 0.0);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_rotated_phone_clip_reports_the_frame_size_ffmpeg_decodes() {
        let Some(mut yolo) = engine() else { return };
        let dir = scratch("rotate");
        let Some(clip) = synthetic_clip(&dir, &["-display_rotation:v", "90"]) else {
            eprintln!("SKIP: this ffmpeg cannot write a display-matrix rotation (-display_rotation needs ffmpeg 6+)");
            return;
        };
        let result = detect(&mut yolo, &clip, request(15, Params::default(), 100)).unwrap();
        assert_eq!(result.get("width").and_then(Value::as_f64), Some(180.0));
        assert_eq!(result.get("height").and_then(Value::as_f64), Some(320.0));
        assert_eq!(result.get("sampled").and_then(Value::as_f64), Some(2.0));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_clip_over_the_frame_limit_is_rejected_before_any_inference() {
        let Some(mut yolo) = engine() else { return };
        let dir = scratch("limit");
        let Some(clip) = synthetic_clip(&dir, &[]) else {
            eprintln!("SKIP: could not generate a test clip with ffmpeg");
            return;
        };
        let started = Instant::now();
        let err = detect(&mut yolo, &clip, request(1, Params::default(), 10)).unwrap_err();
        assert!(matches!(&err, DetectError::TooMany(m) if m.contains("every_n")), "{err:?}");
        assert!(started.elapsed() < Duration::from_secs(2));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn garbage_uploads_are_invalid_not_internal_errors() {
        let Some(mut yolo) = engine() else { return };
        let dir = scratch("garbage");
        let path = dir.join("junk.bin");
        fs::write(&path, vec![0x5au8; 4096]).unwrap();
        let err = detect(&mut yolo, &path, request(1, Params::default(), 10)).unwrap_err();
        assert!(matches!(err, DetectError::Invalid(_)), "{err:?}");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn progress_ends_completed_and_a_cancelled_job_stops() {
        let Some(mut yolo) = engine() else { return };
        let dir = scratch("progress");
        let Some(clip) = synthetic_clip(&dir, &[]) else {
            eprintln!("SKIP: could not generate a test clip with ffmpeg");
            return;
        };
        let progress = Progress::default();
        let mut req = request(3, Params::default(), 100);
        req.detection_id = Some("det_test".to_string());
        let job = Job { input: clip.clone(), request: req, deadline: Instant::now() + Duration::from_secs(60), cancel: Arc::new(AtomicBool::new(false)) };
        yolo.detect(&job, &progress).unwrap();
        let done = progress.get("det_test").unwrap();
        assert_eq!((done.status, done.frame, done.total), ("completed", 30, 30));

        let cancelled = Job {
            input: clip,
            request: request(1, Params::default(), 100),
            deadline: Instant::now() + Duration::from_secs(60),
            cancel: Arc::new(AtomicBool::new(true)),
        };
        assert_eq!(yolo.detect(&cancelled, &progress), Err(DetectError::Cancelled));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn the_repo_video_yields_person_detections_end_to_end() {
        let Some(mut yolo) = engine() else { return };
        let video = root().join("../backend/yolo/assets/video.mov");
        if !video.is_file() {
            eprintln!("SKIP: {} is not present", video.display());
            return;
        }
        // 900 frames sampled every 150th: six inferences.
        let result = detect(&mut yolo, &video, request(150, Params { conf: 0.1, ..Params::default() }, 100)).unwrap();
        assert_eq!(result.get("sampled").and_then(Value::as_f64), Some(6.0));
        assert_eq!(result.get("totalFrames").and_then(Value::as_f64), Some(900.0));
        let frames = result.get("frames").and_then(Value::as_array).unwrap();
        let people: usize = frames
            .iter()
            .map(|f| dets(f.get("detections").unwrap()).iter().filter(|d| d.class == 0).count())
            .sum();
        assert!(people > 0, "the test frame has people in it");
    }

    /// Minimal HTTP client for the round-trip test.
    fn http(addr: std::net::SocketAddr, head: &str, body: &[u8]) -> (u16, String) {
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
        s.write_all(head.as_bytes()).unwrap();
        s.write_all(body).unwrap();
        let mut out = Vec::new();
        let _ = s.read_to_end(&mut out);
        let text = String::from_utf8_lossy(&out).into_owned();
        let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
        (head.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0), body.to_string())
    }

    #[test]
    fn http_round_trip_with_the_real_model() {
        let Some(yolo) = engine() else { return };
        let cfg = Config { port: 0, tmp_dir: scratch("http"), ..Config::default() };
        let server = Server::bind("127.0.0.1:0", cfg, Box::new(yolo), "test".to_string()).unwrap();
        let addr = server.spawn().unwrap();
        let (status, body) = http(addr, "GET /health HTTP/1.1\r\nHost: t\r\n\r\n", b"");
        assert_eq!(status, 200, "{body}");
        let image = fs::read(fixture("frame20.jpg")).unwrap();
        let head = format!("POST /detect?conf=0.5&detection_id=abc HTTP/1.1\r\nHost: t\r\nContent-Length: {}\r\n\r\n", image.len());
        let (status, body) = http(addr, &head, &image);
        assert_eq!(status, 200, "{body}");
        let result = json::parse(&body).unwrap();
        let got = first_frame_dets(&result);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].class, 0);
        let (status, body) = http(addr, "GET /progress/abc HTTP/1.1\r\n\r\n", b"");
        assert_eq!(status, 200);
        assert!(body.contains("\"completed\""), "{body}");
        let (status, body) = http(addr, "POST /detect HTTP/1.1\r\nContent-Length: 4\r\n\r\nabcd", b"");
        assert_eq!(status, 422, "{body}");
    }
}
