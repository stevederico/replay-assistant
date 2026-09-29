//! Routing, limits and the single inference worker.
//!
//! Requests are handled on short-lived threads (capped), but exactly one thread
//! runs inference. A `/detect` request streams its body to a temp file, hands
//! the job to the worker over a bounded queue, and waits for the answer with a
//! deadline. Anything beyond the in-flight cap is turned away with 503 and
//! `Retry-After`, so memory and disk stay bounded.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::classes::NUM_CLASSES;
use crate::detect::{valid_detection_id, DetectError, DetectRequest, Detector, Job, Progress, MODEL_NAME};
use crate::http::{self, Head, HttpError, Response};
use crate::json::Writer;
use crate::yolo::Params;

/// How long a client may take to send the request head.
const HEAD_TIMEOUT: Duration = Duration::from_secs(10);
/// Longest silence tolerated while an upload is streaming.
const BODY_READ_TIMEOUT: Duration = Duration::from_secs(30);
/// How long a response write may block.
const WRITE_TIMEOUT: Duration = Duration::from_secs(30);
/// Seconds a turned-away client is told to wait.
const RETRY_AFTER_SECS: u32 = 5;
/// After an early error response, this much unread upload is drained so the
/// client sees the response instead of a connection reset.
const DRAIN_LIMIT: usize = 256 * 1024;
/// Highest `every_n` accepted.
const MAX_EVERY_N: u32 = 10_000;
/// Highest polygon simplification tolerance accepted, in model pixels.
const MAX_SIMPLIFY: f32 = 20.0;

const LEGACY_MESSAGE: &str =
    "This route does not exist in yolo-server: live MJPEG and the Python service's routes were removed. POST the video or image bytes to /detect instead.";

/// Server limits, all overridable by environment variables (see [`Config::from_lookup`]).
#[derive(Debug, Clone)]
pub struct Config {
    /// TCP port to listen on.
    pub port: u16,
    /// Largest accepted upload in bytes.
    pub max_body_bytes: u64,
    /// Largest number of sampled frames one request may produce.
    pub max_frames: usize,
    /// `every_n` used when the query does not give one.
    pub default_every_n: u32,
    /// `/detect` requests allowed at once (uploading, queued or running).
    pub max_inflight: usize,
    /// Open connections allowed at once.
    pub max_connections: usize,
    /// Total time a `/detect` request may take, queue wait included.
    pub request_timeout: Duration,
    /// Time allowed to receive the upload.
    pub upload_timeout: Duration,
    /// Directory for uploaded files while they are processed.
    pub tmp_dir: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            port: 5001,
            max_body_bytes: 512 * 1024 * 1024,
            max_frames: 3000,
            default_every_n: 1,
            max_inflight: 3,
            max_connections: 32,
            request_timeout: Duration::from_secs(600),
            upload_timeout: Duration::from_secs(300),
            tmp_dir: std::env::temp_dir().join("yolo-server"),
        }
    }
}

impl Config {
    /// Build a config from environment-style lookups, falling back to defaults.
    ///
    /// Variables: `PORT`, `YOLO_MAX_BODY_MB`, `YOLO_MAX_FRAMES`, `YOLO_EVERY_N`,
    /// `YOLO_MAX_INFLIGHT`, `YOLO_REQUEST_TIMEOUT_SECS`, `YOLO_UPLOAD_TIMEOUT_SECS`,
    /// `YOLO_TMP_DIR`.
    ///
    /// # Errors
    /// A message naming the variable when a value is present but not a positive number.
    pub fn from_lookup(get: &dyn Fn(&str) -> Option<String>) -> Result<Config, String> {
        fn positive<T: std::str::FromStr + PartialOrd + Default>(get: &dyn Fn(&str) -> Option<String>, name: &str, fallback: T) -> Result<T, String> {
            match get(name).filter(|v| !v.trim().is_empty()) {
                None => Ok(fallback),
                Some(raw) => raw
                    .trim()
                    .parse::<T>()
                    .ok()
                    .filter(|n| *n > T::default())
                    .ok_or_else(|| format!("{name} must be a positive number, got {raw:?}")),
            }
        }
        let d = Config::default();
        Ok(Config {
            port: positive(get, "PORT", d.port)?,
            max_body_bytes: positive::<u64>(get, "YOLO_MAX_BODY_MB", d.max_body_bytes / (1024 * 1024))? * 1024 * 1024,
            max_frames: positive(get, "YOLO_MAX_FRAMES", d.max_frames)?,
            default_every_n: positive::<u32>(get, "YOLO_EVERY_N", d.default_every_n)?.min(MAX_EVERY_N),
            max_inflight: positive(get, "YOLO_MAX_INFLIGHT", d.max_inflight)?,
            max_connections: d.max_connections,
            request_timeout: Duration::from_secs(positive(get, "YOLO_REQUEST_TIMEOUT_SECS", d.request_timeout.as_secs())?),
            upload_timeout: Duration::from_secs(positive(get, "YOLO_UPLOAD_TIMEOUT_SECS", d.upload_timeout.as_secs())?),
            tmp_dir: get("YOLO_TMP_DIR").filter(|v| !v.trim().is_empty()).map(PathBuf::from).unwrap_or(d.tmp_dir),
        })
    }
}

/// Validate the query string of `POST /detect`.
///
/// Parameters: `every_n`, `conf`, `iou`, `classes` (comma-separated ids),
/// `max_frames`, `simplify`, `masks` (`0`/`false` to skip polygons) and
/// `detection_id`.
///
/// # Errors
/// 400 naming the parameter that is out of range or unparsable.
pub fn parse_detect_request(head: &Head, cfg: &Config) -> Result<DetectRequest, HttpError> {
    fn bad(name: &str, why: &str) -> HttpError {
        HttpError::new(400, format!("query parameter {name} {why}"))
    }
    fn number<T: std::str::FromStr>(head: &Head, name: &str, default: T) -> Result<T, HttpError> {
        match head.query_param(name) {
            None | Some("") => Ok(default),
            Some(raw) => raw.trim().parse().map_err(|_| bad(name, "is not a number")),
        }
    }
    let every_n: u32 = number(head, "every_n", cfg.default_every_n)?;
    if !(1..=MAX_EVERY_N).contains(&every_n) {
        return Err(bad("every_n", &format!("must be between 1 and {MAX_EVERY_N}")));
    }
    let conf: f32 = number(head, "conf", Params::default().conf)?;
    if !conf.is_finite() || !(0.001..=1.0).contains(&conf) {
        return Err(bad("conf", "must be between 0.001 and 1"));
    }
    let iou: f32 = number(head, "iou", Params::default().iou)?;
    if !iou.is_finite() || iou <= 0.0 || iou > 1.0 {
        return Err(bad("iou", "must be greater than 0 and at most 1"));
    }
    let simplify: f32 = number(head, "simplify", Params::default().simplify)?;
    if !simplify.is_finite() || !(0.0..=MAX_SIMPLIFY).contains(&simplify) {
        return Err(bad("simplify", &format!("must be between 0 and {MAX_SIMPLIFY}")));
    }
    let max_frames: usize = number(head, "max_frames", cfg.max_frames)?;
    if max_frames == 0 || max_frames > cfg.max_frames {
        return Err(bad("max_frames", &format!("must be between 1 and {}", cfg.max_frames)));
    }
    let masks = match head.query_param("masks") {
        None | Some("") | Some("1") | Some("true") => true,
        Some("0") | Some("false") => false,
        Some(_) => return Err(bad("masks", "must be 0, 1, true or false")),
    };
    let allowed = match head.query_param("classes").filter(|c| !c.is_empty()) {
        None => None,
        Some(list) => {
            let mut keep = vec![false; NUM_CLASSES];
            for item in list.split(',') {
                let id: usize = item.trim().parse().map_err(|_| bad("classes", "must be comma-separated class ids"))?;
                *keep.get_mut(id).ok_or_else(|| bad("classes", &format!("ids must be below {NUM_CLASSES}")))? = true;
            }
            Some(keep)
        }
    };
    let detection_id = match head.query_param("detection_id").filter(|v| !v.is_empty()) {
        None => None,
        Some(id) if valid_detection_id(id) => Some(id.to_string()),
        Some(_) => return Err(bad("detection_id", "may only use letters, digits, - and _ (64 characters at most)")),
    };
    Ok(DetectRequest { every_n, params: Params { conf, iou, allowed, simplify, masks, ..Params::default() }, max_frames, detection_id })
}

/// Print a log line to stderr.
pub fn log(msg: &str) {
    eprintln!("yolo-server: {msg}");
}

/// Work handed to the inference thread.
struct Work {
    job: Job,
    reply: Sender<Result<String, DetectError>>,
}

/// State shared by every connection thread.
struct App {
    cfg: Config,
    queue: SyncSender<Work>,
    progress: Arc<Progress>,
    inflight: AtomicUsize,
    connections: AtomicUsize,
    temp_counter: AtomicU64,
    ort_version: String,
}

/// Counts one slot of a bounded resource and gives it back on drop.
struct Slot<'a>(&'a AtomicUsize);

impl<'a> Slot<'a> {
    /// Take a slot if fewer than `max` are in use.
    fn acquire(counter: &'a AtomicUsize, max: usize) -> Option<Slot<'a>> {
        counter.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| (n < max).then_some(n + 1)).ok()?;
        Some(Slot(counter))
    }
}

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// An uploaded body on disk, removed when dropped.
struct TempFile {
    path: PathBuf,
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Delete uploads a previous run left behind (a crash or SIGKILL skips `Drop`).
fn clean_stale_temp(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("yolo-") && name.ends_with(".bin") {
            let _ = fs::remove_file(entry.path());
        }
    }
}

/// A bound listener plus the running inference worker.
pub struct Server {
    listener: TcpListener,
    app: Arc<App>,
    _worker: JoinHandle<()>,
}

impl Server {
    /// Bind `addr` and start the inference worker that owns `detector`.
    ///
    /// # Errors
    /// I/O errors from binding or from preparing the temp directory.
    pub fn bind(addr: &str, cfg: Config, detector: Box<dyn Detector>, ort_version: String) -> std::io::Result<Server> {
        fs::create_dir_all(&cfg.tmp_dir)?;
        clean_stale_temp(&cfg.tmp_dir);
        let listener = TcpListener::bind(addr)?;
        let (queue, rx) = mpsc::sync_channel::<Work>(cfg.max_inflight);
        let progress = Arc::new(Progress::default());
        let worker_progress = Arc::clone(&progress);
        let worker = thread::Builder::new().name("inference".into()).spawn(move || worker_loop(rx, detector, worker_progress))?;
        let app = Arc::new(App {
            cfg,
            queue,
            progress,
            inflight: AtomicUsize::new(0),
            connections: AtomicUsize::new(0),
            temp_counter: AtomicU64::new(0),
            ort_version,
        });
        Ok(Server { listener, app, _worker: worker })
    }

    /// Address the listener is bound to (useful with port 0).
    ///
    /// # Errors
    /// If the socket has no local address.
    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// Accept connections until the listener fails. Blocks.
    pub fn serve(self) {
        for stream in self.listener.incoming() {
            let Ok(stream) = stream else { continue };
            let app = Arc::clone(&self.app);
            let Some(slot) = acquire_connection(&app) else {
                turn_away(stream);
                continue;
            };
            let spawned = thread::Builder::new().name("conn".into()).spawn(move || {
                let _slot = slot;
                handle_connection(&app, stream);
            });
            if let Err(e) = spawned {
                log(&format!("cannot start a connection thread: {e}"));
            }
        }
    }

    /// Run [`Server::serve`] on a background thread and return the bound address.
    ///
    /// # Errors
    /// If the socket has no local address or the thread cannot start.
    pub fn spawn(self) -> std::io::Result<SocketAddr> {
        let addr = self.local_addr()?;
        thread::Builder::new().name("accept".into()).spawn(move || self.serve())?;
        Ok(addr)
    }
}

/// An owned connection slot that lives inside the connection thread.
struct OwnedSlot(Arc<App>);

impl Drop for OwnedSlot {
    fn drop(&mut self) {
        self.0.connections.fetch_sub(1, Ordering::AcqRel);
    }
}

fn acquire_connection(app: &Arc<App>) -> Option<OwnedSlot> {
    app.connections
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| (n < app.cfg.max_connections).then_some(n + 1))
        .ok()?;
    Some(OwnedSlot(Arc::clone(app)))
}

/// Answer 503 to a connection we have no capacity for.
fn turn_away(mut stream: TcpStream) {
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
    let res = Response::error(503, "the detection service is at its connection limit; retry shortly")
        .with_header("Retry-After", RETRY_AFTER_SECS.to_string());
    let _ = res.write_to(&mut stream);
}

fn worker_loop(rx: Receiver<Work>, mut detector: Box<dyn Detector>, progress: Arc<Progress>) {
    for work in rx {
        if work.job.cancel.load(Ordering::Relaxed) {
            let _ = work.reply.send(Err(DetectError::Cancelled));
            continue;
        }
        let result = catch_unwind(AssertUnwindSafe(|| detector.detect(&work.job, &progress)))
            .unwrap_or_else(|_| Err(DetectError::Internal("the detector panicked".to_string())));
        // The requester may have timed out and gone; nobody to tell then.
        let _ = work.reply.send(result);
    }
}

fn handle_connection(app: &App, mut stream: TcpStream) {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(HEAD_TIMEOUT));
    let _ = stream.set_write_timeout(Some(WRITE_TIMEOUT));
    let response = match http::read_head(&mut stream) {
        Ok((head, leftover)) => route(app, &head, leftover, &mut stream),
        Err(e) => Response::error(e.status, &e.message),
    };
    let _ = response.write_to(&mut stream);
    close_politely(&mut stream);
}

/// Half-close, then swallow a bounded amount of unread input so an early error
/// response is not lost to a connection reset.
fn close_politely(stream: &mut TcpStream) {
    let _ = stream.shutdown(Shutdown::Write);
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    let mut sink = [0u8; 8192];
    let mut drained = 0;
    while drained < DRAIN_LIMIT {
        match stream.read(&mut sink) {
            Ok(0) | Err(_) => break,
            Ok(n) => drained += n,
        }
    }
}

fn route(app: &App, head: &Head, leftover: Vec<u8>, stream: &mut TcpStream) -> Response {
    let path = head.path.as_str();
    let method = head.method.as_str();
    let is_legacy = matches!(path, "/video_feed" | "/infer" | "/upload_video" | "/analyze_frames" | "/analyze_release")
        || path.starts_with("/detection_progress/");
    if is_legacy {
        return Response::error(501, LEGACY_MESSAGE);
    }
    match (method, path) {
        ("GET", "/health") => health(app),
        ("POST", "/detect") => detect(app, head, leftover, stream),
        ("GET", p) if p.starts_with("/progress/") => progress(app, &p["/progress/".len()..]),
        (_, "/health") | (_, "/detect") => Response::error(405, "method not allowed for this route")
            .with_header("Allow", if path == "/health" { "GET" } else { "POST" }),
        (_, p) if p.starts_with("/progress/") => Response::error(405, "method not allowed for this route").with_header("Allow", "GET"),
        _ => Response::error(404, "no such route"),
    }
}

fn health(app: &App) -> Response {
    let mut w = Writer::new();
    w.begin_object();
    w.key("status");
    w.string("ok");
    w.key("model");
    w.string(MODEL_NAME);
    w.key("onnxruntime");
    w.string(&app.ort_version);
    w.key("inflight");
    w.int(app.inflight.load(Ordering::Relaxed) as i64);
    w.end_object();
    Response::json(200, w.finish())
}

fn progress(app: &App, id: &str) -> Response {
    let Some(entry) = valid_detection_id(id).then(|| app.progress.get(id)).flatten() else {
        return Response::error(404, "no such detection");
    };
    let mut w = Writer::new();
    w.begin_object();
    w.key("frame");
    w.int(entry.frame as i64);
    w.key("total");
    w.int(entry.total as i64);
    w.key("status");
    w.string(entry.status);
    w.end_object();
    Response::json(200, w.finish())
}

fn busy() -> Response {
    Response::error(503, "detection is busy with other requests; retry shortly").with_header("Retry-After", RETRY_AFTER_SECS.to_string())
}

fn detect_error_response(e: &DetectError) -> Response {
    match e {
        DetectError::Invalid(m) | DetectError::TooMany(m) => Response::error(422, m),
        DetectError::TimedOut => Response::error(504, "detection took too long and was stopped; raise every_n or send a shorter clip"),
        DetectError::Cancelled => Response::error(503, "detection was cancelled"),
        DetectError::Internal(m) => {
            log(&format!("detection failed: {m}"));
            Response::error(500, "detection failed on the server")
        }
    }
}

fn detect(app: &App, head: &Head, leftover: Vec<u8>, stream: &mut TcpStream) -> Response {
    let started = Instant::now();
    if head.header("content-type").is_some_and(|t| t.to_ascii_lowercase().starts_with("multipart/")) {
        return Response::error(415, "send the raw video or image bytes as the request body, not multipart");
    }
    let len = match head.content_length() {
        Ok(Some(0)) => return Response::error(400, "the request body is empty"),
        Ok(Some(n)) => n,
        Ok(None) => return Response::error(411, "a Content-Length header is required"),
        Err(e) => return Response::error(e.status, &e.message),
    };
    if len > app.cfg.max_body_bytes {
        return Response::error(413, &format!("the upload is larger than the {} MB limit", app.cfg.max_body_bytes / (1024 * 1024)));
    }
    let request = match parse_detect_request(head, &app.cfg) {
        Ok(r) => r,
        Err(e) => return Response::error(e.status, &e.message),
    };
    let Some(_slot) = Slot::acquire(&app.inflight, app.cfg.max_inflight) else {
        return busy();
    };
    if head.expects_continue() && stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").is_err() {
        return Response::error(400, "connection lost");
    }
    let temp = match store_upload(app, leftover, len, stream) {
        Ok(t) => t,
        Err(e) => return Response::error(e.status, &e.message),
    };

    let cancel = Arc::new(AtomicBool::new(false));
    let deadline = started + app.cfg.request_timeout;
    let job = Job { input: temp.path.clone(), request, deadline, cancel: Arc::clone(&cancel) };
    let (reply, wait) = mpsc::channel();
    match app.queue.try_send(Work { job, reply }) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => return busy(),
        Err(TrySendError::Disconnected(_)) => return Response::error(500, "the inference worker has stopped"),
    }
    match wait.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
        Ok(Ok(body)) => Response::json(200, body),
        Ok(Err(e)) => detect_error_response(&e),
        Err(RecvTimeoutError::Timeout) => {
            cancel.store(true, Ordering::Relaxed);
            detect_error_response(&DetectError::TimedOut)
        }
        Err(RecvTimeoutError::Disconnected) => Response::error(500, "the inference worker stopped while detecting"),
    }
}

/// Stream the request body into a fresh temp file.
fn store_upload(app: &App, leftover: Vec<u8>, len: u64, stream: &mut TcpStream) -> Result<TempFile, HttpError> {
    let n = app.temp_counter.fetch_add(1, Ordering::Relaxed);
    let path = app.cfg.tmp_dir.join(format!("yolo-{}-{n}.bin", std::process::id()));
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| HttpError::new(500, format!("cannot create a temp file: {e}")))?;
    let temp = TempFile { path };
    let _ = stream.set_read_timeout(Some(BODY_READ_TIMEOUT));
    http::copy_body(stream, &leftover, len, &mut file, Instant::now() + app.cfg.upload_timeout)?;
    file.flush().map_err(|e| HttpError::new(500, format!("cannot store upload: {e}")))?;
    Ok(temp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{self, Value};
    use std::sync::Mutex;

    /// Replies with the uploaded bytes' length, or misbehaves on demand.
    struct Fake {
        behaviour: Arc<Mutex<Behaviour>>,
    }

    #[derive(Clone)]
    enum Behaviour {
        Echo,
        Fail(DetectError),
        Sleep(Duration),
        Panic,
    }

    impl Detector for Fake {
        fn detect(&mut self, job: &Job, progress: &Progress) -> Result<String, DetectError> {
            if let Some(id) = &job.request.detection_id {
                progress.set(id, crate::detect::ProgressEntry { frame: 3, total: 9, status: "processing" });
            }
            let behaviour = self.behaviour.lock().unwrap().clone();
            match behaviour {
                Behaviour::Echo => {
                    let bytes = fs::read(&job.input).map_err(|e| DetectError::Internal(e.to_string()))?;
                    Ok(format!(
                        "{{\"bytes\":{},\"everyN\":{},\"conf\":{},\"masks\":{}}}",
                        bytes.len(),
                        job.request.every_n,
                        job.request.params.conf,
                        job.request.params.masks
                    ))
                }
                Behaviour::Fail(e) => Err(e),
                Behaviour::Sleep(d) => {
                    let end = Instant::now() + d;
                    while Instant::now() < end && !job.cancel.load(Ordering::Relaxed) {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(DetectError::TimedOut)
                }
                Behaviour::Panic => panic!("boom"),
            }
        }
    }

    fn test_config(name: &str) -> Config {
        Config {
            port: 0,
            max_body_bytes: 1024,
            max_frames: 100,
            max_inflight: 2,
            tmp_dir: std::env::temp_dir().join(format!("yolo-server-test-{}-{name}", std::process::id())),
            ..Config::default()
        }
    }

    fn start(_name: &str, cfg: Config, behaviour: Behaviour) -> (SocketAddr, Arc<Mutex<Behaviour>>, PathBuf) {
        let tmp = cfg.tmp_dir.clone();
        let shared = Arc::new(Mutex::new(behaviour));
        let fake = Fake { behaviour: Arc::clone(&shared) };
        let server = Server::bind("127.0.0.1:0", cfg, Box::new(fake), "test".into()).unwrap();
        (server.spawn().unwrap(), shared, tmp)
    }

    /// Send a raw request and return `(status, header block, body)`.
    fn raw(addr: SocketAddr, head: &str, body: &[u8]) -> (u16, String, String) {
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        s.write_all(head.as_bytes()).unwrap();
        s.write_all(body).unwrap();
        let mut out = Vec::new();
        let _ = s.read_to_end(&mut out);
        let text = String::from_utf8_lossy(&out).into_owned();
        let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
        let status = head.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
        (status, head.to_string(), body.to_string())
    }

    fn post(addr: SocketAddr, query: &str, body: &[u8]) -> (u16, String, String) {
        let head = format!("POST /detect{query} HTTP/1.1\r\nHost: t\r\nContent-Length: {}\r\n\r\n", body.len());
        raw(addr, &head, body)
    }

    fn error_of(body: &str) -> String {
        json::parse(body).ok().and_then(|v| v.get("error").and_then(Value::as_str).map(str::to_string)).unwrap_or_default()
    }

    fn query_head(q: &str) -> Head {
        http::parse_head(&format!("POST /detect?{q} HTTP/1.1")).unwrap()
    }

    // ---- config ----

    #[test]
    fn config_uses_defaults_when_nothing_is_set() {
        let c = Config::from_lookup(&|_| None).unwrap();
        assert_eq!((c.port, c.max_frames, c.max_inflight, c.default_every_n), (5001, 3000, 3, 1));
        assert_eq!(c.max_body_bytes, 512 * 1024 * 1024);
    }

    #[test]
    fn config_reads_overrides_and_rejects_bad_numbers() {
        let env = |k: &str| match k {
            "PORT" => Some("8123".to_string()),
            "YOLO_MAX_BODY_MB" => Some("64".to_string()),
            _ => None,
        };
        let c = Config::from_lookup(&env).unwrap();
        assert_eq!((c.port, c.max_body_bytes), (8123, 64 * 1024 * 1024));
        for bad in ["0", "-3", "abc", "1.5"] {
            let env = |k: &str| (k == "YOLO_MAX_FRAMES").then(|| bad.to_string());
            assert!(Config::from_lookup(&env).unwrap_err().contains("YOLO_MAX_FRAMES"), "{bad}");
        }
    }

    // ---- query parsing ----

    #[test]
    fn detect_request_defaults_come_from_config() {
        let r = parse_detect_request(&query_head(""), &Config::default()).unwrap();
        assert_eq!((r.every_n, r.max_frames), (1, 3000));
        assert_eq!((r.params.conf, r.params.iou), (0.25, 0.7));
        assert!(r.params.masks && r.params.allowed.is_none() && r.detection_id.is_none());
    }

    #[test]
    fn detect_request_reads_every_parameter() {
        let r = parse_detect_request(
            &query_head("every_n=3&conf=0.1&iou=0.45&classes=0,32&max_frames=50&simplify=0&masks=0&detection_id=det_1"),
            &Config::default(),
        )
        .unwrap();
        assert_eq!((r.every_n, r.max_frames), (3, 50));
        assert_eq!((r.params.conf, r.params.iou, r.params.simplify), (0.1, 0.45, 0.0));
        assert!(!r.params.masks);
        let allowed = r.params.allowed.unwrap();
        assert!(allowed[0] && allowed[32] && !allowed[1]);
        assert_eq!(r.detection_id.as_deref(), Some("det_1"));
    }

    #[test]
    fn detect_request_rejects_out_of_range_values_by_name() {
        let cfg = Config::default();
        for (q, name) in [
            ("every_n=0", "every_n"),
            ("every_n=99999", "every_n"),
            ("every_n=x", "every_n"),
            ("conf=0", "conf"),
            ("conf=1.5", "conf"),
            ("conf=NaN", "conf"),
            ("iou=0", "iou"),
            ("iou=2", "iou"),
            ("classes=80", "classes"),
            ("classes=a", "classes"),
            ("max_frames=0", "max_frames"),
            ("max_frames=999999", "max_frames"),
            ("simplify=-1", "simplify"),
            ("masks=maybe", "masks"),
            ("detection_id=../x", "detection_id"),
        ] {
            let err = parse_detect_request(&query_head(q), &cfg).unwrap_err();
            assert_eq!(err.status, 400, "{q}");
            assert!(err.message.contains(name), "{q}: {}", err.message);
        }
    }

    // ---- routes over a real socket ----

    #[test]
    fn health_reports_model_and_runtime() {
        let (addr, _, _) = start("health", test_config("health"), Behaviour::Echo);
        let (status, _, body) = raw(addr, "GET /health HTTP/1.1\r\nHost: t\r\n\r\n", b"");
        assert_eq!(status, 200);
        let v = json::parse(&body).unwrap();
        assert_eq!(v.get("status").and_then(Value::as_str), Some("ok"));
        assert_eq!(v.get("model").and_then(Value::as_str), Some(MODEL_NAME));
        assert_eq!(v.get("onnxruntime").and_then(Value::as_str), Some("test"));
    }

    #[test]
    fn unknown_routes_are_404_and_wrong_methods_are_405() {
        let (addr, _, _) = start("404", test_config("404"), Behaviour::Echo);
        assert_eq!(raw(addr, "GET /nope HTTP/1.1\r\n\r\n", b"").0, 404);
        let (status, headers, _) = raw(addr, "GET /detect HTTP/1.1\r\n\r\n", b"");
        assert_eq!(status, 405);
        assert!(headers.contains("Allow: POST"));
        assert_eq!(raw(addr, "POST /health HTTP/1.1\r\nContent-Length: 0\r\n\r\n", b"").0, 405);
    }

    #[test]
    fn old_python_routes_answer_501_with_a_clear_message() {
        let (addr, _, _) = start("501", test_config("501"), Behaviour::Echo);
        for path in ["/video_feed", "/infer", "/upload_video", "/analyze_frames", "/analyze_release", "/detection_progress/abc"] {
            let (status, _, body) = raw(addr, &format!("GET {path} HTTP/1.1\r\n\r\n"), b"");
            assert_eq!(status, 501, "{path}");
            assert!(error_of(&body).contains("/detect"), "{path}: {body}");
        }
    }

    #[test]
    fn malformed_requests_get_400_without_killing_the_server() {
        let (addr, _, _) = start("bad", test_config("bad"), Behaviour::Echo);
        assert_eq!(raw(addr, "garbage\r\n\r\n", b"").0, 400);
        assert_eq!(raw(addr, "GET /health HTTP/1.1\r\n\r\n", b"").0, 200);
    }

    #[test]
    fn detect_stores_the_upload_runs_the_job_and_cleans_up() {
        let (addr, _, tmp) = start("ok", test_config("ok"), Behaviour::Echo);
        let (status, _, body) = post(addr, "?every_n=3&conf=0.1&masks=0", b"0123456789");
        assert_eq!(status, 200, "{body}");
        let v = json::parse(&body).unwrap();
        assert_eq!(v.get("bytes").and_then(Value::as_f64), Some(10.0));
        assert_eq!(v.get("everyN").and_then(Value::as_f64), Some(3.0));
        assert_eq!(v.get("masks"), Some(&Value::Bool(false)));
        let leftovers = fs::read_dir(&tmp).unwrap().count();
        assert_eq!(leftovers, 0, "temp upload should be deleted");
    }

    #[test]
    fn detect_handles_expect_100_continue() {
        let (addr, _, _) = start("continue", test_config("continue"), Behaviour::Echo);
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        s.write_all(b"POST /detect HTTP/1.1\r\nContent-Length: 4\r\nExpect: 100-continue\r\n\r\n").unwrap();
        let mut buf = [0u8; 25];
        s.read_exact(&mut buf).unwrap();
        assert!(String::from_utf8_lossy(&buf).starts_with("HTTP/1.1 100 Continue"));
        s.write_all(b"abcd").unwrap();
        let mut rest = String::new();
        s.read_to_string(&mut rest).unwrap();
        assert!(rest.contains("200 OK") && rest.contains("\"bytes\":4"), "{rest}");
    }

    #[test]
    fn detect_rejects_bad_uploads_before_running_anything() {
        let (addr, _, tmp) = start("reject", test_config("reject"), Behaviour::Echo);
        assert_eq!(raw(addr, "POST /detect HTTP/1.1\r\n\r\n", b"").0, 411);
        assert_eq!(post(addr, "", b"").0, 400);
        let (status, _, body) = post(addr, "", &vec![0u8; 2048]);
        assert_eq!(status, 413, "{body}");
        let multipart = "POST /detect HTTP/1.1\r\nContent-Type: multipart/form-data; boundary=x\r\nContent-Length: 3\r\n\r\nabc";
        assert_eq!(raw(addr, multipart, b"").0, 415);
        let (status, _, body) = post(addr, "?conf=9", b"abc");
        assert_eq!(status, 400);
        assert!(error_of(&body).contains("conf"));
        let chunked = "POST /detect HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n";
        assert_eq!(raw(addr, chunked, b"").0, 411);
        assert_eq!(fs::read_dir(&tmp).unwrap().count(), 0, "rejected uploads leave no files");
    }

    #[test]
    fn detect_reports_a_short_upload() {
        let (addr, _, _) = start("short", test_config("short"), Behaviour::Echo);
        let mut s = TcpStream::connect(addr).unwrap();
        s.write_all(b"POST /detect HTTP/1.1\r\nContent-Length: 100\r\n\r\nabc").unwrap();
        s.shutdown(Shutdown::Write).unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        assert!(out.starts_with("HTTP/1.1 400"), "{out}");
    }

    #[test]
    fn detector_errors_map_to_http_statuses() {
        let (addr, behaviour, _) = start("errors", test_config("errors"), Behaviour::Echo);
        for (fail, status) in [
            (DetectError::Invalid("not a video".into()), 422),
            (DetectError::TooMany("too many frames".into()), 422),
            (DetectError::TimedOut, 504),
            (DetectError::Internal("secret detail".into()), 500),
        ] {
            *behaviour.lock().unwrap() = Behaviour::Fail(fail);
            let (got, _, body) = post(addr, "", b"abc");
            assert_eq!(got, status, "{body}");
            assert!(!body.contains("secret detail"), "internal details must not leak: {body}");
        }
    }

    #[test]
    fn a_panicking_detector_is_a_500_and_the_worker_survives() {
        let (addr, behaviour, _) = start("panic", test_config("panic"), Behaviour::Panic);
        assert_eq!(post(addr, "", b"abc").0, 500);
        *behaviour.lock().unwrap() = Behaviour::Echo;
        assert_eq!(post(addr, "", b"abc").0, 200);
    }

    #[test]
    fn slow_detection_times_out_with_504_and_cancels_the_job() {
        let mut cfg = test_config("timeout");
        cfg.request_timeout = Duration::from_millis(300);
        let (addr, _, _) = start("timeout", cfg, Behaviour::Sleep(Duration::from_secs(30)));
        let began = Instant::now();
        let (status, _, body) = post(addr, "", b"abc");
        assert_eq!(status, 504, "{body}");
        assert!(began.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn requests_beyond_the_inflight_cap_get_503_with_retry_after() {
        let mut cfg = test_config("busy");
        cfg.max_inflight = 1;
        let tmp = cfg.tmp_dir.clone();
        let shared = Arc::new(Mutex::new(Behaviour::Sleep(Duration::from_secs(3))));
        let fake = Fake { behaviour: Arc::clone(&shared) };
        let server = Server::bind("127.0.0.1:0", cfg, Box::new(fake), "test".into()).unwrap();
        let addr = server.spawn().unwrap();
        let first = thread::spawn(move || post(addr, "", b"abc"));
        // Let the first request take the only slot.
        thread::sleep(Duration::from_millis(400));
        let (status, headers, body) = post(addr, "", b"abc");
        assert_eq!(status, 503, "{body}");
        assert!(headers.contains("Retry-After: 5"));
        assert!(error_of(&body).contains("busy"));
        assert_eq!(first.join().unwrap().0, 504, "the fake ends as TimedOut once its sleep is over");
        assert_eq!(fs::read_dir(&tmp).unwrap().count(), 0);
    }

    #[test]
    fn progress_is_served_for_known_ids_only() {
        let (addr, _, _) = start("progress", test_config("progress"), Behaviour::Echo);
        assert_eq!(raw(addr, "GET /progress/det_1 HTTP/1.1\r\n\r\n", b"").0, 404);
        assert_eq!(post(addr, "?detection_id=det_1", b"abc").0, 200);
        let (status, _, body) = raw(addr, "GET /progress/det_1 HTTP/1.1\r\n\r\n", b"");
        assert_eq!(status, 200);
        let v = json::parse(&body).unwrap();
        assert_eq!(v.get("frame").and_then(Value::as_f64), Some(3.0));
        assert_eq!(v.get("total").and_then(Value::as_f64), Some(9.0));
        assert_eq!(v.get("status").and_then(Value::as_str), Some("processing"));
        assert_eq!(raw(addr, "GET /progress/..%2Fx HTTP/1.1\r\n\r\n", b"").0, 404);
        assert_eq!(raw(addr, "POST /progress/det_1 HTTP/1.1\r\nContent-Length: 0\r\n\r\n", b"").0, 405);
    }

    #[test]
    fn stale_temp_uploads_are_removed_at_startup() {
        let cfg = test_config("stale");
        fs::create_dir_all(&cfg.tmp_dir).unwrap();
        fs::write(cfg.tmp_dir.join("yolo-1-1.bin"), b"old").unwrap();
        fs::write(cfg.tmp_dir.join("keep.txt"), b"other").unwrap();
        let tmp = cfg.tmp_dir.clone();
        let _ = start("stale", cfg, Behaviour::Echo);
        assert!(!tmp.join("yolo-1-1.bin").exists());
        assert!(tmp.join("keep.txt").exists());
    }
}
