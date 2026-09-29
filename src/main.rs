//! yolo-server entry point: load the model, listen on `$PORT`.
//!
//! `yolo-server healthcheck` probes `GET /health` on localhost and exits 0 or 1,
//! which is what the Docker `HEALTHCHECK` runs (no curl in the image).

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Command, ExitCode};
use std::time::{Duration, Instant};

use yolo_server::detect::{EngineConfig, Yolo};
use yolo_server::ort::Session;
use yolo_server::server::{log, Config, Server};

/// Model file used when `YOLO_MODEL` is not set, relative to the working directory.
const DEFAULT_MODEL: &str = "models/yolo26n-seg.onnx";
/// Default cap on a response body built in memory.
const DEFAULT_MAX_RESPONSE_MB: u64 = 256;
/// Default onnxruntime thread cap. Nano-model inference stops scaling around the
/// physical core count, and hyperthreads plus the ffmpeg process only add
/// contention (measured: 4 threads beat 8 on a 4-core/8-thread CPU).
const DEFAULT_MAX_THREADS: usize = 4;
/// How long the health probe waits for an answer.
const HEALTHCHECK_TIMEOUT: Duration = Duration::from_secs(5);

fn main() -> ExitCode {
    if std::env::args().nth(1).as_deref() == Some("healthcheck") {
        return if healthcheck() { ExitCode::SUCCESS } else { ExitCode::FAILURE };
    }
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("yolo-server: fatal: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Ask the local server for `/health`.
fn healthcheck() -> bool {
    let port = std::env::var("PORT").ok().and_then(|p| p.parse::<u16>().ok()).unwrap_or(5001);
    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) else { return false };
    let _ = stream.set_read_timeout(Some(HEALTHCHECK_TIMEOUT));
    if stream.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n").is_err() {
        return false;
    }
    let mut reply = String::new();
    let _ = stream.read_to_string(&mut reply);
    reply.starts_with("HTTP/1.1 200")
}

/// Fail early with a clear message when an ffmpeg tool is missing.
fn require_tool(name: &str) -> Result<(), String> {
    Command::new(name)
        .arg("-version")
        .output()
        .map(|_| ())
        .map_err(|e| format!("{name} is required but cannot be run: {e}"))
}

fn run() -> Result<(), String> {
    let started = Instant::now();
    let env = |key: &str| std::env::var(key).ok().filter(|v| !v.trim().is_empty());
    let cfg = Config::from_lookup(&env)?;
    let threads = env("YOLO_THREADS")
        .map(|v| v.parse::<usize>().ok().filter(|n| *n > 0).ok_or_else(|| format!("YOLO_THREADS must be a positive number, got {v:?}")))
        .transpose()?
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(DEFAULT_MAX_THREADS, |n| n.get().min(DEFAULT_MAX_THREADS)));
    let max_response_mb = env("YOLO_MAX_RESPONSE_MB")
        .map(|v| v.parse::<u64>().ok().filter(|n| *n > 0).ok_or_else(|| format!("YOLO_MAX_RESPONSE_MB must be a positive number, got {v:?}")))
        .transpose()?
        .unwrap_or(DEFAULT_MAX_RESPONSE_MB);
    let engine = EngineConfig {
        model_path: PathBuf::from(env("YOLO_MODEL").unwrap_or_else(|| DEFAULT_MODEL.to_string())),
        threads,
        ffmpeg: env("FFMPEG").unwrap_or_else(|| "ffmpeg".to_string()),
        ffprobe: env("FFPROBE").unwrap_or_else(|| "ffprobe".to_string()),
        max_response_bytes: (max_response_mb * 1024 * 1024) as usize,
    };
    require_tool(&engine.ffmpeg)?;
    require_tool(&engine.ffprobe)?;
    let yolo = Yolo::load(engine)?;
    log(&format!("model loaded in {} ms ({threads} threads, onnxruntime {})", started.elapsed().as_millis(), Session::version()));
    let port = cfg.port;
    let server = Server::bind(&format!("0.0.0.0:{port}"), cfg, Box::new(yolo), Session::version()).map_err(|e| format!("cannot listen on port {port}: {e}"))?;
    log(&format!("listening on port {port}, ready in {} ms", started.elapsed().as_millis()));
    server.serve();
    Ok(())
}
