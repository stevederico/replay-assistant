//! yolo-server: a zero-crate YOLO26n-seg detection service.
//!
//! Inference runs through the system `libonnxruntime` (C API over FFI, see
//! [`ort`]). Video frames come from the `ffmpeg` command line as piped raw RGB
//! ([`video`]). Everything else is hand-written std: HTTP ([`http`], [`server`]),
//! JSON ([`json`]), letterbox, box decode, NMS and mask decode ([`yolo`]).

pub mod classes;
pub mod detect;
pub mod http;
pub mod json;
pub mod ort;
pub mod server;
pub mod video;
pub mod yolo;
