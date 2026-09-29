//! Just enough blocking HTTP/1.1 for an internal service.
//!
//! One request per connection (`Connection: close`), `Content-Length` bodies
//! only, no keep-alive, no TLS: the service is reached over Railway's private
//! network. Everything is bounded: header size, header count, body size and the
//! time the upload may take.

use std::io::{ErrorKind, Read, Write};
use std::time::{Duration, Instant};

/// Largest request head (request line plus headers) accepted.
pub const MAX_HEAD_BYTES: usize = 16 * 1024;
/// Most header lines accepted.
const MAX_HEADERS: usize = 100;
/// Read size while streaming a body to disk.
const COPY_CHUNK: usize = 64 * 1024;

/// An error that maps straight to an HTTP status and a one-line message.
#[derive(Debug, Clone, PartialEq)]
pub struct HttpError {
    /// Status code to answer with.
    pub status: u16,
    /// Plain-language explanation for the caller.
    pub message: String,
}

impl HttpError {
    /// Build an error.
    pub fn new(status: u16, message: impl Into<String>) -> Self {
        HttpError { status, message: message.into() }
    }
}

/// A parsed request line and headers.
#[derive(Debug, Clone, PartialEq)]
pub struct Head {
    /// Upper-case method, e.g. `POST`.
    pub method: String,
    /// Path without the query string.
    pub path: String,
    /// Decoded query parameters in order of appearance.
    pub query: Vec<(String, String)>,
    /// Headers with lower-cased names.
    pub headers: Vec<(String, String)>,
}

impl Head {
    /// Value of header `name` (case-insensitive).
    pub fn header(&self, name: &str) -> Option<&str> {
        let name = name.to_ascii_lowercase();
        self.headers.iter().find(|(k, _)| *k == name).map(|(_, v)| v.as_str())
    }

    /// First value of query parameter `name`.
    pub fn query_param(&self, name: &str) -> Option<&str> {
        self.query.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    /// The declared body length, if any.
    ///
    /// # Errors
    /// 400 for an unparsable or conflicting `Content-Length`, 411 when the body
    /// uses `Transfer-Encoding`, which this server does not decode.
    pub fn content_length(&self) -> Result<Option<u64>, HttpError> {
        if self.header("transfer-encoding").is_some() {
            return Err(HttpError::new(411, "chunked uploads are not supported; send a Content-Length header"));
        }
        let mut found: Option<u64> = None;
        for (_, value) in self.headers.iter().filter(|(k, _)| k == "content-length") {
            let n: u64 = value.trim().parse().map_err(|_| HttpError::new(400, "invalid Content-Length"))?;
            if found.is_some_and(|prev| prev != n) {
                return Err(HttpError::new(400, "conflicting Content-Length headers"));
            }
            found = Some(n);
        }
        Ok(found)
    }

    /// True when the client waits for `100 Continue` before sending the body.
    pub fn expects_continue(&self) -> bool {
        self.header("expect").is_some_and(|v| v.eq_ignore_ascii_case("100-continue"))
    }
}

/// Decode `%XX` escapes, and `+` as a space when `plus_is_space` is set. Invalid
/// escapes are kept literally; invalid UTF-8 becomes U+FFFD.
pub fn percent_decode(s: &str, plus_is_space: bool) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() && bytes[i + 1].is_ascii_hexdigit() && bytes[i + 2].is_ascii_hexdigit() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("00");
                out.push(u8::from_str_radix(hex, 16).unwrap_or(b'?'));
                i += 3;
            }
            b'+' if plus_is_space => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Split `a=1&b=two` into decoded pairs. A key without `=` gets an empty value.
pub fn parse_query(q: &str) -> Vec<(String, String)> {
    q.split('&')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let (k, v) = part.split_once('=').unwrap_or((part, ""));
            (percent_decode(k, true), percent_decode(v, true))
        })
        .collect()
}

/// Parse the request line and headers (everything before the blank line).
///
/// # Errors
/// 400 for a malformed request, 431 for too many headers, 505 for HTTP/2+.
pub fn parse_head(head: &str) -> Result<Head, HttpError> {
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split(' ');
    let (Some(method), Some(target), Some(version), None) = (parts.next(), parts.next(), parts.next(), parts.next()) else {
        return Err(HttpError::new(400, "malformed request line"));
    };
    if !version.starts_with("HTTP/1.") {
        return Err(HttpError::new(505, "only HTTP/1.x is supported"));
    }
    if method.is_empty() || !method.bytes().all(|b| b.is_ascii_uppercase()) {
        return Err(HttpError::new(400, "malformed request method"));
    }
    if !target.starts_with('/') {
        return Err(HttpError::new(400, "request target must be a path"));
    }
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let mut headers = Vec::new();
    for line in lines {
        if headers.len() >= MAX_HEADERS {
            return Err(HttpError::new(431, "too many headers"));
        }
        let Some((name, value)) = line.split_once(':') else {
            return Err(HttpError::new(400, "malformed header line"));
        };
        if name.is_empty() || name.contains(' ') {
            return Err(HttpError::new(400, "malformed header name"));
        }
        headers.push((name.to_ascii_lowercase(), value.trim().to_string()));
    }
    Ok(Head { method: method.to_string(), path: percent_decode(path, false), query: parse_query(query), headers })
}

/// Read a request head from `r`. Returns it plus any body bytes already read.
///
/// # Errors
/// 408 on a timeout, 431 when the head exceeds [`MAX_HEAD_BYTES`], 400 when the
/// connection closes early or the head is malformed.
pub fn read_head<R: Read>(r: &mut R) -> Result<(Head, Vec<u8>), HttpError> {
    let mut buf: Vec<u8> = Vec::with_capacity(1024);
    let mut chunk = [0u8; 2048];
    loop {
        if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let text = std::str::from_utf8(&buf[..end]).map_err(|_| HttpError::new(400, "request head is not valid UTF-8"))?;
            let head = parse_head(text)?;
            return Ok((head, buf[end + 4..].to_vec()));
        }
        if buf.len() > MAX_HEAD_BYTES {
            return Err(HttpError::new(431, "request head too large"));
        }
        match r.read(&mut chunk) {
            Ok(0) => return Err(HttpError::new(400, "connection closed before the request was complete")),
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                return Err(HttpError::new(408, "timed out waiting for the request"));
            }
            Err(e) => return Err(HttpError::new(400, format!("read failed: {e}"))),
        }
    }
}

/// Copy exactly `len` body bytes to `w`: first `leftover` (already read with the
/// head), then from `r`. Memory use is constant; the deadline bounds the total time.
///
/// # Errors
/// 408 on a timeout or missed deadline, 400 if the client stops early or `w` fails.
pub fn copy_body<R: Read, W: Write>(r: &mut R, leftover: &[u8], len: u64, w: &mut W, deadline: Instant) -> Result<(), HttpError> {
    let take = leftover.len().min(len as usize);
    w.write_all(&leftover[..take]).map_err(|e| HttpError::new(500, format!("cannot store upload: {e}")))?;
    let mut remaining = len - take as u64;
    let mut chunk = vec![0u8; COPY_CHUNK];
    while remaining > 0 {
        if Instant::now() >= deadline {
            return Err(HttpError::new(408, "upload took too long"));
        }
        let want = chunk.len().min(remaining as usize);
        match r.read(&mut chunk[..want]) {
            Ok(0) => return Err(HttpError::new(400, "upload ended before Content-Length bytes arrived")),
            Ok(n) => {
                w.write_all(&chunk[..n]).map_err(|e| HttpError::new(500, format!("cannot store upload: {e}")))?;
                remaining -= n as u64;
            }
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                return Err(HttpError::new(408, "timed out waiting for upload data"));
            }
            Err(e) => return Err(HttpError::new(400, format!("upload read failed: {e}"))),
        }
    }
    Ok(())
}

/// Reason phrase for the statuses this service sends.
pub fn reason(status: u16) -> &'static str {
    match status {
        100 => "Continue",
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        411 => "Length Required",
        413 => "Content Too Large",
        415 => "Unsupported Media Type",
        422 => "Unprocessable Content",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        505 => "HTTP Version Not Supported",
        _ => "Unknown",
    }
}

/// A complete response.
#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    /// Status code.
    pub status: u16,
    /// Body bytes.
    pub body: Vec<u8>,
    /// Headers beyond `Content-Type`, `Content-Length` and `Connection`.
    pub headers: Vec<(&'static str, String)>,
}

impl Response {
    /// A JSON body with the given status.
    pub fn json(status: u16, body: String) -> Response {
        Response { status, body: body.into_bytes(), headers: Vec::new() }
    }

    /// `{"error": message}` with the given status.
    pub fn error(status: u16, message: &str) -> Response {
        let mut w = crate::json::Writer::new();
        w.begin_object();
        w.key("error");
        w.string(message);
        w.end_object();
        Response::json(status, w.finish())
    }

    /// Add a header.
    pub fn with_header(mut self, name: &'static str, value: impl Into<String>) -> Response {
        self.headers.push((name, value.into()));
        self
    }

    /// Serialize onto `w`.
    ///
    /// # Errors
    /// Any write error from `w`.
    pub fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        let mut head = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
            self.status,
            reason(self.status),
            self.body.len()
        );
        for (k, v) in &self.headers {
            head.push_str(&format!("{k}: {v}\r\n"));
        }
        head.push_str("\r\n");
        w.write_all(head.as_bytes())?;
        w.write_all(&self.body)?;
        w.flush()
    }
}

/// A far-future deadline helper for callers that just want "now plus N".
pub fn deadline_in(d: Duration) -> Instant {
    Instant::now() + d
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn head(text: &str) -> Result<Head, HttpError> {
        parse_head(text)
    }

    #[test]
    fn parse_head_reads_method_path_query_and_headers() {
        let h = head("POST /detect?every_n=3&conf=0.1 HTTP/1.1\r\nHost: x\r\nContent-Length: 12\r\nX-Thing:  spaced  ").unwrap();
        assert_eq!(h.method, "POST");
        assert_eq!(h.path, "/detect");
        assert_eq!(h.query_param("every_n"), Some("3"));
        assert_eq!(h.query_param("conf"), Some("0.1"));
        assert_eq!(h.header("HOST"), Some("x"));
        assert_eq!(h.header("x-thing"), Some("spaced"));
        assert_eq!(h.content_length(), Ok(Some(12)));
    }

    #[test]
    fn parse_head_rejects_malformed_request_lines() {
        for bad in ["", "GET", "GET /", "GET / HTTP/1.1 extra", "get / HTTP/1.1", "GET nopath HTTP/1.1"] {
            assert_eq!(head(bad).unwrap_err().status, 400, "{bad:?}");
        }
    }

    #[test]
    fn parse_head_rejects_http2() {
        assert_eq!(head("GET / HTTP/2").unwrap_err().status, 505);
    }

    #[test]
    fn parse_head_rejects_header_lines_without_a_colon() {
        assert_eq!(head("GET / HTTP/1.1\r\nbroken").unwrap_err().status, 400);
    }

    #[test]
    fn parse_head_caps_the_header_count() {
        let mut text = String::from("GET / HTTP/1.1");
        for i in 0..MAX_HEADERS + 1 {
            text.push_str(&format!("\r\nH{i}: v"));
        }
        assert_eq!(head(&text).unwrap_err().status, 431);
    }

    #[test]
    fn percent_decode_handles_escapes_plus_and_garbage() {
        assert_eq!(percent_decode("a%20b+c", true), "a b c");
        assert_eq!(percent_decode("a%20b+c", false), "a b+c");
        assert_eq!(percent_decode("100%", true), "100%");
        assert_eq!(percent_decode("%zz", true), "%zz");
        assert_eq!(percent_decode("%C3%A9", true), "é");
    }

    #[test]
    fn percent_decode_decodes_a_trailing_escape() {
        assert_eq!(percent_decode("x%41", true), "xA");
    }

    #[test]
    fn parse_query_splits_pairs_and_keeps_bare_keys() {
        let q = parse_query("classes=0%2C32&masks&&conf=.5");
        assert_eq!(q, vec![("classes".into(), "0,32".into()), ("masks".into(), "".into()), ("conf".into(), ".5".into())]);
    }

    #[test]
    fn content_length_rejects_junk_conflicts_and_chunked() {
        let ok = |t: &str| head(&format!("POST / HTTP/1.1\r\n{t}")).unwrap();
        assert_eq!(ok("Content-Length: abc").content_length().unwrap_err().status, 400);
        assert_eq!(ok("Content-Length: -1").content_length().unwrap_err().status, 400);
        assert_eq!(ok("Content-Length: 1\r\nContent-Length: 2").content_length().unwrap_err().status, 400);
        assert_eq!(ok("Content-Length: 5\r\nContent-Length: 5").content_length(), Ok(Some(5)));
        assert_eq!(ok("Transfer-Encoding: chunked").content_length().unwrap_err().status, 411);
        assert_eq!(ok("Host: x").content_length(), Ok(None));
    }

    #[test]
    fn expects_continue_is_case_insensitive() {
        assert!(head("POST / HTTP/1.1\r\nExpect: 100-Continue").unwrap().expects_continue());
        assert!(!head("POST / HTTP/1.1\r\nHost: x").unwrap().expects_continue());
    }

    #[test]
    fn read_head_returns_the_head_and_any_body_bytes_already_read() {
        let mut r = Cursor::new(b"POST /detect HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello".to_vec());
        let (h, rest) = read_head(&mut r).unwrap();
        assert_eq!(h.path, "/detect");
        assert_eq!(rest, b"hello");
    }

    #[test]
    fn read_head_fails_on_early_close_and_oversized_heads() {
        let mut r = Cursor::new(b"GET / HTTP/1.1\r\nHost".to_vec());
        assert_eq!(read_head(&mut r).unwrap_err().status, 400);
        let mut big = Cursor::new(vec![b'a'; MAX_HEAD_BYTES + 5000]);
        assert_eq!(read_head(&mut big).unwrap_err().status, 431);
    }

    #[test]
    fn copy_body_uses_leftover_bytes_first_then_the_stream() {
        let mut out = Vec::new();
        let mut r = Cursor::new(b"world".to_vec());
        copy_body(&mut r, b"hello ", 11, &mut out, deadline_in(Duration::from_secs(5))).unwrap();
        assert_eq!(out, b"hello world");
    }

    #[test]
    fn copy_body_ignores_bytes_past_content_length() {
        let mut out = Vec::new();
        let mut r = Cursor::new(Vec::new());
        copy_body(&mut r, b"abcdef", 3, &mut out, deadline_in(Duration::from_secs(5))).unwrap();
        assert_eq!(out, b"abc");
    }

    #[test]
    fn copy_body_reports_a_short_upload() {
        let mut out = Vec::new();
        let mut r = Cursor::new(b"abc".to_vec());
        let err = copy_body(&mut r, b"", 10, &mut out, deadline_in(Duration::from_secs(5))).unwrap_err();
        assert_eq!(err.status, 400);
    }

    #[test]
    fn copy_body_stops_at_the_deadline() {
        let mut out = Vec::new();
        let mut r = Cursor::new(vec![0u8; 100]);
        let err = copy_body(&mut r, b"", 100, &mut out, Instant::now()).unwrap_err();
        assert_eq!(err.status, 408);
    }

    #[test]
    fn response_serializes_status_headers_and_body() {
        let mut out = Vec::new();
        Response::error(429, "busy").with_header("Retry-After", "5").write_to(&mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("HTTP/1.1 429 Too Many Requests\r\n"));
        assert!(text.contains("Content-Length: 16\r\n"));
        assert!(text.contains("Retry-After: 5\r\n"));
        assert!(text.contains("Connection: close\r\n"));
        assert!(text.ends_with("\r\n\r\n{\"error\":\"busy\"}"));
    }
}
