//! Minimal JSON: a streaming [`Writer`] for responses and a small [`parse`]r.
//!
//! The service only ever *writes* JSON (request bodies are raw media and query
//! strings), so the parser exists for tests and for reading the ultralytics
//! reference dump. Both are written by hand to keep the dependency list empty.

use std::fmt::Write as _;

/// Deepest array/object nesting [`parse`] accepts. Bounds recursion on hostile input.
const MAX_DEPTH: usize = 64;

/// Append `s` to `out` as a quoted, escaped JSON string literal.
pub fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Format `v` with at most `decimals` fractional digits and no trailing zeros.
///
/// Non-finite values become `null` because JSON has no NaN or Infinity, and
/// negative zero prints as `0`.
pub fn fmt_num(v: f64, decimals: usize) -> String {
    if !v.is_finite() {
        return "null".to_string();
    }
    let mut s = format!("{v:.decimals$}");
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
    if s == "-0" {
        s = "0".to_string();
    }
    s
}

/// Streaming JSON builder. Commas and key/value separators are handled for you.
///
/// The caller is responsible for balanced `begin_*`/`end_*` calls; `finish`
/// does not check them.
#[derive(Debug, Default)]
pub struct Writer {
    out: String,
    /// One flag per open container: has it already received an item?
    has_items: Vec<bool>,
    /// True right after `key`, so the value that follows gets no comma.
    after_key: bool,
}

impl Writer {
    /// An empty writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bytes written so far. Used to cap response size while streaming.
    pub fn len(&self) -> usize {
        self.out.len()
    }

    /// True when nothing has been written.
    pub fn is_empty(&self) -> bool {
        self.out.is_empty()
    }

    /// Take the finished document.
    pub fn finish(self) -> String {
        self.out
    }

    /// Comma handling shared by every value and container start.
    fn before_value(&mut self) {
        if self.after_key {
            self.after_key = false;
            return;
        }
        if let Some(has) = self.has_items.last_mut() {
            if *has {
                self.out.push(',');
            }
            *has = true;
        }
    }

    /// Start `{`.
    pub fn begin_object(&mut self) {
        self.before_value();
        self.out.push('{');
        self.has_items.push(false);
    }

    /// End `}`.
    pub fn end_object(&mut self) {
        self.has_items.pop();
        self.out.push('}');
    }

    /// Start `[`.
    pub fn begin_array(&mut self) {
        self.before_value();
        self.out.push('[');
        self.has_items.push(false);
    }

    /// End `]`.
    pub fn end_array(&mut self) {
        self.has_items.pop();
        self.out.push(']');
    }

    /// Write an object key. The next call must be a value or a container start.
    pub fn key(&mut self, k: &str) {
        if let Some(has) = self.has_items.last_mut() {
            if *has {
                self.out.push(',');
            }
            *has = true;
        }
        write_str(&mut self.out, k);
        self.out.push(':');
        self.after_key = true;
    }

    /// Write a string value.
    pub fn string(&mut self, v: &str) {
        self.before_value();
        write_str(&mut self.out, v);
    }

    /// Write a number with at most `decimals` fractional digits.
    pub fn number(&mut self, v: f64, decimals: usize) {
        self.before_value();
        self.out.push_str(&fmt_num(v, decimals));
    }

    /// Write an integer.
    pub fn int(&mut self, v: i64) {
        self.before_value();
        let _ = write!(self.out, "{v}");
    }

    /// Write `true` or `false`.
    pub fn boolean(&mut self, v: bool) {
        self.before_value();
        self.out.push_str(if v { "true" } else { "false" });
    }

    /// Write `null`.
    pub fn null(&mut self) {
        self.before_value();
        self.out.push_str("null");
    }
}

/// A parsed JSON value. Objects keep their key order.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Value>),
    Obj(Vec<(String, Value)>),
}

impl Value {
    /// Member `key` of an object, if this is an object that has it.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Obj(members) => members.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The number, if this is one.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Num(n) => Some(*n),
            _ => None,
        }
    }

    /// The string, if this is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The items, if this is an array.
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(items) => Some(items),
            _ => None,
        }
    }

    /// The members, if this is an object.
    pub fn as_object(&self) -> Option<&[(String, Value)]> {
        match self {
            Value::Obj(members) => Some(members),
            _ => None,
        }
    }
}

/// Parse a complete JSON document.
///
/// # Errors
/// A message with the byte offset when the text is not valid JSON, has trailing
/// content, or nests deeper than [`MAX_DEPTH`].
pub fn parse(input: &str) -> Result<Value, String> {
    let mut p = Parser { src: input.as_bytes(), pos: 0 };
    p.skip_ws();
    let value = p.value(0)?;
    p.skip_ws();
    if p.pos != p.src.len() {
        return Err(p.err("trailing content"));
    }
    Ok(value)
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn err(&self, msg: &str) -> String {
        format!("{msg} at byte {}", self.pos)
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, lit: &str, value: Value) -> Result<Value, String> {
        if self.src[self.pos..].starts_with(lit.as_bytes()) {
            self.pos += lit.len();
            Ok(value)
        } else {
            Err(self.err("invalid literal"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, String> {
        if depth > MAX_DEPTH {
            return Err(self.err("nesting too deep"));
        }
        match self.peek() {
            None => Err(self.err("unexpected end")),
            Some(b'n') => self.expect("null", Value::Null),
            Some(b't') => self.expect("true", Value::Bool(true)),
            Some(b'f') => self.expect("false", Value::Bool(false)),
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(b'[') => self.array(depth),
            Some(b'{') => self.object(depth),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.err("unexpected character")),
        }
    }

    fn number(&mut self) -> Result<Value, String> {
        let start = self.pos;
        while matches!(self.peek(), Some(b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')) {
            self.pos += 1;
        }
        let text = std::str::from_utf8(&self.src[start..self.pos]).map_err(|_| self.err("bad number"))?;
        text.parse::<f64>().map(Value::Num).map_err(|_| self.err("bad number"))
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let digits = self.src.get(self.pos..self.pos + 4).ok_or_else(|| self.err("short \\u escape"))?;
        let text = std::str::from_utf8(digits).map_err(|_| self.err("bad \\u escape"))?;
        let n = u32::from_str_radix(text, 16).map_err(|_| self.err("bad \\u escape"))?;
        self.pos += 4;
        Ok(n)
    }

    fn string(&mut self) -> Result<String, String> {
        self.pos += 1; // opening quote
        let mut out = Vec::new();
        loop {
            let Some(b) = self.peek() else {
                return Err(self.err("unterminated string"));
            };
            self.pos += 1;
            match b {
                b'"' => break,
                b'\\' => {
                    let Some(esc) = self.peek() else {
                        return Err(self.err("unterminated escape"));
                    };
                    self.pos += 1;
                    let ch = match esc {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let hi = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&hi) {
                                if self.src[self.pos..].starts_with(b"\\u") {
                                    self.pos += 2;
                                    let lo = self.hex4()?;
                                    0x10000 + ((hi - 0xD800) << 10) + (lo.wrapping_sub(0xDC00) & 0x3FF)
                                } else {
                                    return Err(self.err("lone surrogate"));
                                }
                            } else {
                                hi
                            };
                            char::from_u32(code).ok_or_else(|| self.err("bad code point"))?
                        }
                        _ => return Err(self.err("bad escape")),
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                b if b < 0x20 => return Err(self.err("control character in string")),
                b => out.push(b),
            }
        }
        String::from_utf8(out).map_err(|_| self.err("invalid utf-8"))
    }

    fn array(&mut self, depth: usize) -> Result<Value, String> {
        self.pos += 1;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::Arr(items));
        }
        loop {
            self.skip_ws();
            items.push(self.value(depth + 1)?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Value::Arr(items));
                }
                _ => return Err(self.err("expected , or ]")),
            }
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value, String> {
        self.pos += 1;
        let mut members = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Obj(members));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(self.err("expected string key"));
            }
            let key = self.string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(self.err("expected :"));
            }
            self.pos += 1;
            self.skip_ws();
            members.push((key, self.value(depth + 1)?));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Value::Obj(members));
                }
                _ => return Err(self.err("expected , or }")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_str_escapes_quotes_backslashes_and_controls() {
        let mut out = String::new();
        write_str(&mut out, "a\"b\\c\nd\u{1}");
        assert_eq!(out, "\"a\\\"b\\\\c\\nd\\u0001\"");
    }

    #[test]
    fn write_str_keeps_non_ascii_as_utf8() {
        let mut out = String::new();
        write_str(&mut out, "café");
        assert_eq!(out, "\"café\"");
    }

    #[test]
    fn fmt_num_trims_trailing_zeros() {
        assert_eq!(fmt_num(1.5, 3), "1.5");
        assert_eq!(fmt_num(2.0, 3), "2");
        assert_eq!(fmt_num(0.68361, 4), "0.6836");
    }

    #[test]
    fn fmt_num_maps_non_finite_to_null_and_negative_zero_to_zero() {
        assert_eq!(fmt_num(f64::NAN, 2), "null");
        assert_eq!(fmt_num(f64::INFINITY, 2), "null");
        assert_eq!(fmt_num(-0.0001, 2), "0");
    }

    #[test]
    fn writer_places_commas_between_items_and_members() {
        let mut w = Writer::new();
        w.begin_object();
        w.key("a");
        w.int(1);
        w.key("b");
        w.begin_array();
        w.number(1.25, 2);
        w.string("x");
        w.null();
        w.boolean(true);
        w.end_array();
        w.key("c");
        w.begin_object();
        w.end_object();
        w.end_object();
        assert_eq!(w.finish(), r#"{"a":1,"b":[1.25,"x",null,true],"c":{}}"#);
    }

    #[test]
    fn writer_output_round_trips_through_the_parser() {
        let mut w = Writer::new();
        w.begin_object();
        w.key("name");
        w.string("sports \"ball\"");
        w.key("box");
        w.begin_array();
        w.number(10.5, 2);
        w.number(-3.0, 2);
        w.end_array();
        w.end_object();
        let v = parse(&w.finish()).unwrap();
        assert_eq!(v.get("name").and_then(Value::as_str), Some("sports \"ball\""));
        let arr = v.get("box").and_then(Value::as_array).unwrap();
        assert_eq!(arr[0].as_f64(), Some(10.5));
        assert_eq!(arr[1].as_f64(), Some(-3.0));
    }

    #[test]
    fn parse_reads_nested_documents() {
        let v = parse(r#" {"a":[1,2,{"b":null}],"c":"\u00e9\ud83d\ude00","d":-1.5e2} "#).unwrap();
        assert_eq!(v.get("a").and_then(Value::as_array).map(<[Value]>::len), Some(3));
        assert_eq!(v.get("c").and_then(Value::as_str), Some("é😀"));
        assert_eq!(v.get("d").and_then(Value::as_f64), Some(-150.0));
    }

    #[test]
    fn parse_rejects_malformed_input() {
        for bad in ["", "{", "[1,]", "{\"a\" 1}", "nul", "\"abc", "1 2", "[1 2]", "{1:2}", "\"\\x\""] {
            assert!(parse(bad).is_err(), "{bad:?} should not parse");
        }
    }

    #[test]
    fn parse_rejects_runaway_nesting() {
        let deep = "[".repeat(MAX_DEPTH + 10);
        assert!(parse(&deep).unwrap_err().contains("too deep"));
    }
}
