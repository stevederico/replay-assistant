//! YOLO-seg pre and post processing, written by hand.
//!
//! The math follows ultralytics so the boxes and polygons line up with what the
//! old Python service returned: `LetterBox` for preprocessing,
//! `non_max_suppression` (class-aware, argmax class, `conf` strictly greater)
//! for decode, and `process_mask` + `masks2segments` for polygons.
//!
//! Model layout (YOLO26n-seg, static 640):
//! - input  `[1, 3, 640, 640]` f32 RGB in 0..1, letterboxed with gray 114
//! - output0 `[1, 4 + classes + 32, anchors]`: `cx, cy, w, h`, per-class scores
//!   (already sigmoided), then 32 mask coefficients
//! - output1 `[1, 32, 160, 160]`: mask prototypes

use std::collections::VecDeque;

/// Side of the square model input in pixels.
pub const INPUT_SIZE: usize = 640;
/// Mask coefficients per detection (and prototype channels).
pub const MASK_COEFFS: usize = 32;
/// Gray the padding is filled with, as ultralytics does (114/255).
const PAD_GRAY: f32 = 114.0 / 255.0;
/// Ultralytics keeps at most this many candidates before NMS.
const MAX_NMS_CANDIDATES: usize = 30_000;

/// Round half to even, like Python's `round`, so sizes match ultralytics.
pub fn round_half_even(x: f64) -> f64 {
    if (x - x.trunc()).abs() == 0.5 {
        2.0 * (x / 2.0).round()
    } else {
        x.round()
    }
}

/// Where a source frame sits inside the 640x640 model input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    /// Scale from source pixels to model pixels.
    pub gain: f64,
    /// Width of the resized image inside the input.
    pub new_w: usize,
    /// Height of the resized image inside the input.
    pub new_h: usize,
    /// Left padding in model pixels.
    pub left: usize,
    /// Top padding in model pixels.
    pub top: usize,
}

impl Letterbox {
    /// Geometry for a `src_w` x `src_h` frame (both must be non-zero).
    ///
    /// Mirrors ultralytics `LetterBox(auto=False, scaleup=True)`: scale to fit,
    /// split the padding across both sides with the same `round(d +- 0.1)` rule.
    pub fn new(src_w: usize, src_h: usize) -> Letterbox {
        let size = INPUT_SIZE as f64;
        let gain = (size / src_h as f64).min(size / src_w as f64);
        let new_w = (round_half_even(src_w as f64 * gain) as usize).clamp(1, INPUT_SIZE);
        let new_h = (round_half_even(src_h as f64 * gain) as usize).clamp(1, INPUT_SIZE);
        let dw = (INPUT_SIZE - new_w) as f64 / 2.0;
        let dh = (INPUT_SIZE - new_h) as f64 / 2.0;
        Letterbox {
            gain,
            new_w,
            new_h,
            left: round_half_even(dw - 0.1).max(0.0) as usize,
            top: round_half_even(dh - 0.1).max(0.0) as usize,
        }
    }

    /// Map an x coordinate in model space back to source pixels, clipped to the frame.
    pub fn unmap_x(&self, x: f32, src_w: usize) -> f32 {
        (((x as f64 - self.left as f64) / self.gain) as f32).clamp(0.0, src_w as f32)
    }

    /// Map a y coordinate in model space back to source pixels, clipped to the frame.
    pub fn unmap_y(&self, y: f32, src_h: usize) -> f32 {
        (((y as f64 - self.top as f64) / self.gain) as f32).clamp(0.0, src_h as f32)
    }
}

/// One output column of a bilinear resize: two source columns and the weight of the second.
#[derive(Clone, Copy)]
struct Tap {
    lo: usize,
    hi: usize,
    frac: f32,
}

/// Taps for resizing `src` samples to `dst`, matching OpenCV `INTER_LINEAR`
/// (half-pixel centers, edge clamp, no antialiasing).
fn resize_taps(src: usize, dst: usize) -> Vec<Tap> {
    let scale = src as f64 / dst as f64;
    (0..dst)
        .map(|d| {
            let f = (d as f64 + 0.5) * scale - 0.5;
            let s = f.floor();
            if s < 0.0 {
                Tap { lo: 0, hi: 0, frac: 0.0 }
            } else if s as usize >= src - 1 {
                Tap { lo: src - 1, hi: src - 1, frac: 0.0 }
            } else {
                Tap { lo: s as usize, hi: s as usize + 1, frac: (f - s) as f32 }
            }
        })
        .collect()
}

/// Letterbox an interleaved RGB8 frame into a planar CHW f32 input tensor.
///
/// `dst` must hold `3 * 640 * 640` floats and is fully overwritten. Returns the
/// geometry needed to map detections back to the source frame.
///
/// # Panics
/// If `src` is not `src_w * src_h * 3` bytes or `dst` has the wrong length.
pub fn letterbox_into(src: &[u8], src_w: usize, src_h: usize, dst: &mut [f32]) -> Letterbox {
    assert_eq!(src.len(), src_w * src_h * 3, "source frame size mismatch");
    assert_eq!(dst.len(), 3 * INPUT_SIZE * INPUT_SIZE, "input tensor size mismatch");
    let lb = Letterbox::new(src_w, src_h);
    dst.fill(PAD_GRAY);
    let xs = resize_taps(src_w, lb.new_w);
    let ys = resize_taps(src_h, lb.new_h);
    let plane = INPUT_SIZE * INPUT_SIZE;
    let inv = 1.0 / 255.0;
    for (dy, ty) in ys.iter().enumerate() {
        let row_lo = &src[ty.lo * src_w * 3..(ty.lo + 1) * src_w * 3];
        let row_hi = &src[ty.hi * src_w * 3..(ty.hi + 1) * src_w * 3];
        let out_row = (lb.top + dy) * INPUT_SIZE + lb.left;
        for (dx, tx) in xs.iter().enumerate() {
            for c in 0..3 {
                let a = row_lo[tx.lo * 3 + c] as f32;
                let b = row_lo[tx.hi * 3 + c] as f32;
                let cc = row_hi[tx.lo * 3 + c] as f32;
                let d = row_hi[tx.hi * 3 + c] as f32;
                let top = a + (b - a) * tx.frac;
                let bottom = cc + (d - cc) * tx.frac;
                dst[c * plane + out_row + dx] = (top + (bottom - top) * ty.frac) * inv;
            }
        }
    }
    lb
}

/// Intersection over union of two `[x1, y1, x2, y2]` boxes; 0 when either is empty.
pub fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = iw * ih;
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// A box that passed the confidence threshold, in model (640x640) coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// Best class id (argmax over every class, as ultralytics does).
    pub class: usize,
    /// Score of that class.
    pub score: f32,
    /// `[x1, y1, x2, y2]` in model pixels.
    pub bbox: [f32; 4],
    /// Column in the output tensor, used to fetch mask coefficients.
    pub anchor: usize,
}

/// Turn the raw `[4 + classes + coeffs, anchors]` output into candidates.
///
/// `allowed`, when set, is a per-class keep mask applied to the argmax class
/// (ultralytics `classes=[...]` semantics). A score must be strictly greater
/// than `conf`.
///
/// # Panics
/// If `out0` is shorter than `(4 + num_classes) * num_anchors`.
pub fn decode_candidates(
    out0: &[f32],
    num_anchors: usize,
    num_classes: usize,
    conf: f32,
    allowed: Option<&[bool]>,
) -> Vec<Candidate> {
    assert!(out0.len() >= (4 + num_classes) * num_anchors, "output0 too short");
    let mut best = vec![f32::MIN; num_anchors];
    let mut best_class = vec![0u16; num_anchors];
    // Class-major so every pass reads one contiguous row.
    for c in 0..num_classes {
        let row = &out0[(4 + c) * num_anchors..(5 + c) * num_anchors];
        for (i, &s) in row.iter().enumerate() {
            if s > best[i] {
                best[i] = s;
                best_class[i] = c as u16;
            }
        }
    }
    let mut out = Vec::new();
    for i in 0..num_anchors {
        let score = best[i];
        let class = best_class[i] as usize;
        if score <= conf || !allowed.is_none_or(|a| a.get(class).copied().unwrap_or(false)) {
            continue;
        }
        let (cx, cy) = (out0[i], out0[num_anchors + i]);
        let (w, h) = (out0[2 * num_anchors + i], out0[3 * num_anchors + i]);
        out.push(Candidate {
            class,
            score,
            bbox: [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
            anchor: i,
        });
    }
    out
}

/// Class-aware greedy non-max suppression.
///
/// Sorts by score, keeps a box unless a higher-scored box of the *same class*
/// overlaps it by more than `iou_threshold`, and returns at most `max_det`.
pub fn nms(mut cands: Vec<Candidate>, iou_threshold: f32, max_det: usize) -> Vec<Candidate> {
    cands.sort_by(|a, b| b.score.total_cmp(&a.score));
    cands.truncate(MAX_NMS_CANDIDATES);
    let mut kept: Vec<Candidate> = Vec::new();
    for cand in cands {
        if kept.len() >= max_det {
            break;
        }
        let suppressed = kept
            .iter()
            .any(|k| k.class == cand.class && iou(&k.bbox, &cand.bbox) > iou_threshold);
        if !suppressed {
            kept.push(cand);
        }
    }
    kept
}

/// Mask prototypes, borrowed from the model's second output.
#[derive(Debug, Clone, Copy)]
pub struct Protos<'a> {
    /// `channels * height * width` floats, channel-major.
    pub data: &'a [f32],
    /// Channel count (32).
    pub channels: usize,
    /// Prototype height (160).
    pub height: usize,
    /// Prototype width (160).
    pub width: usize,
}

/// A binary mask cut out around one detection, in model (640x640) pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskRegion {
    /// Left edge of the region in model pixels.
    pub x0: usize,
    /// Top edge of the region in model pixels.
    pub y0: usize,
    /// Region width.
    pub width: usize,
    /// Region height.
    pub height: usize,
    /// Row-major `width * height` flags, true where the mask covers the pixel.
    pub bits: Vec<bool>,
}

/// Decode one detection's mask, following ultralytics `process_mask(upsample=True)`.
///
/// Logits are `coeffs . protos` at prototype resolution, bilinearly upsampled to
/// 640x640 (`align_corners=False`), thresholded at 0, and only then cropped to
/// the box (`x1 <= x < x2`, `y1 <= y < y2`). Cropping after the upsample keeps
/// the edge crisp; cropping first smears it outside the box (ultralytics#24272).
/// Only the box area is materialized, so cost scales with the box, not the frame.
///
/// Returns `None` when the box covers no pixel.
pub fn decode_mask(protos: &Protos, coeffs: &[f32], bbox: &[f32; 4]) -> Option<MaskRegion> {
    let (ph, pw) = (protos.height, protos.width);
    let size = INPUT_SIZE as f32;
    // Output pixels inside the box: `r >= x1 && r < x2`.
    let x0 = bbox[0].clamp(0.0, size).ceil() as usize;
    let x1 = (bbox[2].clamp(0.0, size).ceil() as usize).min(INPUT_SIZE);
    let y0 = bbox[1].clamp(0.0, size).ceil() as usize;
    let y1 = (bbox[3].clamp(0.0, size).ceil() as usize).min(INPUT_SIZE);
    if x0 >= x1 || y0 >= y1 {
        return None;
    }
    let ys = upsample_taps(ph, INPUT_SIZE);
    let xs = upsample_taps(pw, INPUT_SIZE);
    // Prototype cells the bilinear taps of this box read (taps are monotonic).
    let (col_lo, col_hi) = (xs[x0].lo, xs[x1 - 1].hi + 1);
    let (row_lo, row_hi) = (ys[y0].lo, ys[y1 - 1].hi + 1);
    let (cw, ch) = (col_hi - col_lo, row_hi - row_lo);
    let mut logits = vec![0.0f32; cw * ch];
    let plane = ph * pw;
    for (k, &coeff) in coeffs.iter().enumerate().take(protos.channels) {
        for r in 0..ch {
            let src = &protos.data[k * plane + (row_lo + r) * pw + col_lo..][..cw];
            let dst = &mut logits[r * cw..(r + 1) * cw];
            for (d, &s) in dst.iter_mut().zip(src) {
                *d += coeff * s;
            }
        }
    }
    let at = |r: usize, c: usize| logits[(r - row_lo) * cw + (c - col_lo)];
    let (w, h) = (x1 - x0, y1 - y0);
    let mut bits = vec![false; w * h];
    for (j, ty) in ys[y0..y1].iter().enumerate() {
        for (i, tx) in xs[x0..x1].iter().enumerate() {
            let top = at(ty.lo, tx.lo) * (1.0 - tx.frac) + at(ty.lo, tx.hi) * tx.frac;
            let bottom = at(ty.hi, tx.lo) * (1.0 - tx.frac) + at(ty.hi, tx.hi) * tx.frac;
            bits[j * w + i] = top * (1.0 - ty.frac) + bottom * ty.frac > 0.0;
        }
    }
    Some(MaskRegion { x0, y0, width: w, height: h, bits })
}

/// Bilinear taps for PyTorch `interpolate(mode="bilinear", align_corners=False)`.
fn upsample_taps(src: usize, dst: usize) -> Vec<Tap> {
    let scale = src as f32 / dst as f32;
    (0..dst)
        .map(|d| {
            let s = ((d as f32 + 0.5) * scale - 0.5).max(0.0);
            let lo = (s.floor() as usize).min(src - 1);
            Tap { lo, hi: (lo + 1).min(src - 1), frac: s - lo as f32 }
        })
        .collect()
}

/// 8-neighbour offsets, clockwise on screen (y down), starting west.
const DIRS: [(i32, i32); 8] = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)];

fn dir_index(dx: i32, dy: i32) -> usize {
    DIRS.iter().position(|&d| d == (dx, dy)).unwrap_or(0)
}

/// Outer boundary of the 8-connected component whose first raster pixel is
/// `start`, by Moore-neighbour tracing with Jacob's stopping rule. Every
/// consecutive pair of returned points is 8-adjacent.
fn trace_boundary(bits: &[bool], w: usize, h: usize, start: (i32, i32)) -> Vec<(i32, i32)> {
    let fg = |x: i32, y: i32| x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && bits[y as usize * w + x as usize];
    let mut contour = vec![start];
    let mut p = start;
    // `start` is first in raster order, so the cell to its west is background.
    let mut back = 0usize;
    let mut first_move: Option<(i32, i32, usize)> = None;
    let limit = w * h * 4 + 8;
    while contour.len() < limit {
        let Some((q, d)) = (1..=8).find_map(|k| {
            let d = (back + k) % 8;
            let q = (p.0 + DIRS[d].0, p.1 + DIRS[d].1);
            fg(q.0, q.1).then_some((q, d))
        }) else {
            break; // isolated pixel
        };
        if p == start {
            match first_move {
                None => first_move = Some((q.0, q.1, d)),
                Some(f) if f == (q.0, q.1, d) => break,
                Some(_) => {}
            }
        }
        let before = DIRS[(d + 7) % 8];
        let c = (p.0 + before.0, p.1 + before.1);
        back = dir_index(c.0 - q.0, c.1 - q.1);
        p = q;
        if p != start {
            contour.push(p);
        }
    }
    contour
}

/// Drop points in the middle of straight (horizontal, vertical or diagonal)
/// runs, like OpenCV `CHAIN_APPROX_SIMPLE`. The contour is treated as closed.
fn compress_chain(points: &[(i32, i32)]) -> Vec<(i32, i32)> {
    let n = points.len();
    if n < 3 {
        return points.to_vec();
    }
    let step = |a: (i32, i32), b: (i32, i32)| ((b.0 - a.0).signum(), (b.1 - a.1).signum());
    (0..n)
        .filter(|&i| {
            let prev = points[(i + n - 1) % n];
            let next = points[(i + 1) % n];
            step(prev, points[i]) != step(points[i], next)
        })
        .map(|i| points[i])
        .collect()
}

/// Outline of the largest 8-connected blob in `mask`, in region-local pixels.
///
/// Ultralytics keeps the contour with the most points; this keeps the blob with
/// the most pixels, which is the same one for any ordinary mask. Holes are
/// ignored (`RETR_EXTERNAL`). Returns an empty list for an empty mask.
pub fn largest_contour(mask: &MaskRegion) -> Vec<(i32, i32)> {
    let (w, h) = (mask.width, mask.height);
    let mut seen = vec![false; w * h];
    let mut best: Option<(usize, (i32, i32))> = None;
    let mut stack: Vec<usize> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !mask.bits[idx] || seen[idx] {
                continue;
            }
            // Flood fill; (x, y) is the component's first pixel in raster order.
            let mut area = 0usize;
            seen[idx] = true;
            stack.push(idx);
            while let Some(cur) = stack.pop() {
                area += 1;
                let (cx, cy) = ((cur % w) as i32, (cur / w) as i32);
                for (dx, dy) in DIRS {
                    let (nx, ny) = (cx + dx, cy + dy);
                    if nx < 0 || ny < 0 || nx as usize >= w || ny as usize >= h {
                        continue;
                    }
                    let n = ny as usize * w + nx as usize;
                    if mask.bits[n] && !seen[n] {
                        seen[n] = true;
                        stack.push(n);
                    }
                }
            }
            if best.is_none_or(|(a, _)| area > a) {
                best = Some((area, (x as i32, y as i32)));
            }
        }
    }
    match best {
        Some((_, start)) => compress_chain(&trace_boundary(&mask.bits, w, h, start)),
        None => Vec::new(),
    }
}

/// Perpendicular distance from `p` to the segment `a`-`b`.
fn segment_distance(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    if len2 == 0.0 {
        return ((p.0 - a.0).powi(2) + (p.1 - a.1).powi(2)).sqrt();
    }
    let t = (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0);
    let (px, py) = (a.0 + t * dx, a.1 + t * dy);
    ((p.0 - px).powi(2) + (p.1 - py).powi(2)).sqrt()
}

/// Douglas-Peucker on an open polyline; keeps both endpoints.
fn simplify_open(points: &[(f32, f32)], eps: f32) -> Vec<(f32, f32)> {
    let n = points.len();
    if n < 3 {
        return points.to_vec();
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack = VecDeque::from([(0usize, n - 1)]);
    while let Some((lo, hi)) = stack.pop_back() {
        let far = (lo + 1..hi)
            .map(|i| (i, segment_distance(points[i], points[lo], points[hi])))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((i, d)) = far {
            if d > eps {
                keep[i] = true;
                stack.push_back((lo, i));
                stack.push_back((i, hi));
            }
        }
    }
    points.iter().zip(&keep).filter(|(_, &k)| k).map(|(p, _)| *p).collect()
}

/// Douglas-Peucker for a closed polygon. `eps <= 0` returns the input unchanged.
///
/// Anchors the split at the point farthest from the first vertex so the seam
/// cannot land in the middle of a feature.
pub fn simplify_closed(points: &[(f32, f32)], eps: f32) -> Vec<(f32, f32)> {
    if eps <= 0.0 || points.len() <= 4 {
        return points.to_vec();
    }
    let far = (1..points.len())
        .max_by(|&a, &b| {
            let da = (points[a].0 - points[0].0).hypot(points[a].1 - points[0].1);
            let db = (points[b].0 - points[0].0).hypot(points[b].1 - points[0].1);
            da.total_cmp(&db)
        })
        .unwrap_or(0);
    let first = simplify_open(&points[..=far], eps);
    let mut second: Vec<(f32, f32)> = points[far..].to_vec();
    second.push(points[0]);
    let second = simplify_open(&second, eps);
    let mut out = first;
    // `second` starts at points[far] (already the last of `first`) and ends at points[0].
    out.extend(second[1..second.len() - 1].iter().copied());
    if out.len() < 3 {
        return points.to_vec();
    }
    out
}

/// Settings for [`postprocess`].
#[derive(Debug, Clone)]
pub struct Params {
    /// Confidence threshold (a score must be strictly greater).
    pub conf: f32,
    /// NMS IoU threshold.
    pub iou: f32,
    /// Per-class keep mask; `None` keeps every class.
    pub allowed: Option<Vec<bool>>,
    /// Most detections returned per frame.
    pub max_det: usize,
    /// Douglas-Peucker tolerance in model pixels; 0 keeps every contour point.
    pub simplify: f32,
    /// Decode segmentation polygons. Off skips the mask work entirely.
    pub masks: bool,
}

impl Default for Params {
    fn default() -> Self {
        Params { conf: 0.25, iou: 0.7, allowed: None, max_det: 300, simplify: 1.0, masks: true }
    }
}

/// A final detection in source-frame pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// Class id.
    pub class: usize,
    /// Class score, 0..1.
    pub score: f32,
    /// `[x1, y1, x2, y2]` in source pixels.
    pub bbox: [f32; 4],
    /// Mask outline in source pixels; empty when masks are off or the mask is empty.
    pub polygon: Vec<[f32; 2]>,
}

/// Decode, suppress and unmap one frame's model output.
///
/// `out0_shape` and `protos` describe the two output tensors; the class count is
/// derived from `out0_shape[1] - 4 - 32`.
///
/// # Errors
/// When the tensor shapes do not look like a YOLO-seg export.
pub fn postprocess(
    out0: &[f32],
    out0_shape: &[i64],
    protos: Option<Protos>,
    lb: &Letterbox,
    src_w: usize,
    src_h: usize,
    params: &Params,
) -> Result<Vec<Detection>, String> {
    let [_, rows, anchors] = out0_shape else {
        return Err(format!("unexpected output0 rank {}", out0_shape.len()));
    };
    let (rows, anchors) = (*rows as usize, *anchors as usize);
    if rows < 4 + MASK_COEFFS + 1 || out0.len() < rows * anchors {
        return Err(format!("unexpected output0 shape {out0_shape:?}"));
    }
    let num_classes = rows - 4 - MASK_COEFFS;
    let cands = decode_candidates(out0, anchors, num_classes, params.conf, params.allowed.as_deref());
    let kept = nms(cands, params.iou, params.max_det);
    let mut out = Vec::with_capacity(kept.len());
    for cand in kept {
        let bbox = [
            lb.unmap_x(cand.bbox[0], src_w),
            lb.unmap_y(cand.bbox[1], src_h),
            lb.unmap_x(cand.bbox[2], src_w),
            lb.unmap_y(cand.bbox[3], src_h),
        ];
        let polygon = match (params.masks, protos.as_ref()) {
            (true, Some(p)) => {
                let coeffs: Vec<f32> = (0..MASK_COEFFS)
                    .map(|k| out0[(4 + num_classes + k) * anchors + cand.anchor])
                    .collect();
                mask_polygon(p, &coeffs, &cand.bbox, lb, src_w, src_h, params.simplify)
            }
            _ => Vec::new(),
        };
        out.push(Detection { class: cand.class, score: cand.score, bbox, polygon });
    }
    Ok(out)
}

/// Full mask pipeline for one detection: decode, outline, simplify, unmap.
fn mask_polygon(
    protos: &Protos,
    coeffs: &[f32],
    bbox: &[f32; 4],
    lb: &Letterbox,
    src_w: usize,
    src_h: usize,
    simplify: f32,
) -> Vec<[f32; 2]> {
    let Some(region) = decode_mask(protos, coeffs, bbox) else {
        return Vec::new();
    };
    let contour: Vec<(f32, f32)> = largest_contour(&region)
        .into_iter()
        .map(|(x, y)| ((x as usize + region.x0) as f32, (y as usize + region.y0) as f32))
        .collect();
    simplify_closed(&contour, simplify)
        .into_iter()
        .map(|(x, y)| [lb.unmap_x(x, src_w), lb.unmap_y(y, src_h)])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb_frame(w: usize, h: usize, px: [u8; 3]) -> Vec<u8> {
        (0..w * h).flat_map(|_| px).collect()
    }

    fn cand(class: usize, score: f32, bbox: [f32; 4]) -> Candidate {
        Candidate { class, score, bbox, anchor: 0 }
    }

    // ---- letterbox ----

    #[test]
    fn letterbox_of_16_by_9_frame_pads_top_and_bottom() {
        let lb = Letterbox::new(1280, 720);
        assert_eq!((lb.new_w, lb.new_h, lb.left, lb.top), (640, 360, 0, 140));
        assert_eq!(lb.gain, 0.5);
    }

    #[test]
    fn letterbox_of_portrait_frame_pads_left_and_right() {
        let lb = Letterbox::new(720, 1280);
        assert_eq!((lb.new_w, lb.new_h, lb.left, lb.top), (360, 640, 140, 0));
    }

    #[test]
    fn letterbox_upscales_small_frames_like_ultralytics() {
        let lb = Letterbox::new(320, 320);
        assert_eq!((lb.new_w, lb.new_h, lb.left, lb.top), (640, 640, 0, 0));
        assert_eq!(lb.gain, 2.0);
    }

    #[test]
    fn letterbox_splits_odd_padding_with_the_ultralytics_rule() {
        // 640x639 -> gain 1, new_h 639, dh = 0.5: top = round(0.4) = 0.
        let lb = Letterbox::new(640, 639);
        assert_eq!((lb.new_h, lb.top), (639, 0));
    }

    #[test]
    fn letterbox_never_produces_a_zero_sized_image() {
        let lb = Letterbox::new(4096, 2);
        assert!(lb.new_w >= 1 && lb.new_h >= 1);
    }

    #[test]
    fn round_half_even_matches_python_round() {
        assert_eq!(round_half_even(0.5), 0.0);
        assert_eq!(round_half_even(1.5), 2.0);
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(139.9), 140.0);
        assert_eq!(round_half_even(-0.1), 0.0);
    }

    #[test]
    fn letterbox_into_fills_padding_gray_and_image_from_pixels() {
        let src = rgb_frame(1280, 720, [255, 0, 0]);
        let mut dst = vec![0.0f32; 3 * INPUT_SIZE * INPUT_SIZE];
        let lb = letterbox_into(&src, 1280, 720, &mut dst);
        let plane = INPUT_SIZE * INPUT_SIZE;
        let at = |c: usize, x: usize, y: usize| dst[c * plane + y * INPUT_SIZE + x];
        // Padding above the image is gray 114/255 on every channel.
        for c in 0..3 {
            assert!((at(c, 10, 5) - 114.0 / 255.0).abs() < 1e-6);
        }
        // Inside the image a solid red frame stays red.
        let (x, y) = (lb.left + 100, lb.top + 100);
        assert!((at(0, x, y) - 1.0).abs() < 1e-6);
        assert!(at(1, x, y).abs() < 1e-6);
        assert!(at(2, x, y).abs() < 1e-6);
    }

    #[test]
    fn letterbox_into_samples_bilinearly_like_opencv() {
        // 2x1 frame, black then white, upscaled 320x: mid-point blends 50/50.
        let src = vec![0, 0, 0, 255, 255, 255];
        let mut dst = vec![0.0f32; 3 * INPUT_SIZE * INPUT_SIZE];
        let lb = letterbox_into(&src, 2, 1, &mut dst);
        let row = lb.top * INPUT_SIZE;
        assert!(dst[row].abs() < 1e-6, "left edge stays black");
        assert!((dst[row + INPUT_SIZE - 1] - 1.0).abs() < 1e-6, "right edge stays white");
        let mid = dst[row + 319] + dst[row + 320];
        assert!((mid - 1.0).abs() < 0.01, "center straddles 0.5, got {mid}");
    }

    #[test]
    fn unmap_inverts_the_letterbox_and_clips_to_the_frame() {
        let lb = Letterbox::new(1280, 720);
        assert_eq!(lb.unmap_x(320.0, 1280), 640.0);
        assert_eq!(lb.unmap_y(140.0 + 180.0, 720), 360.0);
        assert_eq!(lb.unmap_x(-50.0, 1280), 0.0);
        assert_eq!(lb.unmap_y(9999.0, 720), 720.0);
    }

    // ---- iou ----

    #[test]
    fn iou_of_identical_boxes_is_one() {
        assert_eq!(iou(&[0.0, 0.0, 10.0, 10.0], &[0.0, 0.0, 10.0, 10.0]), 1.0);
    }

    #[test]
    fn iou_of_disjoint_boxes_is_zero() {
        assert_eq!(iou(&[0.0, 0.0, 10.0, 10.0], &[20.0, 20.0, 30.0, 30.0]), 0.0);
    }

    #[test]
    fn iou_of_half_overlap_is_one_third() {
        let v = iou(&[0.0, 0.0, 10.0, 10.0], &[5.0, 0.0, 15.0, 10.0]);
        assert!((v - 1.0 / 3.0).abs() < 1e-6, "got {v}");
    }

    #[test]
    fn iou_of_degenerate_boxes_is_zero_not_nan() {
        let v = iou(&[5.0, 5.0, 5.0, 5.0], &[5.0, 5.0, 5.0, 5.0]);
        assert_eq!(v, 0.0);
    }

    // ---- decode ----

    /// Two anchors, two classes, no real masks: `[4 + 2 + 32, 2]`.
    fn tiny_output() -> Vec<f32> {
        let anchors = 2;
        let rows = 4 + 2 + MASK_COEFFS;
        let mut out = vec![0.0f32; rows * anchors];
        // anchor 0: box center (100,100) size 40x20, class 1 score 0.9
        out[0] = 100.0;
        out[anchors] = 100.0;
        out[2 * anchors] = 40.0;
        out[3 * anchors] = 20.0;
        out[(4 + 1) * anchors] = 0.9;
        // anchor 1: box center (300,200) size 10x10, class 0 score 0.3
        out[1] = 300.0;
        out[anchors + 1] = 200.0;
        out[2 * anchors + 1] = 10.0;
        out[3 * anchors + 1] = 10.0;
        out[4 * anchors + 1] = 0.3;
        out
    }

    #[test]
    fn decode_converts_center_size_to_corners_and_picks_the_argmax_class() {
        let cands = decode_candidates(&tiny_output(), 2, 2, 0.25, None);
        assert_eq!(cands.len(), 2);
        assert_eq!(cands[0].class, 1);
        assert_eq!(cands[0].bbox, [80.0, 90.0, 120.0, 110.0]);
        assert!((cands[0].score - 0.9).abs() < 1e-6);
        assert_eq!(cands[1].class, 0);
        assert_eq!(cands[1].anchor, 1);
    }

    #[test]
    fn decode_requires_score_strictly_above_conf() {
        let cands = decode_candidates(&tiny_output(), 2, 2, 0.3, None);
        assert_eq!(cands.len(), 1, "0.3 is not > 0.3");
        assert_eq!(cands[0].class, 1);
    }

    #[test]
    fn decode_class_filter_applies_to_the_argmax_class() {
        let allowed = [true, false];
        let cands = decode_candidates(&tiny_output(), 2, 2, 0.25, Some(&allowed));
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].class, 0);
    }

    #[test]
    #[should_panic(expected = "output0 too short")]
    fn decode_rejects_a_truncated_tensor() {
        decode_candidates(&[0.0; 10], 2, 2, 0.25, None);
    }

    // ---- nms ----

    #[test]
    fn nms_drops_overlapping_lower_scores_of_the_same_class() {
        let kept = nms(
            vec![
                cand(0, 0.6, [0.0, 0.0, 10.0, 10.0]),
                cand(0, 0.9, [1.0, 1.0, 11.0, 11.0]),
                cand(0, 0.5, [100.0, 100.0, 110.0, 110.0]),
            ],
            0.5,
            300,
        );
        assert_eq!(kept.len(), 2);
        assert!((kept[0].score - 0.9).abs() < 1e-6, "highest score first");
        assert!((kept[1].score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn nms_keeps_overlapping_boxes_of_different_classes() {
        let kept = nms(
            vec![cand(0, 0.9, [0.0, 0.0, 10.0, 10.0]), cand(1, 0.8, [0.0, 0.0, 10.0, 10.0])],
            0.5,
            300,
        );
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn nms_respects_max_det() {
        let cands = (0..10).map(|i| cand(0, 0.5 + i as f32 / 100.0, [i as f32 * 50.0, 0.0, i as f32 * 50.0 + 10.0, 10.0])).collect();
        assert_eq!(nms(cands, 0.5, 3).len(), 3);
    }

    #[test]
    fn nms_threshold_is_strictly_greater_than() {
        // IoU of these two is exactly 1/3; at threshold 1/3 the second survives.
        let a = cand(0, 0.9, [0.0, 0.0, 10.0, 10.0]);
        let b = cand(0, 0.8, [5.0, 0.0, 15.0, 10.0]);
        assert_eq!(nms(vec![a.clone(), b.clone()], 1.0 / 3.0 + 1e-4, 300).len(), 2);
        assert_eq!(nms(vec![a, b], 0.3, 300).len(), 1);
    }

    #[test]
    fn nms_of_nothing_is_nothing() {
        assert!(nms(Vec::new(), 0.5, 300).is_empty());
    }

    // ---- mask decode and outline ----

    /// Prototypes where channel 0 is +1 inside a square and -1 outside.
    fn square_protos(lo: usize, hi: usize) -> Vec<f32> {
        let (h, w) = (160, 160);
        let mut data = vec![0.0f32; MASK_COEFFS * h * w];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = if (lo..hi).contains(&x) && (lo..hi).contains(&y) { 1.0 } else { -1.0 };
            }
        }
        data
    }

    fn one_hot_coeffs() -> Vec<f32> {
        let mut c = vec![0.0; MASK_COEFFS];
        c[0] = 1.0;
        c
    }

    #[test]
    fn decode_mask_covers_the_positive_prototype_area() {
        let data = square_protos(40, 80); // 160..320 in model pixels
        let protos = Protos { data: &data, channels: MASK_COEFFS, height: 160, width: 160 };
        let region = decode_mask(&protos, &one_hot_coeffs(), &[150.0, 150.0, 330.0, 330.0]).unwrap();
        let on = region.bits.iter().filter(|&&b| b).count();
        let expected = 160 * 160;
        assert!(on.abs_diff(expected) < expected / 20, "on={on} expected~{expected}");
    }

    #[test]
    fn decode_mask_is_cropped_to_the_box() {
        // Whole prototype is positive, but the box only spans 200..300.
        let data = vec![1.0f32; MASK_COEFFS * 160 * 160];
        let protos = Protos { data: &data, channels: MASK_COEFFS, height: 160, width: 160 };
        let region = decode_mask(&protos, &one_hot_coeffs(), &[200.0, 200.0, 300.0, 300.0]).unwrap();
        assert_eq!((region.x0, region.x0 + region.width), (200, 300));
        assert_eq!((region.y0, region.y0 + region.height), (200, 300));
        assert!(region.bits.iter().all(|&b| b), "an all-positive prototype fills the whole box");
    }

    #[test]
    fn decode_mask_of_an_empty_box_is_none() {
        let data = vec![1.0f32; MASK_COEFFS * 160 * 160];
        let protos = Protos { data: &data, channels: MASK_COEFFS, height: 160, width: 160 };
        assert!(decode_mask(&protos, &one_hot_coeffs(), &[10.0, 10.0, 10.0, 10.0]).is_none());
    }

    #[test]
    fn decode_mask_crop_bounds_follow_the_ultralytics_comparison() {
        // `x1 <= x < x2`: a fractional box keeps ceil(x1) up to ceil(x2) - 1.
        let data = vec![1.0f32; MASK_COEFFS * 160 * 160];
        let protos = Protos { data: &data, channels: MASK_COEFFS, height: 160, width: 160 };
        let region = decode_mask(&protos, &one_hot_coeffs(), &[200.4, 100.0, 300.5, 101.0]).unwrap();
        assert_eq!((region.x0, region.x0 + region.width), (201, 301));
        assert_eq!((region.y0, region.y0 + region.height), (100, 101));
    }

    #[test]
    fn decode_mask_clips_boxes_that_hang_off_the_frame() {
        let data = vec![1.0f32; MASK_COEFFS * 160 * 160];
        let protos = Protos { data: &data, channels: MASK_COEFFS, height: 160, width: 160 };
        let region = decode_mask(&protos, &one_hot_coeffs(), &[-30.0, -5.0, 700.0, 660.0]).unwrap();
        assert_eq!((region.x0, region.width, region.y0, region.height), (0, 640, 0, 640));
    }

    #[test]
    fn decode_mask_of_all_negative_logits_is_empty() {
        let data = vec![-1.0f32; MASK_COEFFS * 160 * 160];
        let protos = Protos { data: &data, channels: MASK_COEFFS, height: 160, width: 160 };
        let region = decode_mask(&protos, &one_hot_coeffs(), &[200.0, 200.0, 300.0, 300.0]).unwrap();
        assert!(region.bits.iter().all(|&b| !b));
        assert!(largest_contour(&region).is_empty());
    }

    fn region_from_rows(rows: &[&str]) -> MaskRegion {
        let bits: Vec<bool> = rows.iter().flat_map(|r| r.chars().map(|c| c == '#')).collect();
        MaskRegion { x0: 0, y0: 0, width: rows[0].len(), height: rows.len(), bits }
    }

    #[test]
    fn contour_of_a_rectangle_is_its_four_corners() {
        let region = region_from_rows(&["......", ".####.", ".####.", ".####.", "......"]);
        let mut c = largest_contour(&region);
        c.sort();
        assert_eq!(c, vec![(1, 1), (1, 3), (4, 1), (4, 3)]);
    }

    #[test]
    fn contour_of_a_single_pixel_is_that_pixel() {
        let region = region_from_rows(&["...", ".#.", "..."]);
        assert_eq!(largest_contour(&region), vec![(1, 1)]);
    }

    #[test]
    fn contour_of_a_horizontal_line_is_its_two_ends() {
        let region = region_from_rows(&["....", ".###", "...."]);
        let mut c = largest_contour(&region);
        c.sort();
        assert_eq!(c, vec![(1, 1), (3, 1)]);
    }

    #[test]
    fn contour_picks_the_largest_component() {
        let region = region_from_rows(&["#.......", "........", "..#####.", "..#####.", "..#####."]);
        let c = largest_contour(&region);
        assert!(c.iter().all(|&(x, y)| x >= 2 && y >= 2), "picked the small blob: {c:?}");
    }

    #[test]
    fn contour_follows_an_l_shape_and_ignores_holes() {
        let region = region_from_rows(&[
            "#####", //
            "#...#", // inner hole
            "#.#.#", //
            "#...#", //
            "#####",
        ]);
        let c = largest_contour(&region);
        // The outer square's corners are present; hole boundary points are not.
        for corner in [(0, 0), (4, 0), (0, 4), (4, 4)] {
            assert!(c.contains(&corner), "{corner:?} missing from {c:?}");
        }
        assert!(!c.iter().any(|&(x, y)| (1..=3).contains(&x) && (1..=3).contains(&y)));
    }

    #[test]
    fn contour_of_a_diagonal_run_keeps_only_its_ends() {
        let region = region_from_rows(&["#...", ".#..", "..#.", "...#"]);
        let mut c = largest_contour(&region);
        c.sort();
        assert_eq!(c, vec![(0, 0), (3, 3)]);
    }

    #[test]
    fn contour_encloses_the_area_of_a_filled_disc() {
        let n = 41;
        let bits: Vec<bool> = (0..n * n)
            .map(|i| {
                let (x, y) = ((i % n) as f32 - 20.0, (i / n) as f32 - 20.0);
                x * x + y * y <= 15.0 * 15.0
            })
            .collect();
        let region = MaskRegion { x0: 0, y0: 0, width: n, height: n, bits };
        let poly: Vec<(f32, f32)> = largest_contour(&region).into_iter().map(|(x, y)| (x as f32, y as f32)).collect();
        let area = polygon_area(&poly);
        let disc = std::f32::consts::PI * 15.0 * 15.0;
        assert!((area - disc).abs() < disc * 0.1, "area {area} vs disc {disc}");
    }

    fn polygon_area(p: &[(f32, f32)]) -> f32 {
        let n = p.len();
        (0..n).map(|i| p[i].0 * p[(i + 1) % n].1 - p[(i + 1) % n].0 * p[i].1).sum::<f32>().abs() / 2.0
    }

    // ---- simplify ----

    #[test]
    fn simplify_removes_collinear_points_and_keeps_corners() {
        let square: Vec<(f32, f32)> = (0..10)
            .map(|i| (i as f32, 0.0))
            .chain((1..10).map(|i| (9.0, i as f32)))
            .chain((0..9).rev().map(|i| (i as f32, 9.0)))
            .chain((1..9).rev().map(|i| (0.0, i as f32)))
            .collect();
        let out = simplify_closed(&square, 0.5);
        assert!(out.len() <= 5, "{} points left", out.len());
        assert!((polygon_area(&out) - 81.0).abs() < 1.0);
    }

    #[test]
    fn simplify_with_zero_tolerance_is_identity() {
        let pts: Vec<(f32, f32)> = (0..20).map(|i| (i as f32, (i % 3) as f32)).collect();
        assert_eq!(simplify_closed(&pts, 0.0), pts);
    }

    #[test]
    fn simplify_never_returns_fewer_than_three_points() {
        let pts: Vec<(f32, f32)> = (0..30).map(|i| (i as f32 * 0.01, 0.0)).collect();
        assert!(simplify_closed(&pts, 5.0).len() >= 3);
    }

    // ---- postprocess ----

    #[test]
    fn postprocess_maps_boxes_back_to_the_source_frame() {
        let lb = Letterbox::new(1280, 720);
        let dets = postprocess(&tiny_output(), &[1, 38, 2], None, &lb, 1280, 720, &Params { masks: false, ..Params::default() }).unwrap();
        assert_eq!(dets.len(), 2, "different classes do not suppress each other");
        // Best first. Model box [80,90,120,110]: x / 0.5, (y - 140) / 0.5 -> clipped at 0 for y.
        assert_eq!(dets[0].bbox, [160.0, 0.0, 240.0, 0.0]);
        assert_eq!(dets[0].class, 1);
        assert!(dets[0].polygon.is_empty());
        assert_eq!(dets[1].class, 0);
    }

    #[test]
    fn postprocess_rejects_a_tensor_that_is_not_yolo_seg_shaped() {
        let lb = Letterbox::new(640, 640);
        assert!(postprocess(&[0.0; 10], &[1, 3], None, &lb, 640, 640, &Params::default()).is_err());
        assert!(postprocess(&[0.0; 10], &[1, 10, 1], None, &lb, 640, 640, &Params::default()).is_err());
    }

    #[test]
    fn postprocess_builds_a_polygon_from_a_synthetic_mask() {
        // One anchor: box (160..320)^2 in model space, class 0, mask = square protos.
        let anchors = 1;
        let rows = 4 + 1 + MASK_COEFFS;
        let mut out0 = vec![0.0f32; rows * anchors];
        out0[0] = 240.0;
        out0[1] = 240.0;
        out0[2] = 160.0;
        out0[3] = 160.0;
        out0[4] = 0.9;
        out0[5] = 1.0; // coefficient 0
        let data = square_protos(40, 80);
        let protos = Protos { data: &data, channels: MASK_COEFFS, height: 160, width: 160 };
        let lb = Letterbox::new(640, 640);
        let dets = postprocess(&out0, &[1, rows as i64, 1], Some(protos), &lb, 640, 640, &Params::default()).unwrap();
        assert_eq!(dets.len(), 1);
        let poly: Vec<(f32, f32)> = dets[0].polygon.iter().map(|p| (p[0], p[1])).collect();
        assert!(poly.len() >= 4, "polygon has {} points", poly.len());
        let area = polygon_area(&poly);
        assert!((area - 160.0 * 160.0).abs() < 160.0 * 160.0 * 0.08, "area {area}");
    }
}
