//! COCO class names in the id order the YOLO models emit (from `model.names`).

/// Number of classes the shipped model predicts.
pub const NUM_CLASSES: usize = 80;

/// Class id to name, exactly as ultralytics reports them.
pub const COCO_NAMES: [&str; NUM_CLASSES] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// Name for a class id, or `"unknown"` when the id is out of range.
pub fn class_name(id: usize) -> &'static str {
    COCO_NAMES.get(id).copied().unwrap_or("unknown")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_match_the_coco_ids_the_backend_relies_on() {
        assert_eq!(class_name(0), "person");
        assert_eq!(class_name(32), "sports ball");
        assert_eq!(class_name(79), "toothbrush");
    }

    #[test]
    fn out_of_range_id_is_unknown() {
        assert_eq!(class_name(80), "unknown");
    }
}
