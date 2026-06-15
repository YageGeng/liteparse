use crate::preprocess::Letterbox;
use crate::types::LayoutDetection;

#[derive(Debug, Clone, PartialEq)]
pub struct DetectionCandidate {
    pub label: String,
    pub confidence: f32,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

pub fn restore_box_to_page(
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    letterbox: &Letterbox,
    page_width: f32,
    page_height: f32,
) -> (f32, f32, f32, f32) {
    let image_x =
        ((x - letterbox.pad_x) / letterbox.scale).clamp(0.0, letterbox.input_width as f32);
    let image_y =
        ((y - letterbox.pad_y) / letterbox.scale).clamp(0.0, letterbox.input_height as f32);
    let image_right =
        ((x + width - letterbox.pad_x) / letterbox.scale).clamp(0.0, letterbox.input_width as f32);
    let image_bottom = ((y + height - letterbox.pad_y) / letterbox.scale)
        .clamp(0.0, letterbox.input_height as f32);

    let scale_x = page_width / letterbox.input_width as f32;
    let scale_y = page_height / letterbox.input_height as f32;

    (
        image_x * scale_x,
        image_y * scale_y,
        (image_right - image_x).max(0.0) * scale_x,
        (image_bottom - image_y).max(0.0) * scale_y,
    )
}

pub fn non_max_suppression(
    mut candidates: Vec<DetectionCandidate>,
    iou_threshold: f32,
) -> Vec<LayoutDetection> {
    candidates.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<DetectionCandidate> = Vec::new();
    'candidate: for candidate in candidates {
        for existing in &kept {
            if candidate.label == existing.label && iou(&candidate, existing) > iou_threshold {
                continue 'candidate;
            }
        }
        kept.push(candidate);
    }

    kept.into_iter()
        .map(|candidate| LayoutDetection {
            label: candidate.label,
            confidence: candidate.confidence,
            x: candidate.x,
            y: candidate.y,
            width: candidate.width,
            height: candidate.height,
        })
        .collect()
}

fn iou(a: &DetectionCandidate, b: &DetectionCandidate) -> f32 {
    let x_overlap = (a.x + a.width).min(b.x + b.width) - a.x.max(b.x);
    let y_overlap = (a.y + a.height).min(b.y + b.height) - a.y.max(b.y);
    let intersection = x_overlap.max(0.0) * y_overlap.max(0.0);
    let union = a.width * a.height + b.width * b.height - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(label: &str, confidence: f32, x: f32, y: f32) -> DetectionCandidate {
        DetectionCandidate {
            label: label.into(),
            confidence,
            x,
            y,
            width: 100.0,
            height: 100.0,
        }
    }

    #[test]
    fn restores_box_coordinates_from_letterbox_to_page_space() {
        let letterbox = Letterbox::new(1000, 500, 1280);

        let (x, y, width, height) =
            restore_box_to_page(128.0, 448.0, 256.0, 128.0, &letterbox, 500.0, 250.0);

        assert!((x - 50.0).abs() < 0.001);
        assert!((y - 50.0).abs() < 0.001);
        assert!((width - 100.0).abs() < 0.001);
        assert!((height - 50.0).abs() < 0.001);
    }

    #[test]
    fn nms_keeps_highest_confidence_overlapping_detection() {
        let kept = non_max_suppression(
            vec![
                candidate("text", 0.80, 10.0, 10.0),
                candidate("text", 0.95, 12.0, 12.0),
                candidate("table", 0.70, 12.0, 12.0),
                candidate("text", 0.60, 300.0, 300.0),
            ],
            0.5,
        );

        assert_eq!(kept.len(), 3);
        assert_eq!(kept[0].confidence, 0.95);
        assert!(kept.iter().any(|d| d.label == "table"));
        assert!(kept.iter().any(|d| d.x == 300.0));
    }
}
