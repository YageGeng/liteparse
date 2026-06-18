use crate::types::LayoutDetection;

/// Axis used for an XY-cut split over detected layout boxes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CutAxis {
    /// Split left-side boxes from right-side boxes.
    Vertical,
    /// Split upper boxes from lower boxes.
    Horizontal,
}

/// Candidate whitespace split between two non-overlapping box regions.
#[derive(Clone, Copy, Debug)]
struct Cut {
    /// Direction of the whitespace band.
    axis: CutAxis,
    /// Page-space coordinate where boxes are partitioned.
    position: f32,
    /// Width or height of the whitespace band.
    gap: f32,
    /// Number of boxes before the split.
    before_count: usize,
    /// Number of boxes after the split.
    after_count: usize,
}

/// Sort YOLO detections using a recursive XY-cut over their page-space boxes.
///
/// The returned order is intended to approximate human reading order for
/// layout regions. In particular, multi-column pages should read a full left
/// column before moving to the right column.
pub fn order_layout_detections_xy_cut(detections: Vec<LayoutDetection>) -> Vec<LayoutDetection> {
    xy_cut(detections)
}

/// Recursively partition detections by the best whitespace cut.
fn xy_cut(detections: Vec<LayoutDetection>) -> Vec<LayoutDetection> {
    if detections.len() <= 1 {
        return detections;
    }

    let Some(cut) = choose_cut(&detections) else {
        // No reliable whitespace split remains, so fall back to plain top-left
        // ordering inside this local region.
        return sort_top_left(detections);
    };

    let mut before = Vec::new();
    let mut after = Vec::new();

    for detection in detections {
        // Use box midpoints instead of starts so wide titles or tables that
        // straddle a cut are grouped with the side containing their center.
        let midpoint = match cut.axis {
            CutAxis::Vertical => detection.x + detection.width * 0.5,
            CutAxis::Horizontal => detection.y + detection.height * 0.5,
        };

        if midpoint < cut.position {
            before.push(detection);
        } else {
            after.push(detection);
        }
    }

    if before.is_empty() || after.is_empty() {
        // Degenerate midpoint grouping can happen with overlapping or unusual
        // boxes. Avoid infinite recursion and keep deterministic output.
        return sort_top_left(before.into_iter().chain(after).collect());
    }

    let mut ordered = xy_cut(before);
    ordered.extend(xy_cut(after));
    ordered
}

/// Choose the split that best models page reading order for the current region.
fn choose_cut(detections: &[LayoutDetection]) -> Option<Cut> {
    let vertical = find_cut(detections, CutAxis::Vertical);
    let horizontal = find_cut(detections, CutAxis::Horizontal);

    match (vertical, horizontal) {
        // Multi-column documents should be read column-by-column. Prefer a
        // strong vertical split when both sides contain multiple blocks.
        (Some(v), Some(_)) if is_column_cut(v) => Some(v),
        // Otherwise, only prefer a vertical split when its whitespace is
        // materially larger than the horizontal alternative.
        (Some(v), Some(h)) if v.gap > h.gap * 1.25 => Some(v),
        (Some(_), Some(h)) => Some(h),
        (Some(v), None) => Some(v),
        (None, Some(h)) => Some(h),
        (None, None) => None,
    }
}

/// Return whether a cut looks like a true multi-column divider.
fn is_column_cut(cut: Cut) -> bool {
    cut.axis == CutAxis::Vertical && cut.before_count >= 2 && cut.after_count >= 2
}

/// Find the largest usable whitespace gap along one axis.
fn find_cut(detections: &[LayoutDetection], axis: CutAxis) -> Option<Cut> {
    let mut intervals: Vec<(f32, f32)> = detections
        .iter()
        .filter_map(|detection| {
            let (start, end) = match axis {
                CutAxis::Vertical => (detection.x, detection.x + detection.width),
                CutAxis::Horizontal => (detection.y, detection.y + detection.height),
            };

            (end > start).then_some((start, end))
        })
        .collect();

    if intervals.len() < 2 {
        return None;
    }

    intervals.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));

    // Merge projected intervals before searching for whitespace so overlapping
    // boxes do not create false gaps inside a connected visual region.
    let mut merged = Vec::new();
    for (start, end) in intervals {
        match merged.last_mut() {
            Some((_, last_end)) if start <= *last_end => {
                *last_end = last_end.max(end);
            }
            _ => merged.push((start, end)),
        }
    }

    let min_gap = match axis {
        CutAxis::Vertical => region_span(detections, axis) * 0.03,
        CutAxis::Horizontal => region_span(detections, axis) * 0.02,
    }
    .max(8.0);

    let mut best: Option<Cut> = None;

    for pair in merged.windows(2) {
        let gap_start = pair[0].1;
        let gap_end = pair[1].0;
        let gap = gap_end - gap_start;
        if gap < min_gap {
            continue;
        }

        // Split in the middle of the whitespace band, then count actual boxes
        // by midpoint to make sure this cut separates the region.
        let position = (gap_start + gap_end) * 0.5;
        let (before_count, after_count) = split_counts(detections, axis, position);
        if before_count == 0 || after_count == 0 {
            continue;
        }

        let candidate = Cut {
            axis,
            position,
            gap,
            before_count,
            after_count,
        };

        if best.is_none_or(|current| candidate.gap > current.gap) {
            best = Some(candidate);
        }
    }

    best
}

/// Count detections on each side of a candidate split.
fn split_counts(detections: &[LayoutDetection], axis: CutAxis, position: f32) -> (usize, usize) {
    let mut before = 0usize;
    let mut after = 0usize;

    for detection in detections {
        let midpoint = match axis {
            CutAxis::Vertical => detection.x + detection.width * 0.5,
            CutAxis::Horizontal => detection.y + detection.height * 0.5,
        };
        if midpoint < position {
            before += 1;
        } else {
            after += 1;
        }
    }

    (before, after)
}

/// Return the occupied span of all detections along one axis.
fn region_span(detections: &[LayoutDetection], axis: CutAxis) -> f32 {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;

    for detection in detections {
        let (start, end) = match axis {
            CutAxis::Vertical => (detection.x, detection.x + detection.width),
            CutAxis::Horizontal => (detection.y, detection.y + detection.height),
        };
        min = min.min(start);
        max = max.max(end);
    }

    (max - min).max(0.0)
}

/// Deterministic fallback order when XY-cut cannot split a region.
fn sort_top_left(mut detections: Vec<LayoutDetection>) -> Vec<LayoutDetection> {
    detections.sort_by(|a, b| a.y.total_cmp(&b.y).then_with(|| a.x.total_cmp(&b.x)));
    detections
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LayoutLabel;

    // Build a compact page-space detection for reading-order tests.
    fn detection(label: LayoutLabel, x: f32, y: f32) -> LayoutDetection {
        LayoutDetection {
            label,
            confidence: 0.9,
            x,
            y,
            width: 180.0,
            height: 40.0,
        }
    }

    // Verifies that XY-cut reads down the left column before the right column.
    #[test]
    fn xy_cut_reads_left_column_before_right_column() {
        let detections = vec![
            detection(LayoutLabel::Text, 40.0, 40.0),
            detection(LayoutLabel::Title, 320.0, 40.0),
            detection(LayoutLabel::ListItem, 40.0, 120.0),
            detection(LayoutLabel::Table, 320.0, 120.0),
        ];

        let ordered = order_layout_detections_xy_cut(detections);
        let labels: Vec<LayoutLabel> = ordered.iter().map(|detection| detection.label).collect();

        assert_eq!(
            labels,
            vec![
                LayoutLabel::Text,
                LayoutLabel::ListItem,
                LayoutLabel::Title,
                LayoutLabel::Table,
            ]
        );
    }
}
