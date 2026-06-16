use crate::types::LayoutBlock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CutAxis {
    Vertical,
    Horizontal,
}

#[derive(Clone, Copy, Debug)]
struct Cut {
    axis: CutAxis,
    position: f32,
    gap: f32,
    before_count: usize,
    after_count: usize,
}

/// Sort layout blocks using a recursive XY-cut over their page-space bounding boxes.
///
/// This is intentionally block-level only: text reconstruction inside each block
/// remains handled by the existing projection/grid code.
pub fn order_layout_blocks_xy_cut(blocks: Vec<LayoutBlock>) -> Vec<LayoutBlock> {
    let mut ordered = xy_cut(blocks);
    for (id, block) in ordered.iter_mut().enumerate() {
        block.id = id;
    }
    ordered
}

fn xy_cut(blocks: Vec<LayoutBlock>) -> Vec<LayoutBlock> {
    if blocks.len() <= 1 {
        return blocks;
    }

    let Some(cut) = choose_cut(&blocks) else {
        return sort_top_left(blocks);
    };

    let mut before = Vec::new();
    let mut after = Vec::new();

    for block in blocks {
        let midpoint = match cut.axis {
            CutAxis::Vertical => block.x + block.width * 0.5,
            CutAxis::Horizontal => block.y + block.height * 0.5,
        };

        if midpoint < cut.position {
            before.push(block);
        } else {
            after.push(block);
        }
    }

    if before.is_empty() || after.is_empty() {
        return sort_top_left(before.into_iter().chain(after).collect());
    }

    let mut ordered = xy_cut(before);
    ordered.extend(xy_cut(after));
    ordered
}

fn choose_cut(blocks: &[LayoutBlock]) -> Option<Cut> {
    let vertical = find_cut(blocks, CutAxis::Vertical);
    let horizontal = find_cut(blocks, CutAxis::Horizontal);

    match (vertical, horizontal) {
        // Multi-column pages should read one column top-to-bottom before the
        // next column. When a vertical whitespace band splits multiple blocks
        // on both sides, prefer it over horizontal row cuts.
        (Some(v), Some(_)) if is_column_cut(v) => Some(v),
        (Some(v), Some(h)) if v.gap > h.gap * 1.25 => Some(v),
        (Some(_), Some(h)) => Some(h),
        (Some(v), None) => Some(v),
        (None, Some(h)) => Some(h),
        (None, None) => None,
    }
}

fn is_column_cut(cut: Cut) -> bool {
    cut.axis == CutAxis::Vertical && cut.before_count >= 2 && cut.after_count >= 2
}

fn find_cut(blocks: &[LayoutBlock], axis: CutAxis) -> Option<Cut> {
    let mut intervals: Vec<(f32, f32)> = blocks
        .iter()
        .filter_map(|block| {
            let (start, end) = match axis {
                CutAxis::Vertical => (block.x, block.x + block.width),
                CutAxis::Horizontal => (block.y, block.y + block.height),
            };

            (end > start).then_some((start, end))
        })
        .collect();

    if intervals.len() < 2 {
        return None;
    }

    intervals.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));

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
        CutAxis::Vertical => region_span(blocks, axis) * 0.03,
        CutAxis::Horizontal => region_span(blocks, axis) * 0.02,
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

        let position = (gap_start + gap_end) * 0.5;
        let (before_count, after_count) = split_counts(blocks, axis, position);
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

fn split_counts(blocks: &[LayoutBlock], axis: CutAxis, position: f32) -> (usize, usize) {
    let mut before = 0usize;
    let mut after = 0usize;

    for block in blocks {
        let midpoint = match axis {
            CutAxis::Vertical => block.x + block.width * 0.5,
            CutAxis::Horizontal => block.y + block.height * 0.5,
        };
        if midpoint < position {
            before += 1;
        } else {
            after += 1;
        }
    }

    (before, after)
}

fn region_span(blocks: &[LayoutBlock], axis: CutAxis) -> f32 {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;

    for block in blocks {
        let (start, end) = match axis {
            CutAxis::Vertical => (block.x, block.x + block.width),
            CutAxis::Horizontal => (block.y, block.y + block.height),
        };
        min = min.min(start);
        max = max.max(end);
    }

    (max - min).max(0.0)
}

fn sort_top_left(mut blocks: Vec<LayoutBlock>) -> Vec<LayoutBlock> {
    blocks.sort_by(|a, b| {
        a.y.total_cmp(&b.y)
            .then_with(|| a.x.total_cmp(&b.x))
            .then_with(|| a.id.cmp(&b.id))
    });
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(id: usize, label: &str, x: f32, y: f32, width: f32, height: f32) -> LayoutBlock {
        LayoutBlock {
            id,
            label: label.into(),
            confidence: 0.9,
            x,
            y,
            width,
            height,
        }
    }

    #[test]
    fn xy_cut_reads_left_column_before_right_column() {
        let blocks = vec![
            block(0, "left-1", 40.0, 40.0, 180.0, 40.0),
            block(1, "right-1", 320.0, 40.0, 180.0, 40.0),
            block(2, "left-2", 40.0, 120.0, 180.0, 40.0),
            block(3, "right-2", 320.0, 120.0, 180.0, 40.0),
        ];

        let ordered = order_layout_blocks_xy_cut(blocks);
        let labels: Vec<&str> = ordered.iter().map(|block| block.label.as_str()).collect();

        assert_eq!(labels, vec!["left-1", "left-2", "right-1", "right-2"]);
        assert_eq!(
            ordered.iter().map(|block| block.id).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn xy_cut_keeps_full_width_title_before_column_body() {
        let blocks = vec![
            block(0, "left-1", 40.0, 120.0, 180.0, 40.0),
            block(1, "right-1", 320.0, 120.0, 180.0, 40.0),
            block(2, "title", 40.0, 40.0, 460.0, 50.0),
            block(3, "left-2", 40.0, 190.0, 180.0, 40.0),
            block(4, "right-2", 320.0, 190.0, 180.0, 40.0),
        ];

        let ordered = order_layout_blocks_xy_cut(blocks);
        let labels: Vec<&str> = ordered.iter().map(|block| block.label.as_str()).collect();

        assert_eq!(
            labels,
            vec!["title", "left-1", "left-2", "right-1", "right-2"]
        );
        assert_eq!(
            ordered.iter().map(|block| block.id).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
    }
}
