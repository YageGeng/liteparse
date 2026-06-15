use crate::types::{LayoutBlock, TextItem};

const MIN_OVERLAP_RATIO: f32 = 0.5;

pub fn assign_text_items_to_layout_blocks(items: &mut [TextItem], blocks: &[LayoutBlock]) {
    for item in items {
        item.layout_block_id = None;
        item.layout_label = None;

        let mut best: Option<(&LayoutBlock, f32)> = None;
        for block in blocks {
            let item_area = item.width * item.height;
            let overlap_ratio = if item_area > 0.0 {
                intersection_area(
                    item.x,
                    item.y,
                    item.width,
                    item.height,
                    block.x,
                    block.y,
                    block.width,
                    block.height,
                ) / item_area
            } else {
                0.0
            };

            if best.is_none_or(|(_, best_ratio)| overlap_ratio > best_ratio) {
                best = Some((block, overlap_ratio));
            }
        }

        let Some((block, overlap_ratio)) = best else {
            continue;
        };

        if overlap_ratio >= MIN_OVERLAP_RATIO
            || contains_point(block, item.x + item.width / 2.0, item.y + item.height / 2.0)
        {
            item.layout_block_id = Some(block.id);
            item.layout_label = Some(block.label.clone());
        }
    }
}

fn intersection_area(
    ax: f32,
    ay: f32,
    aw: f32,
    ah: f32,
    bx: f32,
    by: f32,
    bw: f32,
    bh: f32,
) -> f32 {
    let overlap_x = (ax + aw).min(bx + bw) - ax.max(bx);
    let overlap_y = (ay + ah).min(by + bh) - ay.max(by);
    overlap_x.max(0.0) * overlap_y.max(0.0)
}

fn contains_point(block: &LayoutBlock, x: f32, y: f32) -> bool {
    x >= block.x && x <= block.x + block.width && y >= block.y && y <= block.y + block.height
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

    fn item(x: f32, y: f32, width: f32, height: f32) -> TextItem {
        TextItem {
            text: "Revenue".into(),
            x,
            y,
            width,
            height,
            ..Default::default()
        }
    }

    #[test]
    fn assigns_text_item_to_block_with_largest_overlap() {
        let mut items = vec![item(12.0, 12.0, 40.0, 10.0)];
        let blocks = vec![
            block(0, "table", 0.0, 0.0, 20.0, 20.0),
            block(1, "text", 10.0, 10.0, 60.0, 20.0),
        ];

        assign_text_items_to_layout_blocks(&mut items, &blocks);

        assert_eq!(items[0].layout_block_id, Some(1));
        assert_eq!(items[0].layout_label.as_deref(), Some("text"));
    }

    #[test]
    fn assigns_text_item_by_center_point_when_overlap_is_small() {
        let mut items = vec![item(48.0, 48.0, 4.0, 4.0)];
        let blocks = vec![block(7, "caption", 0.0, 0.0, 100.0, 100.0)];

        assign_text_items_to_layout_blocks(&mut items, &blocks);

        assert_eq!(items[0].layout_block_id, Some(7));
        assert_eq!(items[0].layout_label.as_deref(), Some("caption"));
    }

    #[test]
    fn leaves_unmatched_text_item_unassigned() {
        let mut items = vec![item(200.0, 200.0, 10.0, 10.0)];
        let blocks = vec![block(0, "text", 0.0, 0.0, 50.0, 50.0)];

        assign_text_items_to_layout_blocks(&mut items, &blocks);

        assert_eq!(items[0].layout_block_id, None);
        assert_eq!(items[0].layout_label, None);
    }
}
