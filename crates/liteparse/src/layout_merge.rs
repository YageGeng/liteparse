use crate::types::{LayoutBlock, Page, TextItem};

const MIN_OVERLAP_RATIO: f32 = 0.5;
const PICTURE_LABEL: &str = "Picture";

pub fn assign_text_items_to_layout_blocks(items: &mut [TextItem], blocks: &[LayoutBlock]) {
    for item in items {
        item.layout_block_id = None;
        item.layout_label = None;
        let item_rect = PageRect::from_text_item(item);

        let mut best: Option<(&LayoutBlock, f32)> = None;
        for block in blocks {
            let item_area = item.width * item.height;
            let overlap_ratio = if item_area > 0.0 {
                item_rect.intersection_area(PageRect::from_layout_block(block)) / item_area
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
            || PageRect::from_layout_block(block)
                .contains_point(item.x + item.width / 2.0, item.y + item.height / 2.0)
        {
            item.layout_block_id = Some(block.id);
            item.layout_label = Some(block.label.clone());
        }
    }
}

/// Remove empty non-picture blocks, then re-assign compact block ids.
///
/// This keeps picture blocks even when OCR/text assignment does not place any
/// text items inside (for image-heavy regions), while dropping empty layout
/// blocks that would otherwise produce noisy, blank output entries.
pub fn compact_layout_blocks(blocks: &mut Vec<LayoutBlock>, items: &mut [TextItem]) {
    let mut text_block_ids = std::collections::HashSet::new();

    for item in items.iter() {
        if let Some(block_id) = item.layout_block_id {
            text_block_ids.insert(block_id);
        }
    }

    let mut retained = Vec::with_capacity(blocks.len());
    let mut remap: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

    for block in blocks.iter() {
        if block.label == PICTURE_LABEL || text_block_ids.contains(&block.id) {
            let new_id = retained.len();
            remap.insert(block.id, new_id);

            let mut compacted = block.clone();
            compacted.id = new_id;
            retained.push(compacted);
        } else {
            remap.insert(block.id, usize::MAX);
        }
    }

    for item in items.iter_mut() {
        if let Some(old_id) = item.layout_block_id {
            match remap.get(&old_id).copied() {
                Some(new_id) if new_id != usize::MAX => {
                    item.layout_block_id = Some(new_id);
                }
                _ => {
                    item.layout_block_id = None;
                    item.layout_label = None;
                }
            }
        }
    }

    for item in items.iter_mut() {
        if let Some(id) = item.layout_block_id {
            item.layout_label = retained.get(id).map(|block| block.label.clone());
        }
    }

    *blocks = retained;
}

pub fn project_layout_text(
    page_number: usize,
    page_width: f32,
    page_height: f32,
    text_items: Vec<TextItem>,
) -> String {
    crate::projection::project_pages_to_grid(vec![Page {
        page_number,
        page_width,
        page_height,
        text_items,
        graphics: Vec::new(),
        struct_nodes: Vec::new(),
        image_refs: Vec::new(),
    }])
    .into_iter()
    .next()
    .map(|page| page.text)
    .unwrap_or_default()
}

#[derive(Debug, Clone, Copy)]
struct PageRect {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

impl PageRect {
    /// Create a page-space rectangle from an extracted text item.
    fn from_text_item(item: &TextItem) -> Self {
        Self {
            x: item.x,
            y: item.y,
            width: item.width,
            height: item.height,
        }
    }

    /// Create a page-space rectangle from a detected layout block.
    fn from_layout_block(block: &LayoutBlock) -> Self {
        Self {
            x: block.x,
            y: block.y,
            width: block.width,
            height: block.height,
        }
    }

    /// Return the overlapping area shared with another page-space rectangle.
    fn intersection_area(self, other: Self) -> f32 {
        let overlap_x = (self.x + self.width).min(other.x + other.width) - self.x.max(other.x);
        let overlap_y = (self.y + self.height).min(other.y + other.height) - self.y.max(other.y);
        overlap_x.max(0.0) * overlap_y.max(0.0)
    }

    /// Return whether the point lies inside this page-space rectangle.
    fn contains_point(self, x: f32, y: f32) -> bool {
        x >= self.x && x <= self.x + self.width && y >= self.y && y <= self.y + self.height
    }
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

    #[test]
    fn compacts_empty_blocks_and_reassigns_ids() {
        let mut items = vec![
            TextItem {
                x: 1.0,
                y: 1.0,
                width: 10.0,
                height: 10.0,
                text: "a".into(),
                ..Default::default()
            },
            TextItem {
                x: 60.0,
                y: 60.0,
                width: 10.0,
                height: 10.0,
                text: "b".into(),
                ..Default::default()
            },
        ];
        items[0].layout_block_id = Some(0);
        items[0].layout_label = Some("Text".into());
        items[1].layout_block_id = Some(2);
        items[1].layout_label = Some("Picture".into());

        let mut blocks = vec![
            block(0, "text", 0.0, 0.0, 20.0, 20.0),
            block(1, "title", 30.0, 30.0, 20.0, 20.0),
            block(2, "Picture", 50.0, 50.0, 30.0, 30.0),
        ];

        compact_layout_blocks(&mut blocks, &mut items);

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].id, 0);
        assert_eq!(blocks[0].label, "text");
        assert_eq!(blocks[1].id, 1);
        assert_eq!(blocks[1].label, "Picture");

        assert_eq!(items[0].layout_block_id, Some(0));
        assert_eq!(items[0].layout_label.as_deref(), Some("text"));
        assert_eq!(items[1].layout_block_id, Some(1));
        assert_eq!(items[1].layout_label.as_deref(), Some("Picture"));
    }

    #[test]
    fn compacts_blocks_using_block_ids_not_vector_indexes() {
        let mut items = vec![TextItem {
            x: 60.0,
            y: 60.0,
            width: 10.0,
            height: 10.0,
            text: "b".into(),
            layout_block_id: Some(20),
            layout_label: Some("Table".into()),
            ..Default::default()
        }];
        let mut blocks = vec![
            block(10, "Text", 0.0, 0.0, 20.0, 20.0),
            block(20, "Table", 50.0, 50.0, 30.0, 30.0),
        ];

        compact_layout_blocks(&mut blocks, &mut items);

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].label, "Table");
        assert_eq!(items[0].layout_block_id, Some(0));
        assert_eq!(items[0].layout_label.as_deref(), Some("Table"));
    }
}
