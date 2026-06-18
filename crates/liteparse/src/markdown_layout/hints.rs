use std::collections::HashSet;

use crate::markdown_layout::Block;
use crate::types::{GraphicPrimitive, LayoutBlock, ParsedPage, ProjectedLine, Rect};

/// Page-level layout guidance derived from generic LiteParse layout blocks.
///
/// This layer deliberately accepts core [`LayoutBlock`] values instead of YOLO
/// detector types. The markdown classifier consumes only these coarse hints, so
/// model-specific labels and thresholds remain outside the heuristic markdown
/// pipeline.
#[derive(Debug, Clone, Default)]
pub(super) struct MarkdownLayoutHints {
    regions: Vec<LayoutHintRegion>,
}

impl MarkdownLayoutHints {
    /// Build markdown hints from detected page layout blocks.
    pub(super) fn from_page(page: &ParsedPage) -> Self {
        let regions = page
            .layout_blocks
            .iter()
            .filter_map(LayoutHintRegion::from_layout_block)
            .collect();

        Self { regions }
    }

    /// Return whether a line sits inside a region that should not be promoted to a heading.
    pub(super) fn suppresses_heading(&self, line: &ProjectedLine) -> bool {
        self.regions
            .iter()
            .any(|region| region.kind.suppresses_heading() && region.matches_line(line))
    }

    /// Return whether two lines should stay in one paragraph due to a text layout block.
    ///
    /// The geometric paragraph heuristic stays conservative by default. A
    /// detector-provided Text region is allowed to relax that heuristic only
    /// when both adjacent lines are inside the same region, which keeps the
    /// model guidance local and avoids changing unrelated markdown flow.
    pub(super) fn continues_text_block(
        &self,
        previous: &ProjectedLine,
        current: &ProjectedLine,
    ) -> bool {
        self.regions.iter().any(|region| {
            region.kind == LayoutHintKind::Text
                && region.matches_line(previous)
                && region.matches_line(current)
        })
    }

    /// Return whether two lines belong to different detector-provided text blocks.
    ///
    /// This is the inverse boundary signal for [`Self::continues_text_block`]:
    /// when both lines are covered by Text regions but there is no shared
    /// region, markdown should keep them as separate paragraph blocks even if
    /// the geometric paragraph heuristic would normally join them.
    pub(super) fn separates_text_blocks(
        &self,
        previous: &ProjectedLine,
        current: &ProjectedLine,
    ) -> bool {
        let mut previous_has_text_region = false;
        let mut current_has_text_region = false;
        let mut shared_text_region = false;

        for region in self
            .regions
            .iter()
            .filter(|region| region.kind == LayoutHintKind::Text)
        {
            let previous_matches = region.matches_line(previous);
            let current_matches = region.matches_line(current);
            previous_has_text_region |= previous_matches;
            current_has_text_region |= current_matches;
            shared_text_region |= previous_matches && current_matches;
        }

        previous_has_text_region && current_has_text_region && !shared_text_region
    }

    /// Detect page-level tables from table-shaped layout regions.
    ///
    /// The existing markdown table detectors still build the final table block.
    /// Layout hints only choose which lines should be tested together when the
    /// projection region tree split a table into multiple leaves.
    pub(super) fn detect_table_interruptions(
        &self,
        lines: &[ProjectedLine],
        graphics: &[GraphicPrimitive],
        page_width: f32,
        page_height: f32,
        excluded: &HashSet<usize>,
    ) -> HintedTableExtraction {
        let mut extracted = HintedTableExtraction::default();

        for region in self
            .regions
            .iter()
            .filter(|region| region.kind == LayoutHintKind::Table)
        {
            let line_indices = region.matching_line_indices(lines, excluded);
            if line_indices.len() < 2 {
                continue;
            }

            // CONTEXT: A table hint is useful only when it groups lines that
            // normal xy-cut region classification may not see together.
            let region_count = line_indices
                .iter()
                .map(|&idx| &lines[idx].region_path)
                .collect::<std::collections::HashSet<_>>()
                .len();
            if region_count < 2 {
                continue;
            }

            let hinted_lines: Vec<ProjectedLine> =
                line_indices.iter().map(|&idx| lines[idx].clone()).collect();
            let table_runs = super::tables::merge_table_runs(
                super::tables::detect_ruled_tables(
                    &hinted_lines,
                    graphics,
                    page_width,
                    page_height,
                ),
                super::tables::detect_tables(&hinted_lines),
            );

            for run in table_runs {
                if !HintedTableExtraction::is_useful_table_block(&run.block) {
                    continue;
                }

                let consumed: Vec<usize> = line_indices[run.start..run.end].to_vec();
                if consumed
                    .iter()
                    .any(|idx| excluded.contains(idx) || extracted.consumed.contains(idx))
                {
                    continue;
                }

                let top_y = consumed
                    .iter()
                    .map(|&idx| lines[idx].bbox.y)
                    .fold(f32::INFINITY, f32::min);
                extracted.consumed.extend(consumed);
                extracted.tables.push((top_y, run.block));
            }
        }

        extracted
    }
}

/// Tables and consumed line indices recovered from layout-guided detection.
#[derive(Debug, Clone, Default)]
pub(super) struct HintedTableExtraction {
    /// Table blocks emitted as page-level interruptions at their top y position.
    pub(super) tables: Vec<(f32, Block)>,
    /// Original line indices consumed by the recovered table blocks.
    pub(super) consumed: HashSet<usize>,
}

impl HintedTableExtraction {
    /// Return whether a table detector result is worth emitting from a layout hint.
    fn is_useful_table_block(block: &Block) -> bool {
        match block {
            Block::Table { header, rows } => {
                let columns = header
                    .as_ref()
                    .map(Vec::len)
                    .or_else(|| rows.first().map(Vec::len))
                    .unwrap_or(0);
                columns >= 2
            }
            Block::GridFallback { lines } => lines.len() >= 2,
            _ => false,
        }
    }
}

/// One page-space hint region consumed by markdown heuristics.
#[derive(Debug, Clone)]
struct LayoutHintRegion {
    kind: LayoutHintKind,
    rect: Rect,
}

impl LayoutHintRegion {
    /// Convert a public layout block into a markdown hint region.
    fn from_layout_block(block: &LayoutBlock) -> Option<Self> {
        let kind = LayoutHintKind::from_label(&block.label)?;
        Some(Self {
            kind,
            rect: Rect {
                x: block.x,
                y: block.y,
                width: block.width,
                height: block.height,
            },
        })
    }

    /// Return whether a projected line belongs to this hint region.
    fn matches_line(&self, line: &ProjectedLine) -> bool {
        let line_rect = &line.bbox;
        let line_area = line_rect.width * line_rect.height;
        if line_area <= 0.0 {
            return self.rect.contains_center_of(line_rect);
        }

        self.rect.intersection_area(line_rect) / line_area >= 0.5
            || self.rect.contains_center_of(line_rect)
    }

    /// Return all non-excluded line indices that belong to this region.
    fn matching_line_indices(
        &self,
        lines: &[ProjectedLine],
        excluded: &HashSet<usize>,
    ) -> Vec<usize> {
        lines
            .iter()
            .enumerate()
            .filter(|(idx, line)| !excluded.contains(idx) && self.matches_line(line))
            .map(|(idx, _)| idx)
            .collect()
    }
}

/// Markdown-relevant layout region kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayoutHintKind {
    Text,
    Table,
    Figure,
    Formula,
}

impl LayoutHintKind {
    /// Convert a layout label string into a markdown hint kind.
    fn from_label(label: &str) -> Option<Self> {
        match label {
            "Text" => Some(Self::Text),
            "Table" => Some(Self::Table),
            "Picture" => Some(Self::Figure),
            "Formula" => Some(Self::Formula),
            _ => None,
        }
    }

    /// Return whether lines in this region should avoid heuristic heading promotion.
    fn suppresses_heading(self) -> bool {
        matches!(self, Self::Table | Self::Figure | Self::Formula)
    }
}

/// Geometry helpers used only by markdown layout hints.
trait RectExt {
    /// Return the overlap area shared by two page-space rectangles.
    fn intersection_area(&self, other: &Rect) -> f32;

    /// Return whether this rectangle contains the center point of another rectangle.
    fn contains_center_of(&self, other: &Rect) -> bool;
}

impl RectExt for Rect {
    fn intersection_area(&self, other: &Rect) -> f32 {
        let overlap_x = (self.x + self.width).min(other.x + other.width) - self.x.max(other.x);
        let overlap_y = (self.y + self.height).min(other.y + other.height) - self.y.max(other.y);
        overlap_x.max(0.0) * overlap_y.max(0.0)
    }

    fn contains_center_of(&self, other: &Rect) -> bool {
        let center_x = other.x + other.width * 0.5;
        let center_y = other.y + other.height * 0.5;
        center_x >= self.x
            && center_x <= self.x + self.width
            && center_y >= self.y
            && center_y <= self.y + self.height
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Anchor, ProjectedLine};

    /// Build a compact projected line for hint geometry tests.
    fn line(text: &str, x: f32, y: f32) -> ProjectedLine {
        ProjectedLine {
            text: text.into(),
            bbox: Rect {
                x,
                y,
                width: 80.0,
                height: 10.0,
            },
            anchor: Anchor::Left,
            indent_x: x,
            dominant_font_size: 10.0,
            font_size_is_estimated: false,
            heading_font_size: None,
            dominant_font_name: None,
            all_bold: false,
            all_italic: false,
            all_mono: false,
            all_strike: false,
            spans: Vec::new(),
            region_path: Vec::new(),
            mcid: None,
            in_figure: false,
        }
    }

    /// Build a compact layout block for hint conversion tests.
    fn block(label: &str) -> LayoutBlock {
        LayoutBlock {
            id: 0,
            label: label.into(),
            confidence: 0.9,
            x: 0.0,
            y: 0.0,
            width: 200.0,
            height: 100.0,
        }
    }

    #[test]
    fn suppresses_heading_inside_picture_formula_or_table_regions() {
        let mut page =
            crate::markdown_layout::test_helpers::page(vec![line("Caption", 20.0, 20.0)]);
        page.layout_blocks = vec![block("Picture")];

        let hints = MarkdownLayoutHints::from_page(&page);

        assert!(hints.suppresses_heading(&page.projected_lines[0]));
    }

    #[test]
    fn ignores_non_markdown_layout_labels() {
        let mut page = crate::markdown_layout::test_helpers::page(vec![line("Body", 20.0, 20.0)]);
        page.layout_blocks = vec![block("Caption")];

        let hints = MarkdownLayoutHints::from_page(&page);

        assert!(!hints.suppresses_heading(&page.projected_lines[0]));
    }

    #[test]
    /// Text layout hints join adjacent projected lines only inside the same region.
    fn continues_lines_inside_the_same_text_region() {
        let mut page = crate::markdown_layout::test_helpers::page(vec![
            line("First text line", 20.0, 20.0),
            line("second text line", 22.0, 70.0),
            line("outside text line", 20.0, 140.0),
        ]);
        page.layout_blocks = vec![block("Text")];

        let hints = MarkdownLayoutHints::from_page(&page);

        assert!(hints.continues_text_block(&page.projected_lines[0], &page.projected_lines[1]));
        assert!(!hints.continues_text_block(&page.projected_lines[1], &page.projected_lines[2]));
    }

    #[test]
    /// Separate Text layout regions keep adjacent projected lines split.
    fn separates_lines_inside_different_text_regions() {
        let mut page = crate::markdown_layout::test_helpers::page(vec![
            line("First text block", 20.0, 20.0),
            line("second text block", 22.0, 70.0),
        ]);
        page.layout_blocks = vec![
            LayoutBlock {
                id: 0,
                label: "Text".into(),
                confidence: 0.9,
                x: 0.0,
                y: 0.0,
                width: 200.0,
                height: 50.0,
            },
            LayoutBlock {
                id: 1,
                label: "Text".into(),
                confidence: 0.9,
                x: 0.0,
                y: 60.0,
                width: 200.0,
                height: 50.0,
            },
        ];

        let hints = MarkdownLayoutHints::from_page(&page);

        assert!(hints.separates_text_blocks(&page.projected_lines[0], &page.projected_lines[1]));
    }

    #[test]
    fn detects_table_across_split_regions_when_table_hint_groups_lines() {
        let mut lines = vec![
            crate::markdown_layout::test_helpers::line_with_spans(
                &[("Name", 50.0), ("Scores", 150.0)],
                20.0,
                10.0,
            ),
            crate::markdown_layout::test_helpers::line_with_spans(
                &[("A", 50.0), ("1", 150.0), ("2", 250.0)],
                35.0,
                10.0,
            ),
            crate::markdown_layout::test_helpers::line_with_spans(
                &[("B", 50.0), ("3", 150.0), ("4", 250.0)],
                50.0,
                10.0,
            ),
            crate::markdown_layout::test_helpers::line_with_spans(
                &[("C", 50.0), ("5", 150.0), ("6", 250.0)],
                65.0,
                10.0,
            ),
        ];
        for (idx, line) in lines.iter_mut().enumerate() {
            line.region_path = vec![idx as u16];
        }
        let mut page = crate::markdown_layout::test_helpers::page(lines);
        page.layout_blocks = vec![block("Table")];

        let hints = MarkdownLayoutHints::from_page(&page);
        let extracted = hints.detect_table_interruptions(
            &page.projected_lines,
            &[],
            page.page_width,
            page.page_height,
            &HashSet::new(),
        );

        assert_eq!(extracted.tables.len(), 1);
        assert_eq!(extracted.consumed.len(), 4);
    }
}
