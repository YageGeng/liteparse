use crate::types::{GraphicPrimitive, ProjectedLine, TextItem};

use super::blocks::Block;
use super::inline::is_bold_span;
use super::paragraphs::collapse_whitespace;

/// Minimum cells per row for a region to qualify as a table.
pub(super) const TABLE_MIN_COLUMNS: usize = 3;

/// Minimum consecutive rows for a region to qualify as a table.
const TABLE_MIN_ROWS: usize = 2;

/// Gap between adjacent spans (in multiples of dominant font size) above which
/// we treat the gap as a cell boundary.
const TABLE_CELL_GAP_FONT_MULTIPLIER: f32 = 1.0;

/// Tolerance (points) for matching a cell's start-x to an existing column
/// track when extending a candidate table run.
const TABLE_TRACK_TOLERANCE_PT: f32 = 6.0;

/// Maximum vertical gap between consecutive table rows, expressed in multiples
/// of the line height. Looser than the paragraph rule because table rows often
/// have more vertical padding than prose lines.
const TABLE_ROW_GAP_MULTIPLIER: f32 = 2.5;

/// Maximum coefficient-of-variation for row spacing within a confident table
/// (rejecting irregular spacing that's more likely prose or a footer block).
const TABLE_ROW_SPACING_MAX_CV: f32 = 0.5;

/// One cell within a tabular row: contributing spans aggregated to text and
/// its leftmost x position, used to align cells across rows into column
/// "tracks".
#[derive(Debug, Clone)]
pub(super) struct TableCell {
    pub(super) start_x: f32,
    /// Right edge of the cell (x of the last span's right). Used by
    /// `recover_merged_cell` to detect cells that straddle two column tracks
    /// when the projection merged two adjacent words into one span.
    pub(super) end_x: f32,
    pub(super) text: String,
    pub(super) bold: bool,
}

/// A contiguous tabular run: line indices `[start, end)` plus the detected
/// rows. Used so the line-classifier can skip the consumed range and so
/// fallback rendering can reach back for the original projected text.
#[derive(Debug, Clone)]
pub(super) struct TableRun {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) block: Block,
}

/// Split a `ProjectedLine`'s spans into cells. A gap larger than
/// `TABLE_CELL_GAP_FONT_MULTIPLIER × font_size` between adjacent spans starts
/// a new cell; otherwise spans join into the same cell with a single space.
pub(super) fn split_cells(line: &ProjectedLine) -> Vec<TableCell> {
    // Skip whitespace-only spans before computing gaps — leading/trailing
    // empty items would otherwise add spurious cell boundaries.
    let mut spans: Vec<&TextItem> = line
        .spans
        .iter()
        .filter(|s| !s.text.trim().is_empty())
        .collect();
    spans.sort_by(|a, b| a.x.total_cmp(&b.x));
    if spans.is_empty() {
        return Vec::new();
    }
    let font_size = if line.dominant_font_size > 0.0 {
        line.dominant_font_size
    } else {
        line.bbox.height.max(1.0)
    };
    let gap_threshold = font_size * TABLE_CELL_GAP_FONT_MULTIPLIER;

    let mut cells: Vec<TableCell> = Vec::new();
    let mut current_text = String::new();
    let mut current_start = spans[0].x;
    let mut current_bold_chars: usize = 0;
    let mut current_total_chars: usize = 0;
    let mut prev_right = spans[0].x;

    for (i, span) in spans.iter().enumerate() {
        let gap = span.x - prev_right;
        let break_cell = i > 0 && gap > gap_threshold;
        if break_cell {
            let bold = current_total_chars > 0 && current_bold_chars * 2 > current_total_chars;
            cells.push(TableCell {
                start_x: current_start,
                end_x: prev_right,
                text: collapse_whitespace(current_text.trim()),
                bold,
            });
            current_text.clear();
            current_start = span.x;
            current_bold_chars = 0;
            current_total_chars = 0;
        }
        if !current_text.is_empty() && !current_text.ends_with(' ') {
            current_text.push(' ');
        }
        current_text.push_str(&span.text);
        let n = span.text.chars().count();
        current_total_chars += n;
        if is_bold_span(span) {
            current_bold_chars += n;
        }
        prev_right = span.x + span.width.max(0.0);
    }
    if !current_text.trim().is_empty() {
        let bold = current_total_chars > 0 && current_bold_chars * 2 > current_total_chars;
        cells.push(TableCell {
            start_x: current_start,
            end_x: prev_right,
            text: collapse_whitespace(current_text.trim()),
            bold,
        });
    }
    cells
}

/// When a candidate row has fewer cells than the established column count,
/// look for cells whose x-range straddles multiple column tracks (likely two
/// or more adjacent words that PDFium merged into a single text run) and
/// split each on internal whitespace at the boundaries nearest to the
/// straddled tracks.
///
/// Returns the patched cells if every short cell could be cleanly split to
/// recover `tracks.len()` cells total; otherwise `None`.
pub(super) fn recover_merged_cell(mut cells: Vec<TableCell>, tracks: &[f32]) -> Option<Vec<TableCell>> {
    let target = tracks.len();
    if cells.len() >= target {
        return None;
    }
    // Repeatedly find the cell that straddles the most tracks (≥2) and split
    // it. Each iteration strictly grows `cells.len()`, so termination is
    // guaranteed; if no cell straddles ≥2 tracks before we hit the target,
    // recovery fails.
    while cells.len() < target {
        let mut best_i: Option<usize> = None;
        let mut best_count: usize = 1;
        let mut best_contained: Vec<f32> = Vec::new();
        for (i, cell) in cells.iter().enumerate() {
            let contained: Vec<f32> = tracks
                .iter()
                .copied()
                .filter(|t| {
                    *t >= cell.start_x - TABLE_TRACK_TOLERANCE_PT
                        && *t <= cell.end_x + TABLE_TRACK_TOLERANCE_PT
                })
                .collect();
            if contained.len() > best_count {
                best_count = contained.len();
                best_i = Some(i);
                best_contained = contained;
            }
        }
        let Some(i) = best_i else {
            return None;
        };
        let cell = cells[i].clone();
        let chars: Vec<char> = cell.text.trim().chars().collect();
        let n = chars.len();
        if n == 0 || best_contained.len() < 2 {
            return None;
        }
        let text_width = (cell.end_x - cell.start_x).max(1.0);
        // For each track after the first, pick the whitespace boundary in
        // `chars` whose linearly-interpolated x is closest to the track.
        let mut split_indices: Vec<usize> = Vec::new();
        for t in best_contained.iter().skip(1) {
            let mut best: Option<(usize, f32)> = None;
            for (k, ch) in chars.iter().enumerate() {
                if !ch.is_whitespace() {
                    continue;
                }
                if split_indices.contains(&k) {
                    continue;
                }
                let frac = k as f32 / n as f32;
                let x = cell.start_x + frac * text_width;
                let d = (x - t).abs();
                if best.as_ref().is_none_or(|b| d < b.1) {
                    best = Some((k, d));
                }
            }
            let (k, _) = best?;
            split_indices.push(k);
        }
        split_indices.sort();
        // Build the split pieces.
        let mut pieces: Vec<String> = Vec::new();
        let mut prev = 0usize;
        for k in &split_indices {
            let piece: String = chars[prev..*k]
                .iter()
                .collect::<String>()
                .trim()
                .to_string();
            if piece.is_empty() {
                return None;
            }
            pieces.push(piece);
            prev = *k;
        }
        let last: String = chars[prev..].iter().collect::<String>().trim().to_string();
        if last.is_empty() {
            return None;
        }
        pieces.push(last);
        if pieces.len() != best_contained.len() {
            return None;
        }
        // Synthesize new TableCells aligned with each track.
        let mut new_cells: Vec<TableCell> = Vec::with_capacity(pieces.len());
        for (p, piece) in pieces.iter().enumerate() {
            let start_x = if p == 0 {
                cell.start_x
            } else {
                best_contained[p]
            };
            let end_x = if p + 1 < best_contained.len() {
                (best_contained[p + 1] - 1.0).max(start_x)
            } else {
                cell.end_x
            };
            new_cells.push(TableCell {
                start_x,
                end_x,
                text: piece.clone(),
                bold: cell.bold,
            });
        }
        cells.remove(i);
        for (offset, c) in new_cells.into_iter().enumerate() {
            cells.insert(i + offset, c);
        }
    }
    if cells.len() == target {
        Some(cells)
    } else {
        None
    }
}

/// Vertical-gap check for table rows. Looser than paragraph continuation
/// because table rows often have extra padding between them.
fn table_rows_adjacent(prev: &ProjectedLine, cur: &ProjectedLine) -> bool {
    // Intentionally don't require region_path equality. Indented sub-group
    // rows (e.g. an indented "MEMORYBANK" row in a grouped academic results
    // table) sometimes land in a different XY-cut leaf than the rest of the
    // table — but the column-track alignment and y-gap checks below are
    // strong enough signals on their own to keep us from spuriously
    // bridging unrelated regions.
    let prev_bottom = prev.bbox.y + prev.bbox.height;
    let gap = cur.bbox.y - prev_bottom;
    let line_height = prev.bbox.height.max(cur.bbox.height).max(1.0);
    gap >= -line_height && gap <= line_height * TABLE_ROW_GAP_MULTIPLIER
}

/// Coefficient of variation (std-dev / mean) of inter-row vertical gaps.
/// Returns 0.0 for runs with <2 gaps (nothing to compare). Used to reject
/// runs whose row spacing is too irregular to be a real table.
fn row_spacing_cv(rows: &[(usize, &ProjectedLine, Vec<TableCell>)]) -> f32 {
    if rows.len() < 3 {
        return 0.0;
    }
    let gaps: Vec<f32> = rows
        .windows(2)
        .map(|w| (w[1].1.bbox.y - w[0].1.bbox.y).abs())
        .collect();
    let mean = gaps.iter().sum::<f32>() / gaps.len() as f32;
    if mean <= 0.0 {
        return f32::INFINITY;
    }
    let var = gaps.iter().map(|g| (g - mean).powi(2)).sum::<f32>() / gaps.len() as f32;
    var.sqrt() / mean
}

/// Try to extend a candidate table starting at `start_idx`. On success returns
/// a `TableRun` with `Block::Table` or `Block::GridFallback`; on failure
/// returns `None` (and the caller should fall through to per-line
/// classification).
fn try_detect_table(lines: &[ProjectedLine], start_idx: usize, floor: usize) -> Option<TableRun> {
    let first_cells = split_cells(&lines[start_idx]);
    if first_cells.len() < TABLE_MIN_COLUMNS {
        return None;
    }

    let mut rows: Vec<(usize, &ProjectedLine, Vec<TableCell>)> =
        vec![(start_idx, &lines[start_idx], first_cells.clone())];
    let column_count = first_cells.len();
    let tracks: Vec<f32> = first_cells.iter().map(|c| c.start_x).collect();

    let mut j = start_idx + 1;
    while j < lines.len() {
        if !table_rows_adjacent(rows.last().unwrap().1, &lines[j]) {
            break;
        }
        let mut cells = split_cells(&lines[j]);
        if cells.len() < column_count && cells.len() >= TABLE_MIN_COLUMNS {
            // PDFium occasionally merges two (or more) adjacent words into one
            // text run when inter-word kerning is tighter than the gap
            // threshold — common in tightly-set numeric tables (e.g. the
            // "MEMORYBANK 5.00 4.77" case on page 6 of the AMEM paper).
            // Recover by splitting straddling cells on internal whitespace.
            if let Some(patched) = recover_merged_cell(cells.clone(), &tracks) {
                cells = patched;
            }
        }
        // Wrapped-continuation merge: a partial-cell line that sits tight
        // beneath (or overlaps) the previous row AND whose cells all align
        // with existing column tracks is a *wrap* of the prior row, not a
        // new row. Common in borderless tables where one column has a
        // multi-line cell while neighbouring columns stay on one line. Merge
        // each wrap cell into its matching track's text rather than
        // breaking the run.
        if cells.len() < column_count && !cells.is_empty() {
            let line_height = lines[j].bbox.height.max(1.0);
            let prev_y_top = rows.last().unwrap().1.bbox.y;
            let centroid_dy = lines[j].bbox.y - prev_y_top;
            let all_align_track = cells.iter().all(|c| {
                tracks
                    .iter()
                    .any(|t| (c.start_x - *t).abs() <= TABLE_TRACK_TOLERANCE_PT)
            });
            if centroid_dy <= line_height * 1.5 && all_align_track {
                let prev_cells = &mut rows.last_mut().unwrap().2;
                for c in &cells {
                    if let Some(idx) = tracks
                        .iter()
                        .position(|t| (c.start_x - *t).abs() <= TABLE_TRACK_TOLERANCE_PT)
                    {
                        if !prev_cells[idx].text.is_empty() && !c.text.is_empty() {
                            prev_cells[idx].text.push(' ');
                        }
                        prev_cells[idx].text.push_str(&c.text);
                    }
                }
                j += 1;
                continue;
            }
        }
        if cells.len() != column_count {
            break;
        }
        // Allow at most one column track to drift out of tolerance, which lets
        // grouped row-labels in academic tables (e.g. an indented "MEMORYBANK"
        // row whose label column shifts right by ~30pt while the numeric
        // columns stay aligned) stay in the same run. Without this slack a
        // single indented label fragments a 6-row table into three 2-row chunks.
        let misaligned = cells
            .iter()
            .zip(tracks.iter())
            .filter(|(c, t)| (c.start_x - **t).abs() > TABLE_TRACK_TOLERANCE_PT)
            .count();
        if misaligned > 1 {
            break;
        }
        rows.push((j, &lines[j], cells));
        j += 1;
    }

    if rows.len() < TABLE_MIN_ROWS {
        return None;
    }

    let cv = row_spacing_cv(&rows);
    let end = j;

    if cv > TABLE_ROW_SPACING_MAX_CV {
        // Suggestive layout but the row cadence is too irregular to trust as a
        // clean table — surface as a fenced fallback so the structure is at
        // least preserved.
        let raw: Vec<String> = rows
            .iter()
            .map(|(_, line, _)| line.text.trim_end().to_string())
            .collect();
        return Some(TableRun {
            start: start_idx,
            end,
            block: Block::GridFallback { lines: raw },
        });
    }

    // Walk back above the detected body and absorb header lines that align to
    // the same column tracks but weren't includable as body rows (merged /
    // partial header cells). Multiple wrapped header lines collapse into one
    // markdown header row, joined per-column top-to-bottom.
    let absorbed = absorb_header_lines(lines, start_idx, &tracks, column_count, floor);

    // Promote the first body row to header iff every cell in it is bold
    // (matches pymupdf4llm's "bold-or-filled" heuristic; fills require fork
    // data). Skipped when we already absorbed an explicit header above.
    let first_row = &rows[0].2;
    let bold_header_qualifies = absorbed.is_none() && first_row.iter().all(|c| c.bold);

    // `row_start` is the index of the first body row within `rows`. When the
    // header came from absorbed lines above, every detected row is body data;
    // only the bold-first-row promotion consumes rows[0].
    let (run_start, header, row_start) = match absorbed {
        Some((hstart, header_texts)) => (hstart, Some(header_texts), 0),
        None if bold_header_qualifies => (
            start_idx,
            Some(first_row.iter().map(|c| c.text.clone()).collect()),
            1,
        ),
        None => (start_idx, None, 0),
    };
    let body_rows: Vec<Vec<String>> = rows[row_start..]
        .iter()
        .map(|(_, _, cells)| cells.iter().map(|c| c.text.clone()).collect())
        .collect();
    if header.is_none() && body_rows.len() < TABLE_MIN_ROWS {
        return None;
    }

    Some(TableRun {
        start: run_start,
        end,
        block: Block::Table {
            header,
            rows: body_rows,
        },
    })
}

/// Walk backward from `start_idx` (not below `floor`), pulling in lines whose
/// cells all align to the table's `tracks` as header rows. Returns the new
/// start index and a single merged header row (`column_count` columns) with
/// each absorbed line's text appended into its nearest column track.
fn absorb_header_lines(
    lines: &[ProjectedLine],
    start_idx: usize,
    tracks: &[f32],
    column_count: usize,
    floor: usize,
) -> Option<(usize, Vec<String>)> {
    let mut absorbed: Vec<Vec<TableCell>> = Vec::new();
    let mut j = start_idx;
    while j > floor {
        let cand = j - 1;
        let cells = split_cells(&lines[cand]);
        // A header line must carry at least two track-aligned cells (a single
        // cell is a title/caption, not a header), no more than column_count,
        // sit tight above the row below it, and have every cell land on a
        // known column track.
        if cells.len() < 2 || cells.len() > column_count {
            break;
        }
        if !table_rows_adjacent(&lines[cand], &lines[j]) {
            break;
        }
        let all_align = cells.iter().all(|c| {
            tracks
                .iter()
                .any(|t| (c.start_x - *t).abs() <= TABLE_TRACK_TOLERANCE_PT)
        });
        if !all_align {
            break;
        }
        absorbed.push(cells);
        j = cand;
    }
    if absorbed.is_empty() {
        return None;
    }
    // Collected bottom-up; reverse so text reads top-to-bottom per column.
    absorbed.reverse();
    let mut header = vec![String::new(); column_count];
    for cells in &absorbed {
        for c in cells {
            let Some(idx) = tracks
                .iter()
                .enumerate()
                .filter(|(_, t)| (c.start_x - **t).abs() <= TABLE_TRACK_TOLERANCE_PT)
                .min_by(|(_, a), (_, b)| {
                    (c.start_x - **a).abs().total_cmp(&(c.start_x - **b).abs())
                })
                .map(|(i, _)| i)
            else {
                continue;
            };
            if !header[idx].is_empty() && !c.text.is_empty() {
                header[idx].push(' ');
            }
            header[idx].push_str(&c.text);
        }
    }
    Some((j, header))
}

/// Scan `lines` once and return all detected tabular regions (sorted by
/// `start`). Caller uses these as cut-points so the per-line classifier never
/// sees lines inside a table.
pub(super) fn detect_tables(lines: &[ProjectedLine]) -> Vec<TableRun> {
    let mut out = Vec::new();
    let mut i = 0;
    let mut floor = 0;
    while i < lines.len() {
        if let Some(run) = try_detect_table(lines, i, floor) {
            floor = run.end;
            i = run.end;
            out.push(run);
        } else {
            i += 1;
        }
    }
    out
}

// ── Ruled-grid table detection ─────────────────────────────────────────────
//
// Detect tables drawn with explicit horizontal + vertical rules (the "Strong"
// mode in MARKDOWN_PLAN.md). Strokes are clustered into H/V grid lines, then
// union-find groups crossing lines into table regions. For each region the
// distinct row/column boundaries form a cell grid; text lines are assigned to
// cells by centroid containment.
//
// Ruled tables are detected before the borderless `detect_tables`. The caller
// merges the two outputs; overlapping ranges defer to the ruled run because
// path-based geometry is a strictly stronger signal than text alignment alone.

/// Horizontal segment in viewport coords (top-left origin). `y` is the rule's
/// y-position; `x_min..x_max` is its horizontal span. Endpoints of multiple
/// short segments sharing a y get unioned into one wider segment during
/// clustering.
#[derive(Debug, Clone, Copy)]
struct HSeg {
    x_min: f32,
    x_max: f32,
    y: f32,
}

#[derive(Debug, Clone, Copy)]
struct VSeg {
    y_min: f32,
    y_max: f32,
    x: f32,
}

/// Strokes are considered "axis-aligned" when the perpendicular delta is at
/// most this many points. Generous to absorb antialiased near-pixel strokes.
const TABLE_AXIS_TOLERANCE_PT: f32 = 1.0;

/// Two H lines (or two V lines) are merged into one grid line when their
/// perpendicular coords are within this many points. Slightly looser than the
/// axis tolerance because rules drawn at the same row can have ±1pt jitter
/// from different stroke widths.
const TABLE_GRID_CLUSTER_PT: f32 = 2.0;

/// Slack added when checking whether a V line "crosses" an H line. Helps
/// when rules don't quite reach the corner because the PDF drew them as
/// individual segments with small gaps.
const TABLE_CROSS_TOLERANCE_PT: f32 = 3.0;

/// Reject ruled-table candidates whose empty-cell fraction exceeds this.
/// NOTE: this can't be loosened to recover blank worksheets/forms — a real
/// sparse table (doc 180, a 4-col Version History, ~75% empty) and a spurious
/// grid from decorative layout boxes (doc 198, a TOC, also ~75% empty) are
/// indistinguishable on empty-fraction, and relaxing it net-regressed TEDS by
/// ~0.09 on the bench (more false tables than real forms recovered).
const TABLE_MAX_EMPTY_CELL_FRACTION: f32 = 0.30;

/// Reject candidates whose grid covers nearly the whole page — almost always
/// a page border, not a real table.
const TABLE_MAX_PAGE_COVERAGE: f32 = 0.95;

/// Extract horizontal and vertical line segments from a page's graphics. Each
/// `Stroke` becomes one HSeg or VSeg depending on orientation; each stroked
/// `Rect` contributes its four edges (cell-border rects, table frames).
fn extract_h_v_segments(graphics: &[GraphicPrimitive]) -> (Vec<HSeg>, Vec<VSeg>) {
    let mut hs = Vec::new();
    let mut vs = Vec::new();
    for g in graphics {
        match g {
            GraphicPrimitive::Stroke { x1, y1, x2, y2, .. } => {
                let (x1, y1, x2, y2) = (*x1, *y1, *x2, *y2);
                let dy = (y1 - y2).abs();
                let dx = (x1 - x2).abs();
                if dy <= TABLE_AXIS_TOLERANCE_PT && dx > 1.0 {
                    hs.push(HSeg {
                        x_min: x1.min(x2),
                        x_max: x1.max(x2),
                        y: (y1 + y2) * 0.5,
                    });
                } else if dx <= TABLE_AXIS_TOLERANCE_PT && dy > 1.0 {
                    vs.push(VSeg {
                        y_min: y1.min(y2),
                        y_max: y1.max(y2),
                        x: (x1 + x2) * 0.5,
                    });
                }
            }
            GraphicPrimitive::Rect { bbox, stroke, .. } => {
                if stroke.is_none() {
                    continue;
                }
                let top = bbox.y;
                let bottom = bbox.y + bbox.height;
                let left = bbox.x;
                let right = bbox.x + bbox.width;
                if bbox.width > 1.0 {
                    hs.push(HSeg {
                        x_min: left,
                        x_max: right,
                        y: top,
                    });
                    hs.push(HSeg {
                        x_min: left,
                        x_max: right,
                        y: bottom,
                    });
                }
                if bbox.height > 1.0 {
                    vs.push(VSeg {
                        y_min: top,
                        y_max: bottom,
                        x: left,
                    });
                    vs.push(VSeg {
                        y_min: top,
                        y_max: bottom,
                        x: right,
                    });
                }
            }
        }
    }
    (hs, vs)
}

/// Cluster H segments sharing a y-coordinate (within `TABLE_GRID_CLUSTER_PT`)
/// into a single wider grid line whose x-extent is the union of the inputs.
fn cluster_h_segments(mut segs: Vec<HSeg>) -> Vec<HSeg> {
    if segs.is_empty() {
        return segs;
    }
    segs.sort_by(|a, b| a.y.total_cmp(&b.y));
    let mut out: Vec<HSeg> = Vec::with_capacity(segs.len());
    for seg in segs {
        if let Some(last) = out.last_mut()
            && (last.y - seg.y).abs() <= TABLE_GRID_CLUSTER_PT
        {
            last.x_min = last.x_min.min(seg.x_min);
            last.x_max = last.x_max.max(seg.x_max);
            continue;
        }
        out.push(seg);
    }
    out
}

fn cluster_v_segments(mut segs: Vec<VSeg>) -> Vec<VSeg> {
    if segs.is_empty() {
        return segs;
    }
    segs.sort_by(|a, b| a.x.total_cmp(&b.x));
    let mut out: Vec<VSeg> = Vec::with_capacity(segs.len());
    for seg in segs {
        if let Some(last) = out.last_mut()
            && (last.x - seg.x).abs() <= TABLE_GRID_CLUSTER_PT
        {
            last.y_min = last.y_min.min(seg.y_min);
            last.y_max = last.y_max.max(seg.y_max);
            continue;
        }
        out.push(seg);
    }
    out
}

/// Union-find root with path compression.
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn uf_union(parent: &mut [usize], a: usize, b: usize) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra != rb {
        parent[ra] = rb;
    }
}

/// Group H/V grid lines that cross each other into connected components.
/// Each component is a candidate ruled table — typically one component per
/// distinct table on the page. Returns `(h_indices, v_indices)` per component,
/// dropping components without ≥2 H and ≥2 V lines (a single L-shape doesn't
/// make a table).
fn find_grid_components(hs: &[HSeg], vs: &[VSeg]) -> Vec<(Vec<usize>, Vec<usize>)> {
    let n_h = hs.len();
    let n_v = vs.len();
    if n_h < 2 || n_v < 2 {
        return Vec::new();
    }
    let n = n_h + n_v;
    let mut parent: Vec<usize> = (0..n).collect();
    let mut connected = vec![false; n];

    let tol = TABLE_CROSS_TOLERANCE_PT;
    for (i, h) in hs.iter().enumerate() {
        for (j, v) in vs.iter().enumerate() {
            let v_crosses_h_x = v.x >= h.x_min - tol && v.x <= h.x_max + tol;
            let h_crosses_v_y = h.y >= v.y_min - tol && h.y <= v.y_max + tol;
            if v_crosses_h_x && h_crosses_v_y {
                uf_union(&mut parent, i, n_h + j);
                connected[i] = true;
                connected[n_h + j] = true;
            }
        }
    }

    use std::collections::HashMap;
    let mut groups: HashMap<usize, (Vec<usize>, Vec<usize>)> = HashMap::new();
    for i in 0..n_h {
        if !connected[i] {
            continue;
        }
        let r = uf_find(&mut parent, i);
        groups.entry(r).or_default().0.push(i);
    }
    for j in 0..n_v {
        if !connected[n_h + j] {
            continue;
        }
        let r = uf_find(&mut parent, n_h + j);
        groups.entry(r).or_default().1.push(j);
    }
    groups
        .into_values()
        .filter(|(h_idx, v_idx)| h_idx.len() >= 2 && v_idx.len() >= 2)
        .collect()
}

/// Build a `TableRun` for one ruled-grid component. Returns `None` if the
/// resulting grid is too small (< 2 cols or < 2 rows), covers nearly the
/// whole page (likely the page border), or is mostly empty cells.
fn build_ruled_table(
    hs: &[HSeg],
    vs: &[VSeg],
    h_indices: &[usize],
    v_indices: &[usize],
    lines: &[ProjectedLine],
    page_width: f32,
    page_height: f32,
) -> Option<TableRun> {
    // Distinct row y-coords (cluster again — multiple H lines may share a y).
    let mut ys: Vec<f32> = h_indices.iter().map(|&i| hs[i].y).collect();
    ys.sort_by(|a, b| a.total_cmp(b));
    dedup_close(&mut ys, TABLE_GRID_CLUSTER_PT);

    let mut xs: Vec<f32> = v_indices.iter().map(|&i| vs[i].x).collect();
    xs.sort_by(|a, b| a.total_cmp(b));
    dedup_close(&mut xs, TABLE_GRID_CLUSTER_PT);

    // Need ≥2 row boundaries (1 row) and ≥2 column boundaries (1 col); but
    // a 1×1 grid is just a callout box, so also require ≥1 inner divider
    // (i.e. ys.len() ≥ 3 for ≥2 rows). Single-column tables (`xs.len() == 2`)
    // are accepted when row evidence is strong enough — extra guards apply
    // below after the empty-row collapse.
    if ys.len() < 3 || xs.len() < 2 {
        return None;
    }

    let n_rows = ys.len() - 1;
    let n_cols = xs.len() - 1;
    let bbox = crate::types::Rect {
        x: xs[0],
        y: ys[0],
        width: xs[n_cols] - xs[0],
        height: ys[n_rows] - ys[0],
    };

    // Reject page-border-as-table.
    if page_width > 0.0 && page_height > 0.0 {
        let coverage = (bbox.width / page_width) * (bbox.height / page_height);
        if coverage > TABLE_MAX_PAGE_COVERAGE {
            return None;
        }
    }

    // Assign each text line to its cell by centroid.
    let mut cells: Vec<Vec<String>> = vec![vec![String::new(); n_cols]; n_rows];
    let mut cell_is_bold: Vec<Vec<bool>> = vec![vec![true; n_cols]; n_rows];
    let mut cell_has_text: Vec<Vec<bool>> = vec![vec![false; n_cols]; n_rows];
    let mut consumed_indices: Vec<usize> = Vec::new();

    for (idx, line) in lines.iter().enumerate() {
        let cx = line.bbox.x + line.bbox.width * 0.5;
        let cy = line.bbox.y + line.bbox.height * 0.5;
        if cy < ys[0] || cy > ys[n_rows] || cx < xs[0] || cx > xs[n_cols] {
            continue;
        }
        let row = match find_bucket(&ys, cy) {
            Some(r) => r,
            None => continue,
        };
        let col = match find_bucket(&xs, cx) {
            Some(c) => c,
            None => continue,
        };
        let txt = line.text.trim();
        if txt.is_empty() {
            continue;
        }
        if !cells[row][col].is_empty() {
            cells[row][col].push(' ');
        }
        cells[row][col].push_str(txt);
        cell_has_text[row][col] = true;
        if !line.all_bold {
            cell_is_bold[row][col] = false;
        }
        consumed_indices.push(idx);
    }

    if consumed_indices.is_empty() {
        return None;
    }

    // Collapse "phantom rows" produced by stacked thin border-strip rects
    // (doc 149 draws each visual table row as: top border strip ~1pt, body
    // rect ~22pt, bottom border strip ~5pt — each contributes y-coords that
    // survive the 2pt clustering as separate grid rows). Rule: drop a row
    // iff (a) it has no text in any cell AND (b) its height is < 50% of the
    // median non-empty row height. The height gate preserves real
    // fill-in-the-blank forms where empty body rows are full-height.
    let row_heights: Vec<f32> = (0..n_rows).map(|r| ys[r + 1] - ys[r]).collect();
    let nonempty_heights: Vec<f32> = (0..n_rows)
        .filter(|r| cell_has_text[*r].iter().any(|t| *t))
        .map(|r| row_heights[r])
        .collect();
    let median_h = if !nonempty_heights.is_empty() {
        let mut s = nonempty_heights.clone();
        s.sort_by(|a, b| a.total_cmp(b));
        s[s.len() / 2]
    } else {
        let mut s = row_heights.clone();
        s.sort_by(|a, b| a.total_cmp(b));
        s[s.len() / 2]
    };
    let keep: Vec<bool> = (0..n_rows)
        .map(|r| {
            let has_text = cell_has_text[r].iter().any(|t| *t);
            has_text || row_heights[r] >= median_h * 0.8
        })
        .collect();
    let cells: Vec<Vec<String>> = (0..n_rows)
        .filter(|r| keep[*r])
        .map(|r| cells[r].clone())
        .collect();
    let cell_has_text: Vec<Vec<bool>> = (0..n_rows)
        .filter(|r| keep[*r])
        .map(|r| cell_has_text[r].clone())
        .collect();
    let cell_is_bold: Vec<Vec<bool>> = (0..n_rows)
        .filter(|r| keep[*r])
        .map(|r| cell_is_bold[r].clone())
        .collect();
    let n_rows = cells.len();
    if n_rows < 2 {
        return None;
    }

    // Mirror the row-collapse rule on columns. Some ruled tables (doc 149)
    // draw their left/right borders as thin strip rects 5pt wide — those
    // become phantom columns with no text. Drop columns that are both empty
    // AND noticeably narrower than the median text-bearing column.
    let col_widths: Vec<f32> = (0..n_cols).map(|c| xs[c + 1] - xs[c]).collect();
    let nonempty_col_widths: Vec<f32> = (0..n_cols)
        .filter(|c| (0..n_rows).any(|r| cell_has_text[r][*c]))
        .map(|c| col_widths[c])
        .collect();
    let median_w = if !nonempty_col_widths.is_empty() {
        let mut s = nonempty_col_widths.clone();
        s.sort_by(|a, b| a.total_cmp(b));
        s[s.len() / 2]
    } else {
        let mut s = col_widths.clone();
        s.sort_by(|a, b| a.total_cmp(b));
        s[s.len() / 2]
    };
    let keep_col: Vec<bool> = (0..n_cols)
        .map(|c| {
            let has_text = (0..n_rows).any(|r| cell_has_text[r][c]);
            has_text || col_widths[c] >= median_w * 0.3
        })
        .collect();
    // Cap how aggressively we drop columns. A real table with one phantom
    // border-strip column (doc 149: 5pt-wide left border) drops exactly one.
    // Anything more than that is almost certainly a chart whose vertical
    // grid-lines got merged with text data (doc 078: a chart's 18 V-lines
    // straddle a table to its left/right and collapse to 1 col), so the
    // "table" is bogus — bail and let the borderless detector handle it.
    // Note: keep_col already only drops columns where both (a) no text AND
    // (b) width < 30% of median text-bearing column.
    let cells: Vec<Vec<String>> = cells
        .into_iter()
        .map(|row| {
            row.into_iter()
                .enumerate()
                .filter(|(c, _)| keep_col[*c])
                .map(|(_, v)| v)
                .collect()
        })
        .collect();
    let cell_has_text: Vec<Vec<bool>> = cell_has_text
        .into_iter()
        .map(|row| {
            row.into_iter()
                .enumerate()
                .filter(|(c, _)| keep_col[*c])
                .map(|(_, v)| v)
                .collect()
        })
        .collect();
    let cell_is_bold: Vec<Vec<bool>> = cell_is_bold
        .into_iter()
        .map(|row| {
            row.into_iter()
                .enumerate()
                .filter(|(c, _)| keep_col[*c])
                .map(|(_, v)| v)
                .collect()
        })
        .collect();
    let n_cols = cells.first().map(|r| r.len()).unwrap_or(0);
    if n_cols == 0 {
        return None;
    }
    // Single-column tables are ambiguous (could be a captioned card) — require
    // ≥3 rows of geometric + textual evidence.
    if n_cols == 1 && n_rows < 3 {
        return None;
    }

    let total = n_rows * n_cols;
    let empty_count = cell_has_text
        .iter()
        .flatten()
        .filter(|filled| !**filled)
        .count();
    if (empty_count as f32) / (total as f32) > TABLE_MAX_EMPTY_CELL_FRACTION {
        return None;
    }

    // Header = first row iff every non-empty cell in it is bold.
    let header_qualifies = cell_has_text[0]
        .iter()
        .zip(cell_is_bold[0].iter())
        .all(|(has, bold)| !has || *bold)
        && cell_has_text[0].iter().any(|has| *has);
    let header = if header_qualifies {
        Some(cells[0].clone())
    } else {
        None
    };
    let body_start = if header.is_some() { 1 } else { 0 };
    let body_rows: Vec<Vec<String>> = cells[body_start..].to_vec();
    if body_rows.is_empty() {
        return None;
    }

    // Line index span this table covers.
    let start = *consumed_indices.iter().min().unwrap();
    let end = *consumed_indices.iter().max().unwrap() + 1;

    Some(TableRun {
        start,
        end,
        block: Block::Table {
            header,
            rows: body_rows,
        },
    })
}

/// In-place dedup of a sorted Vec, collapsing entries within `tol` to the
/// first of each cluster.
fn dedup_close(v: &mut Vec<f32>, tol: f32) {
    if v.len() < 2 {
        return;
    }
    let mut out: Vec<f32> = Vec::with_capacity(v.len());
    for x in v.iter().copied() {
        if let Some(&last) = out.last()
            && (x - last).abs() <= tol
        {
            continue;
        }
        out.push(x);
    }
    *v = out;
}

/// Find the bucket index `i` such that `boundaries[i] <= val < boundaries[i+1]`.
/// Returns `None` if `val` is outside the boundaries.
fn find_bucket(boundaries: &[f32], val: f32) -> Option<usize> {
    if boundaries.len() < 2 || val < boundaries[0] || val > *boundaries.last().unwrap() {
        return None;
    }
    for (i, w) in boundaries.windows(2).enumerate() {
        if val >= w[0] && val <= w[1] {
            return Some(i);
        }
    }
    None
}

/// Detect ruled-grid tables on a page from its vector graphics. Returns runs
/// in document order (sorted by `start`).
pub(super) fn detect_ruled_tables(
    lines: &[ProjectedLine],
    graphics: &[GraphicPrimitive],
    page_width: f32,
    page_height: f32,
) -> Vec<TableRun> {
    let (hs, vs) = extract_h_v_segments(graphics);
    let hs = cluster_h_segments(hs);
    let vs = cluster_v_segments(vs);
    if hs.len() < 2 || vs.len() < 2 {
        return Vec::new();
    }
    let components = find_grid_components(&hs, &vs);
    let mut out = Vec::new();
    for (h_idx, v_idx) in components {
        if let Some(run) =
            build_ruled_table(&hs, &vs, &h_idx, &v_idx, lines, page_width, page_height)
        {
            out.push(run);
        }
    }
    out.sort_by_key(|r| r.start);
    out
}

/// Merge ruled-grid runs with borderless runs into a single sorted list. When
/// ranges overlap the ruled run wins (path-based geometry is strictly stronger
/// than text-alignment heuristics) — overlapping borderless runs are dropped.
pub(super) fn merge_table_runs(mut ruled: Vec<TableRun>, borderless: Vec<TableRun>) -> Vec<TableRun> {
    // A ruled run normally beats an overlapping borderless run (path geometry
    // is a stronger signal than text alignment). But a 1-column ruled run is
    // ambiguous — it can come from a real single-column table (doc 149's
    // stacked cards) OR from a multi-column table whose vertical separators
    // are implicit and only the top/bottom rules were drawn (doc 078). Yield
    // to a multi-column borderless run that covers the same range.
    let mut kept: Vec<TableRun> = Vec::with_capacity(ruled.len());
    for r in ruled.drain(..) {
        let is_one_col = matches!(&r.block, Block::Table { rows, .. } if rows.first().map(|row| row.len()).unwrap_or(0) <= 1);
        if is_one_col {
            let beaten = borderless.iter().any(|b| {
                let overlaps = !(b.end <= r.start || b.start >= r.end);
                if !overlaps {
                    return false;
                }
                matches!(&b.block, Block::Table { rows, .. } if rows.first().map(|row| row.len()).unwrap_or(0) >= 2)
            });
            if beaten {
                continue;
            }
        }
        kept.push(r);
    }
    for b in borderless {
        let overlaps = kept
            .iter()
            .any(|r| !(b.end <= r.start || b.start >= r.end));
        if !overlaps {
            kept.push(b);
        }
    }
    kept.sort_by_key(|r| r.start);
    kept
}

/// Escape `|` and `\n` inside a markdown table cell so the pipe-table grammar
/// stays valid. Newlines should be impossible inside a single cell (we built
/// cells from spans on the same projected line) but guard anyway.
pub(super) fn escape_table_cell(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace('\n', " ")
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::{line, line_with_spans, rect_borders, stroke};
    use super::*;

    #[test]
    fn split_cells_splits_on_wide_gaps() {
        let l = line_with_spans(&[("A", 50.0), ("B", 150.0), ("C", 250.0)], 100.0, 10.0);
        let cells = split_cells(&l);
        assert_eq!(cells.len(), 3);
        assert_eq!(cells[0].text, "A");
        assert_eq!(cells[1].text, "B");
        assert_eq!(cells[2].text, "C");
    }

    #[test]
    fn recover_merged_cell_splits_off_by_one() {
        // Mimics the page-6 case: row 0 establishes 3 tracks at 50/150/250.
        // Row 1's projection merges "MEMORYBANK" + "5.00" into one span at
        // x=50 width=110, so split_cells yields 2 cells while the table
        // expects 3. Recovery must split on whitespace at the missing track.
        let row = vec![
            TableCell {
                start_x: 50.0,
                end_x: 160.0,
                text: "MEMORYBANK 5.00".into(),
                bold: false,
            },
            TableCell {
                start_x: 250.0,
                end_x: 280.0,
                text: "4.77".into(),
                bold: false,
            },
        ];
        let tracks = vec![50.0, 150.0, 250.0];
        let out = recover_merged_cell(row, &tracks).expect("recovery should succeed");
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].text, "MEMORYBANK");
        assert_eq!(out[1].text, "5.00");
        assert_eq!(out[2].text, "4.77");
    }

    #[test]
    fn recover_merged_cell_splits_off_by_two() {
        // Three merged tokens in one cell: "MEMORYBANK 13.18 10.03" straddles
        // tracks at 50/150/250 and the row has only 2 cells, off by 2.
        let row = vec![
            TableCell {
                start_x: 50.0,
                end_x: 260.0,
                text: "MEMORYBANK 13.18 10.03".into(),
                bold: false,
            },
            TableCell {
                start_x: 350.0,
                end_x: 380.0,
                text: "7.61".into(),
                bold: false,
            },
        ];
        let tracks = vec![50.0, 150.0, 250.0, 350.0];
        let out = recover_merged_cell(row, &tracks).expect("recovery should succeed");
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].text, "MEMORYBANK");
        assert_eq!(out[1].text, "13.18");
        assert_eq!(out[2].text, "10.03");
        assert_eq!(out[3].text, "7.61");
    }

    #[test]
    fn recover_merged_cell_bails_without_enough_whitespace() {
        // A cell that straddles two tracks but has no internal whitespace
        // (e.g. a hyphenated token) can't be safely split — return None.
        let row = vec![TableCell {
            start_x: 50.0,
            end_x: 200.0,
            text: "ABC-DEF-GHI".into(),
            bold: false,
        }];
        let tracks = vec![50.0, 150.0];
        assert!(recover_merged_cell(row, &tracks).is_none());
    }

    #[test]
    fn split_cells_keeps_close_spans_together() {
        // Two spans 2pt apart at 10pt font (gap < font_size) → same cell.
        let l = line_with_spans(&[("Hello", 50.0), ("world", 80.0)], 100.0, 10.0);
        let cells = split_cells(&l);
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].text, "Hello world");
    }

    #[test]
    fn absorbs_partial_header_line_above_body() {
        // A header line with only two track-aligned cells sits above a clean
        // 3-column body. It can't start the table on its own (fewer than
        // TABLE_MIN_COLUMNS cells) but should be walked back in as the header.
        let lines = vec![
            line_with_spans(&[("Name", 50.0), ("Scores", 150.0)], 100.0, 10.0),
            line_with_spans(&[("A", 50.0), ("1", 150.0), ("2", 250.0)], 115.0, 10.0),
            line_with_spans(&[("B", 50.0), ("3", 150.0), ("4", 250.0)], 130.0, 10.0),
            line_with_spans(&[("C", 50.0), ("5", 150.0), ("6", 250.0)], 145.0, 10.0),
        ];
        let runs = detect_tables(&lines);
        assert_eq!(runs.len(), 1);
        let run = &runs[0];
        assert_eq!(run.start, 0, "header line should be absorbed into the run");
        assert_eq!(run.end, 4);
        match &run.block {
            Block::Table { header, rows } => {
                let header = header.as_ref().expect("header should be present");
                assert_eq!(header, &vec!["Name".to_string(), "Scores".to_string(), String::new()]);
                // All three body rows survive — the header came from above, so
                // rows[0] is not consumed as a header.
                assert_eq!(rows.len(), 3);
            }
            other => panic!("expected Block::Table, got {other:?}"),
        }
    }

    #[test]
    fn does_not_absorb_single_cell_title_above_body() {
        // A one-cell title/caption above a table is NOT a header row and must
        // not be absorbed.
        let lines = vec![
            line_with_spans(&[("Results", 50.0)], 100.0, 10.0),
            line_with_spans(&[("A", 50.0), ("1", 150.0), ("2", 250.0)], 115.0, 10.0),
            line_with_spans(&[("B", 50.0), ("3", 150.0), ("4", 250.0)], 130.0, 10.0),
            line_with_spans(&[("C", 50.0), ("5", 150.0), ("6", 250.0)], 145.0, 10.0),
        ];
        let runs = detect_tables(&lines);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].start, 1, "single-cell title must stay out of the run");
    }

    #[test]
    fn rejects_table_when_row_count_too_low() {
        let lines = vec![line_with_spans(
            &[("A", 50.0), ("B", 150.0), ("C", 250.0)],
            100.0,
            10.0,
        )];
        let runs = detect_tables(&lines);
        assert!(runs.is_empty());
    }

    #[test]
    fn rejects_table_when_column_count_too_low() {
        let lines = vec![
            line_with_spans(&[("A", 50.0), ("B", 200.0)], 100.0, 10.0),
            line_with_spans(&[("C", 50.0), ("D", 200.0)], 115.0, 10.0),
        ];
        let runs = detect_tables(&lines);
        assert!(runs.is_empty());
    }

    #[test]
    fn escapes_pipe_inside_cell() {
        assert_eq!(escape_table_cell("a|b"), "a\\|b");
    }

    #[test]
    fn ruled_table_2x2_detected() {
        // 2 rows × 2 cols grid: 3 H lines (y=100,140,180), 3 V lines (x=50,150,250)
        // Cell text dropped in the centroid of each cell.
        let mut graphics = Vec::new();
        for y in [100.0_f32, 140.0, 180.0] {
            graphics.push(stroke(50.0, y, 250.0, y, 0.5));
        }
        for x in [50.0_f32, 150.0, 250.0] {
            graphics.push(stroke(x, 100.0, x, 180.0, 0.5));
        }

        // Text lines: one per cell, centered.
        let lines = vec![
            line("a", 90.0, 115.0, 10.0, 10.0),  // row 0, col 0
            line("b", 190.0, 115.0, 10.0, 10.0), // row 0, col 1
            line("c", 90.0, 155.0, 10.0, 10.0),  // row 1, col 0
            line("d", 190.0, 155.0, 10.0, 10.0), // row 1, col 1
        ];

        let runs = detect_ruled_tables(&lines, &graphics, 612.0, 792.0);
        assert_eq!(runs.len(), 1, "expected 1 ruled table, got {runs:?}");
        match &runs[0].block {
            Block::Table { header, rows } => {
                assert!(header.is_none(), "no bold first row → no header");
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0], vec!["a", "b"]);
                assert_eq!(rows[1], vec!["c", "d"]);
            }
            other => panic!("expected Block::Table, got {other:?}"),
        }
    }

    #[test]
    fn ruled_table_rect_borders_detected() {
        // Same 2×2 table but drawn as 4 individual cell rects (each cell is a
        // stroked rectangle). Each rect contributes 4 strokes via
        // extract_h_v_segments.
        let mut graphics = Vec::new();
        graphics.extend(rect_borders(50.0, 100.0, 100.0, 40.0)); // r0 c0
        graphics.extend(rect_borders(150.0, 100.0, 100.0, 40.0)); // r0 c1
        graphics.extend(rect_borders(50.0, 140.0, 100.0, 40.0)); // r1 c0
        graphics.extend(rect_borders(150.0, 140.0, 100.0, 40.0)); // r1 c1

        let lines = vec![
            line("a", 90.0, 115.0, 10.0, 10.0),
            line("b", 190.0, 115.0, 10.0, 10.0),
            line("c", 90.0, 155.0, 10.0, 10.0),
            line("d", 190.0, 155.0, 10.0, 10.0),
        ];
        let runs = detect_ruled_tables(&lines, &graphics, 612.0, 792.0);
        assert_eq!(runs.len(), 1);
    }

    #[test]
    fn ruled_table_page_border_rejected() {
        // Single big rect covering ~the whole page → should NOT be treated as a
        // table even though it has H+V lines on all four sides.
        let graphics = rect_borders(10.0, 10.0, 590.0, 770.0);
        let lines = vec![line("body text", 50.0, 400.0, 10.0, 10.0)];
        let runs = detect_ruled_tables(&lines, &graphics, 612.0, 792.0);
        assert!(
            runs.is_empty(),
            "page-border rect should not become a table, got {runs:?}"
        );
    }

    #[test]
    fn ruled_table_mostly_empty_rejected() {
        // 3×3 grid with text in only one cell — empty fraction 8/9 ≈ 89% >> 30%.
        let mut graphics = Vec::new();
        for y in [100.0_f32, 130.0, 160.0, 190.0] {
            graphics.push(stroke(50.0, y, 350.0, y, 0.5));
        }
        for x in [50.0_f32, 150.0, 250.0, 350.0] {
            graphics.push(stroke(x, 100.0, x, 190.0, 0.5));
        }
        let lines = vec![line("only", 90.0, 115.0, 10.0, 10.0)];
        let runs = detect_ruled_tables(&lines, &graphics, 612.0, 792.0);
        assert!(runs.is_empty());
    }

    #[test]
    fn ruled_table_first_row_bold_becomes_header() {
        // 2×2 with first row text marked all_bold → header promotion.
        let mut graphics = Vec::new();
        for y in [100.0_f32, 140.0, 180.0] {
            graphics.push(stroke(50.0, y, 250.0, y, 0.5));
        }
        for x in [50.0_f32, 150.0, 250.0] {
            graphics.push(stroke(x, 100.0, x, 180.0, 0.5));
        }
        let mut a = line("Name", 90.0, 115.0, 10.0, 10.0);
        let mut b = line("Score", 190.0, 115.0, 10.0, 10.0);
        a.all_bold = true;
        b.all_bold = true;
        let lines = vec![
            a,
            b,
            line("alice", 90.0, 155.0, 10.0, 10.0),
            line("99", 190.0, 155.0, 10.0, 10.0),
        ];
        let runs = detect_ruled_tables(&lines, &graphics, 612.0, 792.0);
        assert_eq!(runs.len(), 1);
        match &runs[0].block {
            Block::Table { header, rows } => {
                assert_eq!(
                    header.as_deref(),
                    Some(&["Name".into(), "Score".into()][..])
                );
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0], vec!["alice", "99"]);
            }
            other => panic!("expected Block::Table, got {other:?}"),
        }
    }

    #[test]
    fn merge_prefers_ruled_when_overlapping() {
        let ruled = vec![TableRun {
            start: 5,
            end: 10,
            block: Block::Table {
                header: None,
                rows: vec![vec!["ruled".into()]],
            },
        }];
        let borderless = vec![TableRun {
            start: 6,
            end: 11,
            block: Block::GridFallback {
                lines: vec!["bl".into()],
            },
        }];
        let merged = merge_table_runs(ruled, borderless);
        assert_eq!(merged.len(), 1);
        assert!(matches!(&merged[0].block, Block::Table { .. }));
    }
}
