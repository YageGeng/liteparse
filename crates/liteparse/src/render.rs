use crate::error::LiteParseError;
use crate::extract::{encode_png, load_document_from_input};
use crate::types::PdfInput;
use pdfium::Library;
use serde::Serialize;

/// A single rendered page as PNG bytes.
#[derive(Debug, Clone)]
pub struct RenderedPage {
    pub page_num: u32,
    pub width: u32,
    pub height: u32,
    pub png_bytes: Vec<u8>,
}

/// A single rendered page as raw RGBA pixels.
#[derive(Debug, Clone)]
pub struct RenderedRgbaPage {
    pub page_num: u32,
    pub width: u32,
    pub height: u32,
    pub rgba_bytes: Vec<u8>,
}

/// Render selected pages from a PDF input to PNG bytes.
///
/// Acquires the process-global PDFium lock for the entire render. The lock
/// is held until this function returns — PNG encoding happens inside the
/// critical section, which is fine because it is pure CPU work with no
/// `.await` points.
pub fn render_pages_to_png(
    input: &PdfInput,
    page_numbers: Option<&[u32]>,
    dpi: f32,
    password: Option<&str>,
) -> Result<Vec<RenderedPage>, LiteParseError> {
    render_pages_to_rgba(input, page_numbers, dpi, password)?
        .into_iter()
        .map(|page| {
            let png_bytes = encode_png(&page.rgba_bytes, page.width, page.height)?;
            Ok(RenderedPage {
                page_num: page.page_num,
                width: page.width,
                height: page.height,
                png_bytes,
            })
        })
        .collect::<Result<Vec<_>, LiteParseError>>()
}

/// Render selected pages from a PDF input to raw RGBA pixels.
pub fn render_pages_to_rgba(
    input: &PdfInput,
    page_numbers: Option<&[u32]>,
    dpi: f32,
    password: Option<&str>,
) -> Result<Vec<RenderedRgbaPage>, LiteParseError> {
    let lib = Library::init();
    let document = load_document_from_input(&lib, input, password)?;
    render_document_pages_to_rgba(&document, page_numbers, dpi)
}

fn render_document_pages_to_rgba(
    document: &pdfium::Document,
    page_numbers: Option<&[u32]>,
    dpi: f32,
) -> Result<Vec<RenderedRgbaPage>, LiteParseError> {
    let page_count = document.page_count() as u32;
    let pages: Vec<u32> = match page_numbers {
        Some(nums) => nums.to_vec(),
        None => (1..=page_count).collect(),
    };

    let mut results = Vec::with_capacity(pages.len());
    for page_num in pages {
        if page_num < 1 || page_num > page_count {
            return Err(LiteParseError::Other(format!(
                "page {page_num} out of range (document has {page_count} pages)"
            )));
        }

        let page = document.page((page_num - 1) as i32)?;
        let bitmap = page.render(dpi)?;
        let width = bitmap.width() as u32;
        let height = bitmap.height() as u32;
        let rgba_bytes = bitmap.to_rgba();

        results.push(RenderedRgbaPage {
            page_num,
            width,
            height,
            rgba_bytes,
        });
    }

    Ok(results)
}

/// Render a single page to a PNG file.
pub fn screenshot(
    pdf_path: &str,
    page_num: u32,
    dpi: f32,
    output_path: &str,
    password: Option<&str>,
) -> Result<(), LiteParseError> {
    let input = PdfInput::Path(pdf_path.to_string());
    let pages = render_pages_to_png(&input, Some(&[page_num]), dpi, password)?;
    let page = pages
        .into_iter()
        .next()
        .ok_or_else(|| LiteParseError::Other("no page rendered".into()))?;

    std::fs::write(output_path, &page.png_bytes)?;

    eprintln!(
        "[rust-bin] rendered page {} at {dpi} DPI → {output_path} ({}×{})",
        page_num, page.width, page.height
    );

    Ok(())
}

#[derive(Debug, Serialize)]
struct ImageBoundsOutput {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

/// Extract image bounding boxes and print as JSON to stdout.
pub fn image_bounds(pdf_path: &str, page_num: Option<u32>) -> Result<(), LiteParseError> {
    let lib = Library::init();
    let document = load_document_from_input(&lib, &PdfInput::Path(pdf_path.to_string()), None)?;
    let page_count = document.page_count();

    for page_index in 0..page_count {
        if let Some(target) = page_num
            && page_index as u32 + 1 != target
        {
            continue;
        }

        let page = document.page(page_index)?;
        let bounds = page.image_bounds(25.0, 0.9);

        let output: Vec<ImageBoundsOutput> = bounds
            .iter()
            .map(|b| ImageBoundsOutput {
                x: b.x,
                y: b.y,
                width: b.width,
                height: b.height,
            })
            .collect();

        let json = serde_json::json!({
            "page_number": page_index + 1,
            "images": output,
        });
        println!("{}", serde_json::to_string(&json)?);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_bounds_output_serializes() {
        let b = ImageBoundsOutput {
            x: 1.0,
            y: 2.0,
            width: 3.0,
            height: 4.0,
        };
        let s = serde_json::to_string(&b).unwrap();
        assert!(s.contains("\"x\":1"));
        assert!(s.contains("\"width\":3"));
    }

    #[test]
    fn test_screenshot_missing_file_errors() {
        let r = screenshot(
            "/nonexistent/path/does_not_exist.pdf",
            1,
            72.0,
            "/tmp/out.png",
            None,
        );
        assert!(r.is_err());
    }

    #[test]
    fn test_image_bounds_missing_file_errors() {
        let r = image_bounds("/nonexistent/path/does_not_exist.pdf", None);
        assert!(r.is_err());
    }
}
