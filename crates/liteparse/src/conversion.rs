use std::path::Path;

/// Supported file extensions for conversion (non-PDF formats).
const OFFICE_EXTENSIONS: &[&str] = &[
    "doc", "docx", "docm", "dot", "dotm", "dotx", "odt", "ott", "rtf", "pages",
];
const PRESENTATION_EXTENSIONS: &[&str] = &[
    "ppt", "pptx", "pptm", "pot", "potm", "potx", "odp", "otp", "key",
];
const SPREADSHEET_EXTENSIONS: &[&str] = &[
    "xls", "xlsx", "xlsm", "xlsb", "ods", "ots", "csv", "tsv", "numbers",
];
const IMAGE_EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "webp", "svg",
];

/// Check if a file is a PDF (no conversion needed).
pub fn is_pdf(path: &str) -> bool {
    Path::new(path)
        .extension()
        .map(|ext| ext.eq_ignore_ascii_case("pdf"))
        .unwrap_or(false)
}

/// Check if a file extension is supported (either PDF or convertible).
pub fn is_supported_extension(path: &str) -> bool {
    let ext = match Path::new(path).extension().and_then(|e| e.to_str()) {
        Some(e) => e.to_lowercase(),
        None => return false,
    };

    if ext == "pdf" {
        return true;
    }

    OFFICE_EXTENSIONS.contains(&ext.as_str())
        || PRESENTATION_EXTENSIONS.contains(&ext.as_str())
        || SPREADSHEET_EXTENSIONS.contains(&ext.as_str())
        || IMAGE_EXTENSIONS.contains(&ext.as_str())
}

/// Attempt to convert a non-PDF file to PDF.
///
/// Currently stubbed out — returns an error directing users to install
/// LibreOffice (for office documents) or ImageMagick (for images).
pub fn convert_to_pdf(path: &str, _password: Option<&str>) -> Result<String, Box<dyn std::error::Error>> {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    if ext == "pdf" {
        return Ok(path.to_string());
    }

    let tool = if OFFICE_EXTENSIONS.contains(&ext.as_str())
        || PRESENTATION_EXTENSIONS.contains(&ext.as_str())
        || SPREADSHEET_EXTENSIONS.contains(&ext.as_str())
    {
        "LibreOffice"
    } else if IMAGE_EXTENSIONS.contains(&ext.as_str()) {
        "ImageMagick"
    } else {
        return Err(format!("unsupported file format: .{}", ext).into());
    };

    Err(format!(
        "conversion from .{ext} to PDF requires {tool}, which is not yet supported in the Rust port. \
         Convert the file to PDF manually, or use the TypeScript version of LiteParse."
    )
    .into())
}
