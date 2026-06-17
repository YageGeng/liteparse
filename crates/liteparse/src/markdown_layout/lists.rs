/// Roughly one indent step in PDF points. Used to bucket list items into
/// nesting levels relative to the first item of the list.
pub(super) const LIST_INDENT_STEP_PT: f32 = 12.0;

/// Characters recognized as bullet markers when followed by whitespace.
/// Limited to glyphs that are unlikely to appear at line-start in normal prose.
const BULLET_CHARS: &[char] = &['•', '·', '◦', '▪', '▸', '▶', '●', '○', '■', '□'];

/// Detect a list marker at the start of `text`. Returns `(ordered, marker_str,
/// remainder)` when matched; otherwise `None`.
///
/// Recognizes:
/// - Unicode bullet characters (`BULLET_CHARS`) followed by whitespace.
/// - Decimal-prefix markers like `1.` / `1)` / `12.` / `12)` followed by
///   whitespace — kept strict (digits only) so things like footnote callers
///   (`1` alone) and section refs (`A.1`) don't match.
pub(super) fn parse_list_marker(text: &str) -> Option<(bool, String, &str)> {
    let trimmed = text.trim_start();
    if trimmed.is_empty() {
        return None;
    }

    let mut chars = trimmed.chars();
    let first = chars.next()?;

    // Unicode bullet
    if BULLET_CHARS.contains(&first) {
        let rest = chars.as_str();
        if let Some(rest_trim) = rest.strip_prefix(|c: char| c.is_whitespace()) {
            return Some((false, first.to_string(), rest_trim.trim_start()));
        }
    }

    // Decimal: 1. / 1) / 12. / 12)
    if first.is_ascii_digit() {
        let mut digit_end = 1;
        for c in trimmed[1..].chars() {
            if c.is_ascii_digit() {
                digit_end += c.len_utf8();
            } else {
                break;
            }
        }
        // Cap to keep us from matching page-number-like prefixes
        if digit_end <= 3 {
            let after_digits = &trimmed[digit_end..];
            let mut after_iter = after_digits.chars();
            if let Some(punct) = after_iter.next()
                && (punct == '.' || punct == ')')
            {
                let after_punct = after_iter.as_str();
                if let Some(rest_trim) = after_punct.strip_prefix(|c: char| c.is_whitespace()) {
                    let marker = format!("{}{}", &trimmed[..digit_end], punct);
                    return Some((true, marker, rest_trim.trim_start()));
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_list_marker_bullets() {
        let (ordered, marker, rest) = parse_list_marker("• item one").unwrap();
        assert!(!ordered);
        assert_eq!(marker, "•");
        assert_eq!(rest, "item one");
    }

    #[test]
    fn parse_list_marker_decimal() {
        let (ordered, marker, rest) = parse_list_marker("1. first").unwrap();
        assert!(ordered);
        assert_eq!(marker, "1.");
        assert_eq!(rest, "first");

        let (ordered, marker, rest) = parse_list_marker("12) twelfth").unwrap();
        assert!(ordered);
        assert_eq!(marker, "12)");
        assert_eq!(rest, "twelfth");
    }

    #[test]
    fn parse_list_marker_rejects_prose() {
        assert!(parse_list_marker("This sentence.").is_none());
        // Bare digit with no terminator → not a list
        assert!(parse_list_marker("2023 was a year").is_none());
        // Footnote caller / page number style — no whitespace after
        assert!(parse_list_marker("1.5x growth").is_none());
    }
}
