use crate::error::LiteParseError;
use crate::types::LayoutBlock;
use image::ImageEncoder;

const PALETTE: [[u8; 3]; 12] = [
    [230, 57, 70],
    [29, 78, 216],
    [22, 163, 74],
    [217, 119, 6],
    [147, 51, 234],
    [8, 145, 178],
    [219, 39, 119],
    [101, 163, 13],
    [79, 70, 229],
    [234, 88, 12],
    [15, 118, 110],
    [190, 24, 93],
];

pub fn annotate_layout_png(
    rgba: &mut [u8],
    image_width: u32,
    image_height: u32,
    page_width: f32,
    page_height: f32,
    blocks: &[LayoutBlock],
) -> Result<Vec<u8>, LiteParseError> {
    if rgba.len() != image_width as usize * image_height as usize * 4 {
        return Err(LiteParseError::Other(format!(
            "invalid RGBA buffer length: got {}, expected {}",
            rgba.len(),
            image_width as usize * image_height as usize * 4
        )));
    }

    if page_width > 0.0 && page_height > 0.0 {
        for block in blocks {
            draw_block(
                rgba,
                image_width,
                image_height,
                page_width,
                page_height,
                block,
            );
        }
    }

    encode_png(rgba, image_width, image_height)
}

fn draw_block(
    rgba: &mut [u8],
    image_width: u32,
    image_height: u32,
    page_width: f32,
    page_height: f32,
    block: &LayoutBlock,
) {
    let color = label_color(&block.label);
    let scale_x = image_width as f32 / page_width;
    let scale_y = image_height as f32 / page_height;

    let x0 = clamp_i32(
        (block.x * scale_x).floor() as i32,
        0,
        image_width as i32 - 1,
    );
    let y0 = clamp_i32(
        (block.y * scale_y).floor() as i32,
        0,
        image_height as i32 - 1,
    );
    let x1 = clamp_i32(
        ((block.x + block.width) * scale_x).ceil() as i32,
        0,
        image_width as i32 - 1,
    );
    let y1 = clamp_i32(
        ((block.y + block.height) * scale_y).ceil() as i32,
        0,
        image_height as i32 - 1,
    );

    if x1 <= x0 || y1 <= y0 {
        return;
    }

    draw_rect_outline(rgba, image_width, image_height, x0, y0, x1, y1, color);
    draw_label(rgba, image_width, image_height, x0, y0, &block.label, color);
}

fn draw_rect_outline(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: [u8; 3],
) {
    for offset in 0..2 {
        draw_hline(rgba, width, height, x0, x1, y0 + offset, color);
        draw_hline(rgba, width, height, x0, x1, y1 - offset, color);
        draw_vline(rgba, width, height, x0 + offset, y0, y1, color);
        draw_vline(rgba, width, height, x1 - offset, y0, y1, color);
    }
}

fn draw_label(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    label: &str,
    color: [u8; 3],
) {
    let label = label.trim();
    if label.is_empty() {
        return;
    }

    let text_width = (label.chars().count() as i32 * 6).max(1);
    let box_width = (text_width + 6).min(width as i32 - x);
    let box_height = 13;
    let box_y = if y >= box_height { y - box_height } else { y };

    fill_rect_alpha(
        rgba,
        width,
        height,
        x,
        box_y,
        x + box_width,
        box_y + box_height,
        color,
        220,
    );
    draw_text(
        rgba,
        width,
        height,
        x + 3,
        box_y + 3,
        label,
        [255, 255, 255],
    );
}

fn draw_text(rgba: &mut [u8], width: u32, height: u32, x: i32, y: i32, text: &str, color: [u8; 3]) {
    let mut cursor = x;
    for ch in text.chars() {
        if cursor >= width as i32 {
            break;
        }
        draw_char(rgba, width, height, cursor, y, ch, color);
        cursor += 6;
    }
}

fn draw_char(rgba: &mut [u8], width: u32, height: u32, x: i32, y: i32, ch: char, color: [u8; 3]) {
    let glyph = glyph(ch);
    for (row, bits) in glyph.iter().enumerate() {
        for col in 0..5 {
            if bits & (1 << (4 - col)) != 0 {
                set_pixel(rgba, width, height, x + col, y + row as i32, color);
            }
        }
    }
}

fn draw_hline(rgba: &mut [u8], width: u32, height: u32, x0: i32, x1: i32, y: i32, color: [u8; 3]) {
    for x in x0..=x1 {
        set_pixel(rgba, width, height, x, y, color);
    }
}

fn draw_vline(rgba: &mut [u8], width: u32, height: u32, x: i32, y0: i32, y1: i32, color: [u8; 3]) {
    for y in y0..=y1 {
        set_pixel(rgba, width, height, x, y, color);
    }
}

fn fill_rect_alpha(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: [u8; 3],
    alpha: u8,
) {
    for y in y0..y1 {
        for x in x0..x1 {
            blend_pixel(rgba, width, height, x, y, color, alpha);
        }
    }
}

fn set_pixel(rgba: &mut [u8], width: u32, height: u32, x: i32, y: i32, color: [u8; 3]) {
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return;
    }
    let idx = ((y as u32 * width + x as u32) * 4) as usize;
    rgba[idx] = color[0];
    rgba[idx + 1] = color[1];
    rgba[idx + 2] = color[2];
    rgba[idx + 3] = 255;
}

fn blend_pixel(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    color: [u8; 3],
    alpha: u8,
) {
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return;
    }
    let idx = ((y as u32 * width + x as u32) * 4) as usize;
    let alpha = alpha as u16;
    let inverse = 255 - alpha;
    for channel in 0..3 {
        rgba[idx + channel] =
            ((color[channel] as u16 * alpha + rgba[idx + channel] as u16 * inverse) / 255) as u8;
    }
    rgba[idx + 3] = 255;
}

fn label_color(label: &str) -> [u8; 3] {
    let hash = label.bytes().fold(0usize, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(byte as usize)
    });
    PALETTE[hash % PALETTE.len()]
}

fn clamp_i32(value: i32, min: i32, max: i32) -> i32 {
    value.max(min).min(max)
}

fn encode_png(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, LiteParseError> {
    let mut png = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(&mut png);
    encoder.write_image(rgba, width, height, image::ColorType::Rgba8.into())?;
    Ok(png)
}

fn glyph(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        'A' => [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'B' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
        ],
        'C' => [
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ],
        'D' => [
            0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
        'E' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
        'F' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'G' => [
            0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110,
        ],
        'H' => [
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'I' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111,
        ],
        'J' => [
            0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100,
        ],
        'K' => [
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
        'L' => [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
        'M' => [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
        'N' => [
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ],
        'O' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'P' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'Q' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ],
        'R' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
        'S' => [
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        'T' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'U' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'V' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
        ],
        'W' => [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010,
        ],
        'X' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ],
        'Y' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'Z' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ],
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
        '3' => [
            0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b00001, 0b00001, 0b11110,
        ],
        '6' => [
            0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110,
        ],
        '-' => [
            0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000,
        ],
        '_' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111,
        ],
        ' ' => [0; 7],
        _ => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b00100, 0b00000, 0b00100,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(label: &str, x: f32, y: f32, width: f32, height: f32) -> LayoutBlock {
        LayoutBlock {
            id: 1,
            label: label.into(),
            confidence: 0.9,
            x,
            y,
            width,
            height,
        }
    }

    #[test]
    fn annotation_draws_layout_box_pixels() {
        let mut rgba = vec![255; 80 * 60 * 4];
        let png = annotate_layout_png(
            &mut rgba,
            80,
            60,
            80.0,
            60.0,
            &[block("Title", 10.0, 12.0, 20.0, 16.0)],
        )
        .unwrap();

        assert!(!png.is_empty());
        let top_left = ((12 * 80 + 10) * 4) as usize;
        assert_ne!(&rgba[top_left..top_left + 3], &[255, 255, 255]);
    }

    #[test]
    fn annotation_returns_png_without_blocks() {
        let mut rgba = vec![255; 32 * 24 * 4];
        let png = annotate_layout_png(&mut rgba, 32, 24, 32.0, 24.0, &[]).unwrap();

        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
    }
}
