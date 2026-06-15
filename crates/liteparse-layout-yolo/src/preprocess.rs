use crate::error::LayoutError;
use crate::types::PageImage;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    pub input_width: u32,
    pub input_height: u32,
    pub target_size: u32,
    pub resized_width: u32,
    pub resized_height: u32,
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
}

impl Letterbox {
    pub fn new(input_width: u32, input_height: u32, target_size: u32) -> Self {
        let scale =
            (target_size as f32 / input_width as f32).min(target_size as f32 / input_height as f32);
        let resized_width = (input_width as f32 * scale).round() as u32;
        let resized_height = (input_height as f32 * scale).round() as u32;
        let pad_x = (target_size - resized_width) as f32 / 2.0;
        let pad_y = (target_size - resized_height) as f32 / 2.0;

        Self {
            input_width,
            input_height,
            target_size,
            resized_width,
            resized_height,
            scale,
            pad_x,
            pad_y,
        }
    }
}

pub fn validate_page_image(image: &PageImage<'_>) -> Result<(), LayoutError> {
    let expected = image.width as usize * image.height as usize * 3;
    let actual = image.rgb.len();
    if actual != expected {
        return Err(LayoutError::InvalidImageBuffer { expected, actual });
    }
    Ok(())
}

pub fn letterbox_rgb_to_chw_f32(
    image: &PageImage<'_>,
    target_size: u32,
) -> Result<(Vec<f32>, Letterbox), LayoutError> {
    validate_page_image(image)?;
    let letterbox = Letterbox::new(image.width, image.height, target_size);
    let target_size = target_size as usize;
    let mut input = vec![1.0; 3 * target_size * target_size];

    for target_y in 0..letterbox.resized_height {
        let source_y = ((target_y as f32 + 0.5) / letterbox.scale)
            .floor()
            .clamp(0.0, image.height.saturating_sub(1) as f32) as usize;
        let output_y = target_y as usize + letterbox.pad_y.round() as usize;

        for target_x in 0..letterbox.resized_width {
            let source_x = ((target_x as f32 + 0.5) / letterbox.scale)
                .floor()
                .clamp(0.0, image.width.saturating_sub(1) as f32)
                as usize;
            let output_x = target_x as usize + letterbox.pad_x.round() as usize;
            let source_offset = (source_y * image.width as usize + source_x) * 3;
            let output_offset = output_y * target_size + output_x;

            input[output_offset] = image.rgb[source_offset] as f32 / 255.0;
            input[target_size * target_size + output_offset] =
                image.rgb[source_offset + 1] as f32 / 255.0;
            input[2 * target_size * target_size + output_offset] =
                image.rgb[source_offset + 2] as f32 / 255.0;
        }
    }

    Ok((input, letterbox))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_letterbox_for_wide_page() {
        let letterbox = Letterbox::new(1000, 500, 1280);

        assert_eq!(letterbox.resized_width, 1280);
        assert_eq!(letterbox.resized_height, 640);
        assert_eq!(letterbox.pad_x, 0.0);
        assert_eq!(letterbox.pad_y, 320.0);
        assert!((letterbox.scale - 1.28).abs() < 0.0001);
    }

    #[test]
    fn validates_rgb_buffer_length() {
        let image = PageImage {
            rgb: &[0; 12],
            width: 2,
            height: 2,
            page_width: 2.0,
            page_height: 2.0,
            dpi: 72.0,
        };

        assert!(validate_page_image(&image).is_ok());
    }

    #[test]
    fn letterboxes_rgb_to_chw_normalized_float_input() {
        let rgb = [
            255, 0, 0, //
            0, 255, 0,
        ];
        let image = PageImage {
            rgb: &rgb,
            width: 2,
            height: 1,
            page_width: 2.0,
            page_height: 1.0,
            dpi: 72.0,
        };

        let (input, letterbox) = letterbox_rgb_to_chw_f32(&image, 4).unwrap();

        assert_eq!(letterbox.resized_width, 4);
        assert_eq!(letterbox.resized_height, 2);
        assert_eq!(input.len(), 3 * 4 * 4);
        assert_eq!(input[0], 1.0);
        assert_eq!(input[16], 1.0);
        assert_eq!(input[32], 1.0);
        assert_eq!(input[4], 1.0);
        assert_eq!(input[20], 0.0);
        assert_eq!(input[36], 0.0);
        assert_eq!(input[6], 0.0);
        assert_eq!(input[22], 1.0);
        assert_eq!(input[38], 0.0);
    }
}
