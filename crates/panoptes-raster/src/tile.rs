//! Tile reading — load image tiles into tensors.

use image::{DynamicImage, ImageReader};
use ndarray::Array3;
use std::path::Path;
use thiserror::Error;

use panoptes_core::tensor::ImageTensor;

/// Errors during tile operations.
#[derive(Debug, Error)]
pub enum TileError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Image decode error: {0}")]
    Image(#[from] image::ImageError),
    #[error("Invalid dimensions: {0}")]
    Dimensions(String),
}

/// A loaded raster tile with metadata.
#[derive(Debug, Clone)]
pub struct Tile {
    /// Image data in CHW format.
    pub data: ImageTensor,
    /// Tile origin X (in pixel coordinates relative to full image).
    pub origin_x: u32,
    /// Tile origin Y.
    pub origin_y: u32,
    /// Original image width.
    pub source_width: u32,
    /// Original image height.
    pub source_height: u32,
}

/// Load an image file as a CHW tensor.
pub fn load_image(path: &Path) -> Result<ImageTensor, TileError> {
    let img = ImageReader::open(path)?.decode()?;
    Ok(image_to_tensor(&img))
}

/// Convert a DynamicImage to a CHW f32 tensor.
pub fn image_to_tensor(img: &DynamicImage) -> ImageTensor {
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let mut tensor = Array3::zeros((3, h, w));

    for y in 0..h {
        for x in 0..w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            tensor[[0, y, x]] = pixel[0] as f32;
            tensor[[1, y, x]] = pixel[1] as f32;
            tensor[[2, y, x]] = pixel[2] as f32;
        }
    }
    tensor
}

/// Convert a CHW tensor back to a DynamicImage (RGB).
pub fn tensor_to_image(tensor: &ImageTensor) -> DynamicImage {
    let shape = tensor.shape();
    let (h, w) = (shape[1], shape[2]);
    let mut imgbuf = image::RgbImage::new(w as u32, h as u32);

    for y in 0..h {
        for x in 0..w {
            let r = tensor[[0, y, x]].clamp(0.0, 255.0) as u8;
            let g = tensor[[1.min(shape[0] - 1), y, x]].clamp(0.0, 255.0) as u8;
            let b = tensor[[2.min(shape[0] - 1), y, x]].clamp(0.0, 255.0) as u8;
            imgbuf.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }
    DynamicImage::ImageRgb8(imgbuf)
}

/// Create a solid-color test tile.
pub fn create_test_tile(width: usize, height: usize, color: [f32; 3]) -> ImageTensor {
    let mut tensor = Array3::zeros((3, height, width));
    for (c, &val) in color.iter().enumerate() {
        tensor.slice_mut(ndarray::s![c, .., ..]).fill(val);
    }
    tensor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_test_tile() {
        let tile = create_test_tile(64, 64, [255.0, 128.0, 0.0]);
        assert_eq!(tile.shape(), &[3, 64, 64]);
        assert_eq!(tile[[0, 0, 0]], 255.0);
        assert_eq!(tile[[1, 0, 0]], 128.0);
        assert_eq!(tile[[2, 0, 0]], 0.0);
    }

    #[test]
    fn test_tensor_image_roundtrip() {
        let original = create_test_tile(32, 32, [100.0, 150.0, 200.0]);
        let img = tensor_to_image(&original);
        let back = image_to_tensor(&img);
        assert_eq!(back[[0, 0, 0]], 100.0);
        assert_eq!(back[[1, 0, 0]], 150.0);
        assert_eq!(back[[2, 0, 0]], 200.0);
    }
}
