//! Image pyramid — multi-resolution analysis.

use ndarray::Array3;
use panoptes_core::tensor::ImageTensor;

/// A level in the image pyramid.
#[derive(Debug, Clone)]
pub struct PyramidLevel {
    /// Level index (0 = original resolution).
    pub level: usize,
    /// Scale factor relative to original (1.0 = original, 0.5 = half-res).
    pub scale: f32,
    /// Image data at this level.
    pub data: ImageTensor,
}

/// Generate a multi-resolution pyramid using 2x downsampling (average pooling).
pub fn build_pyramid(image: &ImageTensor, levels: usize) -> Vec<PyramidLevel> {
    let mut pyramid = Vec::with_capacity(levels);
    pyramid.push(PyramidLevel {
        level: 0,
        scale: 1.0,
        data: image.clone(),
    });

    let mut current = image.clone();
    for i in 1..levels {
        current = downsample_2x(&current);
        pyramid.push(PyramidLevel {
            level: i,
            scale: 1.0 / (1 << i) as f32,
            data: current.clone(),
        });
    }

    pyramid
}

/// Downsample an image by 2x using average pooling.
fn downsample_2x(image: &ImageTensor) -> ImageTensor {
    let shape = image.shape();
    let (c, h, w) = (shape[0], shape[1], shape[2]);
    let new_h = h / 2;
    let new_w = w / 2;

    let mut result = Array3::zeros((c, new_h, new_w));
    for ci in 0..c {
        for y in 0..new_h {
            for x in 0..new_w {
                let avg = (image[[ci, y * 2, x * 2]]
                    + image[[ci, y * 2 + 1, x * 2]]
                    + image[[ci, y * 2, x * 2 + 1]]
                    + image[[ci, y * 2 + 1, x * 2 + 1]])
                    / 4.0;
                result[[ci, y, x]] = avg;
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_build_pyramid() {
        let image = Array3::from_elem((3, 256, 256), 100.0_f32);
        let pyramid = build_pyramid(&image, 4);
        assert_eq!(pyramid.len(), 4);
        assert_eq!(pyramid[0].data.shape(), &[3, 256, 256]);
        assert_eq!(pyramid[1].data.shape(), &[3, 128, 128]);
        assert_eq!(pyramid[2].data.shape(), &[3, 64, 64]);
        assert_eq!(pyramid[3].data.shape(), &[3, 32, 32]);
    }

    #[test]
    fn test_downsample_preserves_value() {
        let image = Array3::from_elem((3, 64, 64), 42.0_f32);
        let down = downsample_2x(&image);
        assert_eq!(down.shape(), &[3, 32, 32]);
        // Uniform image should stay the same value
        assert!((down[[0, 0, 0]] - 42.0).abs() < 1e-5);
    }
}
