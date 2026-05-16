//! Sliding window — extract overlapping tiles from large images.

use panoptes_core::tensor::ImageTensor;

use crate::tile::Tile;

/// Configuration for sliding window extraction.
#[derive(Debug, Clone)]
pub struct WindowConfig {
    /// Tile width in pixels.
    pub tile_width: usize,
    /// Tile height in pixels.
    pub tile_height: usize,
    /// Overlap in pixels (stride = tile_size - overlap).
    pub overlap: usize,
}

impl WindowConfig {
    pub fn new(tile_size: usize, overlap: usize) -> Self {
        Self {
            tile_width: tile_size,
            tile_height: tile_size,
            overlap,
        }
    }

    /// Compute the stride (step size).
    pub fn stride_x(&self) -> usize {
        self.tile_width - self.overlap
    }

    pub fn stride_y(&self) -> usize {
        self.tile_height - self.overlap
    }
}

/// Extract tiles from a large image using a sliding window.
pub fn extract_tiles(image: &ImageTensor, config: &WindowConfig) -> Vec<Tile> {
    let shape = image.shape();
    let (_, img_h, img_w) = (shape[0], shape[1], shape[2]);
    let mut tiles = Vec::new();

    let stride_y = config.stride_y();
    let stride_x = config.stride_x();

    let mut y = 0;
    while y + config.tile_height <= img_h {
        let mut x = 0;
        while x + config.tile_width <= img_w {
            let tile_data = image
                .slice(ndarray::s![
                    ..,
                    y..y + config.tile_height,
                    x..x + config.tile_width
                ])
                .to_owned();

            tiles.push(Tile {
                data: tile_data,
                origin_x: x as u32,
                origin_y: y as u32,
                source_width: img_w as u32,
                source_height: img_h as u32,
            });

            x += stride_x;
        }
        y += stride_y;
    }

    tiles
}

/// Compute number of tiles that will be generated.
pub fn tile_count(img_width: usize, img_height: usize, config: &WindowConfig) -> usize {
    let cols = if img_width >= config.tile_width {
        (img_width - config.tile_width) / config.stride_x() + 1
    } else {
        0
    };
    let rows = if img_height >= config.tile_height {
        (img_height - config.tile_height) / config.stride_y() + 1
    } else {
        0
    };
    rows * cols
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_extract_tiles_no_overlap() {
        let image = Array3::zeros((3, 128, 128));
        let config = WindowConfig::new(64, 0);
        let tiles = extract_tiles(&image, &config);
        assert_eq!(tiles.len(), 4); // 2x2 grid
        assert_eq!(tiles[0].origin_x, 0);
        assert_eq!(tiles[0].origin_y, 0);
        assert_eq!(tiles[1].origin_x, 64);
        assert_eq!(tiles[3].origin_x, 64);
        assert_eq!(tiles[3].origin_y, 64);
    }

    #[test]
    fn test_extract_tiles_with_overlap() {
        let image = Array3::zeros((3, 128, 128));
        let config = WindowConfig::new(64, 32);
        let tiles = extract_tiles(&image, &config);
        // stride = 32, so (128-64)/32 + 1 = 3 per axis = 9 tiles
        assert_eq!(tiles.len(), 9);
    }

    #[test]
    fn test_tile_count() {
        let config = WindowConfig::new(64, 0);
        assert_eq!(tile_count(128, 128, &config), 4);
        assert_eq!(tile_count(256, 256, &config), 16);
    }
}
