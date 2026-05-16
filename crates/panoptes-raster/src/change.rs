//! Change detection — compare two temporal raster images.

use ndarray::Array2;
use panoptes_core::tensor::{ImageTensor, ProbabilityMap, SegmentationMask};

/// Result of a change detection analysis.
#[derive(Debug, Clone)]
pub struct ChangeResult {
    /// Binary mask of changed pixels.
    pub change_mask: SegmentationMask,
    /// Per-pixel change magnitude (0.0 to 1.0).
    pub magnitude: ProbabilityMap,
    /// Percentage of pixels that changed.
    pub change_ratio: f32,
}

/// Perform simple pixel-difference change detection between two images.
///
/// Both images must have the same dimensions (CHW).
/// Returns a change mask and magnitude map.
pub fn detect_change(before: &ImageTensor, after: &ImageTensor, threshold: f32) -> ChangeResult {
    let shape = before.shape();
    let (c, h, w) = (shape[0], shape[1], shape[2]);

    let mut magnitude = Array2::zeros((h, w));
    let mut change_mask = Array2::zeros((h, w));
    let mut changed_count = 0u64;

    for y in 0..h {
        for x in 0..w {
            let mut diff_sum = 0.0_f32;
            for ci in 0..c {
                let d = (after[[ci, y, x]] - before[[ci, y, x]]).abs();
                diff_sum += d;
            }
            let avg_diff = diff_sum / c as f32;
            // Normalize assuming pixel values 0-255
            let normalized = (avg_diff / 255.0).min(1.0);
            magnitude[[y, x]] = normalized;

            if normalized >= threshold {
                change_mask[[y, x]] = 1;
                changed_count += 1;
            }
        }
    }

    let total = (h * w) as f32;
    ChangeResult {
        change_mask,
        magnitude,
        change_ratio: changed_count as f32 / total,
    }
}

/// Compute per-band statistics of change.
pub fn band_change_stats(before: &ImageTensor, after: &ImageTensor) -> Vec<f32> {
    let shape = before.shape();
    let (c, h, w) = (shape[0], shape[1], shape[2]);
    let pixel_count = (h * w) as f32;

    (0..c)
        .map(|ci| {
            let sum: f32 = (0..h)
                .flat_map(|y| (0..w).map(move |x| (y, x)))
                .map(|(y, x)| (after[[ci, y, x]] - before[[ci, y, x]]).abs())
                .sum();
            sum / pixel_count
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_no_change() {
        let img = Array3::from_elem((3, 32, 32), 100.0_f32);
        let result = detect_change(&img, &img, 0.1);
        assert_eq!(result.change_ratio, 0.0);
        assert_eq!(result.change_mask.iter().filter(|&&v| v == 1).count(), 0);
    }

    #[test]
    fn test_full_change() {
        let before = Array3::from_elem((3, 32, 32), 0.0_f32);
        let after = Array3::from_elem((3, 32, 32), 255.0_f32);
        let result = detect_change(&before, &after, 0.5);
        assert!((result.change_ratio - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_band_change_stats() {
        let before = Array3::from_elem((3, 4, 4), 0.0_f32);
        let after = Array3::from_elem((3, 4, 4), 10.0_f32);
        let stats = band_change_stats(&before, &after);
        assert_eq!(stats.len(), 3);
        assert!((stats[0] - 10.0).abs() < 1e-5);
    }
}
