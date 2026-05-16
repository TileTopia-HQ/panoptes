//! Explainability maps — visualize model attention and feature importance.

use ndarray::Array2;
use panoptes_core::tensor::{ImageTensor, ProbabilityMap, SegmentationMask};

/// Type of explainability visualization.
#[derive(Debug, Clone, Copy)]
pub enum ExplainMethod {
    /// Gradient-weighted Class Activation Mapping.
    GradCam,
    /// Simple pixel saliency (input gradient magnitude).
    Saliency,
    /// Occlusion sensitivity (mask regions and observe confidence drop).
    Occlusion,
}

/// Explainability result for a single image.
#[derive(Debug, Clone)]
pub struct ExplanationMap {
    /// The attention/importance heatmap (0.0 to 1.0).
    pub heatmap: ProbabilityMap,
    /// Method used to generate the explanation.
    pub method: ExplainMethod,
    /// Target class this explanation is for.
    pub target_class: u8,
}

/// Generate an occlusion-based explanation map.
///
/// Slides a patch over the image and measures confidence drop for the target class.
/// Higher values indicate regions more important for the prediction.
pub fn occlusion_sensitivity<F>(
    image: &ImageTensor,
    target_class: u8,
    patch_size: usize,
    stride: usize,
    predict_fn: F,
) -> ExplanationMap
where
    F: Fn(&ImageTensor) -> (SegmentationMask, ProbabilityMap),
{
    let shape = image.shape();
    let (_, h, w) = (shape[0], shape[1], shape[2]);

    // Get baseline prediction confidence
    let (_, base_conf) = predict_fn(image);
    let base_mean = base_conf.mean().unwrap_or(0.5);

    let mut heatmap = Array2::<f32>::zeros((h, w));
    let mut counts = Array2::<f32>::zeros((h, w));

    let mut y = 0;
    while y + patch_size <= h {
        let mut x = 0;
        while x + patch_size <= w {
            // Occlude the patch (set to zero/mean)
            let mut occluded = image.clone();
            occluded
                .slice_mut(ndarray::s![.., y..y + patch_size, x..x + patch_size])
                .fill(0.0);

            // Measure confidence with occlusion
            let (_, occ_conf) = predict_fn(&occluded);
            let occ_mean = occ_conf.mean().unwrap_or(0.5);

            // Confidence drop = importance
            let importance = (base_mean - occ_mean).max(0.0);

            // Assign importance to occluded region
            for dy in 0..patch_size {
                for dx in 0..patch_size {
                    heatmap[[y + dy, x + dx]] += importance;
                    counts[[y + dy, x + dx]] += 1.0;
                }
            }

            x += stride;
        }
        y += stride;
    }

    // Normalize by count
    for y in 0..h {
        for x in 0..w {
            if counts[[y, x]] > 0.0 {
                heatmap[[y, x]] /= counts[[y, x]];
            }
        }
    }

    // Normalize to 0-1 range
    let max_val = heatmap.iter().cloned().fold(0.0_f32, f32::max);
    if max_val > 0.0 {
        heatmap.mapv_inplace(|v| v / max_val);
    }

    ExplanationMap {
        heatmap,
        method: ExplainMethod::Occlusion,
        target_class,
    }
}

/// Generate a simple gradient-based saliency map from prediction confidence.
///
/// Approximates gradient magnitude via finite differences on the confidence map.
pub fn saliency_from_confidence(confidence: &ProbabilityMap) -> ExplanationMap {
    let (h, w) = (confidence.shape()[0], confidence.shape()[1]);
    let mut saliency = Array2::<f32>::zeros((h, w));

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let dx = (confidence[[y, x + 1]] - confidence[[y, x - 1]]).abs();
            let dy = (confidence[[y + 1, x]] - confidence[[y - 1, x]]).abs();
            saliency[[y, x]] = (dx * dx + dy * dy).sqrt();
        }
    }

    // Normalize
    let max_val = saliency.iter().cloned().fold(0.0_f32, f32::max);
    if max_val > 0.0 {
        saliency.mapv_inplace(|v| v / max_val);
    }

    ExplanationMap {
        heatmap: saliency,
        method: ExplainMethod::Saliency,
        target_class: 0,
    }
}

/// Overlay a heatmap on an image for visualization.
///
/// Returns an RGB image with the heatmap blended using a jet colormap.
pub fn overlay_heatmap(image: &ImageTensor, heatmap: &ProbabilityMap, alpha: f32) -> ImageTensor {
    let shape = image.shape();
    let (_, h, w) = (shape[0], shape[1], shape[2]);
    let mut result = image.clone();

    for y in 0..h {
        for x in 0..w {
            let val = heatmap[[y, x]];
            let (hr, hg, hb) = jet_colormap(val);

            result[[0, y, x]] = result[[0, y, x]] * (1.0 - alpha) + hr * alpha;
            result[[1, y, x]] = result[[1, y, x]] * (1.0 - alpha) + hg * alpha;
            result[[2, y, x]] = result[[2, y, x]] * (1.0 - alpha) + hb * alpha;
        }
    }

    result
}

/// Jet colormap: maps 0.0-1.0 to RGB (0-255).
fn jet_colormap(value: f32) -> (f32, f32, f32) {
    let v = value.clamp(0.0, 1.0);
    let r = (1.5 - (v - 0.75).abs() * 4.0).clamp(0.0, 1.0) * 255.0;
    let g = (1.5 - (v - 0.5).abs() * 4.0).clamp(0.0, 1.0) * 255.0;
    let b = (1.5 - (v - 0.25).abs() * 4.0).clamp(0.0, 1.0) * 255.0;
    (r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    #[test]
    fn test_saliency_from_confidence() {
        // Create a confidence map with a gradient
        let conf = Array2::from_shape_fn((16, 16), |(_y, x)| x as f32 / 16.0);
        let result = saliency_from_confidence(&conf);
        assert_eq!(result.heatmap.shape(), &[16, 16]);
        // Interior pixels should have non-zero saliency (horizontal gradient)
        assert!(result.heatmap[[8, 8]] > 0.0);
    }

    #[test]
    fn test_jet_colormap_range() {
        let (r, g, b) = jet_colormap(0.0);
        assert!((0.0..=255.0).contains(&r));
        assert!((0.0..=255.0).contains(&g));
        assert!((0.0..=255.0).contains(&b));

        let (r, _, _) = jet_colormap(1.0);
        assert!((0.0..=255.0).contains(&r));
    }

    #[test]
    fn test_overlay_heatmap() {
        let image = Array3::from_elem((3, 8, 8), 128.0_f32);
        let heatmap = Array2::from_elem((8, 8), 0.5_f32);
        let result = overlay_heatmap(&image, &heatmap, 0.5);
        assert_eq!(result.shape(), &[3, 8, 8]);
    }

    #[test]
    fn test_occlusion_sensitivity() {
        let image = Array3::from_elem((3, 32, 32), 200.0_f32);

        // Mock predict function that returns higher confidence for brighter pixels
        let predict_fn = |img: &ImageTensor| {
            let mean_val = img.mean().unwrap_or(0.0) / 255.0;
            let mask = Array2::from_elem((32, 32), 1u8);
            let conf = Array2::from_elem((32, 32), mean_val);
            (mask, conf)
        };

        let result = occlusion_sensitivity(&image, 1, 8, 8, predict_fn);
        assert_eq!(result.heatmap.shape(), &[32, 32]);
        assert!(matches!(result.method, ExplainMethod::Occlusion));
    }
}
