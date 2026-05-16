//! Confidence scoring and quality metrics for predictions.

use crate::tensor::{ProbabilityMap, SegmentationMask};

/// Quality metrics for a segmentation prediction.
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Mean confidence across all pixels.
    pub mean_confidence: f32,
    /// Percentage of pixels above the confidence threshold.
    pub high_confidence_ratio: f32,
    /// Number of distinct classes predicted.
    pub classes_present: usize,
    /// Percentage of pixels belonging to each class.
    pub class_distribution: Vec<(u8, f32)>,
}

/// Compute quality metrics for a segmentation result.
pub fn compute_quality(
    mask: &SegmentationMask,
    confidence: &ProbabilityMap,
    threshold: f32,
    num_classes: usize,
) -> QualityMetrics {
    let total_pixels = (mask.shape()[0] * mask.shape()[1]) as f32;
    let mean_confidence = confidence.mean().unwrap_or(0.0);

    let high_confidence_count = confidence.iter().filter(|&&v| v >= threshold).count();
    let high_confidence_ratio = high_confidence_count as f32 / total_pixels;

    let mut class_counts = vec![0u64; num_classes];
    for &class_id in mask.iter() {
        if (class_id as usize) < num_classes {
            class_counts[class_id as usize] += 1;
        }
    }

    let classes_present = class_counts.iter().filter(|&&c| c > 0).count();
    let class_distribution: Vec<(u8, f32)> = class_counts
        .iter()
        .enumerate()
        .filter(|&(_, &count)| count > 0)
        .map(|(id, &count)| (id as u8, count as f32 / total_pixels))
        .collect();

    QualityMetrics {
        mean_confidence,
        high_confidence_ratio,
        classes_present,
        class_distribution,
    }
}

/// Compute Intersection over Union (IoU) between predicted and ground truth masks.
pub fn iou(predicted: &SegmentationMask, ground_truth: &SegmentationMask, class_id: u8) -> f32 {
    let mut intersection = 0u64;
    let mut union = 0u64;

    for (p, g) in predicted.iter().zip(ground_truth.iter()) {
        let p_match = *p == class_id;
        let g_match = *g == class_id;
        if p_match && g_match {
            intersection += 1;
        }
        if p_match || g_match {
            union += 1;
        }
    }

    if union == 0 {
        1.0 // Both empty — perfect match
    } else {
        intersection as f32 / union as f32
    }
}

/// Compute mean IoU across all classes.
pub fn mean_iou(
    predicted: &SegmentationMask,
    ground_truth: &SegmentationMask,
    num_classes: usize,
) -> f32 {
    let mut sum = 0.0_f32;
    let mut count = 0;

    for class_id in 0..num_classes {
        let class_iou = iou(predicted, ground_truth, class_id as u8);
        sum += class_iou;
        count += 1;
    }

    if count == 0 { 0.0 } else { sum / count as f32 }
}

/// Compute pixel accuracy.
pub fn pixel_accuracy(predicted: &SegmentationMask, ground_truth: &SegmentationMask) -> f32 {
    let total = predicted.len();
    let correct = predicted
        .iter()
        .zip(ground_truth.iter())
        .filter(|(p, g)| p == g)
        .count();
    correct as f32 / total as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_quality_metrics() {
        let mask = Array2::from_shape_fn((4, 4), |(y, _)| if y < 2 { 0u8 } else { 1 });
        let conf = Array2::from_elem((4, 4), 0.9_f32);
        let metrics = compute_quality(&mask, &conf, 0.5, 2);
        assert!((metrics.mean_confidence - 0.9).abs() < 0.01);
        assert_eq!(metrics.classes_present, 2);
        assert!((metrics.high_confidence_ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_iou_perfect() {
        let mask = Array2::from_elem((4, 4), 1u8);
        let gt = Array2::from_elem((4, 4), 1u8);
        assert!((iou(&mask, &gt, 1) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_iou_no_overlap() {
        let pred = Array2::from_elem((4, 4), 0u8);
        let gt = Array2::from_elem((4, 4), 1u8);
        assert!((iou(&pred, &gt, 1) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_pixel_accuracy() {
        let pred = Array2::from_shape_fn((4, 4), |(y, _)| if y < 2 { 0u8 } else { 1 });
        let gt = Array2::from_shape_fn((4, 4), |(y, _)| if y < 3 { 0u8 } else { 1 });
        // 12 correct (rows 0-1 both 0, row 3 both 1), 4 wrong (row 2: pred=1, gt=0)
        let acc = pixel_accuracy(&pred, &gt);
        assert!((acc - 0.75).abs() < 0.01);
    }
}
