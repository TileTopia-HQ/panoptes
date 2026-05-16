//! Distributed inference — parallel tile processing across threads.

use std::sync::Arc;

use ndarray::Array2;
use rayon::prelude::*;

use panoptes_core::inference::{InferenceEngine, InferenceResult};
use panoptes_core::model::ModelConfig;
use panoptes_core::tensor::{ImageTensor, ProbabilityMap, SegmentationMask};

use crate::tile::Tile;
use crate::window::{WindowConfig, extract_tiles};

/// Configuration for distributed inference.
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Number of worker threads (0 = auto-detect).
    pub num_workers: usize,
    /// Whether to merge overlapping tile results.
    pub merge_overlaps: bool,
    /// Tile overlap blending mode.
    pub blend_mode: BlendMode,
}

/// How to blend overlapping tile predictions.
#[derive(Debug, Clone, Copy)]
pub enum BlendMode {
    /// Use the maximum confidence value.
    Max,
    /// Average overlapping predictions.
    Average,
    /// Use center-weighted blending (higher weight in center, lower at edges).
    CenterWeighted,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            num_workers: 0,
            merge_overlaps: true,
            blend_mode: BlendMode::CenterWeighted,
        }
    }
}

/// Result of distributed inference over a full image.
#[derive(Debug)]
pub struct MergedResult {
    /// Final merged segmentation mask.
    pub mask: SegmentationMask,
    /// Final merged confidence map.
    pub confidence: ProbabilityMap,
    /// Number of tiles processed.
    pub tiles_processed: usize,
}

/// Run inference across tiles in parallel using rayon.
pub fn parallel_inference(
    image: &ImageTensor,
    engine: Arc<dyn InferenceEngine + Send + Sync>,
    config: &ModelConfig,
    window_config: &WindowConfig,
    dist_config: &DistributedConfig,
) -> MergedResult {
    // Configure thread pool if num_workers > 0
    let pool = if dist_config.num_workers > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(dist_config.num_workers)
                .build()
                .ok(),
        )
    } else {
        None
    };

    let tiles = extract_tiles(image, window_config);
    let shape = image.shape();
    let (_, img_h, img_w) = (shape[0], shape[1], shape[2]);

    // Run inference on all tiles in parallel
    let tile_results: Vec<(Tile, InferenceResult)> = if let Some(Some(pool)) = &pool {
        pool.install(|| {
            tiles
                .into_par_iter()
                .map(|tile| {
                    let result = engine.predict(&tile.data, config);
                    (tile, result)
                })
                .collect()
        })
    } else {
        tiles
            .into_par_iter()
            .map(|tile| {
                let result = engine.predict(&tile.data, config);
                (tile, result)
            })
            .collect()
    };

    let tiles_processed = tile_results.len();

    // Merge tile results into a single mask and confidence map
    let merged = merge_tiles(
        &tile_results,
        img_h,
        img_w,
        window_config,
        dist_config.blend_mode,
    );

    MergedResult {
        mask: merged.0,
        confidence: merged.1,
        tiles_processed,
    }
}

/// Merge per-tile segmentation results into a full-image result.
fn merge_tiles(
    tile_results: &[(Tile, InferenceResult)],
    img_h: usize,
    img_w: usize,
    _window_config: &WindowConfig,
    blend_mode: BlendMode,
) -> (SegmentationMask, ProbabilityMap) {
    let mut confidence_accum = Array2::<f32>::zeros((img_h, img_w));
    let mut weight_accum = Array2::<f32>::zeros((img_h, img_w));
    let mut class_votes = Array2::from_elem((img_h, img_w), Vec::<(u8, f32)>::new());

    for (tile, result) in tile_results {
        if let InferenceResult::Segmentation { mask, confidence } = result {
            let ox = tile.origin_x as usize;
            let oy = tile.origin_y as usize;
            let th = mask.shape()[0];
            let tw = mask.shape()[1];

            for y in 0..th {
                for x in 0..tw {
                    let gx = ox + x;
                    let gy = oy + y;
                    if gx >= img_w || gy >= img_h {
                        continue;
                    }

                    let weight = match blend_mode {
                        BlendMode::Max => 1.0,
                        BlendMode::Average => 1.0,
                        BlendMode::CenterWeighted => {
                            // Distance from center normalized to 0-1
                            let cx = (x as f32 / tw as f32 - 0.5).abs() * 2.0;
                            let cy = (y as f32 / th as f32 - 0.5).abs() * 2.0;
                            let dist = (cx * cx + cy * cy).sqrt().min(1.0);
                            1.0 - dist * 0.5 // Center = 1.0, edges = 0.5
                        }
                    };

                    let conf = confidence[[y, x]];
                    confidence_accum[[gy, gx]] += conf * weight;
                    weight_accum[[gy, gx]] += weight;
                    class_votes[[gy, gx]].push((mask[[y, x]], conf * weight));
                }
            }
        }
    }

    // Finalize: pick best class per pixel, compute average confidence
    let mut final_mask = Array2::<u8>::zeros((img_h, img_w));
    let mut final_confidence = Array2::<f32>::zeros((img_h, img_w));

    for y in 0..img_h {
        for x in 0..img_w {
            if weight_accum[[y, x]] > 0.0 {
                final_confidence[[y, x]] = confidence_accum[[y, x]] / weight_accum[[y, x]];

                // Weighted majority vote for class
                let votes = &class_votes[[y, x]];
                if !votes.is_empty() {
                    let mut class_weights = std::collections::HashMap::new();
                    for &(class_id, weight) in votes {
                        *class_weights.entry(class_id).or_insert(0.0_f32) += weight;
                    }
                    final_mask[[y, x]] = *class_weights
                        .iter()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(k, _)| k)
                        .unwrap_or(&0);
                }
            }
        }
    }

    (final_mask, final_confidence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use panoptes_core::inference::ThresholdEngine;
    use std::sync::Arc;

    #[test]
    fn test_parallel_inference() {
        let image = Array3::from_elem((3, 128, 128), 200.0_f32);
        let engine: Arc<dyn InferenceEngine + Send + Sync> =
            Arc::new(ThresholdEngine::new(vec![128.0]));
        let config = panoptes_core::model::ModelConfig {
            name: "test".to_string(),
            version: "1.0".to_string(),
            task: panoptes_core::model::TaskType::Segmentation,
            input: panoptes_core::model::InputSpec {
                width: 64,
                height: 64,
                channels: 3,
                imagenet_normalize: false,
            },
            classes: vec![
                panoptes_core::model::ClassDef {
                    id: 0,
                    name: "bg".to_string(),
                    color: [0, 0, 0],
                },
                panoptes_core::model::ClassDef {
                    id: 1,
                    name: "fg".to_string(),
                    color: [255, 0, 0],
                },
            ],
            confidence_threshold: 0.5,
            model_path: None,
        };
        let window_config = WindowConfig::new(64, 16);
        let dist_config = DistributedConfig::default();

        let result = parallel_inference(&image, engine, &config, &window_config, &dist_config);
        assert_eq!(result.mask.shape(), &[128, 128]);
        assert_eq!(result.confidence.shape(), &[128, 128]);
        assert!(result.tiles_processed > 0);
    }
}
