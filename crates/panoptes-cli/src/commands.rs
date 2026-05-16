//! CLI command implementations.

use std::path::Path;

use panoptes_core::confidence;
use panoptes_core::inference::ThresholdEngine;
use panoptes_models::catalog;
use panoptes_models::pipeline::Pipeline;
use panoptes_raster::change::detect_change;
use panoptes_raster::tile::load_image;
use panoptes_vector::geojson_io::to_geojson_string;
use panoptes_vector::polygonize::polygonize_class;

pub fn segment(input: &Path, output: &Path, model: &str, tile_size: usize, min_area: f64) {
    let config = match model {
        "buildings" => catalog::building_segmentation(),
        "roads" => catalog::road_segmentation(),
        "vegetation" => catalog::vegetation_detection(),
        "landcover" => catalog::land_cover_classification(),
        _ => {
            eprintln!(
                "Unknown model: {}. Use 'panoptes models' to list available models.",
                model
            );
            std::process::exit(1);
        }
    };

    let image = match load_image(input) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to load image: {}", e);
            std::process::exit(1);
        }
    };

    println!("Loaded image: {:?}", image.shape());
    println!("Using model: {}", config.name);

    let mut pipeline = Pipeline::new(config.clone());
    pipeline.window_config = panoptes_raster::window::WindowConfig::new(tile_size, tile_size / 4);
    pipeline.min_area = min_area;

    // Use threshold engine as placeholder for real ONNX inference
    let engine = ThresholdEngine::new(vec![128.0]);
    let result = pipeline.process(&image, &engine);

    println!("Processed {} tiles", result.tile_results.len());
    println!("Extracted {} features", result.features.len());

    let geojson = to_geojson_string(&result.features);
    if let Err(e) = std::fs::write(output, &geojson) {
        eprintln!("Failed to write output: {}", e);
        std::process::exit(1);
    }
    println!("Output written to: {}", output.display());
}

pub fn change(before: &Path, after: &Path, output: &Path, threshold: f32) {
    let before_img = match load_image(before) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to load before image: {}", e);
            std::process::exit(1);
        }
    };

    let after_img = match load_image(after) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to load after image: {}", e);
            std::process::exit(1);
        }
    };

    println!("Before image: {:?}", before_img.shape());
    println!("After image: {:?}", after_img.shape());

    let result = detect_change(&before_img, &after_img, threshold);
    println!("Change ratio: {:.1}%", result.change_ratio * 100.0);

    // Polygonize the change mask
    let features = match polygonize_class(&result.change_mask, 1, 10.0) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Polygonization failed: {}", e);
            std::process::exit(1);
        }
    };

    println!("Detected {} change regions", features.len());

    let geojson = to_geojson_string(&features);
    if let Err(e) = std::fs::write(output, &geojson) {
        eprintln!("Failed to write output: {}", e);
        std::process::exit(1);
    }
    println!("Output written to: {}", output.display());
}

pub fn list_models() {
    let models = catalog::list_models();
    println!("Available models ({}):", models.len());
    println!();
    for model in &models {
        println!(
            "  {} ({})",
            model.name,
            format!("{:?}", model.task).to_lowercase()
        );
        println!(
            "    Input: {}x{}x{}",
            model.input.width, model.input.height, model.input.channels
        );
        println!(
            "    Classes: {}",
            model
                .classes
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("    Threshold: {}", model.confidence_threshold);
        println!();
    }
}

pub fn evaluate(prediction: &Path, ground_truth: &Path, num_classes: usize) {
    let pred_img = match load_image(prediction) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to load prediction: {}", e);
            std::process::exit(1);
        }
    };

    let gt_img = match load_image(ground_truth) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to load ground truth: {}", e);
            std::process::exit(1);
        }
    };

    // Use first channel as mask
    let pred_shape = pred_img.shape();
    let gt_shape = gt_img.shape();
    let (h, w) = (pred_shape[1], pred_shape[2]);

    let pred_mask = ndarray::Array2::from_shape_fn((h, w), |(y, x)| pred_img[[0, y, x]] as u8);
    let gt_mask = ndarray::Array2::from_shape_fn((gt_shape[1], gt_shape[2]), |(y, x)| {
        gt_img[[0, y, x]] as u8
    });

    let acc = confidence::pixel_accuracy(&pred_mask, &gt_mask);
    let miou = confidence::mean_iou(&pred_mask, &gt_mask, num_classes);

    println!("Evaluation Results:");
    println!("  Pixel Accuracy: {:.2}%", acc * 100.0);
    println!("  Mean IoU: {:.4}", miou);
    println!();

    for class_id in 0..num_classes {
        let class_iou = confidence::iou(&pred_mask, &gt_mask, class_id as u8);
        println!("  Class {} IoU: {:.4}", class_id, class_iou);
    }
}
