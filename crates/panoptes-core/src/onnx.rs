//! ONNX Runtime inference engine — run real models via ort.
//!
//! Enable with `--features onnx`. Requires ONNX Runtime shared library
//! to be available on the system (automatically downloaded by ort crate).

use std::path::Path;

use ndarray::Array3;
use ort::Session;
use thiserror::Error;

use crate::inference::{Detection, InferenceEngine, InferenceResult};
use crate::model::{ModelConfig, TaskType};
use crate::tensor::{ImageTensor, argmax_mask, confidence_map, normalize, softmax_chw};

/// Errors from ONNX inference.
#[derive(Debug, Error)]
pub enum OnnxError {
    #[error("ONNX Runtime error: {0}")]
    Runtime(#[from] ort::Error),
    #[error("Model file not found: {0}")]
    ModelNotFound(String),
    #[error("Unsupported task type for ONNX: {0:?}")]
    UnsupportedTask(TaskType),
    #[error("Shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },
}

/// ONNX Runtime-based inference engine.
pub struct OnnxEngine {
    session: Session,
    config: ModelConfig,
}

impl OnnxEngine {
    /// Load an ONNX model from disk.
    pub fn load(model_path: &Path, config: ModelConfig) -> Result<Self, OnnxError> {
        if !model_path.exists() {
            return Err(OnnxError::ModelNotFound(model_path.display().to_string()));
        }

        let session = Session::builder()?.commit_from_file(model_path)?;
        Ok(Self { session, config })
    }

    /// Run inference on a single CHW image tensor.
    pub fn infer(&self, input: &ImageTensor) -> Result<InferenceResult, OnnxError> {
        let shape = input.shape();
        let (c, h, w) = (shape[0], shape[1], shape[2]);

        // Normalize input
        let normalized = if self.config.input.imagenet_normalize {
            normalize(input)
        } else {
            let mut n = input.clone();
            n.mapv_inplace(|v| v / 255.0);
            n
        };

        // Create batch: [1, C, H, W]
        let batch = normalized
            .into_shape_with_order((1, c, h, w))
            .map_err(|e| OnnxError::ShapeMismatch {
                expected: format!("[1, {c}, {h}, {w}]"),
                actual: e.to_string(),
            })?;

        // Run the model
        let outputs = self.session.run(ort::inputs![batch]?)?;

        // Parse output
        match self.config.task {
            TaskType::Segmentation | TaskType::ChangeDetection => self.parse_segmentation(&outputs),
            TaskType::Detection => self.parse_detection(&outputs),
            TaskType::Classification => self.parse_classification(&outputs),
        }
    }

    fn parse_segmentation(
        &self,
        outputs: &ort::SessionOutputs<'_, '_>,
    ) -> Result<InferenceResult, OnnxError> {
        let output = &outputs[0];
        let tensor = output.try_extract_tensor::<f32>()?;
        let view = tensor.view();
        let out_shape = view.shape();

        // Output: [1, num_classes, H, W]
        let num_classes = out_shape[1];
        let out_h = out_shape[2];
        let out_w = out_shape[3];

        let logits = Array3::from_shape_fn((num_classes, out_h, out_w), |(ci, y, x)| {
            view[[0, ci, y, x]]
        });

        let probs = softmax_chw(&logits);
        let mask = argmax_mask(&probs);
        let confidence = confidence_map(&probs);

        Ok(InferenceResult::Segmentation { mask, confidence })
    }

    fn parse_detection(
        &self,
        outputs: &ort::SessionOutputs<'_, '_>,
    ) -> Result<InferenceResult, OnnxError> {
        let output = &outputs[0];
        let tensor = output.try_extract_tensor::<f32>()?;
        let view = tensor.view();
        let out_shape = view.shape();

        // Output: [1, N, 6] (x1, y1, x2, y2, confidence, class_id)
        let num_detections = out_shape[1];
        let mut detections = Vec::new();

        for i in 0..num_detections {
            let conf = view[[0, i, 4]];
            if conf >= self.config.confidence_threshold {
                detections.push(Detection {
                    class_id: view[[0, i, 5]] as u8,
                    confidence: conf,
                    bbox: [
                        view[[0, i, 0]],
                        view[[0, i, 1]],
                        view[[0, i, 2]],
                        view[[0, i, 3]],
                    ],
                });
            }
        }

        Ok(InferenceResult::Detection { detections })
    }

    fn parse_classification(
        &self,
        outputs: &ort::SessionOutputs<'_, '_>,
    ) -> Result<InferenceResult, OnnxError> {
        let output = &outputs[0];
        let tensor = output.try_extract_tensor::<f32>()?;
        let view = tensor.view();

        let mut max_idx = 0u8;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &v) in view.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i as u8;
            }
        }

        Ok(InferenceResult::Classification {
            class_id: max_idx,
            confidence: max_val,
        })
    }
}

impl InferenceEngine for OnnxEngine {
    fn predict(&self, input: &ImageTensor, _config: &ModelConfig) -> InferenceResult {
        match self.infer(input) {
            Ok(result) => result,
            Err(_) => InferenceResult::Detection { detections: vec![] },
        }
    }

    fn name(&self) -> &str {
        "onnx-runtime"
    }
}
