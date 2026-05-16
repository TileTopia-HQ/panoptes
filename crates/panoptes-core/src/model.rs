//! Model definitions — architecture metadata and class labels.

use serde::{Deserialize, Serialize};

/// Supported model task types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    /// Semantic segmentation (per-pixel classification).
    Segmentation,
    /// Object detection (bounding boxes + classes).
    Detection,
    /// Binary classification (single label per tile).
    Classification,
    /// Change detection (compare two inputs).
    ChangeDetection,
}

/// Model input specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSpec {
    /// Number of channels expected (e.g., 3 for RGB, 4 for RGBN).
    pub channels: usize,
    /// Expected height in pixels.
    pub height: usize,
    /// Expected width in pixels.
    pub width: usize,
    /// Whether to apply ImageNet normalization.
    pub imagenet_normalize: bool,
}

/// A class definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassDef {
    pub id: u8,
    pub name: String,
    pub color: [u8; 3],
}

/// Complete model metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub version: String,
    pub task: TaskType,
    pub input: InputSpec,
    pub classes: Vec<ClassDef>,
    /// Minimum confidence threshold for predictions.
    pub confidence_threshold: f32,
    /// Path to the model file (ONNX).
    pub model_path: Option<String>,
}

impl ModelConfig {
    pub fn num_classes(&self) -> usize {
        self.classes.len()
    }

    pub fn class_name(&self, id: u8) -> Option<&str> {
        self.classes
            .iter()
            .find(|c| c.id == id)
            .map(|c| c.name.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_serialization() {
        let config = ModelConfig {
            name: "building-footprints".to_string(),
            version: "1.0.0".to_string(),
            task: TaskType::Segmentation,
            input: InputSpec {
                channels: 3,
                height: 256,
                width: 256,
                imagenet_normalize: true,
            },
            classes: vec![
                ClassDef {
                    id: 0,
                    name: "background".to_string(),
                    color: [0, 0, 0],
                },
                ClassDef {
                    id: 1,
                    name: "building".to_string(),
                    color: [255, 0, 0],
                },
            ],
            confidence_threshold: 0.5,
            model_path: None,
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "building-footprints");
        assert_eq!(back.num_classes(), 2);
        assert_eq!(back.class_name(1), Some("building"));
    }
}
