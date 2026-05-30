//! # panoptes-core
//!
//! AI feature extraction engine core: tensor operations, model inference,
//! prediction types for geospatial raster analysis, and satellite image processing.

pub mod confidence;
pub mod inference;
pub mod model;
#[cfg(feature = "onnx")]
pub mod onnx;
pub mod satellite;
pub mod tensor;
