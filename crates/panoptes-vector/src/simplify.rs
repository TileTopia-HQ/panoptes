//! Polygon simplification — reduce vertex count while preserving shape.

use geo::Simplify;
use geo_types::Polygon;

use crate::polygonize::VectorFeature;

/// Simplify a polygon using the Ramer-Douglas-Peucker algorithm.
pub fn simplify_polygon(polygon: &Polygon<f64>, tolerance: f64) -> Polygon<f64> {
    polygon.simplify(&tolerance)
}

/// Simplify all features in a collection.
pub fn simplify_features(features: &mut [VectorFeature], tolerance: f64) {
    for feature in features.iter_mut() {
        feature.geometry = simplify_polygon(&feature.geometry, tolerance);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Coord, LineString, Polygon};

    #[test]
    fn test_simplify_reduces_points() {
        // Create a roughly rectangular polygon with many intermediate points
        let mut coords: Vec<Coord<f64>> = Vec::new();
        // Top edge with many intermediate points
        for i in 0..50 {
            coords.push(Coord {
                x: i as f64,
                y: 0.0 + (i as f64 * 0.001), // tiny deviations
            });
        }
        // Right edge
        for i in 0..50 {
            coords.push(Coord {
                x: 49.0 + (i as f64 * 0.001),
                y: i as f64,
            });
        }
        // Bottom edge back
        for i in (0..50).rev() {
            coords.push(Coord {
                x: i as f64,
                y: 49.0 - (i as f64 * 0.001),
            });
        }
        // Left edge back
        for i in (0..50).rev() {
            coords.push(Coord {
                x: 0.0 - (i as f64 * 0.001),
                y: i as f64,
            });
        }
        // Close the ring
        coords.push(coords[0]);

        let original_len = coords.len();
        let polygon = Polygon::new(LineString::from(coords), vec![]);
        let simplified = simplify_polygon(&polygon, 1.0);
        assert!(simplified.exterior().0.len() < original_len);
    }
}
