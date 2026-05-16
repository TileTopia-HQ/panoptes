//! GeoJSON input/output for vector features.

use geojson::{Feature, FeatureCollection, GeoJson, Geometry, Value};
use serde_json::json;

use crate::polygonize::VectorFeature;

/// Convert vector features to a GeoJSON FeatureCollection.
pub fn features_to_geojson(features: &[VectorFeature]) -> GeoJson {
    let geojson_features: Vec<Feature> = features
        .iter()
        .map(|f| {
            let coords: Vec<Vec<Vec<f64>>> = vec![
                f.geometry
                    .exterior()
                    .0
                    .iter()
                    .map(|c| vec![c.x, c.y])
                    .collect(),
            ];

            Feature {
                bbox: None,
                geometry: Some(Geometry::new(Value::Polygon(coords))),
                id: None,
                properties: Some(serde_json::Map::from_iter([
                    ("class_id".to_string(), json!(f.class_id)),
                    ("area_px".to_string(), json!(f.area_px)),
                    ("confidence".to_string(), json!(f.confidence)),
                ])),
                foreign_members: None,
            }
        })
        .collect();

    GeoJson::FeatureCollection(FeatureCollection {
        bbox: None,
        features: geojson_features,
        foreign_members: None,
    })
}

/// Serialize features to a GeoJSON string.
pub fn to_geojson_string(features: &[VectorFeature]) -> String {
    features_to_geojson(features).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Coord, LineString, Polygon};

    #[test]
    fn test_to_geojson() {
        let features = vec![VectorFeature {
            class_id: 1,
            geometry: Polygon::new(
                LineString::from(vec![
                    Coord { x: 0.0, y: 0.0 },
                    Coord { x: 1.0, y: 0.0 },
                    Coord { x: 1.0, y: 1.0 },
                    Coord { x: 0.0, y: 0.0 },
                ]),
                vec![],
            ),
            area_px: 100.0,
            confidence: 0.95,
        }];

        let geojson_str = to_geojson_string(&features);
        assert!(geojson_str.contains("Polygon"));
        assert!(geojson_str.contains("class_id"));
        assert!(geojson_str.contains("confidence"));
    }
}
