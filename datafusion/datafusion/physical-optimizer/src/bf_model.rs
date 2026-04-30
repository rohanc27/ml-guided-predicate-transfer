// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! GBT model loader and inference for the ML Bloom filter budget policy.

use std::sync::OnceLock;

use serde::Deserialize;

// ── Deserialization structs ───────────────────────────────────────────────────

#[derive(Debug, Deserialize, Clone)]
pub struct Node {
    pub feature: i32,
    pub threshold: Option<f64>,
    pub left: i32,
    pub right: i32,
    pub value: Option<f64>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Tree {
    pub nodes: Vec<Node>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct GbtModel {
    pub feature_names: Vec<String>,
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub init_score: f64,
    pub trees: Vec<Tree>,
}

// ── Global one-time model cache ───────────────────────────────────────────────

static MODEL_CACHE: OnceLock<Option<GbtModel>> = OnceLock::new();

/// Load and parse the GBT model from `path` exactly once per process.
/// Returns `None` if the file is missing or unparseable.
pub fn load_model_once(path: &str) -> Option<&'static GbtModel> {
    MODEL_CACHE
        .get_or_init(|| {
            let text = std::fs::read_to_string(path)
                .map_err(|e| log::warn!("bf_model: cannot read '{}': {}", path, e))
                .ok()?;
            serde_json::from_str::<GbtModel>(&text)
                .map_err(|e| log::warn!("bf_model: JSON parse error in '{}': {}", path, e))
                .ok()
        })
        .as_ref()
}

// ── Inference ─────────────────────────────────────────────────────────────────

fn walk_tree(nodes: &[Node], features: &[f64]) -> f64 {
    let mut idx = 0usize;
    loop {
        let node = &nodes[idx];
        if node.feature == -1 {
            return node.value.unwrap_or(0.0);
        }
        let feat_val = features.get(node.feature as usize).copied().unwrap_or(0.0);
        let threshold = node.threshold.unwrap_or(0.0);
        idx = if feat_val <= threshold {
            node.left as usize
        } else {
            node.right as usize
        };
    }
}

/// Returns P(beneficial = 1) in [0, 1].
/// `features` must be in the same order as `model.feature_names`.
pub fn predict_proba(features: &[f64], model: &GbtModel) -> f64 {
    let mut score = model.init_score;
    for tree in &model.trees {
        score += model.learning_rate * walk_tree(&tree.nodes, features);
    }
    // sigmoid
    1.0 / (1.0 + (-score).exp())
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn load_test_fixture() -> GbtModel {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/bf_model_test.json"
        );
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|_| panic!("missing fixture: {}", path));
        serde_json::from_str(&text).expect("fixture parse error")
    }

    // Hand-verifiable: features = [50, 50, 100, 5, 1.0]
    // Tree 1: build_cardinality=50 ≤ 100 → left → value=0.5
    // Tree 2: distinct_estimate=50 ≤ 60  → left → value=0.8
    // score = 0.0 + 0.1*0.5 + 0.1*0.8 = 0.13
    // sigmoid(0.13) ≈ 0.53246
    #[test]
    fn test_predict_low_cardinality() {
        let model = load_test_fixture();
        let features = vec![50.0, 50.0, 100.0, 5.0, 1.0];
        let p = predict_proba(&features, &model);
        let expected = 1.0_f64 / (1.0 + (-0.13_f64).exp());
        assert!((p - expected).abs() < 1e-9, "got {p}, expected {expected}");
    }

    // features = [200, 50, 100, 5, 1.0]
    // Tree 1: build_cardinality=200 > 100 → right → value=-0.5
    // Tree 2: distinct_estimate=50 ≤ 60   → left  → value=0.8
    // score = 0.0 + 0.1*(-0.5) + 0.1*0.8 = 0.03
    // sigmoid(0.03) ≈ 0.50750
    #[test]
    fn test_predict_high_cardinality() {
        let model = load_test_fixture();
        let features = vec![200.0, 50.0, 100.0, 5.0, 1.0];
        let p = predict_proba(&features, &model);
        let expected = 1.0_f64 / (1.0 + (-0.03_f64).exp());
        assert!((p - expected).abs() < 1e-9, "got {p}, expected {expected}");
    }

    // Low-cardinality score (0.13) > high-cardinality score (0.03).
    #[test]
    fn test_ordering() {
        let model = load_test_fixture();
        let p_low = predict_proba(&[50.0, 50.0, 100.0, 5.0, 1.0], &model);
        let p_high = predict_proba(&[200.0, 50.0, 100.0, 5.0, 1.0], &model);
        assert!(p_low > p_high, "p_low={p_low} should exceed p_high={p_high}");
    }

    // init_score = -inf equivalent: score always dominated by tree sum.
    #[test]
    fn test_sigmoid_bounds() {
        let model = load_test_fixture();
        let features = vec![50.0, 50.0, 100.0, 5.0, 1.0];
        let p = predict_proba(&features, &model);
        assert!(p > 0.0 && p < 1.0, "probability must be in (0,1)");
    }

    // Deserialisation: model has exactly 2 trees and 5 feature names.
    #[test]
    fn test_fixture_structure() {
        let model = load_test_fixture();
        assert_eq!(model.trees.len(), 2);
        assert_eq!(model.feature_names.len(), 5);
        assert_eq!(model.learning_rate, 0.1);
        assert_eq!(model.init_score, 0.0);
    }
}
