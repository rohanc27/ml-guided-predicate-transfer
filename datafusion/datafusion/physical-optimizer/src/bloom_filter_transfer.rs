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

//! Multi-hop predicate transfer via runtime Bloom filters.

use std::any::Any;
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, BooleanArray, Int32Array, Int64Array, StringArray};
use arrow::compute::filter_record_batch;
use arrow::record_batch::RecordBatch;
use bloomfilter::Bloom;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{DataFusionError, JoinType, Result};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::joins::HashJoinExec;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
};
use futures::StreamExt;
use tokio::sync::broadcast;

use crate::bf_model::{self, GbtModel};
use crate::PhysicalOptimizerRule;

// Shared Bloom Filter Handle

#[derive(Debug, Clone)]
pub struct BloomPayload {
    pub filter: Arc<Bloom<String>>,
    pub build_cardinality: usize,
    pub distinct_estimate: usize,
    pub filter_size_bytes: usize,
    pub build_time_ms: f64,
}

type SharedBloom = Arc<broadcast::Sender<Arc<BloomPayload>>>;

// Execution Metrics

#[derive(Debug)]
pub struct FilterMetrics {
    pub query_id: String,
    pub col_name: String,
    pub build_cardinality: usize,
    pub distinct_estimate: usize,
    pub filter_size_bytes: usize,
    pub build_time_ms: f64,
    pub probe_batches: usize,
    pub rows_in: usize,
    pub rows_out: usize,
    pub rows_eliminated: usize,
    pub probe_time_ms: f64,
}

impl FilterMetrics {
    pub fn elimination_rate(&self) -> f64 {
        if self.rows_in == 0 { return 0.0; }
        self.rows_eliminated as f64 / self.rows_in as f64
    }

    pub fn net_benefit_ms(&self) -> f64 {
        let scan_ms_per_row = 0.001;
        let saved_ms = self.rows_eliminated as f64 * scan_ms_per_row;
        saved_ms - self.build_time_ms - self.probe_time_ms
    }

    pub fn was_beneficial(&self) -> u8 {
        if self.net_benefit_ms() > 0.0 { 1 } else { 0 }
    }

    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{:.4},{},{},{},{},{:.4},{:.4},{}\n",
            self.query_id,
            self.col_name,
            self.build_cardinality,
            self.distinct_estimate,
            self.filter_size_bytes,
            self.build_time_ms,
            self.probe_batches,
            self.rows_in,
            self.rows_out,
            self.rows_eliminated,
            self.probe_time_ms,
            self.net_benefit_ms(),
            self.was_beneficial()
        )
    }
}

pub fn write_metrics(metrics: &FilterMetrics, path: &str) {
    let write_header = !std::path::Path::new(path).exists() || std::fs::metadata(path).map(|m| m.len() == 0).unwrap_or(true);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    if write_header {
        writeln!(file, "query_id,col_name,build_cardinality,distinct_estimate,filter_size_bytes,build_time_ms,probe_batches,rows_in,rows_out,rows_eliminated,probe_time_ms,net_benefit_ms,was_beneficial").unwrap();
    }
    write!(file, "{}", metrics.to_csv_row()).unwrap();
}

// Rule Struct

#[derive(Default, Debug)]
pub struct MultiHopBloomFilterRule {}

impl MultiHopBloomFilterRule {
    pub fn new() -> Self {
        Self {}
    }
}

// JoinEdge

#[derive(Debug, Clone)]
pub struct JoinEdge {
    pub left_col: String,
    pub right_col: String,
}

// Join Graph Collection

pub fn collect_join_edges(plan: &Arc<dyn ExecutionPlan>) -> Vec<JoinEdge> {
    let mut edges = Vec::new();
    collect_recursive(plan, &mut edges);
    edges
}

fn collect_recursive(plan: &Arc<dyn ExecutionPlan>, edges: &mut Vec<JoinEdge>) {
    if let Some(hash_join) = plan.as_any().downcast_ref::<HashJoinExec>() {
        if *hash_join.join_type() == JoinType::Inner {
            for (left_expr, right_expr) in hash_join.on() {
                edges.push(JoinEdge {
                    left_col: format!("{}", left_expr),
                    right_col: format!("{}", right_expr),
                });
            }
        }
    }
    for child in plan.children() {
        collect_recursive(child, edges);
    }
}

// BloomFilterBuildExec

#[derive(Debug)]
pub struct BloomFilterBuildExec {
    input: Arc<dyn ExecutionPlan>,
    key_column_name: String,
    sender: SharedBloom,
}

impl BloomFilterBuildExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        key_column_name: String,
        sender: SharedBloom,
    ) -> Self {
        Self { input, key_column_name, sender }
    }
}

impl DisplayAs for BloomFilterBuildExec {
    fn fmt_as(
        &self,
        _t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "BloomFilterBuildExec: key_col={}", self.key_column_name)
    }
}

impl ExecutionPlan for BloomFilterBuildExec {
    fn name(&self) -> &'static str {
        "BloomFilterBuildExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(BloomFilterBuildExec::new(
            Arc::clone(&children[0]),
            self.key_column_name.clone(),
            Arc::clone(&self.sender),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let schema = self.input.schema();
        let key_column = schema
            .fields()
            .iter()
            .position(|f| f.name() == &self.key_column_name)
            .ok_or_else(|| DataFusionError::Plan(
                format!("BloomFilterBuildExec: column '{}' not found", self.key_column_name)
            ))?;

        let mut input_stream = self.input.execute(partition, context)?;
        let sender = Arc::clone(&self.sender);
        let schema_ref = self.input.schema();
        let key_column_name = self.key_column_name.clone();

        let output_stream = async_stream::stream! {
            let build_start = Instant::now();
            let mut all_values: Vec<String> = Vec::new();
            let mut batches: Vec<RecordBatch> = Vec::new();

            while let Some(batch_result) = input_stream.next().await {
                let batch: RecordBatch = match batch_result {
                    Ok(b) => b,
                    Err(e) => { yield Err(e); return; }
                };

                let col = batch.column(key_column);
                if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
                    for i in 0..arr.len() {
                        if !arr.is_null(i) { all_values.push(arr.value(i).to_string()); }
                    }
                } else if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                    for i in 0..arr.len() {
                        if !arr.is_null(i) { all_values.push(arr.value(i).to_string()); }
                    }
                } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                    for i in 0..arr.len() {
                        if !arr.is_null(i) { all_values.push(arr.value(i).to_string()); }
                    }
                }
                batches.push(batch);
            }

            let build_cardinality = all_values.len();
            let distinct_estimate = {
                let mut seen = std::collections::HashSet::new();
                for v in &all_values { seen.insert(v.clone()); }
                seen.len()
            };

            let mut bf = Bloom::new_for_fp_rate(build_cardinality.max(1), 0.01).unwrap();
            for v in &all_values {
                bf.set(v);
            }
            let filter_size_bytes = build_cardinality * 2;
            let build_time_ms = build_start.elapsed().as_secs_f64() * 1000.0;

            let payload = BloomPayload {
                filter: Arc::new(bf),
                build_cardinality,
                distinct_estimate,
                filter_size_bytes,
                build_time_ms,
            };
            let _ = sender.send(Arc::new(payload)); 

            for batch in batches {
                yield Ok(batch);
            }
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema_ref, output_stream)))
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&dyn PhysicalExpr) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }
}

// BloomFilterProbeExec

#[derive(Debug)]
pub struct BloomFilterProbeExec {
    input: Arc<dyn ExecutionPlan>,
    key_column_name: String,
    sender: SharedBloom,
}

impl BloomFilterProbeExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        key_column_name: String,
        sender: SharedBloom,
    ) -> Self {
        Self { input, key_column_name, sender }
    }
}

impl DisplayAs for BloomFilterProbeExec {
    fn fmt_as(
        &self,
        _t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "BloomFilterProbeExec: key_col={}", self.key_column_name)
    }
}

impl ExecutionPlan for BloomFilterProbeExec {
    fn name(&self) -> &'static str {
        "BloomFilterProbeExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(BloomFilterProbeExec::new(
            Arc::clone(&children[0]),
            self.key_column_name.clone(),
            Arc::clone(&self.sender),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let schema = self.input.schema();
        let key_column = schema
            .fields()
            .iter()
            .position(|f| f.name() == &self.key_column_name)
            .ok_or_else(|| DataFusionError::Plan(
                format!("BloomFilterProbeExec: column '{}' not found", self.key_column_name)
            ))?;

        let mut input_stream = self.input.execute(partition, context)?;
        let mut receiver = self.sender.subscribe();
        let schema_ref = self.input.schema();
        let key_column_name = self.key_column_name.clone();

        let output_stream = async_stream::stream! {
            let payload = match receiver.recv().await {
                Ok(p) => p,
                Err(_) => {
                    while let Some(batch_result) = input_stream.next().await {
                        match batch_result {
                            Ok(b) => yield Ok(b),
                            Err(e) => yield Err(e),
                        }
                    }
                    return;
                }
            };

            let probe_start = Instant::now();
            let mut probe_batches = 0usize;
            let mut total_rows_in = 0usize;
            let mut total_rows_out = 0usize;

            while let Some(batch_result) = input_stream.next().await {
                let batch: RecordBatch = match batch_result {
                    Ok(b) => b,
                    Err(e) => { yield Err(e); continue; }
                };

                let col = batch.column(key_column);
                let keep: Vec<bool> = (0..col.len()).map(|i| {
                    if col.is_null(i) {
                        false
                    } else {
                        let val = if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
                            arr.value(i).to_string()
                        } else if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                            arr.value(i).to_string()
                        } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                            arr.value(i).to_string()
                        } else {
                            return true;
                        };
                        payload.filter.check(&val)
                    }
                }).collect();

                let rows_before = batch.num_rows();
                let mask = BooleanArray::from(keep);
                match filter_record_batch(&batch, &mask) {
                    Ok(filtered) => {
                        let rows_after = filtered.num_rows();
                        total_rows_in += rows_before;
                        total_rows_out += rows_after;
                        probe_batches += 1;
                        let eliminated = rows_before - rows_after;
                        if eliminated > 0 {
                            println!(
                                "[BloomFilterProbeExec] col='{}' eliminated {}/{} rows ({:.1}%)",
                                key_column_name, eliminated, rows_before,
                                (eliminated as f64 / rows_before as f64) * 100.0
                            );
                        }
                        yield Ok(filtered)
                    },
                    Err(e) => yield Err(DataFusionError::ArrowError(Box::new(e), None)),
                }
            }

            let probe_time_ms = probe_start.elapsed().as_secs_f64() * 1000.0;
            let rows_eliminated = total_rows_in - total_rows_out;

            let csv_path = std::env::var("BF_METRICS_PATH")
                .unwrap_or_else(|_| "/tmp/bf_metrics.csv".to_string());
            let build_cardinality = payload.build_cardinality;
            let distinct_estimate = payload.distinct_estimate;
            let filter_size_bytes = payload.filter_size_bytes;
            let build_time_ms = payload.build_time_ms;

            let query_id = std::env::var("BF_QUERY_ID")
                .unwrap_or_else(|_| "unknown".to_string());

            let metrics = FilterMetrics {
                query_id,
                col_name: key_column_name.clone(),
                build_cardinality,
                distinct_estimate,
                filter_size_bytes,
                build_time_ms,
                probe_batches,
                rows_in: total_rows_in,
                rows_out: total_rows_out,
                rows_eliminated,
                probe_time_ms,
            };

            write_metrics(&metrics, &csv_path);
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema_ref, output_stream)))
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&dyn PhysicalExpr) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }
}

// Plan Rewriting

fn insert_bloom_filters(
    plan: Arc<dyn ExecutionPlan>,
    edges: &[JoinEdge],
) -> Result<Arc<dyn ExecutionPlan>> {
    if edges.is_empty() {
        return Ok(plan);
    }

    if let Some(hash_join) = plan.as_any().downcast_ref::<HashJoinExec>() {
        if *hash_join.join_type() == JoinType::Inner {
            let left = Arc::clone(hash_join.left());
            let right = Arc::clone(hash_join.right());

            let left = insert_bloom_filters(left, edges)?;
            let right = insert_bloom_filters(right, edges)?;

            let mut new_left = left;
            let mut new_right = right;

            for (join_left_expr, join_right_expr) in hash_join.on() {
                let left_col_name = format!("{}", join_left_expr);
                let right_col_name = format!("{}", join_right_expr);

                let left_schema = new_left.schema();
                let right_schema = new_right.schema();

                let left_field_name = left_schema
                    .fields()
                    .iter()
                    .find(|f| left_col_name.contains(f.name()))
                    .map(|f| f.name().clone());

                let right_field_name = right_schema
                    .fields()
                    .iter()
                    .find(|f| right_col_name.contains(f.name()))
                    .map(|f| f.name().clone());

                if let (Some(left_field_name), Some(right_field_name)) =
                    (left_field_name, right_field_name)
                {
                    let (sender, _) = broadcast::channel(1);
                    let sender = Arc::new(sender);

                    new_left = Arc::new(BloomFilterBuildExec::new(
                        new_left,
                        left_field_name.clone(),
                        Arc::clone(&sender),
                    ));

                    new_right = Arc::new(BloomFilterProbeExec::new(
                        new_right,
                        right_field_name.clone(),
                        Arc::clone(&sender),
                    ));

                    println!(
                        "Inserted BF: Build On Col '{}', Probe On Col '{}'",
                        left_field_name, right_field_name
                    );
                }
            }

            let new_plan = plan.with_new_children(vec![new_left, new_right])?;
            return Ok(new_plan);
        }
    }

    let children = plan.children();
    if children.is_empty() {
        return Ok(plan);
    }

    let new_children = children
        .into_iter()
        .map(|c| insert_bloom_filters(Arc::clone(c), edges))
        .collect::<Result<Vec<_>>>()?;

    plan.with_new_children(new_children)
}

// ML Policy Support

/// Estimated build-side features for a join key column.
/// Order matches `bf_model::GbtModel::feature_names`:
/// [build_cardinality, distinct_estimate, filter_size_bytes, build_time_ms, distinctness_ratio]
fn extract_features(build_side: &Arc<dyn ExecutionPlan>) -> [f64; 5] {
    const FALLBACK: f64 = 10_000.0;

    let build_cardinality = build_side
        .statistics()
        .ok()
        .and_then(|s| s.num_rows.get_value().copied())
        .map(|n| n as f64)
        .unwrap_or(FALLBACK);

    // Heuristic: assume 80% distinct values when statistics are absent.
    let distinct_estimate = build_cardinality * 0.8;
    // filter_size_bytes: 2 bytes per row (same estimate used in BloomFilterBuildExec).
    let filter_size_bytes = build_cardinality * 2.0;
    // build_time_ms: measured empirically at ~0.0017 ms per row.
    let build_time_ms = build_cardinality * 0.0017;
    let distinctness_ratio = distinct_estimate / build_cardinality.max(1.0);

    [
        build_cardinality,
        distinct_estimate,
        filter_size_bytes,
        build_time_ms,
        distinctness_ratio,
    ]
}

struct MlCandidate {
    key: String,
    build_col: String,
    probe_col: String,
    weight_bytes: usize,
    pred_prob: f64,
}

/// Walk plan to collect one `MlCandidate` per join key that has resolvable
/// field names on both sides.
fn collect_ml_candidates(
    plan: &Arc<dyn ExecutionPlan>,
    model: &GbtModel,
    candidates: &mut Vec<MlCandidate>,
) {
    if let Some(hash_join) = plan.as_any().downcast_ref::<HashJoinExec>() {
        if *hash_join.join_type() == JoinType::Inner {
            let left = Arc::clone(hash_join.left());
            let right = Arc::clone(hash_join.right());

            // Recurse into children first so outer joins wrap inner ones.
            collect_ml_candidates(&left, model, candidates);
            collect_ml_candidates(&right, model, candidates);

            let left_schema = left.schema();
            let right_schema = right.schema();

            let features = extract_features(&left);
            let weight_bytes = features[2] as usize; // filter_size_bytes
            let pred_prob = bf_model::predict_proba(&features, model);

            for (left_expr, right_expr) in hash_join.on() {
                let left_name_str = format!("{}", left_expr);
                let right_name_str = format!("{}", right_expr);

                let build_col = left_schema
                    .fields()
                    .iter()
                    .find(|f| left_name_str.contains(f.name()))
                    .map(|f| f.name().clone());
                let probe_col = right_schema
                    .fields()
                    .iter()
                    .find(|f| right_name_str.contains(f.name()))
                    .map(|f| f.name().clone());

                if let (Some(build_col), Some(probe_col)) = (build_col, probe_col) {
                    // Use a counter suffix to keep keys unique when the same
                    // column appears in multiple joins.
                    let key = format!("{}_{}", build_col, candidates.len());
                    candidates.push(MlCandidate {
                        key,
                        build_col,
                        probe_col,
                        weight_bytes,
                        pred_prob,
                    });
                }
            }
            return;
        }
    }

    for child in plan.children() {
        collect_ml_candidates(child, model, candidates);
    }
}

/// 0/1 knapsack (KB-scaled weights) — returns the set of selected candidate keys.
fn knapsack_select(candidates: &[MlCandidate], budget_bytes: usize) -> HashSet<String> {
    if candidates.is_empty() || budget_bytes == 0 {
        return HashSet::new();
    }

    let to_kb = |b: usize| -> usize { (b / 1024).max(1) };

    let n = candidates.len();
    let w_cap = to_kb(budget_bytes);
    let ws: Vec<usize> = candidates.iter().map(|c| to_kb(c.weight_bytes)).collect();
    let vs: Vec<f64> = candidates.iter().map(|c| c.pred_prob).collect();

    let mut dp = vec![0.0f64; w_cap + 1];
    let mut chosen = vec![vec![false; w_cap + 1]; n];

    for i in 0..n {
        for w in (ws[i]..=w_cap).rev() {
            let candidate = dp[w - ws[i]] + vs[i];
            if candidate > dp[w] {
                dp[w] = candidate;
                chosen[i][w] = true;
            }
        }
    }

    let mut selected = HashSet::new();
    let mut w = w_cap;
    for i in (0..n).rev() {
        if chosen[i][w] {
            selected.insert(candidates[i].key.clone());
            w -= ws[i];
        }
    }
    selected
}

/// Write one row per candidate to the predictions CSV for Python parity checks.
fn write_predictions_csv(
    candidates: &[MlCandidate],
    selected_keys: &HashSet<String>,
    path: &str,
) {
    let write_header = !std::path::Path::new(path).exists()
        || std::fs::metadata(path).map(|m| m.len() == 0).unwrap_or(true);
    let mut file = match OpenOptions::new().create(true).append(true).open(path) {
        Ok(f) => f,
        Err(e) => {
            log::warn!("bf_model: cannot open predictions CSV '{}': {}", path, e);
            return;
        }
    };
    if write_header {
        let _ = writeln!(
            file,
            "query_id,build_col,probe_col,pred_prob,weight_bytes,selected"
        );
    }
    let query_id = std::env::var("BF_QUERY_ID").unwrap_or_else(|_| "unknown".to_string());
    for c in candidates {
        let selected = if selected_keys.contains(&c.key) { 1 } else { 0 };
        let _ = writeln!(
            file,
            "{},{},{},{:.6},{},{}",
            query_id, c.build_col, c.probe_col, c.pred_prob, c.weight_bytes, selected,
        );
    }
}

/// Two-pass ML-guided plan rewrite:
/// 1. Collect candidates with predicted probabilities.
/// 2. Run knapsack to select within budget.
/// 3. Insert BF execs only for selected candidates.
fn insert_bloom_filters_ml(
    plan: Arc<dyn ExecutionPlan>,
    model: &GbtModel,
    budget_bytes: usize,
) -> Result<Arc<dyn ExecutionPlan>> {
    // Pass 1 — collect candidates.
    let mut candidates: Vec<MlCandidate> = Vec::new();
    collect_ml_candidates(&plan, model, &mut candidates);

    if candidates.is_empty() {
        return Ok(plan);
    }

    log::info!(
        "bf_ml: {} candidates, budget {} bytes",
        candidates.len(),
        budget_bytes
    );
    for c in &candidates {
        log::info!(
            "  candidate key={} build_col={} pred_prob={:.4} weight={}B",
            c.key, c.build_col, c.pred_prob, c.weight_bytes
        );
    }

    let selected_keys = knapsack_select(&candidates, budget_bytes);
    log::info!(
        "bf_ml: knapsack selected {}/{} filters",
        selected_keys.len(),
        candidates.len()
    );

    // Optional prediction log (written after knapsack so selected flag is accurate).
    if std::env::var("BF_LOG_PREDICTIONS").unwrap_or_default() == "1" {
        let csv_path = std::env::var("BF_PREDICTIONS_PATH")
            .unwrap_or_else(|_| "/tmp/bf_predictions.csv".to_string());
        write_predictions_csv(&candidates, &selected_keys, &csv_path);
    }

    // Build a lookup: build_col → probe_col for selected candidates.
    let selected_pairs: HashSet<(String, String)> = candidates
        .iter()
        .filter(|c| selected_keys.contains(&c.key))
        .map(|c| (c.build_col.clone(), c.probe_col.clone()))
        .collect();

    // Pass 2 — insert BF execs only for selected pairs.
    insert_bloom_filters_selected(plan, &selected_pairs)
}

fn insert_bloom_filters_selected(
    plan: Arc<dyn ExecutionPlan>,
    selected: &HashSet<(String, String)>,
) -> Result<Arc<dyn ExecutionPlan>> {
    if selected.is_empty() {
        return Ok(plan);
    }

    if let Some(hash_join) = plan.as_any().downcast_ref::<HashJoinExec>() {
        if *hash_join.join_type() == JoinType::Inner {
            let left = Arc::clone(hash_join.left());
            let right = Arc::clone(hash_join.right());

            let left = insert_bloom_filters_selected(left, selected)?;
            let right = insert_bloom_filters_selected(right, selected)?;

            let mut new_left = left;
            let mut new_right = right;

            for (left_expr, right_expr) in hash_join.on() {
                let left_name_str = format!("{}", left_expr);
                let right_name_str = format!("{}", right_expr);

                let left_schema = new_left.schema();
                let right_schema = new_right.schema();

                let build_col = left_schema
                    .fields()
                    .iter()
                    .find(|f| left_name_str.contains(f.name()))
                    .map(|f| f.name().clone());
                let probe_col = right_schema
                    .fields()
                    .iter()
                    .find(|f| right_name_str.contains(f.name()))
                    .map(|f| f.name().clone());

                if let (Some(build_col), Some(probe_col)) = (build_col, probe_col) {
                    if selected.contains(&(build_col.clone(), probe_col.clone())) {
                        let (sender, _) = broadcast::channel(1);
                        let sender = Arc::new(sender);

                        new_left = Arc::new(BloomFilterBuildExec::new(
                            new_left,
                            build_col.clone(),
                            Arc::clone(&sender),
                        ));
                        new_right = Arc::new(BloomFilterProbeExec::new(
                            new_right,
                            probe_col.clone(),
                            Arc::clone(&sender),
                        ));

                        println!(
                            "[bf_ml] Selected BF: build='{}' probe='{}'",
                            build_col, probe_col
                        );
                    } else {
                        println!(
                            "[bf_ml] Skipped BF: build='{}' probe='{}'",
                            build_col, probe_col
                        );
                    }
                }
            }

            let new_plan = plan.with_new_children(vec![new_left, new_right])?;
            return Ok(new_plan);
        }
    }

    let children = plan.children();
    if children.is_empty() {
        return Ok(plan);
    }

    let new_children = children
        .into_iter()
        .map(|c| insert_bloom_filters_selected(Arc::clone(c), selected))
        .collect::<Result<Vec<_>>>()?;

    plan.with_new_children(new_children)
}

// Optimizer Rule Implementation

impl PhysicalOptimizerRule for MultiHopBloomFilterRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let policy = std::env::var("BF_POLICY").unwrap_or_else(|_| "always_build".to_string());

        if policy == "ml_guided" {
            let model_path = std::env::var("BF_MODEL_PATH")
                .unwrap_or_else(|_| "ml/bf_model.json".to_string());
            let budget_bytes: usize = std::env::var("BF_MEMORY_BUDGET")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(25_000);
            let prob_floor: f64 = std::env::var("BF_PROB_FLOOR")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);

            match bf_model::load_model_once(&model_path) {
                Some(model) => {
                    println!(
                        "MultiHopBloomFilterRule: ml_guided policy, budget={}B prob_floor={}",
                        budget_bytes, prob_floor
                    );
                    return insert_bloom_filters_ml(plan, model, budget_bytes);
                }
                None => {
                    println!(
                        "MultiHopBloomFilterRule: ml_guided requested but model '{}' \
                         unavailable — falling back to always_build",
                        model_path
                    );
                }
            }
        }

        // always_build (default) or fallback.
        let edges = collect_join_edges(&plan);
        println!("MultiHopBloomFilterRule: Found {} Join Edge(s)", edges.len());
        for edge in &edges {
            println!("  edge: {} = {}", edge.left_col, edge.right_col);
        }

        if edges.is_empty() {
            return Ok(plan);
        }

        insert_bloom_filters(plan, &edges)
    }

    fn name(&self) -> &str {
        "multi_hop_bloom_filter"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_name() {
        let rule = MultiHopBloomFilterRule::new();
        assert_eq!(rule.name(), "multi_hop_bloom_filter");
    }

    fn make_candidate(key: &str, prob: f64, weight_bytes: usize) -> MlCandidate {
        MlCandidate {
            key: key.to_string(),
            build_col: key.to_string(),
            probe_col: key.to_string(),
            weight_bytes,
            pred_prob: prob,
        }
    }

    // All three filters fit within a generous budget.
    #[test]
    fn test_knapsack_all_fit() {
        let candidates = vec![
            make_candidate("a", 0.9, 1000),
            make_candidate("b", 0.7, 1500),
            make_candidate("c", 0.5, 2000),
        ];
        let selected = knapsack_select(&candidates, 10_000);
        assert_eq!(selected.len(), 3);
        assert!(selected.contains("a") && selected.contains("b") && selected.contains("c"));
    }

    // Tight budget: only the two highest-value filters fit.
    #[test]
    fn test_knapsack_tight_budget() {
        let candidates = vec![
            make_candidate("best1", 0.9, 3000),
            make_candidate("best2", 0.8, 2500),
            make_candidate("medium", 0.5, 4000),
            make_candidate("low1", 0.3, 5000),
        ];
        let selected = knapsack_select(&candidates, 5_600);
        assert!(selected.contains("best1"), "best1 must be selected");
        assert!(selected.contains("best2"), "best2 must be selected");
        assert!(!selected.contains("medium"), "medium must not fit");
        assert!(!selected.contains("low1"), "low1 must not fit");
    }

    // Empty candidates list returns empty set.
    #[test]
    fn test_knapsack_empty() {
        let selected = knapsack_select(&[], 100_000);
        assert!(selected.is_empty());
    }

    // Zero budget returns empty set.
    #[test]
    fn test_knapsack_zero_budget() {
        let candidates = vec![make_candidate("x", 0.9, 1000)];
        let selected = knapsack_select(&candidates, 0);
        assert!(selected.is_empty());
    }
}