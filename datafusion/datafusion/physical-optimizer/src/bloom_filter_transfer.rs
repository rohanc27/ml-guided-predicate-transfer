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
use std::sync::Arc;

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

use crate::PhysicalOptimizerRule;

// Shared Bloom Filter Handle

type SharedBloom = Arc<broadcast::Sender<Arc<Bloom<String>>>>;

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
                format!("BloomFilterBuildExec: Column '{}' Not Found", self.key_column_name)
            ))?;

        let mut input_stream = self.input.execute(partition, context)?;
        let sender = Arc::clone(&self.sender);
        let schema_ref = self.input.schema();

        let output_stream = async_stream::stream! {
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

            let mut bf = Bloom::new_for_fp_rate(all_values.len().max(1), 0.01).unwrap();
            for v in &all_values {
                bf.set(v);
            }
            let _ = sender.send(Arc::new(bf));

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
                format!("BloomFilterProbeExec: Column '{}' Not Found", self.key_column_name)
            ))?;

        let mut input_stream = self.input.execute(partition, context)?;
        let mut receiver = self.sender.subscribe();
        let schema_ref = self.input.schema();
        let key_column_name = self.key_column_name.clone();

        let output_stream = async_stream::stream! {
            let bf = match receiver.recv().await {
                Ok(b) => b,
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
                        bf.check(&val)
                    }
                }).collect();

                let rows_before = batch.num_rows();
                let mask = BooleanArray::from(keep);
                match filter_record_batch(&batch, &mask) {
                    Ok(filtered) => {
                        let rows_after = filtered.num_rows();
                        let eliminated = rows_before - rows_after;
                        if eliminated > 0 {
                            println!(
                                "[BloomFilterProbeExec] col='{}' Eliminated {}/{} Rows ({:.1}%)",
                                key_column_name, eliminated, rows_before,
                                (eliminated as f64 / rows_before as f64) * 100.0
                            );
                        }
                        yield Ok(filtered)
                    },
                    Err(e) => yield Err(DataFusionError::ArrowError(Box::new(e), None)),
                }
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

// Optimizer Rule Implementation

impl PhysicalOptimizerRule for MultiHopBloomFilterRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
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
}