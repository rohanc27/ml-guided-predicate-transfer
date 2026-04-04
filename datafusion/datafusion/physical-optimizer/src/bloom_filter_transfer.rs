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

//! multi-hop predicate transfer via runtime bloom filters.

use std::any::Any;
use std::sync::Arc;

use arrow::array::{Array, BooleanArray, Int32Array, Int64Array, StringArray};
use arrow::compute::filter_record_batch;
use arrow::record_batch::RecordBatch;
use bloomfilter::Bloom;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::{DataFusionError, JoinType, Result};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::joins::HashJoinExec;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
};
use futures::StreamExt;
use tokio::sync::Mutex;

use crate::PhysicalOptimizerRule;

// Shared Bloom Filter Handle

type SharedBloom = Arc<Mutex<Option<Bloom<String>>>>;

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
    key_column: usize,
    bloom: SharedBloom,
}

impl BloomFilterBuildExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        key_column: usize,
        bloom: SharedBloom,
    ) -> Self {
        Self { input, key_column, bloom }
    }

    pub fn bloom(&self) -> SharedBloom {
        Arc::clone(&self.bloom)
    }
}

impl DisplayAs for BloomFilterBuildExec {
    fn fmt_as(
        &self,
        _t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "BloomFilterBuildExec: key_col={}", self.key_column)
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
            self.key_column,
            Arc::clone(&self.bloom),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let mut input_stream = self.input.execute(partition, context)?;
        let bloom = Arc::clone(&self.bloom);
        let key_column = self.key_column;
        let schema = self.input.schema();

        let output_stream = async_stream::stream! {
            while let Some(batch_result) = input_stream.next().await {
                let batch: RecordBatch = match batch_result {
                    Ok(b) => b,
                    Err(e) => { yield Err(e); continue; }
                };

                let col = batch.column(key_column);
                let values: Vec<String> = if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
                    (0..arr.len()).filter(|&i| !arr.is_null(i)).map(|i| arr.value(i).to_string()).collect()
                } else if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                    (0..arr.len()).filter(|&i| !arr.is_null(i)).map(|i| arr.value(i).to_string()).collect()
                } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                    (0..arr.len()).filter(|&i| !arr.is_null(i)).map(|i| arr.value(i).to_string()).collect()
                } else {
                    vec![]
                };

                {
                    let mut guard = bloom.lock().await;
                    let bf = guard.get_or_insert_with(|| {
                        Bloom::new_for_fp_rate(100_000, 0.01).unwrap()
                    });
                    for v in &values {
                        bf.set(v);
                    }
                }

                yield Ok(batch);
            }
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, output_stream)))
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
    key_column: usize,
    bloom: SharedBloom,
}

impl BloomFilterProbeExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        key_column: usize,
        bloom: SharedBloom,
    ) -> Self {
        Self { input, key_column, bloom }
    }
}

impl DisplayAs for BloomFilterProbeExec {
    fn fmt_as(
        &self,
        _t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "BloomFilterProbeExec: key_col={}", self.key_column)
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
            self.key_column,
            Arc::clone(&self.bloom),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let mut input_stream = self.input.execute(partition, context)?;
        let bloom = Arc::clone(&self.bloom);
        let key_column = self.key_column;
        let schema = self.input.schema();

        let output_stream = async_stream::stream! {
            while let Some(batch_result) = input_stream.next().await {
                let batch: RecordBatch = match batch_result {
                    Ok(b) => b,
                    Err(e) => { yield Err(e); continue; }
                };

                let col = batch.column(key_column);

                let keep: Vec<bool> = {
                    let guard = bloom.lock().await;
                    if let Some(bf) = guard.as_ref() {
                        (0..col.len()).map(|i| {
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
                        }).collect()
                    } else {
                        vec![true; col.len()]
                    }
                };

                let mask = BooleanArray::from(keep);
                match filter_record_batch(&batch, &mask) {
                    Ok(filtered) => yield Ok(filtered),
                    Err(e) => yield Err(DataFusionError::ArrowError(Box::new(e), None)),

                }
            }
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, output_stream)))
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&dyn PhysicalExpr) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }
}

// Optimizer Rule Implementation

impl PhysicalOptimizerRule for MultiHopBloomFilterRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let edges = collect_join_edges(&plan);

        println!("MultiHopBloomFilterRule: found {} join edge(s)", edges.len());
        for edge in &edges {
            println!("  edge: {} = {}", edge.left_col, edge.right_col);
        }

        plan.transform_up(|node| Ok(Transformed::no(node))).map(|t| t.data)
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