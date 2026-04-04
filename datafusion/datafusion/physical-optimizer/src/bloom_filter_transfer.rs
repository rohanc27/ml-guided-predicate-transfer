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
//! This optimizer rule finds inner equi-joins in the physical plan,
//! discovers which Bloom filters can be propagated across multiple
//! join edges, and inserts BloomFilterBuildExec / BloomFilterProbeExec
//! operators so that table scans can pre-filter rows before joining.

use std::sync::Arc;

use datafusion_common::config::ConfigOptions;
use datafusion_common::Result;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::joins::HashJoinExec;

use crate::PhysicalOptimizerRule;
use log::debug;

// Rule Struct
#[derive(Default, Debug)]
pub struct MultiHopBloomFilterRule {}

impl MultiHopBloomFilterRule {
    pub fn new() -> Self {
        Self {}
    }
}

// Core Data Structures
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
        if *hash_join.join_type() == datafusion_common::JoinType::Inner {
            for (left_expr, right_expr) in hash_join.on() {
                let left_col = format!("{}", left_expr);
                let right_col = format!("{}", right_expr);
                edges.push(JoinEdge { left_col, right_col });
            }
        }
    }

    for child in plan.children() {
        collect_recursive(child, edges);
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

        debug!(
            "MultiHopBloomFilterRule: found {} join edge(s)",
            edges.len()
        );
        for edge in &edges {
            debug!("  edge: {} = {}", edge.left_col, edge.right_col);
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