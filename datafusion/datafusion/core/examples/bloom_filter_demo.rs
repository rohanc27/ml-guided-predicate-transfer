//! run with:
//!   cargo run --example bloom_filter_demo

use datafusion::execution::context::SessionConfig;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::*;
use datafusion_physical_optimizer::bloom_filter_transfer::MultiHopBloomFilterRule;
use std::sync::Arc;
use std::time::Instant;

#[tokio::main]
async fn main() -> datafusion_common::Result<()> {
    println!("Correctness Check (Small Dataset)\n");
    let baseline = run_query(false, 20).await?;
    let optimized = run_query(true, 20).await?;
    println!("Baseline:  {} rows", baseline.len());
    println!("Optimized: {} rows", optimized.len());
    if baseline == optimized {
        println!("PASS: Results Match\n");
    } else {
        println!("FAIL: Results Differ\n");
    }

    println!("Timing Benchmark (5 Runs Each)\n");

    for scale in [1000, 10_000, 100_000] {
        println!("{} Orders, 3 Matching Customers", scale);

        let mut baseline_times = vec![];
        let mut optimized_times = vec![];

        for _ in 0..5 {
            let start = Instant::now();
            run_query(false, scale).await?;
            baseline_times.push(start.elapsed().as_millis());
        }

        for _ in 0..5 {
            let start = Instant::now();
            run_query(true, scale).await?;
            optimized_times.push(start.elapsed().as_millis());
        }

        let b_avg = baseline_times.iter().sum::<u128>() / 5;
        let o_avg = optimized_times.iter().sum::<u128>() / 5;

        println!("Baseline  Average: {}ms", b_avg);
        println!("Optimized Average: {}ms", o_avg);

        if o_avg < b_avg {
            let speedup = b_avg as f64 / o_avg as f64;
            println!("Speedup: {:.2}x\n", speedup);
        } else {
            println!(" No Speedup At This Scale (Overhead Dominates)\n");
        }
    }

    Ok(())
}

async fn run_query(
    use_bloom: bool,
    order_count: usize,
) -> datafusion_common::Result<Vec<Vec<String>>> {
    let ctx = make_context(use_bloom);

    let orders_values: String = (1..=order_count)
        .map(|i| format!("({},{},{})", i, i, i as f64 * 10.0))
        .collect::<Vec<_>>()
        .join(",");

    ctx.sql(&format!(
        "CREATE TABLE orders (order_id BIGINT, customer_id BIGINT, amount FLOAT) AS VALUES {}",
        orders_values
    ))
    .await?.collect().await?;

    ctx.sql("CREATE TABLE customers (customer_id BIGINT, name VARCHAR, nation_id BIGINT) AS VALUES
        (1, 'Anna', 1), (2, 'Bob', 2), (3, 'Chen', 1)")
        .await?.collect().await?;

    ctx.sql("CREATE TABLE nations (nation_id BIGINT, nation_name VARCHAR) AS VALUES
        (1, 'Germany'), (2, 'France')")
        .await?.collect().await?;

    let query = "
        SELECT o.order_id, c.name, n.nation_name, o.amount
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN nations n ON c.nation_id = n.nation_id
        ORDER BY o.order_id
    ";

    let df = ctx.sql(query).await?;
    let batches = df.collect().await?;

    let mut rows: Vec<Vec<String>> = Vec::new();
    for batch in &batches {
        for row_idx in 0..batch.num_rows() {
            let mut row = Vec::new();
            for col_idx in 0..batch.num_columns() {
                let col = batch.column(col_idx);
                let val = arrow::util::display::array_value_to_string(col, row_idx)?;
                row.push(val);
            }
            rows.push(row);
        }
    }

    Ok(rows)
}

fn make_context(use_bloom: bool) -> SessionContext {
    if use_bloom {
        let rule = Arc::new(MultiHopBloomFilterRule::new());
        let state = SessionStateBuilder::new()
            .with_config(SessionConfig::new())
            .with_runtime_env(Arc::new(RuntimeEnv::default()))
            .with_default_features()
            .with_physical_optimizer_rule(rule)
            .build();
        SessionContext::new_with_state(state)
    } else {
        SessionContext::new()
    }
}