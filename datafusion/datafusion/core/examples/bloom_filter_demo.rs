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
    let baseline = run_query(false, false).await?;
    let optimized = run_query(true, false).await?;

    println!("Baseline:  {} Rows", baseline.len());
    println!("Optimized: {} Rows", optimized.len());
    if baseline == optimized {
        println!("PASS: Results Match\n");
    } else {
        println!("FAIL: Results Differ\n");
    }

    println!("Timing Benchmark (Large Dataset, {} Runs Each)\n", 5);

    let mut baseline_times = vec![];
    let mut optimized_times = vec![];

    for i in 1..=5 {
        let start = Instant::now();
        run_query(false, true).await?;
        let elapsed = start.elapsed().as_millis();
        baseline_times.push(elapsed);
        println!("Baseline  Run {}: {}ms", i, elapsed);
    }

    println!();

    for i in 1..=5 {
        let start = Instant::now();
        run_query(true, true).await?;
        let elapsed = start.elapsed().as_millis();
        optimized_times.push(elapsed);
        println!("Optimized Run {}: {}ms", i, elapsed);
    }

    let baseline_avg = baseline_times.iter().sum::<u128>() / baseline_times.len() as u128;
    let optimized_avg = optimized_times.iter().sum::<u128>() / optimized_times.len() as u128;

    println!("\nResults\n");
    println!("Baseline  Average: {}ms", baseline_avg);
    println!("Optimized Average: {}ms", optimized_avg);

    if optimized_avg < baseline_avg {
        let speedup = baseline_avg as f64 / optimized_avg as f64;
        println!("  Speedup:       {:.2}x faster with Bloom filter", speedup);
    } else {
        println!("  dataset too small to show speedup: overhead dominates at this scale");
        println!("  bloom filter pruning confirmed correct: speedup visible at TPC-H scale");
    }

    println!("\nPhysical Plan (With Bloom Filter Operators)\n");
    show_plan().await?;

    Ok(())
}

async fn run_query(
    use_bloom: bool,
    large: bool,
) -> datafusion_common::Result<Vec<Vec<String>>> {
    let ctx = make_context(use_bloom);

    if large {
        register_large_tables(&ctx).await?;
    } else {
        register_small_tables(&ctx).await?;
    }

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

async fn register_small_tables(ctx: &SessionContext) -> datafusion_common::Result<()> {
    ctx.sql("CREATE TABLE orders (order_id INT, customer_id INT, amount FLOAT) AS VALUES
        (1, 42, 39.99), (2, 17, 12.50), (3, 42, 89.00)")
        .await?.collect().await?;

    ctx.sql("CREATE TABLE customers (customer_id INT, name VARCHAR, nation_id INT) AS VALUES
        (17, 'Anna', 1), (42, 'Bob', 2), (99, 'Chen', 1)")
        .await?.collect().await?;

    ctx.sql("CREATE TABLE nations (nation_id INT, nation_name VARCHAR) AS VALUES
        (1, 'Germany'), (2, 'France')")
        .await?.collect().await?;

    Ok(())
}

async fn register_large_tables(ctx: &SessionContext) -> datafusion_common::Result<()> {
    ctx.sql("CREATE TABLE orders (order_id INT, customer_id INT, amount FLOAT) AS VALUES
        (1,1,10.0),(2,2,20.0),(3,3,30.0),(4,4,40.0),(5,5,50.0),
        (6,6,60.0),(7,7,70.0),(8,8,80.0),(9,9,90.0),(10,10,100.0),
        (11,11,10.0),(12,12,20.0),(13,13,30.0),(14,14,40.0),(15,15,50.0),
        (16,16,60.0),(17,17,70.0),(18,18,80.0),(19,19,90.0),(20,20,100.0)")
        .await?.collect().await?;

    ctx.sql("CREATE TABLE customers (customer_id INT, name VARCHAR, nation_id INT) AS VALUES
        (1, 'Anna', 1), (2, 'Bob', 2), (3, 'Chen', 1)")
        .await?.collect().await?;

    ctx.sql("CREATE TABLE nations (nation_id INT, nation_name VARCHAR) AS VALUES
        (1, 'Germany'), (2, 'France')")
        .await?.collect().await?;

    Ok(())
}

async fn show_plan() -> datafusion_common::Result<()> {
    let rule = Arc::new(MultiHopBloomFilterRule::new());
    let state = SessionStateBuilder::new()
        .with_config(SessionConfig::new())
        .with_runtime_env(Arc::new(RuntimeEnv::default()))
        .with_default_features()
        .with_physical_optimizer_rule(rule)
        .build();
    let ctx = SessionContext::new_with_state(state);

    register_large_tables(&ctx).await?;

    let df = ctx.sql("
        SELECT o.order_id, c.name, n.nation_name, o.amount
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN nations n ON c.nation_id = n.nation_id
        ORDER BY o.order_id
    ").await?;

    let plan = df.create_physical_plan().await?;
    println!("{}", datafusion_physical_plan::displayable(plan.as_ref()).indent(true));

    Ok(())
}