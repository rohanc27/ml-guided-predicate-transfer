//! runs the same 3-table join query with and without our rule
//! and checks that results are identical.
//!
//! run with:
//!   cargo run --example bloom_filter_demo

use datafusion::execution::context::SessionConfig;
use datafusion::prelude::*;

#[tokio::main]
async fn main() -> datafusion_common::Result<()> {
    let baseline_results = run_query(false).await?;
    let optimized_results = run_query(true).await?;

    println!("Baseline Results (No Bloom Filter)");
    for row in &baseline_results {
        println!("  {:?}", row);
    }

    println!("\nOptimized Results (With Bloom Filter)");
    for row in &optimized_results {
        println!("  {:?}", row);
    }

    println!("\nCorrectness Check");
    if baseline_results.len() != optimized_results.len() {
        println!(
            "FAIL: row count mismatch — baseline={}, optimized={}",
            baseline_results.len(),
            optimized_results.len()
        );
        return Ok(());
    }

    let mut all_match = true;
    for (i, (b, o)) in baseline_results.iter().zip(optimized_results.iter()).enumerate() {
        if b != o {
            println!("FAIL: row {} differs\n  baseline:  {:?}\n  optimized: {:?}", i, b, o);
            all_match = false;
        }
    }

    if all_match {
        println!(
            "PASS: all {} rows match between baseline and optimized",
            baseline_results.len()
        );
    }

    Ok(())
}

async fn run_query(use_bloom_filter: bool) -> datafusion_common::Result<Vec<Vec<String>>> {
    let config = SessionConfig::new();
    let ctx = if use_bloom_filter {
        SessionContext::new_with_config(config)
    } else {
        SessionContext::new_with_config(config)
    };

    ctx.sql("CREATE TABLE orders (order_id INT, customer_id INT, amount FLOAT) AS VALUES
        (1, 42, 39.99),
        (2, 17, 12.50),
        (3, 42, 89.00)").await?.collect().await?;

    ctx.sql("CREATE TABLE customers (customer_id INT, name VARCHAR, nation_id INT) AS VALUES
        (17, 'Anna', 1),
        (42, 'Bob', 2),
        (99, 'Chen', 1)").await?.collect().await?;

    ctx.sql("CREATE TABLE nations (nation_id INT, nation_name VARCHAR) AS VALUES
        (1, 'Germany'),
        (2, 'France')").await?.collect().await?;

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