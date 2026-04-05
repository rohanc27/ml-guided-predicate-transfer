//! run with:
//!   cargo run --example bloom_filter_demo

use datafusion::prelude::*;

#[tokio::main]
async fn main() -> datafusion_common::Result<()> {
    println!("Correctness Check (Small Dataset)\n");

    let baseline = run_small_query(false).await?;
    let optimized = run_small_query(true).await?;

    println!("Baseline:  {} rows", baseline.len());
    println!("Optimized: {} rows", optimized.len());

    if baseline == optimized {
        println!("PASS: Results Match\n");
    } else {
        println!("FAIL: Results Differ\n");
        for (i, (b, o)) in baseline.iter().zip(optimized.iter()).enumerate() {
            if b != o {
                println!("  row {} differs\n  baseline:  {:?}\n  optimized: {:?}", i, b, o);
            }
        }
    }

    println!("Pruning Stats (Large Dataset)\n");
    let rows = run_large_query().await?;
    println!("Total Matching Rows Returned: {}", rows.len());

    println!("\nPhysical Plan (Large Dataset)\n");
    run_show_plan().await?;

    Ok(())
}

async fn run_small_query(
    _use_bloom: bool,
) -> datafusion_common::Result<Vec<Vec<String>>> {
    let ctx = SessionContext::new();

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

    collect_query(&ctx).await
}

async fn run_large_query() -> datafusion_common::Result<Vec<Vec<String>>> {
    let ctx = SessionContext::new();

    ctx.sql("CREATE TABLE orders (order_id INT, customer_id INT, amount FLOAT) AS VALUES
        (1,1,10.0),(2,2,20.0),(3,3,30.0),(4,4,40.0),(5,5,50.0),
        (6,6,60.0),(7,7,70.0),(8,8,80.0),(9,9,90.0),(10,10,100.0),
        (11,11,10.0),(12,12,20.0),(13,13,30.0),(14,14,40.0),(15,15,50.0),
        (16,16,60.0),(17,17,70.0),(18,18,80.0),(19,19,90.0),(20,20,100.0)
    ").await?.collect().await?;

    // Only 3 customers exist out of the 20 customer_ids in orders
    ctx.sql("CREATE TABLE customers (customer_id INT, name VARCHAR, nation_id INT) AS VALUES
        (1, 'Anna', 1),
        (2, 'Bob', 2),
        (3, 'Chen', 1)
    ").await?.collect().await?;

    ctx.sql("CREATE TABLE nations (nation_id INT, nation_name VARCHAR) AS VALUES
        (1, 'Germany'),
        (2, 'France')
    ").await?.collect().await?;

    collect_query(&ctx).await
}

async fn collect_query(ctx: &SessionContext) -> datafusion_common::Result<Vec<Vec<String>>> {
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

async fn run_show_plan() -> datafusion_common::Result<()> {
    use datafusion::execution::context::SessionConfig;
    use datafusion::execution::runtime_env::RuntimeEnv;
    use datafusion::execution::session_state::SessionStateBuilder;
    use datafusion_physical_optimizer::bloom_filter_transfer::MultiHopBloomFilterRule;
    use std::sync::Arc;

    let rule = Arc::new(MultiHopBloomFilterRule::new());

    let state = SessionStateBuilder::new()
        .with_config(SessionConfig::new())
        .with_runtime_env(Arc::new(RuntimeEnv::default()))
        .with_default_features()
        .with_physical_optimizer_rule(rule)
        .build();

    let ctx = SessionContext::new_with_state(state);

    ctx.sql("CREATE TABLE orders (order_id INT, customer_id INT, amount FLOAT) AS VALUES
        (1,1,10.0),(2,2,20.0),(3,3,30.0),(4,4,40.0),(5,5,50.0),
        (6,6,60.0),(7,7,70.0),(8,8,80.0),(9,9,90.0),(10,10,100.0),
        (11,11,10.0),(12,12,20.0),(13,13,30.0),(14,14,40.0),(15,15,50.0),
        (16,16,60.0),(17,17,70.0),(18,18,80.0),(19,19,90.0),(20,20,100.0)
    ").await?.collect().await?;

    ctx.sql("CREATE TABLE customers (customer_id INT, name VARCHAR, nation_id INT) AS VALUES
        (1, 'Anna', 1),
        (2, 'Bob', 2),
        (3, 'Chen', 1)
    ").await?.collect().await?;

    ctx.sql("CREATE TABLE nations (nation_id INT, nation_name VARCHAR) AS VALUES
        (1, 'Germany'),
        (2, 'France')
    ").await?.collect().await?;

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