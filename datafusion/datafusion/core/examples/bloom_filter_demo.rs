//! run MultiHopBloomFilterRule on a 3-table join query
//! and print the join edges it discovers.
//!
//! run with:
//!   cargo run --example bloom_filter_demo

use datafusion::prelude::*;
use datafusion_physical_optimizer::bloom_filter_transfer::MultiHopBloomFilterRule;
use datafusion_physical_optimizer::PhysicalOptimizerRule;
use datafusion_common::config::ConfigOptions;

#[tokio::main]
async fn main() -> datafusion_common::Result<()> {
    let ctx = SessionContext::new();

    // a small star-schema (like TPC-H region -> nation -> customer)
    let orders_sql = "CREATE TABLE orders (order_id INT, customer_id INT, amount FLOAT) AS VALUES
        (1, 42, 39.99),
        (2, 17, 12.50),
        (3, 42, 89.00)";

    let customers_sql = "CREATE TABLE customers (customer_id INT, name VARCHAR, nation_id INT) AS VALUES
        (17, 'Anna', 1),
        (42, 'Bob', 2),
        (99, 'Chen', 1)";

    let nations_sql = "CREATE TABLE nations (nation_id INT, nation_name VARCHAR) AS VALUES
        (1, 'Germany'),
        (2, 'France')";

    ctx.sql(orders_sql).await?.collect().await?;
    ctx.sql(customers_sql).await?.collect().await?;
    ctx.sql(nations_sql).await?.collect().await?;

    // A 3-table inner join query: this is what predicate transfer targets
    let query = "
        SELECT o.order_id, c.name, n.nation_name, o.amount
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN nations n ON c.nation_id = n.nation_id
    ";

    let df = ctx.sql(query).await?;
    let physical_plan = df.create_physical_plan().await?;

    println!("Physical Plan");
    println!("{}", datafusion_physical_plan::displayable(physical_plan.as_ref()).indent(true));

    println!("\nRunning MultiHopBloomFilterRule");
    let rule = MultiHopBloomFilterRule::new();
    let config = ConfigOptions::new();
    let _new_plan = rule.optimize(physical_plan, &config)?;

    println!("\nDone!!!");
    Ok(())
}