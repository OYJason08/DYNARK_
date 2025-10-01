To build this project

```rust 
cargo build --release
```

Testing Branches are in `ark-groth16/examples/bench_v1.rs` (for preprocessing)and `ark-groth16/examples/bench_v2.rs` (for proving).

To get the specific test cases, you should find it in the `main` function, run:
```
cargo run --example bench_v1 --features parallel --release

cargo run --example bench_v2 --features parallel --release
```