# Reference implementation of Dynark: Making Groth16 Dynamic

This is an implementation of [Dynark: Making Groth16 Dynamic](https://eprint.iacr.org/2025/1897) under `Bls12_381` curve.

## Benchmarks

```rust 
cargo build --release
```

Testing Branches are in `ark-groth16/examples/bench_v1.rs` (for preprocessing)and `ark-groth16/examples/bench_v2.rs` (for proving).

To get test semi dynamic and fully dynamic cases respectively, run:
```
cargo run --example semi_dynamic --features parallel --release

cargo run --example fully_dynamic --features parallel --release
```

For some other benchmarks that perform figures, run:
```
cargo run --example bench_v1 --features parallel --release

cargo run --example bench_v2 --features parallel --release
```
## Customizing Setting

For different parameters on benchmark, you can simply modify on the corresponding codes in `ark-groth16/examples`. You can also build specific R1CS scenario, and also build consistent proof-generating and updating functionalities.


## Acknowledgement

We implemented our approach based on [arkworks' implementation of Groth16](https://github.com/arkworks-rs/groth16.git).