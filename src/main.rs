//! cmac-study — Forgotten Architectures #2.
//!
//! Stage entrypoints:
//!   cargo run --release --bin stage1_gate     # faithful tiling + local-gen gate
//!   cargo run --release --bin stage2_native   # 1-seed sanity
//!   cargo run --release --bin stage2_full     # 3-seed native
//!   cargo run --release --bin stage3_mnist    # MNIST hashing-CMAC comparability

fn main() {
    println!("cmac-study bins:");
    println!("  --bin stage1_gate    faithful tiling local-gen gate");
    println!("  --bin stage2_native  native 1-seed");
    println!("  --bin stage2_full    native 3-seed (freeze)");
    println!("  --bin stage3_mnist   MNIST hash-CMAC vs wisard/mlp");
}
