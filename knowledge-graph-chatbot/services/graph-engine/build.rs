// build.rs — Compile protobuf definitions into Rust code at build time.
// tonic-build generates server traits and message structs from .proto files.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile all proto files with the proto/ directory as the include root.
    // This generates Rust code in OUT_DIR that we include via tonic::include_proto!
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile(
            &[
                "../../proto/common.proto",
                "../../proto/graph.proto",
                "../../proto/ai.proto",
            ],
            &["../../proto"],
        )?;
    Ok(())
}
