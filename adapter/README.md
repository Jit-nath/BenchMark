# Adapters

Database-specific implementations share the `VectorStore` contract in `src/main.rs`.
Move an implementation into this directory as the adapter set grows; keep the
operation names and input semantics unchanged so results remain comparable.
