# Issue #898 research-only Rust comparator

Profiling research for [#898](https://github.com/fol2/newsroom/issues/898). This
crate is not a product dependency, Cargo workspace member, runtime route,
daemon, or CI contract. It reads a copied proving snapshot. Mode `r1` emits an
observation-scan manifest. Mode `r2` selects bounded proving rows from
`(run_id, source_id, observation_digest)` coordinates and computes useful
output from bodies; it must not receive retained `unit_refs` identity fields.
It has no authority-store write, credential, provider, Neo4j, or publication
capability.
