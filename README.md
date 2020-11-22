# Dense


[![Crates.io](https://img.shields.io/crates/v/dense)](https://crates.io/crates/dense)
[![lib.rs.io](https://img.shields.io/crates/v/dense?color=blue&label=lib.rs)](https://lib.rs/crates/dense)
[![docs](https://img.shields.io/crates/v/dense?color=yellow&label=docs)](https://docs.rs/dense)

An encoder/decoder to/from dense files.

A file format simpler and denser than MNIST, a dense file is binary file of seqeuantial training examples and nothing else (example->label->example->label->etc.).

Format | Size | Size on disk
--- | --- | ---
MNIST | 7,850,024 | 7,856,128
Dense | 7,850,000 | 7,852,032
