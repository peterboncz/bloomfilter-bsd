# (Classic) Bloom filter

A classic Bloom filter implementation that supports batch-lookups.
The lookup code makes use of SIMD instructions, but without any
divergence handling (like in [1]).


## References
[1] Orestis Polychroniou and Kenneth A. Ross, 
    _Vectorized Bloom filters for advanced SIMD processors_, 
    DaMoN'14 
