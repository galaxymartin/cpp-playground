simd — SIMD helpers and runtime dispatch

Purpose
- Contains AVX-512 and SSE4.1 helpers for common vectorized operations on `float` and `double`.
- Exposes a small runtime dispatch layer so consumers call `simd_*` APIs and the implementation chooses the best available ISA.

Public API (headers)
- simd/dispatch.hpp
  - `enum class SimdLevel { AUTO, SCALAR, SSE42, AVX512 };`
  - `void init_simd();` — optional early initialization
  - `SimdLevel get_simd_level();`
  - `simd_dot_float`, `simd_dot_double` — matrix-vector dot helpers used by `Matrix::multiply`
  - `simd_add_float`, `simd_add_double` — elementwise add
  - `simd_sub_float`, `simd_sub_double` — elementwise subtract

Usage
- Include `simd/dispatch.hpp` and call the `simd_*` functions directly or rely on `Matrix` which uses the dispatch API.
- You can force paths for testing via `Matrix<T>::setForceScalarAdd/Multiply/Sub` and `setForceSSE42Add/Multiply/Sub`.

Notes
- Implementations follow IEEE-754 semantics for `float`/`double` (NaN/Inf handled per hardware).
- The library is compiled into a static `simd` target and linked by top-level targets.

Contact
- See repository `matrix.cpp` and `matrix.hpp` for examples of how `Matrix` uses the `simd` API.
