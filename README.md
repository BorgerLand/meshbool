# Meshbool

Meshbool is a pure-Rust implementation/port of Manifold's state of the art **mesh boolean algorithm**, known for its guarantee that, given manifold input, will always produce manifold output: solid, watertight, correct. It enables robust [CSG (Constructive Solid Geometry) operations](https://en.wikipedia.org/wiki/Constructive_solid_geometry) on 3D models.

Note that many features are currently unimplemented, and completeness is not the primary goal. I will accept PR's to port more features, especially those that increase the number of passing tests, but the main focus of this crate is the boolean algorithm.

This repo is up to date with [this Manifold commit](https://github.com/elalish/manifold/tree/81e94c86c513d045ea86c0856e34618c5c4ec13e) (v3.5.2) and passes 409 tests when [linked to the original C++ test suite](/test/README.md). I consider what is here to be reliably complete.

### Why does this exist?

If you're just looking to use Manifold in Rust, you have probably better options:

- [Another Rust port](https://github.com/larsbrubaker/manifold-rust), looks more feature-complete and attempts to be as faithful as possible to the original
- [Rust bindings to the original library](https://github.com/zmerlynn/manifold-csg), if you're able to stomach dual C++/Rust runtime

So why bother?

- I would be pleased to never see a line of C++ again. This cannot be overstated.
- `wasm-bindgen` ecosystem - Rust and C++ in the same WASM binary requires 2 runtimes, 2 standard libraries, emscripten, wasm bindgen, mixed ABI, yay!
- I plan to continue researching [symbolic perturbation](https://github.com/elalish/manifold/issues/1430) improvements, refactoring into more idiomatic Rust, and optimizing for my specific use case.

### Installation

```TOML
#crates.io coming soon? maybe?
[dependencies]
meshbool = { git = "https://github.com/BorgerLand/meshbool.git" }
```

### Example

```Rust
let cube1 = MeshBool::cube(Vector3::new(1.0, 1.0, 1.0), true)?;
let cube2 = MeshBool::cube(Vector3::new(1.0, 1.0, 1.0), false)?;

let union = cube1.union(&cube2)?;
let difference = cube1.difference(&cube2)?;
let intersection = cube1.intersection(&cube2)?;

//now convert the output into a format suitable for rendering
let mesh = union.to_meshgl(0);
```

### Performance:

- Parallelized algorithms haven't been ported yet, so this table compares single-threaded mode for each implementation.
- There is some unknown amount of FFI overhead incurred from copying data between Rust/C++ vecs
- Halfedge tangent calculations are unimplemented in Rust. C++ always computes them, whether you need them or not
- Benching was done on an Intel i5-4210M, a force to be reckoned with.

| Test                                      | C++ duration (ms) | Rust duration (ms) | C++ peak memory (kb) | Rust peak memory (kb) |
| ----------------------------------------- | ----------------- | ------------------ | -------------------- | --------------------- |
| Properties.MingapAfterTransformations     | 2925              | 2596               | 117768               | 71680                 |
| Properties.MingapStretchyBracelet         | 4930              | 4541               | 44920                | 33124                 |
| Properties.ToleranceSphere                | 18955             | 11804              | 3663036              | 2505952               |
| Boolean.CreatePropertiesSlow              | 2087              | 1658               | 543664               | 437800                |
| Samples.Bracelet                          | 936               | 914                | 36620                | 25844                 |
| Samples.Sponge4                           | 39666             | 44067              | 1110176              | 1145340               |
| Samples.CondensedMatter16                 | 4381              | 3643               | 128952               | 97984                 |
| Samples.CondensedMatter64                 | 71959             | 64679              | 1253484              | 848140                |
| BooleanComplex.Close                      | 2715              | 3044               | 59124                | 72488                 |
| BooleanComplex.GenericTwinBooleanTest7081 | 25007             | 31687              | 26440                | 26424                 |
| Polygon.Zebra                             | 1592              | 1740               | 57776                | 68464                 |
| Polygon.Zebra3                            | 1686              | 1824               | 62220                | 69856                 |
| **Total**                                 | **176846**        | **172205**         | **3663036**          | **2505952**           |
