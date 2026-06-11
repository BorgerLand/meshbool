# Meshbool

Meshbool is a pure-Rust implementation/port of Manifold's state of the art **mesh boolean algorithm**, known for its guarantee that, given manifold input, will always produce manifold output: solid, watertight, correct. It enables robust [CSG (Constructive Solid Geometry) operations](https://en.wikipedia.org/wiki/Constructive_solid_geometry) on 3D models.

Note that many features are currently unimplemented, and completeness is not the primary goal. I will accept PR's to port more features, especially those that increase the number of passing tests, but the main focus of this crate is the boolean algorithm.

This repo is up to date with [this Manifold commit](https://github.com/elalish/manifold/tree/db042ab153bf8e5dbef2eb00990e8024c2f272ec) (v3.5.1) and passes 245 tests when [linked to the original C++ test suite](/test/README.md). I consider what is here to be reliably complete.

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
//note you currently need the nalgebra crate to construct these linear algebra objects
let cube1 = MeshBool::cube(Vector3::new(1.0, 1.0, 1.0), true);
let cube2 = MeshBool::cube(Vector3::new(1.0, 1.0, 1.0), false);

let union = &cube1 + &cube2;
let difference = &cube1 - &cube2;
let intersection = &cube1 ^ &cube2;

//now convert the output into a format suitable for rendering
let mesh = union.get_mesh_gl(0);
```

### Performance:

- Parallelized algorithms haven't been ported yet, so this table compares single-threaded mode for each implementation.
- There is some unknown amount of FFI overhead incurred from copying data between Rust/C++ vecs
- Halfedge tangent calculations are unimplemented in Rust. C++ always computes them, whether you need them or not
- The CSG tree, essentially an optimization for operating on batches of meshes, is also unimplemented.
- Benching was done on an Intel i5-4210M, a force to be reckoned with.

| Test                                  | C++ (ms) | Rust (ms) |
| ------------------------------------- | -------- | --------- |
| Properties.MingapAfterTransformations | 2916     | 2747      |
| Properties.ToleranceSphere            | 18649    | 18009     |
| Boolean.CreatePropertiesSlow          | 2038     | 2049      |
| Samples.CondensedMatter16             | 4384     | 8718      |
| Samples.CondensedMatter64             | 71583    | 141745    |
| BooleanComplex.Close                  | 2708     | 3763      |
| Polygon.Zebra                         | 1617     | 1820      |
| Polygon.Zebra3                        | 1687     | 1908      |
