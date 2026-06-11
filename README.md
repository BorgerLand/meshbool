# Meshbool

### ⚠️ Currently experimental; not production-ready ⚠️

Meshbool is a pure-Rust implementation/port of Manifold's state of the art **mesh boolean algorithm**, known for its guarantee that, given manifold input, will always produce manifold output: solid, watertight, correct. It enables robust [CSG (Constructive Solid Geometry) operations](https://en.wikipedia.org/wiki/Constructive_solid_geometry) on 3D models.

This repo is up to date with [this Manifold commit](https://github.com/elalish/manifold/tree/db042ab153bf8e5dbef2eb00990e8024c2f272ec) (v3.5.1).

### Example:

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

```TOML
#currently not published to crates.io until testing suite has been ported
[dependencies]
meshbool = { git = "https://github.com/luisfonsivevo/meshbool.git" }
```

In its **current state**, meshbool is utter chaos:

- It was done line by line, and so is offensively unidiomatic, reeks of C++/OOP/implicit number conversion, etc.
- It was done as quickly as possible without attempting to understand it, and likely contains glaring typos and translation errors
- No parallelization, so single core performance only for now
- Because the line by line strategy was prioritized over anything else, it likely performs worse than single-threaded Manifold due to borrow checker fighting and bounds checking. No benchmarking has been performed yet.
- The test suite is in progress. **No guarantees it's even working properly!**

However, it **unlocks new doors:**

- Modern tooling
- Memory safety - potentially a big one, considering the sheer volume of array indexing this library does
- wasm-bindgen ecosystem - this is ultimately why I decided to commit to this project

I'm aware of the **[manifold-rs crate](https://github.com/WilstonOreo/manifold-rs)**. If you need a reliable boolean algorithm in Rust right now, it's probably your best bet (though it looks slightly outdated). It brings the battle-tested, original C++ algorithm straight to you via cxx bindings. But:

- Unlike the Manifold bindings for other languages where you can just drop in a precompiled binary, in Rust you're constantly recompiling from C++ source and relinking the 2 languages together. This introduces a great amount of build complexity.
- It's seemingly wholly incompatible with WASM; at least not without blowing up your binary with 2 runtimes, 2 standard libraries, emscripten, wasm bindgen, mixed C++ and Rust ABI, yay! Assuming you can even get it to build in the first place (I couldn't)

**Roadmap priorities:**

- Make the tests pass
- Standard stuff: CI, code formatting, publish on crates.io
- Port parallelization (Rayon?)
- General cleanup/idiomatic refactor: I'm most looking forward to removing all classes that make up the algorithm's pipeline. Pure functional, chronological order
- Continued maintenance: keep up to date with the original library
