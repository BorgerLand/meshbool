# Manifold Test Suite

This directory bridges the crate to the upstream Manifold C++ test suite. The C++ headers translate the Manifold API surface to calls into MeshBool via [Zngur](https://hkalbasi.github.io/zngur/)-generated FFI bindings.

> **Note:** These bindings are for testing only and are not optimized for general use. They are not the main focus of the project.

---

## Prerequisites

| Tool         | Version                          |
| ------------ | -------------------------------- |
| Rust         | 1.91+                            |
| CMake        | 3.23+                            |
| C++ compiler | C++17, e.g. GCC 11+ or Clang 14+ |

---

## Building

### 1. Clone submodules

```bash
#if you haven't cloned the main repo yet:
git clone --recurse-submodules https://github.com/BorgerLand/meshbool.git
cd meshbool

#if you've already cloned the main repo:
cd meshbool
git submodule update --init --recursive
```

### 2. Build

```bash
mkdir -p test/manifold/build && cd test/manifold/build
#THOROUGH MODE:
cmake .. -DCMAKE_BUILD_TYPE=Release -DMANIFOLD_DEBUG=ON -DMANIFOLD_ASSERT=ON && make manifold_test -j$(nproc)
#BENCH MODE (you can use this same command on the upstream manifold repo to bench against the C++ implementation):
cmake .. -DCMAKE_BUILD_TYPE=Release -DMANIFOLD_DEBUG=OFF -DMANIFOLD_ASSERT=OFF && make manifold_test -j$(nproc)
```

### 3. Run the tests

```bash
cd test #from inside test/manifold/build
#THOROUGH MODE (requires thorough build):
RUST_BACKTRACE=1 ./manifold_test
#BENCH MODE (requires bench build):
./manifold_test --gtest_filter='Properties.MingapAfterTransformations:Properties.MingapStretchyBracelet:Properties.ToleranceSphere:Boolean.CreatePropertiesSlow:Samples.Bracelet:Samples.CondensedMatter16:Samples.CondensedMatter64:Samples.Sponge4:BooleanComplex.Close:BooleanComplex.GenericTwinBooleanTest7081:Polygon.Zebra:Polygon.Zebra3'
```

---

## Modified tests

- `Manifold.MeshDeterminism`: MeshBool's output is deterministic like Manifold, but not bit-identical
- `Manifold.MeshRelationRefinePrecision`: Had a crashing bug - now uses assert instead of expect as an array length guard
- `Manifold.InvalidInput*`, `Manifold.Merge`: Removed meaningless `IsEmpty()` check on the errored mesh

## Disabled tests

There are a few upstream tests that prevent other tests from running due to features not yet implemented in MeshBool. Additionally, some suites are entirely built around unimplemented features.

| Disabled test                                | Reason                  | Method                         |
| -------------------------------------------- | ----------------------- | ------------------------------ |
| All of `Manifold.ErrorPropagation*`          | N/A to Result-based API | `#if 0`                        |
| `Manifold.DeepChainDoesNotOverflowNumLeaves` | Does not compile        | `#if 0`                        |
| All of `CBIND.execution_context_*`           | Does not compile        | `#if 0`                        |
| All of `context_test.cpp`                    | Does not compile        | `manifold/test/CMakeLists.txt` |
| All of `sdf_test.cpp`                        | Unimplemented           | `manifold/test/CMakeLists.txt` |
| All of `smooth_test.cpp`                     | Unimplemented           | `manifold/test/CMakeLists.txt` |
| All of `hull_test.cpp`                       | Unimplemented           | `manifold/test/CMakeLists.txt` |

Known failures, treating them as potential bugs in Manifold/its test suite's expectations:

- Boolean.Precision: `CsgLeafNode::Compose` optimization is not implemented, and the unoptimized full boolean path calls simplify_topology with nonzero first_new_vert, causing the tiny cube to not run through degenerate removal
- Boolean.BatchBoolean: Output looks visually correct, again error may be due to unimplemented `CsgLeafNode::Compose` optimization, though it's weird that they don't produce the same number of triangles out
- Samples.Sponge4: Changing the halfedges' order causes a different triangulation with different number of degenerate triangles. Unclear to me whether this is a true failure or an overly strict test
