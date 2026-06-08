# Manifold Test Suite

This directory bridges the [MeshBool](https://github.com/JaminKoke/meshbool)
Rust crate to the upstream
[Manifold](https://github.com/BorgerLand/manifold/tree/meshbool-test) C++ test
suite. The C++ wrapper headers translate the Manifold API surface to calls into
MeshBool via [Zngur](https://github.com/HKalbasi/zngur)-generated FFI bindings.

> **Note:** These bindings are for testing only and are not optimized for
> general use. They are not the main focus of the project.

---

## Prerequisites

| Tool         | Version                          |
| ------------ | -------------------------------- |
| Rust         | 1.91+                            |
| CMake        | 3.23+                            |
| C++ compiler | C++17, e.g. GCC 11+ or Clang 14+ |

Clipper2, GoogleTest, and Corrosion (Rust/CMake integration) are fetched automatically by CMake.

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
cmake .. -DCMAKE_BUILD_TYPE=Release -DMANIFOLD_CBIND=OFF -DMANIFOLD_DEBUG=ON -DMANIFOLD_ASSERT=ON && make manifold_test -j$(nproc)
```

### 3. Run the tests

```bash
cd test #from inside test/manifold/build
./manifold_test
```

---

## Disabled tests

There are a few upstream tests that prevent other tests from running due to features not yet implemented in MeshBool. Tests that are currently failing gracefully without interfering with other tests are kept enabled.

| Disabled test                                | Reason           | Method                         |
| -------------------------------------------- | ---------------- | ------------------------------ |
| `Manifold.DeepChainDoesNotOverflowNumLeaves` | Does not compile | `#if 0`                        |
| All of `context_test.cpp`                    | Does not compile | `manifold/test/CMakeLists.txt` |
| All of `manifoldc_test.cpp`                  | Does not compile | CMake flag                     |
| `Manifold.MeshRelationRefinePrecision`       | Crashes          | `#if 0`                        |
| `Smooth.Manual`                              | Crashes          | `#if 0`                        |
