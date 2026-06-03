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
| Rust         | stable (1.80+)                   |
| CMake        | 3.23+                            |
| C++ compiler | C++17, e.g. GCC 11+ or Clang 14+ |

Clipper2, GoogleTest, and Corrosion (Rust/CMake integration) are fetched automatically by CMake.

---

## Building

### 1. Clone submodules

```bash
#if you haven't cloned the main repo yet:
git clone --recurse-submodules https://github.com/JaminKoke/meshbool.git
cd meshbool

#if you've already cloned the main repo:
cd meshbool
git submodule update --init --recursive
```

### 2. Build

```bash
mkdir -p test/manifold/build && cd test/manifold/build
cmake .. -DMANIFOLD_CBIND=OFF -DMANIFOLD_DEBUG=ON -DMANIFOLD_ASSERT=ON && make manifold_test -j$(nproc)
```

### 3. Run the tests

```bash
cd test #from inside test/manifold/build
./manifold_test
```

---

## Disabled tests

A few upstream tests exercise features not yet implemented in the MeshBool
backend. Two are wrapped in `#if 0` in the test files, and one entire file is
excluded from the build in `test/CMakeLists.txt`:

| Feature                                | Disabled                                                                              |
| -------------------------------------- | ------------------------------------------------------------------------------------- |
| `ExecutionContext` / `WithContext`     | entire `context_test.cpp`; `DeepChainDoesNotOverflowNumLeaves` in `manifold_test.cpp` |
| `halfedgeTangents` / `InvalidTangents` | `InvalidTangents` in `smooth_test.cpp`                                                |
