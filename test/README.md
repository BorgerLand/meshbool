# Manifold Test Suite

This directory contains files for testing MeshBool against the Manifold test
suite using C++ bindings that map to the Manifold API.

These bindings should ***not*** be used directly in new C++ projects and is
recommended to use [Zngur](https://github.com/HKalbasi/zngur) as these bindings
are not optimized and not the main focus of the project. Official bindings for
general use are not currently being worked on.


## Build

```
mkdir build && cd build
cmake ..
make
```

The test bin is at `build/manifold_wrapper/test/manifold_test`
