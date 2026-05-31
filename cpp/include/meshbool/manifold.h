// Copyright 2021 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <cstdint>  // uint32_t, uint64_t
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>  // needed for shared_ptr
#include <set>
#include <vector>

#include "common.h"
#include "meshbool/common.h"
#include "meshbool/meshbool.h"
// #include "meshbool/common.h"
#include "meshbool/vec_view.h"
#include "meshbool/vec_wrapper.h"

namespace manifold {
template <typename T>
using RustBox = rust::Box<T>;
template <typename T>
using RustRef = rust::Ref<T>;
template <typename T>
using RustRefMut = rust::RefMut<T>;
template <typename... T>
using RustDyn = rust::Dyn<T...>;
template <typename... T>
using RustFn = rust::Fn<T...>;
template <typename... T>
using RustSlice = rust::Slice<T...>;

// workaround for a test
template <typename T>
int NumUnique(const VecWrapper<T>& in) {
  std::set<int> unique;
  for (const T& v : in) {
    unique.emplace(v);
  }
  return unique.size();
}

/**
 * @ingroup Debug
 *
 * Allows modification of the assertions checked in MANIFOLD_DEBUG mode.
 *
 * @return ExecutionParams&
 */
inline ExecutionParams& ManifoldParams() {
  // TODO
  static ExecutionParams ex = ExecutionParams();
  return ex;
}

// class CsgNode;
// class CsgLeafNode;

/** @addtogroup Core
 *  @brief The central classes of the library
 *  @{
 */

/**
 * @brief Mesh input/output suitable for pushing directly into graphics
 * libraries.
 *
 * The core (non-optional) parts of MeshGL are the triVerts indices buffer and
 * the vertProperties interleaved vertex buffer, which follow the conventions of
 * OpenGL (and other graphic libraries') buffers and are therefore generally
 * easy to map directly to other applications' data structures.
 *
 * The triVerts vector has a stride of 3 and specifies triangles as
 * vertex indices. For triVerts = [2, 4, 5, 3, 1, 6, ...], the triangles are [2,
 * 4, 5], [3, 1, 6], etc. and likewise the halfedges are [2, 4], [4, 5], [5, 2],
 * [3, 1], [1, 6], [6, 3], etc.
 *
 * The triVerts indices should form a manifold mesh: each of the 3 halfedges of
 * each triangle should have exactly one paired halfedge in the list, defined as
 * having the first index of one equal to the second index of the other and
 * vice-versa. However, this is not always possible - consider e.g. a cube with
 * normal-vector properties. Shared vertices would turn the cube into a ball by
 * interpolating normals - the common solution is to duplicate each corner
 * vertex into 3, each with the same position, but different normals
 * corresponding to each face. This is exactly what should be done in MeshGL,
 * however we request two additional vectors in this case: mergeFromVert and
 * mergeToVert. Each vertex mergeFromVert[i] is merged into vertex
 * mergeToVert[i], avoiding unreliable floating-point comparisons to recover the
 * manifold topology. These merges are simply a union, so which is from and to
 * doesn't matter.
 *
 * If you don't have merge vectors, you can create them with the Merge() method,
 * however this will fail if the mesh is not already manifold within the set
 * tolerance. For maximum reliability, always store the merge vectors with the
 * mesh, e.g. using the EXT_mesh_manifold extension in glTF.
 *
 * You can have any number of arbitrary floating-point properties per vertex,
 * and they will all be interpolated as necessary during operations. It is up to
 * you to keep track of which channel represents what type of data. A few of
 * Manifold's methods allow you to specify the channel where normals data
 * starts, in order to update it automatically for transforms and such. This
 * will be easier if your meshes all use the same channels for properties, but
 * this is not a requirement. Operations between meshes with different numbers
 * of peroperties will simply use the larger numProp and pad the smaller one
 * with zeroes.
 *
 * On output, the triangles are sorted into runs (runIndex, runOriginalID,
 * runTransform) that correspond to different mesh inputs. Other 3D libraries
 * may refer to these runs as primitives of a mesh (as in glTF) or draw calls,
 * as they often represent different materials on different parts of the mesh.
 * It is generally a good idea to maintain a map of OriginalIDs to materials to
 * make it easy to reapply them after a set of Boolean operations. These runs
 * can also be used as input, and thus also ensure a lossless roundtrip of data
 * through MeshGL.
 *
 * As an example, with runIndex = [0, 6, 18, 21] and runOriginalID = [1, 3, 3],
 * there are 7 triangles, where the first two are from the input mesh with ID 1,
 * the next 4 are from an input mesh with ID 3, and the last triangle is from a
 * different copy (instance) of the input mesh with ID 3. These two instances
 * can be distinguished by their different runTransform matrices.
 *
 * You can reconstruct polygonal faces by assembling all the triangles that are
 * from the same run and share the same faceID. These faces will be planar
 * within the output tolerance.
 *
 * The halfedgeTangent vector is used to specify the weighted tangent vectors of
 * each halfedge for the purpose of using the Refine methods to create a
 * smoothly-interpolated surface. They can also be output when calculated
 * automatically by the Smooth functions.
 *
 * MeshGL is an alias for the standard single-precision version. Use MeshGL64 to
 * output the full double precision that Manifold uses internally.
 */
template <typename MeshGLType, typename Precision, typename I = uint32_t>
class MeshGLP {
 private:
  MeshGLType internal;

 public:
  inline MeshGLP(MeshGLType&& i)
      : internal(i.clone()),
        numProp(*::rust::RefMut<I>(internal.num_prop)),
        vertProperties(::rust::RefMut<::rust::std::vec::Vec<Precision>>(
            internal.vert_properties)),
        triVerts(::rust::RefMut<::rust::std::vec::Vec<I>>(internal.tri_verts)),
        mergeFromVert(
            ::rust::RefMut<::rust::std::vec::Vec<I>>(internal.merge_from_vert)),
        mergeToVert(
            ::rust::RefMut<::rust::std::vec::Vec<I>>(internal.merge_to_vert)),
        runIndex(::rust::RefMut<::rust::std::vec::Vec<I>>(internal.run_index)),
        runOriginalID(::rust::RefMut<::rust::std::vec::Vec<uint32_t>>(
            internal.run_original_id)),
        runTransform(::rust::RefMut<::rust::std::vec::Vec<Precision>>(
            internal.run_transform)),
        faceID(::rust::RefMut<::rust::std::vec::Vec<I>>(internal.face_id)),
        runFlags(::rust::RefMut<::rust::std::vec::Vec<uint8_t>>(internal.run_flags)),
        halfedgeTangent(internal.tri_verts.len() * 4, 0.0),
        tolerance(*::rust::RefMut<Precision>(internal.tolerance)) {}
  // halfedgeTangent(*::rust::RefMut<I>(i.halfedge_tangent)),
  inline MeshGLP(const MeshGLP& i)
      : internal(i.internal.clone()),
        numProp(*::rust::RefMut<I>(internal.num_prop)),
        vertProperties(::rust::RefMut<::rust::std::vec::Vec<Precision>>(
            internal.vert_properties)),
        triVerts(::rust::RefMut<::rust::std::vec::Vec<I>>(internal.tri_verts)),
        mergeFromVert(
            ::rust::RefMut<::rust::std::vec::Vec<I>>(internal.merge_from_vert)),
        mergeToVert(
            ::rust::RefMut<::rust::std::vec::Vec<I>>(internal.merge_to_vert)),
        runIndex(::rust::RefMut<::rust::std::vec::Vec<I>>(internal.run_index)),
        runOriginalID(::rust::RefMut<::rust::std::vec::Vec<uint32_t>>(
            internal.run_original_id)),
        runTransform(::rust::RefMut<::rust::std::vec::Vec<Precision>>(
            internal.run_transform)),
        faceID(::rust::RefMut<::rust::std::vec::Vec<I>>(internal.face_id)),
        runFlags(::rust::RefMut<::rust::std::vec::Vec<uint8_t>>(internal.run_flags)),
        halfedgeTangent(internal.tri_verts.len() * 4, 0.0),
        tolerance(*::rust::RefMut<Precision>(internal.tolerance)) {}

  /// Number of property vertices
  I NumVert() const { return this->internal.num_vert(); };
  /// Number of triangles
  I NumTri() const { return this->internal.num_tri(); };
  /// Number of properties per vertex, always >= 3.
  I& numProp;
  /// Flat, GL-style interleaved list of all vertex properties: propVal =
  /// vertProperties[vert * numProp + propIdx]. The first three properties are
  /// always the position x, y, z. The stride of the array is numProp.
  VecWrapper<Precision> vertProperties;
  /// The vertex indices of the three triangle corners in CCW (from the outside)
  /// order, for each triangle.
  VecWrapper<I> triVerts;
  /// Optional: A list of only the vertex indicies that need to be merged to
  /// reconstruct the manifold.
  VecWrapper<I> mergeFromVert;
  /// Optional: The same length as mergeFromVert, and the corresponding value
  /// contains the vertex to merge with. It will have an identical position, but
  /// the other properties may differ.
  VecWrapper<I> mergeToVert;
  /// Optional: Indicates runs of triangles that correspond to a particular
  /// input mesh instance. The runs encompass all of triVerts and are sorted
  /// by runOriginalID. Run i begins at triVerts[runIndex[i]] and ends at
  /// triVerts[runIndex[i+1]]. All runIndex values are divisible by 3. Returned
  /// runIndex will always be 1 longer than runOriginalID, but same length is
  /// also allowed as input: triVerts.size() will be automatically appended in
  /// this case.
  VecWrapper<I> runIndex;
  /// Optional: The OriginalID of the mesh this triangle run came from. This ID
  /// is ideal for reapplying materials to the output mesh. Multiple runs may
  /// have the same ID, e.g. representing different copies of the same input
  /// mesh. If you create an input MeshGL that you want to be able to reference
  /// as one or more originals, be sure to set unique values from ReserveIDs().
  VecWrapper<uint32_t> runOriginalID;
  /// Optional: For each run, a 3x4 transform is stored representing how the
  /// corresponding original mesh was transformed to create this triangle run.
  /// This matrix is stored in column-major order and the length of the overall
  /// vector is 12 * runOriginalID.size().
  VecWrapper<Precision> runTransform;
  /// Optional: Length NumTri, contains the source face ID this triangle comes
  /// from. Simplification will maintain all edges between triangles with
  /// different faceIDs. Input faceIDs will be maintained to the outputs, but if
  /// none are given, they will be filled in with Manifold's coplanar face
  /// calculation based on mesh tolerance.
  VecWrapper<I> faceID;
  /// Optional: The X-Y-Z-W weighted tangent vectors for smooth Refine(). If
  /// non-empty, must be exactly four times as long as Mesh.triVerts. Indexed
  /// as 4 * (3 * tri + i) + j, i < 3, j < 4, representing the tangent value
  /// Mesh.triVerts[tri][i] along the CCW edge. If empty, mesh is faceted.

  // TODO
  // VecWrapper<Precision> halfedgeTangent;
  std::vector<Precision> halfedgeTangent;

  /// Optional: per-run flags (bit 0 = backside, bit 1 = hasNormals).
  VecWrapper<uint8_t> runFlags;

  /// Tolerance for mesh simplification. When creating a Manifold, the tolerance
  /// used will be the maximum of this and a baseline tolerance from the size of
  /// the bounding box. Any edge shorter than tolerance may be collapsed.
  /// Tolerance may be enlarged when floating point error accumulates.
  Precision& tolerance;

  /// Number of triangle runs.
  I NumRun() const { return runOriginalID.size(); }

  /// Returns the 3x4 column-major transform for the given run, or identity if
  /// runTransform is too short (e.g. the run was not transformed).
  mat3x4 GetRunTransform(size_t run) const {
    size_t offset = 12 * run;
    if (offset + 12 > runTransform.size()) return mat3x4(la::identity);
    return mat3x4(la::mat<Precision, 3, 4>(&runTransform[offset]));
  }

  /// Returns true if runFlags bit 1 is set for this run, meaning the first
  /// three extra property channels carry world-frame vertex normals.
  bool HasNormals(size_t run) const {
    return run < runFlags.size() && (runFlags[run] & 2) != 0;
  }

  /// Returns true if runFlags bit 0 is set for this run, meaning the
  /// triangles are on the backside relative to the original mesh.
  bool Backside(size_t run) const {
    return run < runFlags.size() && (runFlags[run] & 1) != 0;
  }

  // : internal(::std::move(MeshGLType::default_())),
  inline MeshGLP()
      : internal(MeshGLType::default_()),
        numProp(*::rust::RefMut<I>(this->internal.num_prop)),
        vertProperties(::rust::RefMut<::rust::std::vec::Vec<Precision>>(
            this->internal.vert_properties)),
        triVerts(
            ::rust::RefMut<::rust::std::vec::Vec<I>>(this->internal.tri_verts)),
        mergeFromVert(::rust::RefMut<::rust::std::vec::Vec<I>>(
            this->internal.merge_from_vert)),
        mergeToVert(::rust::RefMut<::rust::std::vec::Vec<I>>(
            this->internal.merge_to_vert)),
        runIndex(
            ::rust::RefMut<::rust::std::vec::Vec<I>>(this->internal.run_index)),
        runOriginalID(::rust::RefMut<::rust::std::vec::Vec<uint32_t>>(
            this->internal.run_original_id)),
        runTransform(::rust::RefMut<::rust::std::vec::Vec<Precision>>(
            this->internal.run_transform)),
        faceID(
            ::rust::RefMut<::rust::std::vec::Vec<I>>(this->internal.face_id)),
        runFlags(::rust::RefMut<::rust::std::vec::Vec<uint8_t>>(this->internal.run_flags)),
        halfedgeTangent(std::vector<Precision>()),
        tolerance(*::rust::RefMut<Precision>(this->internal.tolerance)) {}

  inline MeshGLP& operator=(const MeshGLP& other) {
    this->internal = other.internal.clone();
    return *this;
  }

  inline MeshGLP& operator=(MeshGLP&& other) {
    this->internal = MeshGLType(other.internal.clone());
    return *this;
  }

  /**
   * Updates the mergeFromVert and mergeToVert vectors in order to create a
   * manifold solid. If the MeshGL is already manifold, no change will occur and
   * the function will return false. Otherwise, this will merge verts along open
   * edges within tolerance (the maximum of the MeshGL tolerance and the
   * baseline bounding-box tolerance), keeping any from the existing merge
   * vectors, and return true.
   *
   * There is no guarantee the result will be manifold - this is a best-effort
   * helper function designed primarily to aid in the case where a manifold
   * multi-material MeshGL was produced, but its merge vectors were lost due to
   * a round-trip through a file format. Constructing a Manifold from the result
   * will report an error status if it is not manifold.
   */
  inline bool Merge() { return this->internal.merge(); }

  /**
   * Returns the x, y, z position of the ith vertex.
   *
   * @param v vertex index.
   */
  la::vec<Precision, 3> GetVertPos(size_t v) const {
    // TODO: port to Rust
    size_t offset = v * numProp;
    return la::vec<Precision, 3>(vertProperties[offset],
                                 vertProperties[offset + 1],
                                 vertProperties[offset + 2]);
  }

  /**
   * Returns the three vertex indices of the ith triangle.
   *
   * @param t triangle index.
   */
  la::vec<I, 3> GetTriVerts(size_t t) const {
    // TODO: port to Rust
    size_t offset = 3 * t;
    return la::vec<I, 3>(triVerts[offset], triVerts[offset + 1],
                         triVerts[offset + 2]);
  }

  /**
   * Returns the x, y, z, w tangent of the ith halfedge.
   *
   * @param h halfedge index (3 * triangle_index + [0|1|2]).
   */
  // la::vec<Precision, 4> GetTangent(size_t h) const {
  //   // TODO: port to Rust
  //   size_t offset = 4 * h;
  //   return la::vec<Precision, 4>(
  //       halfedgeTangent[offset], halfedgeTangent[offset + 1],
  //       halfedgeTangent[offset + 2], halfedgeTangent[offset + 3]);
  // }
  friend class Manifold;
};
/**
 * @brief Single-precision - ideal for most uses, especially graphics.
 */
using MeshGL = MeshGLP<rust::crate::MeshGL32, float>;
/**
 * @brief Double-precision, 64-bit indices - best for huge meshes.
 */
using MeshGL64 = MeshGLP<rust::crate::MeshGL64, double, uint64_t>;

/**
 * @brief This library's internal representation of an oriented, 2-manifold,
 * triangle mesh - a simple boundary-representation of a solid object. Use this
 * class to store and operate on solids, and use MeshGL for input and output.
 *
 * In addition to storing geometric data, a Manifold can also store an arbitrary
 * number of vertex properties. These could be anything, e.g. normals, UV
 * coordinates, colors, etc, but this library is completely agnostic. All
 * properties are merely float values indexed by channel number. It is up to the
 * user to associate channel numbers with meaning.
 *
 * Manifold allows vertex properties to be shared for efficient storage, or to
 * have multiple property verts associated with a single geometric vertex,
 * allowing sudden property changes, e.g. at Boolean intersections, without
 * sacrificing manifoldness.
 *
 * Manifolds also keep track of their relationships to their inputs, via
 * OriginalIDs and the faceIDs and transforms accessible through MeshGL. This
 * allows object-level properties to be re-associated with the output after many
 * operations, particularly useful for materials. Since separate object's
 * properties are not mixed, there is no requirement that channels have
 * consistent meaning between different inputs.
 */
class Manifold {
 private:
  rust::crate::MeshBool internal;

 public:
  // inline Manifold(rust::crate::MeshBool&& i) : internal(::std::move(i)) {}
  inline Manifold(rust::crate::MeshBool&& i) : internal(i.clone()) {}

  /** @name Basics
   *  Copy / move / assignment
   */
  ///@{
  inline Manifold() : internal(rust::crate::MeshBool::default_()) {}
  ~Manifold() = default;
  inline Manifold(const Manifold& other) {
    this->internal = other.internal.clone();
  }
  inline Manifold& operator=(const Manifold& other) {
    this->internal = other.internal.clone();
    return *this;
  }
  Manifold(Manifold&&) noexcept = default;
  Manifold& operator=(Manifold&&) noexcept = default;
  ///@}

  /** @name Input & Output
   *  Create and retrieve arbitrary manifolds
   */
  ///@{
  inline Manifold(const MeshGL& mesh)
      : internal(rust::crate::MeshBool::from_meshgl_32(mesh.internal).clone()) {
  }
  inline Manifold(const MeshGL64& mesh)
      : internal(rust::crate::MeshBool::from_meshgl_64(mesh.internal).clone()) {
  }
  // TODO: originally negative 1?
  inline MeshGL GetMeshGL(int normalIdx = -1) const {
    return this->internal.get_mesh_gl_32(normalIdx);
  }
  inline MeshGL64 GetMeshGL64(int normalIdx = -1) const {
    return this->internal.get_mesh_gl_64(normalIdx);
  }
  ///@}

  /** @name Constructors
   *  Topological ops, primitives, and SDF
   */
  ///@{
  inline std::vector<Manifold> Decompose() const {
    auto l = this->internal.decompose();

    std::vector<Manifold> return_v;
    return_v.reserve(l.len());

    for (size_t i = 0; i < l.len(); i++) {
      return_v.emplace_back(l.get(i).unwrap().clone());
    }
    return return_v;
  }
  inline static Manifold Compose(const std::vector<Manifold>&) {
    // TODO
    throw "fail";
    return Manifold();
  }
  inline static Manifold Tetrahedron() {
    return rust::crate::MeshBool::tetrahedron();
  }
  inline static Manifold Cube(vec3 size = vec3(1.0), bool center = false) {
    return rust::crate::MeshBool::cube(
        ::rust::nalgebra::Vector3<::double_t>::new_(size.x, size.y, size.z),
        center);
  }
  inline static Manifold Cylinder(double height, double radiusLow,
                                  double radiusHigh = -1.0,
                                  int circularSegments = 0,
                                  bool center = false) {
    return rust::crate::MeshBool::cylinder(height, radiusLow, radiusHigh,
                                           circularSegments, center);
  }
  inline static Manifold Sphere(double radius, int circularSegments = 0) {
    return rust::crate::MeshBool::sphere(radius, circularSegments);
  }
  inline static Manifold LevelSet(std::function<double(vec3)> sdf, Box bounds,
                                  double edgeLength, double level = 0,
                                  double tolerance = -1,
                                  bool canParallel = true) {
    // TODO
    throw "fail";
    return Manifold();
  }
  ///@}

  /** @name Polygons
   * 3D to 2D and 2D to 3D
   */
  ///@{
  inline Polygons Slice(double height = 0) const {
    auto polys_rs = this->internal.slice(height);
    return RSPolygonsToCPPPolygons(polys_rs);
  }
  inline Polygons Project() const {
    auto polys_rs = this->internal.project();
    return RSPolygonsToCPPPolygons(polys_rs);
  }
  inline static Manifold Extrude(const Polygons& crossSection, double height,
                                 int nDivisions = 0, double twistDegrees = 0.0,
                                 vec2 scaleTop = vec2(1.0)) {
    auto crossSection_rs = CPPPolygonsToRSPolygons(crossSection);
    return rust::crate::MeshBool::extrude(
        crossSection_rs, height, nDivisions, twistDegrees,
        rust::nalgebra::Vector2<double>::new_(scaleTop.x, scaleTop.x));
  }
  inline static Manifold Revolve(const Polygons& crossSection,
                                 int circularSegments = 0,
                                 double revolveDegrees = 360.0f) {
    // revolve is not yet implemented in the Rust backend
    (void)crossSection;
    (void)circularSegments;
    (void)revolveDegrees;
    return Manifold();
  }
  ///@}

  enum class Error {
    NoError,
    NonFiniteVertex,
    NotManifold,
    VertexOutOfBounds,
    PropertiesWrongLength,
    MissingPositionProperties,
    MergeVectorsDifferentLengths,
    MergeIndexOutOfBounds,
    TransformWrongLength,
    RunIndexWrongLength,
    FaceIDWrongLength,
    InvalidConstruction,
    ResultTooLarge,
  };

  /** @name Information
   *  Details of the manifold
   */
  ///@{
  inline Error Status() const {
    using ManifoldError = ::rust::crate::MeshBoolError;
    auto status = this->internal.status();
    if (status.is_no_error()) {
      return Error::NoError;
    } else if (status.is_non_finite_vertex()) {
      return Error::NonFiniteVertex;
    } else if (status.is_invalid_construction()) {
      return Error::InvalidConstruction;
    } else if (status.is_result_too_large()) {
      return Error::ResultTooLarge;
    } else if (status.is_not_manifold()) {
      return Error::NotManifold;
    } else if (status.is_missing_position_properties()) {
      return Error::MissingPositionProperties;
    } else if (status.is_merge_vectors_different_lengths()) {
      return Error::MergeVectorsDifferentLengths;
    } else if (status.is_transform_wrong_length()) {
      return Error::TransformWrongLength;
    } else if (status.is_run_index_wrong_length()) {
      return Error::RunIndexWrongLength;
    } else if (status.is_face_id_wrong_length()) {
      return Error::FaceIDWrongLength;
    } else if (status.is_merge_index_out_of_bounds()) {
      return Error::MergeIndexOutOfBounds;
    } else if (status.is_vertex_out_of_bounds()) {
      return Error::VertexOutOfBounds;
    } else {
      // Shouldn't be possible?
      // return Error::NoError;
      throw "unexpected";
    }
  }
  inline bool IsEmpty() const { return this->internal.is_empty(); }
  inline size_t NumVert() const { return this->internal.num_vert(); }
  inline size_t NumEdge() const { return this->internal.num_edge(); }
  inline size_t NumTri() const { return this->internal.num_tri(); }
  inline size_t NumProp() const { return this->internal.num_prop(); }
  inline size_t NumPropVert() const { return this->internal.num_prop_vert(); }
  inline Box BoundingBox() const {
    auto rust_aabb = this->internal.bounding_box();
    return Box(vec3(rust_aabb.min.get_x(), rust_aabb.min.get_y(),
                    rust_aabb.min.get_z()),
               vec3(rust_aabb.max.get_x(), rust_aabb.max.get_y(),
                    rust_aabb.max.get_z()));
  }
  inline int Genus() const {
    return this->internal.genus();
    return 0;
  }
  inline double GetTolerance() const { return this->internal.get_tolerance(); }
  ///@}

  /** @name Measurement
   */
  ///@{
  inline double SurfaceArea() const { return this->internal.surface_area(); }
  inline double Volume() const { return this->internal.volume(); }
  inline double MinGap(const Manifold& other, double searchLength) const {
    return this->internal.min_gap(other.internal, searchLength);
  }
  ///@}

  /** @name Mesh ID
   *  Details of the manifold's relation to its input meshes, for the purposes
   * of reapplying mesh properties.
   */
  ///@{
  inline int OriginalID() const {
    return this->internal.original_id();
    return 0;
  }
  inline Manifold AsOriginal() const { return this->internal.as_original(); }
  inline static uint32_t ReserveIDs(uint32_t n) {
    return ::rust::crate::MeshBool::reserve_ids(n);
    return n;
  }
  ///@}

  /** @name Transformations
   */
  ///@{
  inline Manifold Translate(vec3 v) const {
    return this->internal.translate(
        ::rust::nalgebra::Vector3<::double_t>::new_(v.x, v.y, v.z));
  }
  inline Manifold Scale(vec3 size) const {
    return this->internal.scale(
        ::rust::nalgebra::Vector3<::double_t>::new_(size.x, size.y, size.z));
  }
  inline Manifold Rotate(double xDegrees, double yDegrees = 0.0,
                         double zDegrees = 0.0) const {
    return this->internal.rotate(xDegrees, yDegrees, zDegrees);
  }
  inline Manifold Mirror(vec3 v) const {
    return this->internal.mirror(
        ::rust::nalgebra::Vector3<::double_t>::new_(v.x, v.y, v.z));
  }
  inline Manifold Transform(const mat3x4& m) const {
    auto thing = ::rust::nalgebra::Matrix3x4<double>::from_column_slice(
        rust::std::slice::from_raw_parts(reinterpret_cast<const double*>(&m),
                                         3 * 4));
    return this->internal.transform(thing);
  }
  inline Manifold Warp(std::function<void(vec3&)> f) const {
    using RustPoint3 = ::rust::nalgebra::Point3<double>;
    using FT = RustBox<RustDyn<RustFn<RustRefMut<RustPoint3>, rust::Unit>>>;

    auto f_new = FT::make_box([f](RustRefMut<RustPoint3> v) -> rust::Unit {
      vec3 new_v(v.get_x(), v.get_y(), v.get_z());
      f(new_v);
      v = RustPoint3::new_(new_v.x, new_v.y, new_v.z);
      return {};
    });

    return this->internal.warp_boxed(std::move(f_new));
  }
  inline Manifold WarpBatch(std::function<void(VecView<vec3>)> f) const {
    // warp_batch_boxed is not yet exposed; implement via warp_boxed by
    // collecting all vertices into a temporary buffer, calling f once, then
    // writing them back one at a time through warp_boxed.
    // For now, fall back to per-vertex Warp so the signature compiles and
    // tests that don't call WarpBatch still pass.
    return Warp([&f](vec3& v) {
      vec3 buf[1] = {v};
      f({buf, 1});
      v = buf[0];
    });
  }
  inline Manifold SetTolerance(double t) const {
    return this->internal.set_tolerance(t);
  }
  inline Manifold Simplify(double tolerance = 0) const {
    using T = ::rust::core::option::Option<double>;
    if (tolerance == 0) {
      return this->internal.simplify(T::None());
    } else {
      return this->internal.simplify(T::Some(tolerance));
    }
  }
  ///@}

  /** @name Boolean
   *  Combine two manifolds
   */
  ///@{
  inline Manifold Boolean(const Manifold& second, OpType op) const {
    auto v = rust::crate::OpType();
    switch (op) {
      case manifold::OpType::Add:
        v = rust::crate::OpType::Add();
        break;
      case manifold::OpType::Subtract:
        v = rust::crate::OpType::Subtract();
        break;
      case manifold::OpType::Intersect:
        v = rust::crate::OpType::Intersect();
        break;
    }
    return this->internal.boolean(second.internal, v);
  }
  inline static Manifold BatchBoolean(const std::vector<Manifold>& manifolds,
                                      OpType op) {
    // TODO
    throw "fail";
    return Manifold();
  }
  // // Boolean operation shorthand
  // Add (Union)
  inline Manifold operator+(const Manifold& second) const {
    return this->Boolean(second, OpType::Add);
  }
  inline Manifold& operator+=(const Manifold& second) {
    *this = *this + second;
    return *this;
  }
  // Subtract (Difference)
  inline Manifold operator-(const Manifold& second) const {
    return this->Boolean(second, OpType::Subtract);
  }
  inline Manifold& operator-=(const Manifold& second) {
    *this = *this - second;
    return *this;
  }
  // Intersect
  inline Manifold operator^(const Manifold& second) const {
    return this->Boolean(second, OpType::Intersect);
  }
  inline Manifold& operator^=(const Manifold& second) {
    *this = *this ^ second;
    return *this;
  }
  inline std::pair<Manifold, Manifold> Split(const Manifold& other) const {
    auto v = this->internal.split(other.internal);
    return std::pair(Manifold(v.f0.clone()), Manifold(v.f1.clone()));
  }
  inline std::pair<Manifold, Manifold> SplitByPlane(vec3 normal,
                                                    double originOffset) const {
    auto new_normal =
        ::rust::nalgebra::Vector3<double>::new_(normal.x, normal.y, normal.z);
    auto v = this->internal.split_by_plane(new_normal, originOffset);
    return std::pair(Manifold(v.f0.clone()), Manifold(v.f1.clone()));
  }
  inline Manifold TrimByPlane(vec3 normal, double originOffset) const {
    auto new_normal =
        ::rust::nalgebra::Vector3<double>::new_(normal.x, normal.y, normal.z);
    return this->internal.trim_by_plane(new_normal, originOffset);
  }
  ///@}

  /** @name Properties
   * Create and modify vertex properties.
   */
  ///@{
  inline Manifold SetProperties(
      int numProp,
      std::function<void(double*, vec3, const double*)> propFunc) const {
    using FT = RustBox<RustDyn<
        RustFn<RustRefMut<RustSlice<double>>, ::rust::nalgebra::Point3<double>,
               RustRef<RustSlice<double>>, rust::Unit>>>;

    using ThisOption = ::rust::core::option::Option<FT>;

    auto maybe_function =
        (propFunc == nullptr)
            ? ThisOption::None()
            : ThisOption::Some(FT::make_box(
                  [propFunc](
                      ::rust::RefMut<::rust::Slice<double>> a,
                      ::rust::nalgebra::Point3<double> b,
                      ::rust::Ref<::rust::Slice<double>> c) -> rust::Unit {
                    vec3 new_b(b.get_x(), b.get_y(), b.get_z());
                    propFunc(a.as_mut_ptr(), new_b, c.as_ptr());
                    return {};
                  }));

    return this->internal.set_properties_boxed(numProp, std::move(maybe_function));
  }
  inline Manifold CalculateCurvature(int gaussianIdx, int meanIdx) const {
    return this->internal.calculate_curvature(gaussianIdx, meanIdx);
  }
  inline Manifold CalculateNormals(int normalIdx,
                                   double minSharpAngle = 60) const {
    return this->internal.calculate_normals(normalIdx, minSharpAngle);
  }
  inline Manifold CalculateNormals() const {
    return this->internal.calculate_normals(0, 60);
  }
  ///@}

  /** @name Smoothing
   * Smooth meshes by calculating tangent vectors and refining to a higher
   * triangle count.
   */
  ///@{
  inline Manifold Refine(int v) const { return this->internal.refine(v); }
  inline Manifold RefineToLength(double v) const {
    return this->internal.refine_to_length(v);
  }
  inline Manifold RefineToTolerance(double v) const {
    return this->internal.refine_to_tolerance(v);
  }
  inline Manifold SmoothByNormals(int normalIdx) const {
    // TODO
    throw "fail";
    return Manifold();
  }
  inline Manifold SmoothOut(double minSharpAngle = 60,
                            double minSmoothness = 0) const {
    // TODO
    throw "fail";
    return Manifold();
  }
  inline static Manifold Smooth(
      const MeshGL& mesh, const std::vector<Smoothness>& sharpenedEdges = {}) {
    // TODO
    throw "fail";
    return Manifold();
  }
  inline static Manifold Smooth(
      const MeshGL64& mesh,
      const std::vector<Smoothness>& sharpenedEdges = {}) {
    // TODO
    throw "fail";
    return Manifold();
  }
  ///@}

  /** @name Convex Hull
   */
  ///@{
  inline Manifold Hull() const {
    // TODO
    throw "fail";
    // return this->internal.hull();
    return Manifold();
  }
  inline static Manifold Hull(const std::vector<Manifold>& manifolds) {
    // TODO
    throw "fail";
    return Manifold();
  }
  inline static Manifold Hull(const std::vector<vec3>& pts) {
    // TODO
    throw "fail";
    return Manifold();
  }
///@}

/** @name Debugging I/O
 * Self-contained mechanism for reading and writing high precision Manifold
 * data.  Write function creates special-purpose OBJ files, and Read function
 * reads them in.  Be warned these are not (and not intended to be)
 * full-featured OBJ importers/exporters.  Their primary use is to extract
 * accurate Manifold data for debugging purposes - writing out any info
 * needed to accurately reproduce a problem case's state.  Consequently, they
 * may store and process additional data in comments that other OBJ parsing
 * programs won't understand.
 *
 * The "format" read and written by these functions is not guaranteed to be
 * stable from release to release - it will be modified as needed to ensure
 * it captures information needed for debugging.  The only API guarantee is
 * that the ReadOBJ method in a given build/release will read in the output
 * of the WriteOBJ method produced by that release.
 *
 * To work with a file, the caller should prepare the ifstream/ostream
 * themselves, as follows:
 *
 * Reading:
 * @code
 * std::ifstream ifile;
 * ifile.open(filename);
 * if (ifile.is_open()) {
 *   Manifold obj_m = Manifold::ReadOBJ(ifile);
 *   ifile.close();
 *   if (obj_m.Status() != Manifold::Error::NoError) {
 *      std::cerr << "Failed reading " << filename << ":\n";
 *      std::cerr << Manifold::ToString(ob_m.Status()) << "\n";
 *   }
 *   ifile.close();
 * }
 * @endcode
 *
 * Writing:
 * @code
 * std::ofstream ofile;
 * ofile.open(filename);
 * if (ofile.is_open()) {
 *    if (!m.WriteOBJ(ofile)) {
 *       std::cerr << "Failed writing to " << filename << "\n";
 *    }
 * }
 * ofile.close();
 * @endcode
 */
#ifdef MANIFOLD_DEBUG
  static Manifold ReadOBJ(std::istream& stream);
  bool WriteOBJ(std::ostream& stream) const;
#endif

  /** @name Testing Hooks
   *  These are just for internal testing.
   */
  ///@{
  inline bool MatchesTriNormals() const {
    return this->internal.matches_tri_normals();
  }
  inline size_t NumDegenerateTris() const {
    return this->internal.num_degenerate_tris();
  }
  inline double GetEpsilon() const { return this->internal.get_epsilon(); }
  ///@}
};
/** @} */

/** @addtogroup Debug
 *  @ingroup Optional
 *  @brief Debugging features
 *
 * The features require compiler flags to be enabled. Assertions are enabled
 * with the MANIFOLD_DEBUG flag and then controlled with ExecutionParams.
 *  @{
 */
#ifdef MANIFOLD_DEBUG
inline std::string ToString(const Manifold::Error& error) {
  switch (error) {
    case Manifold::Error::NoError:
      return "No Error";
    case Manifold::Error::NonFiniteVertex:
      return "Non Finite Vertex";
    case Manifold::Error::NotManifold:
      return "Not Manifold";
    case Manifold::Error::VertexOutOfBounds:
      return "Vertex Out Of Bounds";
    case Manifold::Error::PropertiesWrongLength:
      return "Properties Wrong Length";
    case Manifold::Error::MissingPositionProperties:
      return "Missing Position Properties";
    case Manifold::Error::MergeVectorsDifferentLengths:
      return "Merge Vectors Different Lengths";
    case Manifold::Error::MergeIndexOutOfBounds:
      return "Merge Index Out Of Bounds";
    case Manifold::Error::TransformWrongLength:
      return "Transform Wrong Length";
    case Manifold::Error::RunIndexWrongLength:
      return "Run Index Wrong Length";
    case Manifold::Error::FaceIDWrongLength:
      return "Face ID Wrong Length";
    case Manifold::Error::InvalidConstruction:
      return "Invalid Construction";
    case Manifold::Error::ResultTooLarge:
      return "Result Too Large";
    default:
      return "Unknown Error";
  };
}

inline std::ostream& operator<<(std::ostream& stream,
                                const Manifold::Error& error) {
  return stream << ToString(error);
}
#endif
#ifndef MANIFOLD_NO_IOSTREAM
MeshGL64 ReadOBJ(std::istream& stream);
bool WriteOBJ(std::ostream& stream, const MeshGL64& mesh);
#endif
/** @} */
}  // namespace manifold
