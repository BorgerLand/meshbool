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
#include "common.h"
#include "meshbool/common.h"
#include "meshbool/meshbool.h"
#include "meshbool/vec_view.h"

namespace manifold {

/** @addtogroup Structs
 *  @{
 */

/**
 * @brief Polygon vertex.
 */
struct PolyVert {
  /// X-Y position
  vec2 pos;
  /// ID or index into another vertex vector
  int idx;
};

/**
 * @brief Single polygon contour, wound CCW, with indices. First and last point
 * are implicitly connected. Should ensure all input is
 * [&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid).
 */
using SimplePolygonIdx = std::vector<PolyVert>;

/**
 * @brief Set of indexed polygons with holes. Order of contours is arbitrary.
 * Can contain any depth of nested holes and any number of separate polygons.
 * Should ensure all input is
 * [&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid).
 */
using PolygonsIdx = std::vector<SimplePolygonIdx>;
/** @} */

/** @addtogroup Triangulation
 *  @ingroup Core
 * @brief Polygon triangulation
 *  @{
 */
inline std::vector<ivec3> TriangulateIdx(const PolygonsIdx& polys,
                                         double epsilon = -1,
                                         bool allowConvex = true) {
  using RustPolyVert = rust::crate::polygon::PolyVert;
  using RustPoint2 = rust::nalgebra::Point2<double>;
  auto new_p =
      ::rust::std::vec::Vec<::rust::std::vec::Vec<RustPolyVert>>::new_();
  for (const auto& vec : polys) {
    auto new_p_sub = ::rust::std::vec::Vec<RustPolyVert>::new_();
    for (const auto& p : vec) {
      new_p_sub.push(
          RustPolyVert::new_(RustPoint2::new_(p.pos.x, p.pos.y), p.idx));
    }
    new_p.push(std::move(new_p_sub));
  }

  auto return_val =
      rust::crate::polygon::triangulate_idx(new_p, epsilon, allowConvex);

  std::vector<ivec3> new_cpp;
  new_cpp.reserve(return_val.len());
  for (size_t i = 0; i < return_val.len(); i++) {
    auto vec = return_val.get(i).unwrap();
    new_cpp.push_back(ivec3(vec.get_x(), vec.get_y(), vec.get_z()));(const Polygons& polygons,
                                      double epsilon = -1,
                                      bool allowConvex = true) {
  auto new_p = CPPPolygonsToRSPolygons(polygons);

  auto return_val =
      rust::crate::polygon::triangulate(new_p, epsilon, allowConvex);

  std::vector<ivec3> new_cpp;
  new_cpp.reserve(return_val.len());
  for (size_t i = 0; i < return_val.len(); i++) {
    auto vec = return_val.get(i).unwrap();
    new_cpp.push_back(ivec3(vec.get_x(), vec.get_y(), vec.get_z()));
  }
  return new_cpp;
}
/** @} */
}  // namespace manifold
