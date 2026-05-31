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
//
// Extracted from upstream manifold src/impl.cpp (commit 9f01818 / HEAD)
// to provide the free ReadOBJ / WriteOBJ functions that were part of the
// upstream manifold core library (deleted from this repo's build by
// file(REMOVE_RECURSE manifold/src)).

#ifndef MANIFOLD_NO_IOSTREAM

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <optional>
#include <regex>
#include <sstream>
#include <string>

#include "manifold/manifold.h"

namespace {

template <typename T>
double FromChars(T buffer) {
  double tmp;
  std::istringstream iss(buffer);
  iss >> std::setprecision(19);
  iss >> tmp;
  return tmp;
}

}  // namespace

namespace manifold {

static std::ostream& WriteOBJWithEpsilon(std::ostream& stream,
                                         const MeshGL64& mesh,
                                         std::optional<double> epsilon) {
  auto useHexFloat = []() {
    const char* v = std::getenv("MANIFOLD_OBJ_HEX_FLOAT");
    if (v == nullptr) return false;
    return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
           std::strcmp(v, "TRUE") == 0 || std::strcmp(v, "on") == 0 ||
           std::strcmp(v, "ON") == 0;
  };
  const bool hexFloat = useHexFloat();
  auto writeValue = [&](double value) {
    if (hexFloat) {
      char buf[128];
      std::snprintf(buf, sizeof(buf), "%.13a", value);
      stream << buf;
    } else {
      stream << value;
    }
  };

  stream << std::setprecision(19);
  if (!hexFloat) {
    stream << std::fixed;
  }
  stream << "# ======= begin mesh ======" << std::endl;
  stream << "# float_format = " << (hexFloat ? "hexfloat" : "fixed")
         << std::endl;
  stream << "# tolerance = ";
  writeValue(mesh.tolerance);
  stream << std::endl;
  if (epsilon.has_value()) {
    stream << "# epsilon = ";
    writeValue(epsilon.value());
    stream << std::endl;
  }
  for (size_t i = 0; i < mesh.NumVert(); i++) {
    stream << "v";
    size_t offset = i * mesh.numProp;
    for (size_t j : {0, 1, 2}) {
      stream << " ";
      writeValue(mesh.vertProperties[offset + j]);
    }
    stream << std::endl;
  }
  std::vector<std::array<uint64_t, 3>> triangles;
  triangles.reserve(mesh.NumTri());
  for (size_t i = 0; i < mesh.NumTri(); i++)
    triangles.push_back({mesh.triVerts[3 * i] + 1,
                         mesh.triVerts[3 * i + 1] + 1,
                         mesh.triVerts[3 * i + 2] + 1});
  sort(triangles.begin(), triangles.end());
  for (const auto& tri : triangles)
    stream << "f " << tri[0] << " " << tri[1] << " " << tri[2] << std::endl;
  stream << "# ======== end mesh =======" << std::endl;
  return stream;
}

static std::pair<MeshGL64, std::optional<double>> ReadOBJWithEpsilon(
    std::istream& stream) {
  static const std::string FLOAT_PATTERN =
      "(-?\\d+(?:\\.\\d*)?(?:[eE][+\\-]?\\d+)?)";
  static const std::string FACE_ELEMENT = "(\\d+)(?:\\S+)?";
  static const std::string TRAILING_SPACES = "(?:\\s*)";
  static const std::string SEPARATOR = "\\s+";
  static const std::regex TOLERANCE_COMMENT_PATTERN(
      "^# tolerance = " + FLOAT_PATTERN + TRAILING_SPACES);
  static const std::regex EPSILON_COMMENT_PATTERN(
      "^# epsilon = " + FLOAT_PATTERN + TRAILING_SPACES);
  static const std::regex VERTEX_PATTERN("^v" + SEPARATOR + FLOAT_PATTERN +
                                         SEPARATOR + FLOAT_PATTERN + SEPARATOR +
                                         FLOAT_PATTERN + TRAILING_SPACES);
  static const std::regex FACE_PATTERN("^f" + SEPARATOR + FACE_ELEMENT +
                                       SEPARATOR + FACE_ELEMENT + SEPARATOR +
                                       FACE_ELEMENT + TRAILING_SPACES);

  MeshGL64 mesh;
  std::optional<double> epsilon;
  if (!stream.good()) return std::make_pair(mesh, epsilon);

  constexpr size_t BUFFER_SIZE = 1000;
  std::array<char, BUFFER_SIZE> buffer;
  std::cmatch m;

  while (!stream.eof()) {
    size_t i = 0;
    char c;
    while (!stream.eof() && (c = stream.get()) != '\n' && c != '\r')
      if (i < BUFFER_SIZE) buffer[i++] = c;
    if (i == BUFFER_SIZE) continue;
    buffer[i] = '\0';
    if (std::regex_match(buffer.data(), m, TOLERANCE_COMMENT_PATTERN)) {
      mesh.tolerance = FromChars(m[1]);
    } else if (std::regex_match(buffer.data(), m, EPSILON_COMMENT_PATTERN)) {
      epsilon = {FromChars(m[1])};
    } else if (std::regex_match(buffer.data(), m, VERTEX_PATTERN)) {
      for (int j : {0, 1, 2})
        mesh.vertProperties.push_back(FromChars(m[j + 1]));
    } else if (std::regex_match(buffer.data(), m, FACE_PATTERN)) {
      for (int j : {0, 1, 2})
        mesh.triVerts.push_back(std::stoi(m[j + 1].str()) - 1);
    }
  }

  return std::make_pair(mesh, epsilon);
}

MeshGL64 ReadOBJ(std::istream& stream) {
  return ReadOBJWithEpsilon(stream).first;
}

bool WriteOBJ(std::ostream& stream, const MeshGL64& mesh) {
  if (!stream.good()) return false;
  WriteOBJWithEpsilon(stream, mesh, {});
  return true;
}

}  // namespace manifold

#endif  // MANIFOLD_NO_IOSTREAM
