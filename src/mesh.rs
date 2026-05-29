use crate::common::{LossyFrom, LossyInto};

///@brief Mesh input/output suitable for pushing directly into graphics
///libraries.
///
///The core (non-optional) parts of MeshGL are the triVerts indices buffer and
///the vertProperties interleaved vertex buffer, which follow the conventions of
///OpenGL (and other graphic libraries') buffers and are therefore generally
///easy to map directly to other applications' data structures.
///
///The triVerts vector has a stride of 3 and specifies triangles as
///vertex indices. For triVerts = [2, 4, 5, 3, 1, 6, ...], the triangles are [2,
///4, 5], [3, 1, 6], etc. and likewise the halfedges are [2, 4], [4, 5], [5, 2],
///[3, 1], [1, 6], [6, 3], etc.
///
///The triVerts indices should form a manifold mesh: each of the 3 halfedges of
///each triangle should have exactly one paired halfedge in the list, defined as
///having the first index of one equal to the second index of the other and
///vice-versa. However, this is not always possible - consider e.g. a cube with
///normal-vector properties. Shared vertices would turn the cube into a ball by
///interpolating normals - the common solution is to duplicate each corner
///vertex into 3, each with the same position, but different normals
///corresponding to each face. This is exactly what should be done in MeshGL,
///however we request two additional vectors in this case: mergeFromVert and
///mergeToVert. Each vertex mergeFromVert[i] is merged into vertex
///mergeToVert[i], avoiding unreliable floating-point comparisons to recover the
///manifold topology. These merges are simply a union, so which is from and to
///doesn't matter.
///
///If you don't have merge vectors, you can create them with the Merge() method,
///however this will fail if the mesh is not already manifold within the set
///tolerance. For maximum reliability, always store the merge vectors with the
///mesh, e.g. using the EXT_mesh_manifold extension in glTF.
///
///You can have any number of arbitrary floating-point properties per vertex,
///and they will all be interpolated as necessary during operations. It is up to
///you to keep track of which channel represents what type of data. A few of
///Manifold's methods allow you to specify the channel where normals data
///starts, in order to update it automatically for transforms and such. This
///will be easier if your meshes all use the same channels for properties, but
///this is not a requirement. Operations between meshes with different numbers
///of properties will simply use the larger numProp and pad the smaller one
///with zeroes.
///
///On output, the triangles are sorted into runs (runIndex, runOriginalID,
///runTransform) that correspond to different mesh inputs. Other 3D libraries
///may refer to these runs as primitives of a mesh (as in glTF) or draw calls,
///as they often represent different materials on different parts of the mesh.
///It is generally a good idea to maintain a map of OriginalIDs to materials to
///make it easy to reapply them after a set of Boolean operations. These runs
///can also be used as input, and thus also ensure a lossless roundtrip of data
///through MeshGL.
///
///As an example, with runIndex = [0, 6, 18, 21] and runOriginalID = [1, 3, 3],
///there are 7 triangles, where the first two are from the input mesh with ID 1,
///the next 4 are from an input mesh with ID 3, and the last triangle is from a
///different copy (instance) of the input mesh with ID 3. These two instances
///can be distinguished by their different runTransform matrices.
///
///You can reconstruct polygonal faces by assembling all the triangles that are
///from the same run and share the same faceID. These faces will be planar
///within the output tolerance.
///
///The halfedgeTangent vector is used to specify the weighted tangent vectors of
///each halfedge for the purpose of using the Refine methods to create a
///smoothly-interpolated surface. They can also be output when calculated
///automatically by the Smooth functions.
///
///MeshGL is an alias for the standard single-precision version. Use MeshGL64 to
///output the full double precision that Manifold uses internally.
//
// MeshGLP / MeshGL / MeshGL64 are forward-declared in common.h; the
// default `I = uint32_t` lives on the forward decl.
#[derive(Debug, Clone)]
pub struct MeshGLP<F, I>
where
	F: LossyFrom<f64>,
	I: LossyFrom<usize>,
{
	/// Number of properties per vertex, always >= 3.
	pub num_prop: I,
	/// Flat, GL-style interleaved list of all vertex properties: propVal =
	/// vertProperties[vert * numProp + propIdx]. The first three properties are
	/// always the position x, y, z. The stride of the array is numProp.
	pub vert_properties: Vec<F>,
	/// The vertex indices of the three triangle corners in CCW (from the outside)
	/// order, for each triangle.
	pub tri_verts: Vec<I>,
	/// Optional: A list of only the vertex indicies that need to be merged to
	/// reconstruct the manifold.
	pub merge_from_vert: Vec<I>,
	/// Optional: The same length as mergeFromVert, and the corresponding value
	/// contains the vertex to merge with. It will have an identical position, but
	/// the other properties may differ.
	pub merge_to_vert: Vec<I>,
	/// Optional: Indicates runs of triangles that correspond to a particular
	/// input mesh instance. The runs encompass all of triVerts and are sorted
	/// by runOriginalID. Run i begins at triVerts[runIndex[i]] and ends at
	/// triVerts[runIndex[i+1]]. All runIndex values are divisible by 3. Returned
	/// runIndex will always be 1 longer than runOriginalID, but same length is
	/// also allowed as input: triVerts.size() will be automatically appended in
	/// this case.
	pub run_index: Vec<I>,
	/// Optional: The OriginalID of the mesh this triangle run came from. This ID
	/// is ideal for reapplying materials to the output mesh. Multiple runs may
	/// have the same ID, e.g. representing different copies of the same input
	/// mesh. If you create an input MeshGL that you want to be able to reference
	/// as one or more originals, be sure to set unique values from ReserveIDs().
	pub run_original_id: Vec<u32>,
	/// Optional: For each run, a 3x4 transform is stored representing how the
	/// corresponding original mesh was transformed to create this triangle run.
	/// This matrix is stored in column-major order and the length of the overall
	/// vector is 12 * runOriginalID.size().
	pub run_transform: Vec<F>,
	/// Optional: For each run, defines a set of flags giving extra information
	/// about the run. See the corresponding getter functions for details on the
	/// specific flags. These are primarily used on output.
	pub run_flags: Vec<u8>,
	/// Optional: Length NumTri, contains the source face ID this triangle comes
	/// from. Simplification will maintain all edges between triangles with
	/// different faceIDs. Input faceIDs will be maintained to the outputs, but if
	/// none are given, they will be filled in with Manifold's coplanar face
	/// calculation based on mesh tolerance.
	pub face_id: Vec<I>,
	/// Tolerance for mesh simplification. When creating a Manifold, the tolerance
	/// used will be the maximum of this and a baseline tolerance from the size of
	/// the bounding box. Any edge shorter than tolerance may be collapsed.
	/// Tolerance may be enlarged when floating point error accumulates.
	pub tolerance: F,
}

impl<F, I> Default for MeshGLP<F, I>
where
	F: LossyFrom<f64>,
	I: LossyFrom<usize>,
{
	fn default() -> Self {
		Self {
			num_prop: I::lossy_from(3),
			tolerance: F::lossy_from(0.0),
			vert_properties: Vec::default(),
			tri_verts: Vec::default(),
			merge_from_vert: Vec::default(),
			merge_to_vert: Vec::default(),
			run_index: Vec::default(),
			run_original_id: Vec::default(),
			run_transform: Vec::default(),
			run_flags: Vec::default(),
			face_id: Vec::default(),
		}
	}
}

impl<F, I> MeshGLP<F, I>
where
	F: LossyFrom<f64>,
	I: LossyFrom<usize> + Copy,
{
	pub fn num_vert(&self) -> I {
		I::lossy_from(self.vert_properties.len()) / self.num_prop
	}

	pub fn num_tri(&self) -> I {
		(self.tri_verts.len() / 3).lossy_into()
	}

	pub fn num_run(&self) -> I {
		self.run_original_id.len().lossy_into()
	}

	///Returns true if this triangle run is on the backside compared to the
	///original mesh, e.g. from a subtraction. Informational only - the framework
	///already orients stored normals so the standard `getMesh()` flow returns
	///world-frame values regardless of this bit.
	///
	///@param run The index of the triangle run (0 <= run < runFlags.size()).
	pub fn backside(&self, run: usize) -> bool {
		run < self.run_flags.len() && (self.run_flags[run] & 1) != 0
	}

	///Returns true if the first three extra-property channels (slots 3, 4, 5)
	///of this run carry world-frame vertex normals (set by
	///`Manifold::CalculateNormals(0)` and round-tripped via `runFlags` bit 1).
	///Consumers should treat the slot as normals and skip re-applying
	///`runTransform` to it.
	///
	///hasNormals is per-run, so different runs may set it differently.
	///Behavior is undefined when a single propVert is shared by triangles
	///from runs that disagree - the slot has one interpretation, and a
	///Transform rotates it for hasNormals=true and clobbers any
	///hasNormals=false sharer. Standard `CalculateNormals` / Boolean /
	///Compose outputs never produce that shape.
	///
	///@param run The index of the triangle run (0 <= run < runFlags.size()).
	pub fn has_normals(&self, run: usize) -> bool {
		run < self.run_flags.len() && (self.run_flags[run] & 2) != 0
	}
}
