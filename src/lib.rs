pub use crate::meshgl::{MeshGL, MeshGLError};
pub use crate::ops::boolean::BooleanError;
pub use crate::ops::proc_gen::ConstructorError;
pub use crate::spatial::aabb::Box3D;
pub use crate::triangulation::Polygons;
pub use crate::util::segment_resolution::SegmentResolution;
pub use nalgebra::{Matrix3x4, Point3, Vector3};

use crate::halfedge::Halfedges;
use crate::mesh_relations::{InstanceRelation, TriRelation};
use crate::spatial::bvh_collider::BVHCollider;
use crate::util::hash_table::DeterministicMap;
use crate::util::math::K_PRECISION;

mod halfedge;
mod mesh_relations;
mod meshgl;
mod postprocessing;
mod triangulation;

mod ops {
	pub mod boolean;
	pub mod measure;
	pub mod misc;
	pub mod proc_gen;
	pub mod properties;
	pub mod section;
	pub mod subdivision;
	pub mod transform;
}

mod util {
	pub mod disjoint_sets;
	pub mod hash_table;
	pub mod math;
	pub mod multiset;
	pub mod num_convert;
	pub mod segment_resolution;
	pub mod tri_dst;
	pub mod vec_ext;
}

mod spatial {
	pub mod aabb;
	pub mod bvh_collider;
	pub mod tree2d;
}

#[cfg(feature = "test")]
mod test;

///@brief This library's internal representation of an oriented, 2-manifold,
///triangle mesh - a simple boundary-representation of a solid object. Use this
///class to store and operate on solids, and use MeshGL for input and output.
///
///In addition to storing geometric data, a Manifold can also store an arbitrary
///number of vertex properties. These could be anything, e.g. normals, UV
///coordinates, colors, etc, but this library is completely agnostic. All
///properties are merely float values indexed by channel number. It is up to the
///user to associate channel numbers with meaning.
///
///Manifold allows vertex properties to be shared for efficient storage, or to
///have multiple property verts associated with a single geometric vertex,
///allowing sudden property changes, e.g. at Boolean intersections, without
///sacrificing manifoldness.
///
///Manifolds also keep track of their relationships to their inputs, via
///OriginalIDs and the faceIDs and transforms accessible through MeshGL. This
///allows object-level properties to be re-associated with the output after many
///operations, particularly useful for materials. Since separate object's
///properties are not mixed, there is no requirement that channels have
///consistent meaning between different inputs.
#[derive(Debug)]
pub struct MeshBool {
	///The original ID of this mesh if it is an original (constructed either by
	///loading a MeshGL or a procedurally generating a primitive shape), or None
	original_id: Option<u32>,
	precision: Precision,
	vert_pos: Vec<Point3<f64>>,
	properties: Properties,
	tri: Triangles,
	///Maps <instance id, instance metadata> to look up how each mesh instance
	///relates back to its original
	instance_relation: DeterministicMap<u32, InstanceRelation>,
	collider: BVHCollider,
}

impl Clone for MeshBool {
	fn clone(&self) -> Self {
		Self {
			original_id: None,
			precision: self.precision.clone(),
			vert_pos: self.vert_pos.clone(),
			properties: self.properties.clone(),
			tri: self.tri.clone(),
			instance_relation: self.instance_relation.clone(),
			collider: self.collider.clone(),
		}
	}
}

impl MeshBool {
	///If this mesh is an original, this returns its ID that can be referenced
	///by product manifolds' MeshRelation. If this manifold is a product, this
	///returns none.
	pub fn original_id(&self) -> Option<u32> {
		self.original_id
	}

	///Returns the epsilon value of this Manifold's vertices, which tracks the
	///approximate rounding error over all the transforms and operations that have
	///led to this state. This is the value of &epsilon; defining
	///[&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid).
	pub fn get_epsilon(&self) -> f64 {
		self.precision.epsilon
	}

	///Returns the tolerance value of this Manifold. Triangles that are coplanar
	///within tolerance tend to be merged and edges shorter than tolerance tend to
	///be collapsed.
	pub fn get_tolerance(&self) -> f64 {
		self.precision.tolerance
	}

	///The number of vertices in the Manifold.
	pub fn num_vert(&self) -> usize {
		self.vert_pos.len()
	}

	///The number of properties per vertex in the Manifold.
	pub fn prop_stride(&self) -> usize {
		self.properties.stride
	}

	///The number of property vertices in the Manifold. This will always be >=
	///NumVert, as some physical vertices may be duplicated to account for different
	///properties on different neighboring triangles.
	pub fn num_prop_vert(&self) -> usize {
		if self.prop_stride() == 0 {
			self.num_vert()
		} else {
			self.properties.data.len() / self.prop_stride()
		}
	}

	///The number of edges in the Manifold.
	pub fn num_edge(&self) -> usize {
		self.tri.halfedge.num_edge()
	}

	///The number of triangles in the Manifold.
	pub fn num_tri(&self) -> usize {
		self.tri.halfedge.num_tri()
	}

	///Returns the axis-aligned bounding box of all the Manifold's vertices.
	pub fn bounding_box(&self) -> Box3D {
		self.collider.get_bounding_box()
	}

	///Does the Manifold have any triangles?
	pub fn is_empty(&self) -> bool {
		self.num_tri() == 0
	}
}

#[derive(Debug, Clone, Copy)]
struct Precision {
	epsilon: f64,
	tolerance: f64,
}

impl Precision {
	fn from_box(bbox: Box3D) -> Self {
		Self::new(bbox, -1.0, false)
	}

	///Sets epsilon based on the bounding box, and limits its minimum value
	///by the optional input.
	fn new(bbox: Box3D, min_tolerance: f64, use_f32: bool) -> Self {
		let epsilon = K_PRECISION * bbox.scale();
		let tolerance = min_tolerance.max(if use_f32 {
			epsilon.max(f32::EPSILON as f64 * bbox.scale())
		} else {
			epsilon
		});

		Self { epsilon, tolerance }
	}
}

#[derive(Debug, Clone, Default)]
struct Properties {
	data: Vec<f64>,
	stride: usize,
}

//structure of arrays
#[derive(Debug, Clone, Default)]
struct Triangles {
	halfedge: Halfedges,
	normal: Vec<Vector3<f64>>,
	///Maps each triangle to the instance it comes from to look up how each
	///triangle relates back to its original
	relation: Vec<TriRelation>,
}

impl Triangles {
	fn partial(&mut self) -> TrianglesPartial<'_> {
		TrianglesPartial {
			halfedge: &mut self.halfedge,
			normal: Some(&mut self.normal),
			relation: Some(&mut self.relation),
		}
	}
}

//some algorithms that operate on toplogy optionally accept the
//normal/relation columns of the triangles soa if they were
//already eagerly computed, in order to keep them in sync with
//halfedge. if not eagerly computed, this optimization allows
//skipping the sync and waiting to populate normal+relation
//until halfedge is already finalized. having normal+relation
//both none is the ideal fast scenario. the c++ version
//implements this same optimization by checking vec.len() > 0 or
//vec.len() == num_tri() in select areas, which requires a full
//mental map of the entire codebase to understand whether the
//columns should exist yet or not
struct TrianglesPartial<'a> {
	halfedge: &'a mut Halfedges,
	normal: Option<&'a mut Vec<Vector3<f64>>>,
	relation: Option<&'a mut Vec<TriRelation>>,
}

impl<'a> TrianglesPartial<'a> {
	fn reborrow<'b>(&'b mut self) -> TrianglesPartial<'b>
	where
		'a: 'b,
	{
		TrianglesPartial {
			halfedge: self.halfedge,
			normal: self.normal.as_deref_mut(),
			relation: self.relation.as_deref_mut(),
		}
	}
}
