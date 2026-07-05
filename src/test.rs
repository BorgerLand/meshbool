use crate::mesh_relations::reserve_original_id;
use crate::meshgl::MeshGL;
use crate::postprocessing::sort::get_tri_box_morton;
use crate::spatial::bvh_collider::SimpleRecorder;
use crate::triangulation::PolyVert;
use crate::util::num_convert::LossyFrom;
use crate::util::segment_resolution::SegmentResolution;
use crate::util::tri_dst::distance_triangle_triangle_squared;
use crate::*;
use nalgebra::{Matrix3x4, Point2, Point3, Vector2, Vector3};
use std::sync::atomic::{AtomicBool, Ordering};

include!(concat!(env!("OUT_DIR"), "/test.rs"));

/// Perform extra sanity checks and assertions on the intermediate data
/// structures.
static INTERMEDIATE_CHECKS: AtomicBool = AtomicBool::new(false);
/// Perform 3D mesh self-intersection test on intermediate boolean results to
/// test for ϵ-validity. For debug purposes only.
static SELF_INTERSECTION_CHECKS: AtomicBool = AtomicBool::new(false);
/// If processOverlaps is false, a geometric check will be performed to assert
/// all triangles are CCW.
static PROCESS_OVERLAPS: AtomicBool = AtomicBool::new(true);

//need a wrapper purely so that status can exist
#[derive(Clone)]
pub struct MeshBoolTestWrapper {
	pub mesh: MeshBool,
	pub status: MeshBoolError,
}

pub fn get_intermediate_checks() -> bool {
	INTERMEDIATE_CHECKS.load(Ordering::Relaxed)
}

pub fn set_intermediate_checks(value: bool) {
	INTERMEDIATE_CHECKS.store(value, Ordering::Relaxed);
}

pub fn get_self_intersection_checks() -> bool {
	SELF_INTERSECTION_CHECKS.load(Ordering::Relaxed)
}

pub fn set_self_intersection_checks(value: bool) {
	SELF_INTERSECTION_CHECKS.store(value, Ordering::Relaxed);
}

pub fn get_process_overlaps() -> bool {
	PROCESS_OVERLAPS.load(Ordering::Relaxed)
}

pub fn set_process_overlaps(value: bool) {
	PROCESS_OVERLAPS.store(value, Ordering::Relaxed);
}

impl Default for MeshBoolTestWrapper {
	fn default() -> Self {
		Self {
			mesh: MeshBool {
				original_id: None,
				precision: Precision {
					epsilon: -1.0,
					tolerance: -1.0,
				},
				vert_pos: Vec::default(),
				properties: Properties::default(),
				tri: Triangles::default(),
				instance_relation: DeterministicMap::default(),
				collider: BVHCollider::default(),
			},
			status: MeshBoolError::default(),
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeshBoolError {
	#[default]
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
	InvalidTangents,
	Cancelled,
}

impl MeshBoolError {
	pub fn is_no_error(self) -> bool {
		self == MeshBoolError::NoError
	}
	pub fn is_non_finite_vertex(self) -> bool {
		self == MeshBoolError::NonFiniteVertex
	}
	pub fn is_not_manifold(self) -> bool {
		self == MeshBoolError::NotManifold
	}
	pub fn is_vertex_out_of_bounds(self) -> bool {
		self == MeshBoolError::VertexOutOfBounds
	}
	pub fn is_properties_wrong_length(self) -> bool {
		self == MeshBoolError::PropertiesWrongLength
	}
	pub fn is_missing_position_properties(self) -> bool {
		self == MeshBoolError::MissingPositionProperties
	}
	pub fn is_merge_vectors_different_lengths(self) -> bool {
		self == MeshBoolError::MergeVectorsDifferentLengths
	}
	pub fn is_merge_index_out_of_bounds(self) -> bool {
		self == MeshBoolError::MergeIndexOutOfBounds
	}
	pub fn is_transform_wrong_length(self) -> bool {
		self == MeshBoolError::TransformWrongLength
	}
	pub fn is_run_index_wrong_length(self) -> bool {
		self == MeshBoolError::RunIndexWrongLength
	}
	pub fn is_face_id_wrong_length(self) -> bool {
		self == MeshBoolError::FaceIDWrongLength
	}
	pub fn is_invalid_construction(self) -> bool {
		self == MeshBoolError::InvalidConstruction
	}
	pub fn is_result_too_large(self) -> bool {
		self == MeshBoolError::ResultTooLarge
	}
	pub fn is_invalid_tangents(self) -> bool {
		self == MeshBoolError::InvalidTangents
	}
	pub fn is_cancelled(self) -> bool {
		self == MeshBoolError::Cancelled
	}
}

impl From<MeshGLError> for MeshBoolError {
	fn from(error: MeshGLError) -> Self {
		match error {
			MeshGLError::NonFiniteVertex => MeshBoolError::NonFiniteVertex,
			MeshGLError::InvalidConstruction => MeshBoolError::InvalidConstruction,
			MeshGLError::NotManifold => MeshBoolError::NotManifold,
			MeshGLError::MissingPositionProperties => MeshBoolError::MissingPositionProperties,
			MeshGLError::MergeVectorsDifferentLengths => {
				MeshBoolError::MergeVectorsDifferentLengths
			}
			MeshGLError::TransformWrongLength => MeshBoolError::TransformWrongLength,
			MeshGLError::RunIndexWrongLength => MeshBoolError::RunIndexWrongLength,
			MeshGLError::FaceIDWrongLength => MeshBoolError::FaceIDWrongLength,
			MeshGLError::MergeIndexOutOfBounds => MeshBoolError::MergeIndexOutOfBounds,
			MeshGLError::VertexOutOfBounds => MeshBoolError::VertexOutOfBounds,
		}
	}
}

impl From<BooleanError> for MeshBoolError {
	fn from(error: BooleanError) -> Self {
		match error {
			BooleanError::ResultTooLarge => MeshBoolError::ResultTooLarge,
		}
	}
}

impl From<ConstructorError> for MeshBoolError {
	fn from(error: ConstructorError) -> Self {
		match error {
			ConstructorError::InvalidConstruction => MeshBoolError::InvalidConstruction,
		}
	}
}

impl From<MeshBool> for MeshBoolTestWrapper {
	fn from(mesh: MeshBool) -> Self {
		Self {
			mesh,
			status: MeshBoolError::NoError,
		}
	}
}

impl<E> From<Result<MeshBool, E>> for MeshBoolTestWrapper
where
	E: Into<MeshBoolError>,
{
	fn from(result: Result<MeshBool, E>) -> Self {
		match result {
			Ok(mesh) => Self {
				mesh,
				status: MeshBoolError::NoError,
			},
			Err(error) => Self {
				mesh: MeshBool {
					original_id: None,
					precision: Precision {
						epsilon: -1.0,
						tolerance: -1.0,
					},
					vert_pos: Vec::default(),
					properties: Properties::default(),
					tri: Triangles::default(),
					instance_relation: DeterministicMap::default(),
					collider: BVHCollider::default(),
				},
				status: error.into(),
			},
		}
	}
}

impl MeshBoolTestWrapper {
	fn propagate_error(&self, mesh: impl FnOnce() -> MeshBool) -> Self {
		if self.status == MeshBoolError::NoError {
			Self {
				mesh: mesh(),
				status: self.status,
			}
		} else {
			eprintln!(
				"Attempted to operate on mesh with error status ({:?}). Returning clone instead.",
				self.status
			);
			self.clone()
		}
	}

	pub fn tetrahedron() -> Self {
		MeshBool::tetrahedron().into()
	}

	pub fn cube(size: Vector3<f64>, center: bool) -> Self {
		MeshBool::cube(size, center).into()
	}

	pub fn cylinder(
		height: f64,
		radius_low: f64,
		radius_high: f64,
		circular_segments: i32,
		center: bool,
	) -> Self {
		let quality = SegmentResolution {
			circular_segments: circular_segments as u32,
			..SegmentResolution::default()
		};
		MeshBool::cylinder(height, radius_low, radius_high, quality, center).into()
	}

	pub fn sphere(radius: f64, circular_segments: i32) -> Self {
		let quality = SegmentResolution {
			circular_segments: circular_segments as u32,
			..SegmentResolution::default()
		};
		MeshBool::sphere(radius, quality).into()
	}

	pub fn extrude(
		cross_section: &Polygons,
		height: f64,
		n_divisions: u32,
		twist_degrees: f64,
		scale_top: Vector2<f64>,
	) -> Self {
		MeshBool::extrude(cross_section, height, n_divisions, twist_degrees, scale_top).into()
	}

	pub fn revolve(cross_section: &Polygons, circular_segments: i32, revolve_degrees: f64) -> Self {
		let quality = SegmentResolution {
			circular_segments: circular_segments as u32,
			..SegmentResolution::default()
		};
		MeshBool::revolve(cross_section, quality, revolve_degrees).into()
	}

	pub fn reserve_ids(n: u32) -> u32 {
		let first = reserve_original_id();
		for _ in 1..n {
			reserve_original_id();
		}
		first
	}

	pub fn from_meshgl<F, I>(gl: &MeshGL<F, I>) -> Self
	where
		F: LossyFrom<f64> + Copy + 'static,
		f64: From<F>,
		I: LossyFrom<usize> + Copy,
		usize: LossyFrom<I>,
		i32: LossyFrom<I>,
	{
		MeshBool::try_from(gl).into()
	}

	pub fn simplify(&self, tolerance: Option<f64>) -> Self {
		self.propagate_error(|| self.mesh.simplify(tolerance))
	}

	pub fn as_original(&self) -> Self {
		self.propagate_error(|| self.mesh.as_original())
	}

	pub fn translate(&self, offset: Vector3<f64>) -> Self {
		self.propagate_error(|| self.mesh.translate(offset))
	}

	pub fn scale(&self, scale: Vector3<f64>) -> Self {
		self.propagate_error(|| self.mesh.scale(scale))
	}

	pub fn rotate(&self, x_degrees: f64, y_degrees: f64, z_degrees: f64) -> Self {
		self.propagate_error(|| self.mesh.rotate(x_degrees, y_degrees, z_degrees))
	}

	pub fn transform(&self, transform: Matrix3x4<f64>) -> Self {
		self.propagate_error(|| self.mesh.transform(transform))
	}

	pub fn mirror(&self, normal: Vector3<f64>) -> Self {
		self.propagate_error(|| self.mesh.mirror(normal))
	}

	pub fn set_properties(
		&self,
		prop_stride: usize,
		prop_func: Option<Box<dyn FnMut(&mut [f64], Point3<f64>, &[f64])>>,
	) -> Self {
		self.propagate_error(|| self.mesh.set_properties(prop_stride, prop_func))
	}

	pub fn warp(&self, warp_func: Box<dyn FnMut(&mut Point3<f64>)>) -> Self {
		self.propagate_error(|| self.mesh.warp(warp_func))
	}

	pub fn warp_batch(&self, warp_func: Box<dyn FnMut(&mut [Point3<f64>])>) -> Self {
		self.propagate_error(|| self.mesh.warp_batch(warp_func))
	}

	pub fn calculate_curvature(&self, gaussian_idx: i32, mean_idx: i32) -> Self {
		let gaussian = if gaussian_idx < 0 {
			None
		} else {
			Some(gaussian_idx as usize)
		};
		let mean = if mean_idx < 0 {
			None
		} else {
			Some(mean_idx as usize)
		};
		self.propagate_error(|| self.mesh.calculate_curvature(gaussian, mean))
	}

	pub fn calculate_normals(&self, normal_idx: i32, min_sharp_angle: f64) -> Self {
		self.propagate_error(|| {
			self.mesh
				.calculate_normals(normal_idx.max(0) as usize, min_sharp_angle)
		})
	}

	pub fn boolean(&self, other: &Self, op: OpType) -> Self {
		let result = match op {
			OpType::Add => self.mesh.union(&other.mesh),
			OpType::Subtract => self.mesh.difference(&other.mesh),
			OpType::Intersect => self.mesh.intersect(&other.mesh),
		};
		result.into()
	}

	pub fn decompose(&self) -> Vec<Self> {
		self.mesh.decompose().into_iter().map(Into::into).collect()
	}

	pub fn split(&self, cutter: &Self) -> (Self, Self) {
		match self.mesh.split(&cutter.mesh) {
			Ok((a, b)) => (
				Ok::<_, BooleanError>(a).into(),
				Ok::<_, BooleanError>(b).into(),
			),
			Err(error) => (Err(error).into(), Err(error).into()),
		}
	}

	pub fn split_by_plane(&self, normal: Vector3<f64>, origin_offset: f64) -> (Self, Self) {
		match self.mesh.split_by_plane(normal, origin_offset) {
			Ok((a, b)) => (
				Ok::<_, BooleanError>(a).into(),
				Ok::<_, BooleanError>(b).into(),
			),
			Err(error) => (Err(error).into(), Err(error).into()),
		}
	}

	pub fn trim_by_plane(&self, normal: Vector3<f64>, origin_offset: f64) -> Self {
		self.mesh.trim_by_plane(normal, origin_offset).into()
	}

	pub fn set_tolerance(&self, tolerance: f64) -> Self {
		self.propagate_error(|| self.mesh.set_tolerance(tolerance))
	}

	pub fn genus(&self) -> usize {
		self.mesh.genus()
	}

	pub fn surface_area(&self) -> f64 {
		self.mesh.surface_area()
	}

	pub fn volume(&self) -> f64 {
		self.mesh.volume()
	}

	pub fn original_id(&self) -> i32 {
		self.mesh.original_id().map(|id| id as i32).unwrap_or(-1)
	}

	pub fn matches_tri_normals(&self) -> bool {
		self.mesh.matches_tri_normals()
	}

	pub fn num_degenerate_tris(&self) -> usize {
		self.mesh.num_degenerate_tris()
	}

	pub fn slice(&self, height: f64) -> Vec<Vec<Point2<f64>>> {
		self.mesh.slice(height)
	}

	pub fn project(&self) -> Vec<Vec<Point2<f64>>> {
		self.mesh.project()
	}

	pub fn get_mesh_gl_32(&self, normal_idx: i32) -> MeshGL32 {
		self.mesh.to_meshgl::<f32, u32>(normal_idx)
	}

	pub fn get_mesh_gl_64(&self, normal_idx: i32) -> MeshGL64 {
		self.mesh.to_meshgl::<f64, u64>(normal_idx)
	}

	pub fn is_empty(&self) -> bool {
		self.mesh.is_empty()
	}

	pub fn status(&self) -> MeshBoolError {
		self.status
	}

	pub fn num_vert(&self) -> usize {
		self.mesh.num_vert()
	}

	pub fn num_edge(&self) -> usize {
		self.mesh.num_edge()
	}

	pub fn num_tri(&self) -> usize {
		self.mesh.num_tri()
	}

	pub fn prop_stride(&self) -> usize {
		self.mesh.prop_stride()
	}

	pub fn num_prop_vert(&self) -> usize {
		self.mesh.num_prop_vert()
	}

	pub fn bounding_box(&self) -> Box3D {
		self.mesh.bounding_box()
	}

	pub fn get_epsilon(&self) -> f64 {
		self.mesh.get_epsilon()
	}

	pub fn get_tolerance(&self) -> f64 {
		self.mesh.get_tolerance()
	}

	pub fn min_gap(&self, other: &Self, search_length: f64) -> f64 {
		self.mesh.min_gap(&other.mesh, search_length)
	}
}

impl MeshBool {
	/// Returns true if this manifold is self-intersecting. Called internally by
	/// the boolean pipeline when self-intersection checks are enabled.
	/// Note that this is not checking for epsilon-validity.
	pub fn is_self_intersecting(&self) -> bool {
		let ep = 2.0 * self.precision.epsilon;
		let epsilon_sq = ep * ep;
		let (tri_box, _) = get_tri_box_morton(&self.tri.halfedge, &self.vert_pos, None);

		let intersecting = AtomicBool::new(false);

		let mut f = |tri0: i32, tri1: i32| {
			let mut tri_verts0: [Point3<f64>; 3] = [Point3::default(); 3];
			let mut tri_verts1: [Point3<f64>; 3] = [Point3::default(); 3];
			for i in 0..3 {
				tri_verts0[i as usize] =
					self.vert_pos[self.tri.halfedge.start(3 * tri0 + i) as usize];
				tri_verts1[i as usize] =
					self.vert_pos[self.tri.halfedge.start(3 * tri1 + i) as usize];
			}
			// if triangles tri0 and tri1 share a vertex, return true to skip the
			// check. we relax the sharing criteria a bit to allow for at most
			// distance epsilon squared
			for i in 0..3 {
				for j in 0..3 {
					if (tri_verts1[j] - tri_verts0[i]).magnitude_squared() <= epsilon_sq {
						return;
					}
				}
			}

			if distance_triangle_triangle_squared(&tri_verts0, &tri_verts1) == 0.0 {
				// try to move the triangles around the normal of the other face
				let mut tmp0: [Point3<f64>; 3] = [Point3::default(); 3];
				let mut tmp1: [Point3<f64>; 3] = [Point3::default(); 3];
				for i in 0..3 {
					tmp0[i] = tri_verts0[i] + ep * self.tri.normal[tri1 as usize];
				}
				if distance_triangle_triangle_squared(&tmp0, &tri_verts1) > 0.0 {
					return;
				}
				for i in 0..3 {
					tmp0[i] = tri_verts0[i] - ep * self.tri.normal[tri1 as usize];
				}
				if distance_triangle_triangle_squared(&tmp0, &tri_verts1) > 0.0 {
					return;
				}
				for i in 0..3 {
					tmp1[i] = tri_verts1[i] + ep * self.tri.normal[tri0 as usize];
				}
				if distance_triangle_triangle_squared(&tri_verts0, &tmp1) > 0.0 {
					return;
				}
				for i in 0..3 {
					tmp1[i] = tri_verts1[i] - ep * self.tri.normal[tri0 as usize];
				}
				if distance_triangle_triangle_squared(&tri_verts0, &tmp1) > 0.0 {
					return;
				}

				intersecting.store(true, Ordering::SeqCst);
			}
		};

		let mut recorder = SimpleRecorder::new(&mut f);
		self.collider
			.collisions_from_slice::<true, _>(&mut recorder, &tri_box, true);

		intersecting.load(Ordering::SeqCst)
	}
}

pub struct Quality;

impl Quality {
	pub fn get_circular_segments(radius: f64) -> u32 {
		SegmentResolution::default().get_circular_segments(radius)
	}
}

impl PolyVert {
	pub fn new(pos: Point2<f64>, idx: i32) -> Self {
		Self { pos, idx }
	}
}

pub type MeshGL32 = MeshGL<f32, u32>;
pub type MeshGL64 = MeshGL<f64, u64>;

/// `merge` is a reserved keyword in zngur, so expose it under a different name.
impl<F, I> MeshGL<F, I>
where
	F: LossyFrom<f64> + Copy + 'static,
	I: LossyFrom<usize> + Copy,
	usize: LossyFrom<I>,
	u64: LossyFrom<I>,
	i32: LossyFrom<I>,
	f64: LossyFrom<F>,
{
	pub fn merge_glp(&mut self) -> bool {
		self.merge()
	}
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OpType {
	Add,
	Subtract,
	Intersect,
}

pub trait Vector3i32Coords {
	fn get_x(&self) -> i32;
	fn get_y(&self) -> i32;
	fn get_z(&self) -> i32;
}

impl Vector3i32Coords for Vector3<i32> {
	fn get_x(&self) -> i32 {
		self.x
	}
	fn get_y(&self) -> i32 {
		self.y
	}
	fn get_z(&self) -> i32 {
		self.z
	}
}

pub trait Point2f64Coords {
	fn get_x(&self) -> f64;
	fn get_y(&self) -> f64;
}

impl Point2f64Coords for Point2<f64> {
	fn get_x(&self) -> f64 {
		self.x
	}
	fn get_y(&self) -> f64 {
		self.y
	}
}

pub trait Point3f64Coords {
	fn get_x(&self) -> f64;
	fn get_y(&self) -> f64;
	fn get_z(&self) -> f64;
}

impl Point3f64Coords for Point3<f64> {
	fn get_x(&self) -> f64 {
		self.x
	}
	fn get_y(&self) -> f64 {
		self.y
	}
	fn get_z(&self) -> f64 {
		self.z
	}
}
