use crate::mesh_relations::reserve_original_id;
use crate::meshgl::MeshGL;
use crate::postprocessing::sort::get_tri_box_morton;
use crate::triangulation::PolyVert;
use crate::util::num_convert::LossyFrom;
use crate::util::segment_resolution::SegmentResolution;
use crate::util::tri_dst::distance_triangle_triangle_squared;
use crate::*;
use nalgebra::{Matrix3x4, Point2, Point3, Vector2, Vector3};
use std::backtrace::Backtrace;
use std::mem;
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

pub fn get_intermediate_checks() -> bool {
	INTERMEDIATE_CHECKS.load(Ordering::Relaxed)
}

fn set_intermediate_checks(value: bool) {
	INTERMEDIATE_CHECKS.store(value, Ordering::Relaxed);
}

pub fn get_self_intersection_checks() -> bool {
	SELF_INTERSECTION_CHECKS.load(Ordering::Relaxed)
}

fn set_self_intersection_checks(value: bool) {
	SELF_INTERSECTION_CHECKS.store(value, Ordering::Relaxed);
}

pub fn get_process_overlaps() -> bool {
	PROCESS_OVERLAPS.load(Ordering::Relaxed)
}

fn set_process_overlaps(value: bool) {
	PROCESS_OVERLAPS.store(value, Ordering::Relaxed);
}

//need a wrapper because c++ doesn't have separate types for
//meshes and expressions, and includes a status field instead
//of using result<> equivalent
#[derive(Clone, Debug)]
enum MeshBoolTestWrapper {
	Leaf(MeshBool, MeshBoolError),
	Node(CSGExpression),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum MeshBoolError {
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
	fn is_no_error(self) -> bool {
		self == MeshBoolError::NoError
	}
	fn is_non_finite_vertex(self) -> bool {
		self == MeshBoolError::NonFiniteVertex
	}
	fn is_not_manifold(self) -> bool {
		self == MeshBoolError::NotManifold
	}
	fn is_vertex_out_of_bounds(self) -> bool {
		self == MeshBoolError::VertexOutOfBounds
	}
	fn is_properties_wrong_length(self) -> bool {
		self == MeshBoolError::PropertiesWrongLength
	}
	fn is_missing_position_properties(self) -> bool {
		self == MeshBoolError::MissingPositionProperties
	}
	fn is_merge_vectors_different_lengths(self) -> bool {
		self == MeshBoolError::MergeVectorsDifferentLengths
	}
	fn is_merge_index_out_of_bounds(self) -> bool {
		self == MeshBoolError::MergeIndexOutOfBounds
	}
	fn is_transform_wrong_length(self) -> bool {
		self == MeshBoolError::TransformWrongLength
	}
	fn is_run_index_wrong_length(self) -> bool {
		self == MeshBoolError::RunIndexWrongLength
	}
	fn is_face_id_wrong_length(self) -> bool {
		self == MeshBoolError::FaceIDWrongLength
	}
	fn is_invalid_construction(self) -> bool {
		self == MeshBoolError::InvalidConstruction
	}
	fn is_result_too_large(self) -> bool {
		self == MeshBoolError::ResultTooLarge
	}
	fn is_invalid_tangents(self) -> bool {
		self == MeshBoolError::InvalidTangents
	}
	fn is_cancelled(self) -> bool {
		self == MeshBoolError::Cancelled
	}
}

impl From<MeshGLError> for MeshBoolError {
	fn from(oops: MeshGLError) -> Self {
		match oops {
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
	fn from(oops: BooleanError) -> Self {
		match oops {
			BooleanError::ResultTooLarge => MeshBoolError::ResultTooLarge,
		}
	}
}

impl From<ConstructorError> for MeshBoolError {
	fn from(oops: ConstructorError) -> Self {
		match oops {
			ConstructorError::InvalidConstruction => MeshBoolError::InvalidConstruction,
		}
	}
}

impl Default for MeshBoolTestWrapper {
	fn default() -> Self {
		Self::Leaf(MeshBool::default(), MeshBoolError::default())
	}
}

impl Default for MeshBool {
	fn default() -> Self {
		Self {
			original_id: Some(reserve_original_id()),
			precision: Precision {
				epsilon: -1.0,
				tolerance: -1.0,
			},
			vert_pos: Rc::new(Vec::default()),
			properties: Rc::new(Properties::default()),
			tri: Triangles::default(),
			instance_relation: Rc::new(Vec::default()),
			collider: Rc::new(BVHCollider::default()),
		}
	}
}

impl From<MeshBool> for MeshBoolTestWrapper {
	fn from(mesh: MeshBool) -> Self {
		Self::Leaf(mesh, MeshBoolError::default())
	}
}

impl From<CSGExpression> for MeshBoolTestWrapper {
	fn from(expr: CSGExpression) -> Self {
		Self::Node(expr)
	}
}

impl<E: Into<MeshBoolError>> From<E> for MeshBoolTestWrapper {
	fn from(oops: E) -> Self {
		let ret = Self::Leaf(MeshBool::default(), oops.into());
		ret.warn_status();
		ret
	}
}

impl<E: Into<MeshBoolError>> From<Result<MeshBool, E>> for MeshBoolTestWrapper {
	fn from(result: Result<MeshBool, E>) -> Self {
		match result {
			Ok(mesh) => mesh.into(),
			Err(oops) => oops.into().into(),
		}
	}
}

impl MeshBoolTestWrapper {
	fn eval(&mut self) -> &MeshBool {
		self.warn_status();
		if let Self::Node(expr) = self {
			*self = mem::replace(expr, CSGExpression::temporary_dud())
				.eval()
				.into()
		}

		match self {
			MeshBoolTestWrapper::Leaf(mesh, _) => mesh,
			_ => unreachable!(),
		}
	}

	fn into_expr(&self) -> CSGExpression {
		self.warn_status();
		match self {
			Self::Leaf(mesh, _) => mesh.clone().into(),
			Self::Node(expr) => expr.clone(),
		}
	}

	fn warn_status(&self) {
		if let &Self::Leaf(_, oops) = self
			&& oops != MeshBoolError::NoError
		{
			eprintln!("ERROR STATUS ({:?}), Result will be meaningless.", oops);
			eprintln!("{}", Backtrace::capture());
		}
	}

	fn invalid() -> Self {
		Self::Leaf(MeshBool::default(), MeshBoolError::InvalidConstruction)
	}

	fn tetrahedron() -> Self {
		MeshBool::tetrahedron().into()
	}

	fn cube(size: Vector3<f64>, center: bool) -> Self {
		MeshBool::cube(size, center).into()
	}

	fn cylinder(
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

	fn sphere(radius: f64, circular_segments: i32) -> Self {
		let quality = SegmentResolution {
			circular_segments: circular_segments as u32,
			..SegmentResolution::default()
		};
		MeshBool::sphere(radius, quality).into()
	}

	fn extrude(
		cross_section: &Polygons,
		height: f64,
		n_divisions: u32,
		twist_degrees: f64,
		scale_top: Vector2<f64>,
	) -> Self {
		MeshBool::extrude(cross_section, height, n_divisions, twist_degrees, scale_top).into()
	}

	fn revolve(cross_section: &Polygons, circular_segments: i32, revolve_degrees: f64) -> Self {
		let quality = SegmentResolution {
			circular_segments: circular_segments as u32,
			..SegmentResolution::default()
		};
		MeshBool::revolve(cross_section, quality, revolve_degrees).into()
	}

	fn reserve_ids(n: u32) -> u32 {
		let first = reserve_original_id();
		for _ in 1..n {
			reserve_original_id();
		}
		first
	}

	fn from_meshgl<F, I>(gl: &MeshGL<F, I>) -> Self
	where
		F: LossyFrom<f64> + Copy + 'static,
		f64: From<F>,
		I: LossyFrom<usize> + Copy,
		usize: LossyFrom<I>,
		i32: LossyFrom<I>,
	{
		MeshBool::try_from(gl).into()
	}

	//---- deferred ops: build the tree, evaluate nothing ----

	fn translate(&self, offset: Vector3<f64>) -> Self {
		self.into_expr().translate(offset).into()
	}

	fn scale(&self, scale: Vector3<f64>) -> Self {
		self.into_expr().scale(scale).into()
	}

	fn rotate(&self, x_degrees: f64, y_degrees: f64, z_degrees: f64) -> Self {
		self.into_expr()
			.rotate(x_degrees, y_degrees, z_degrees)
			.into()
	}

	fn transform(&self, transform: Matrix3x4<f64>) -> Self {
		self.into_expr().transform(transform).into()
	}

	fn mirror(&self, normal: Vector3<f64>) -> Self {
		self.into_expr().mirror(normal).into()
	}

	fn boolean(&self, other: &Self, op: OpType) -> Self {
		let lhs = self.into_expr();
		let rhs = other.into_expr();
		(match op {
			OpType::Add => lhs + rhs,
			OpType::Subtract => lhs - rhs,
			OpType::Intersect => lhs ^ rhs,
		})
		.into()
	}

	//---- forcing ops: collapse, then consume an owned copy of the leaf ----

	fn simplify(&mut self, tolerance: f64) -> Self {
		self.eval()
			.clone()
			.simplify(if tolerance == 0.0 {
				None
			} else {
				Some(tolerance)
			})
			.into()
	}

	fn as_original(&mut self) -> Self {
		self.eval().clone().as_original().into()
	}

	fn set_properties(
		&mut self,
		prop_stride: usize,
		prop_func: Option<Box<dyn FnMut(&mut [f64], Point3<f64>, &[f64])>>,
	) -> Self {
		self.eval()
			.clone()
			.set_properties(prop_stride, prop_func)
			.into()
	}

	fn warp(&mut self, warp_func: Box<dyn FnMut(&mut Point3<f64>)>) -> Self {
		self.eval().clone().warp(warp_func).into()
	}

	fn warp_batch(&mut self, warp_func: Box<dyn FnMut(&mut [Point3<f64>])>) -> Self {
		self.eval().clone().warp_batch(warp_func).into()
	}

	fn calculate_curvature(&mut self, gaussian_idx: i32, mean_idx: i32) -> Self {
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

		self.eval()
			.clone()
			.calculate_curvature(gaussian, mean)
			.into()
	}

	fn calculate_normals(&mut self, normal_idx: i32, min_sharp_angle: f64) -> Self {
		self.eval()
			.clone()
			.calculate_normals(normal_idx.max(0) as usize, min_sharp_angle)
			.into()
	}

	fn set_tolerance(&mut self, tolerance: f64) -> Self {
		self.eval().clone().set_tolerance(tolerance).into()
	}

	fn decompose(&mut self) -> Vec<Self> {
		self.eval()
			.clone()
			.decompose()
			.into_iter()
			.map(Into::into)
			.collect()
	}

	fn split(&mut self, cutter: &mut Self) -> (Self, Self) {
		let cutter = cutter.eval().clone();
		match self.eval().clone().split(cutter) {
			Ok((a, b)) => (
				Ok::<_, BooleanError>(a).into(),
				Ok::<_, BooleanError>(b).into(),
			),
			Err(error) => (Err(error).into(), Err(error).into()),
		}
	}

	fn split_by_plane(&mut self, normal: Vector3<f64>, origin_offset: f64) -> (Self, Self) {
		match self.eval().clone().split_by_plane(normal, origin_offset) {
			Ok((a, b)) => (
				Ok::<_, BooleanError>(a).into(),
				Ok::<_, BooleanError>(b).into(),
			),
			Err(error) => (Err(error).into(), Err(error).into()),
		}
	}

	fn trim_by_plane(&mut self, normal: Vector3<f64>, origin_offset: f64) -> Self {
		self.eval()
			.clone()
			.trim_by_plane(normal, origin_offset)
			.into()
	}

	//---- forcing queries ----

	fn genus(&mut self) -> i32 {
		self.warn_status();
		self.eval().genus()
	}

	fn surface_area(&mut self) -> f64 {
		self.warn_status();
		self.eval().surface_area()
	}

	fn volume(&mut self) -> f64 {
		self.warn_status();
		self.eval().volume()
	}

	fn original_id(&mut self) -> i32 {
		self.warn_status();
		self.eval().original_id().map(|id| id as i32).unwrap_or(-1)
	}

	fn matches_tri_normals(&mut self) -> bool {
		self.warn_status();
		self.eval().matches_tri_normals()
	}

	fn num_degenerate_tris(&mut self) -> usize {
		self.warn_status();
		self.eval().num_degenerate_tris()
	}

	fn slice(&mut self, height: f64) -> Vec<Vec<Point2<f64>>> {
		self.warn_status();
		self.eval().slice(height)
	}

	fn project(&mut self) -> Vec<Vec<Point2<f64>>> {
		self.warn_status();
		self.eval().project()
	}

	fn get_mesh_gl_32(&mut self, normal_idx: i32) -> MeshGL32 {
		self.warn_status();
		self.eval().to_meshgl::<f32, u32>(normal_idx)
	}

	fn get_mesh_gl_64(&mut self, normal_idx: i32) -> MeshGL64 {
		self.warn_status();
		self.eval().to_meshgl::<f64, u64>(normal_idx)
	}

	fn is_empty(&mut self) -> bool {
		self.warn_status();
		self.eval().is_empty()
	}

	///Forces evaluation: this is the documented observation point for a
	///deferred tree, so an error raised during eval has to surface here.
	fn status(&mut self) -> MeshBoolError {
		self.eval();
		match self {
			MeshBoolTestWrapper::Leaf(_, oops) => *oops,
			_ => unreachable!(),
		}
	}

	fn num_vert(&mut self) -> usize {
		self.warn_status();
		self.eval().num_vert()
	}

	fn num_edge(&mut self) -> usize {
		self.warn_status();
		self.eval().num_edge()
	}

	fn num_tri(&mut self) -> usize {
		self.warn_status();
		self.eval().num_tri()
	}

	fn prop_stride(&mut self) -> usize {
		self.warn_status();
		self.eval().prop_stride()
	}

	fn num_prop_vert(&mut self) -> usize {
		self.warn_status();
		self.eval().num_prop_vert()
	}

	fn bounding_box(&mut self) -> Box3D {
		self.warn_status();
		self.eval().bounding_box()
	}

	fn get_epsilon(&mut self) -> f64 {
		self.warn_status();
		self.eval().get_epsilon()
	}

	fn get_tolerance(&mut self) -> f64 {
		self.warn_status();
		self.eval().get_tolerance()
	}

	fn min_gap(&mut self, other: &mut Self, search_length: f64) -> f64 {
		self.warn_status();
		let other = other.eval().clone();
		self.eval().min_gap(&other, search_length)
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

		let mut intersecting = false;

		self.collider.collisions_from_slice::<true, _>(
			|tri0, tri1| {
				let mut tri_verts0: [Point3<f64>; 3] = [Point3::default(); 3];
				let mut tri_verts1: [Point3<f64>; 3] = [Point3::default(); 3];
				for i in 0..3 {
					tri_verts0[i] = self.vert_pos[self.tri.halfedge.start[3 * tri0 + i] as usize];
					tri_verts1[i] = self.vert_pos[self.tri.halfedge.start[3 * tri1 + i] as usize];
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
						tmp0[i] = tri_verts0[i] + ep * self.tri.normal[tri1];
					}
					if distance_triangle_triangle_squared(&tmp0, &tri_verts1) > 0.0 {
						return;
					}
					for i in 0..3 {
						tmp0[i] = tri_verts0[i] - ep * self.tri.normal[tri1];
					}
					if distance_triangle_triangle_squared(&tmp0, &tri_verts1) > 0.0 {
						return;
					}
					for i in 0..3 {
						tmp1[i] = tri_verts1[i] + ep * self.tri.normal[tri0];
					}
					if distance_triangle_triangle_squared(&tri_verts0, &tmp1) > 0.0 {
						return;
					}
					for i in 0..3 {
						tmp1[i] = tri_verts1[i] - ep * self.tri.normal[tri0];
					}
					if distance_triangle_triangle_squared(&tri_verts0, &tmp1) > 0.0 {
						return;
					}

					intersecting = true;
				}
			},
			&tri_box,
			true,
		);

		intersecting
	}
}

struct Quality;

impl Quality {
	fn get_circular_segments(radius: f64) -> u32 {
		SegmentResolution::default().get_circular_segments(radius)
	}
}

impl PolyVert {
	fn new(pos: Point2<f64>, idx: i32) -> Self {
		Self { pos, idx }
	}
}

type MeshGL32 = MeshGL<f32, u32>;
type MeshGL64 = MeshGL<f64, u64>;

///`to_meshgl` narrows the library's internal f64 into the output float type,
///so it bounds that type on `LossyFrom<f64>`. `MeshGL64` instantiates it at
///`F = f64`, where the narrowing is the identity. Kept here rather than in
///num_convert because only the bindings need it. Note this is the f64 -> f64
///identity only: f32 -> f64 is an upsize, is lossless, and is deliberately
///absent so it goes through the standard From/Into impls instead.
impl LossyFrom<f64> for f64 {
	fn lossy_from(other: f64) -> Self {
		other
	}
}

/// `merge` is a reserved keyword in zngur, so expose it under a different name.
impl<F, I> MeshGL<F, I>
where
	F: LossyFrom<f64> + Copy + 'static,
	I: LossyFrom<usize> + Copy,
	usize: LossyFrom<I>,
	u64: LossyFrom<I>,
	i32: LossyFrom<I>,
	f64: From<F>,
{
	fn merge_glp(&mut self) -> bool {
		self.merge()
	}
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OpType {
	Add,
	Subtract,
	Intersect,
}

trait Vector3i32Coords {
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

trait Point2f64Coords {
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

trait Point3f64Coords {
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
