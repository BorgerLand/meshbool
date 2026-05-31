//! Stubs required by the zngur C++ binding layer (test feature only).

use nalgebra::{Point2, Point3, Vector3, Vector4};
use crate::{MeshBool, MeshBoolError, MeshGL32, MeshGL64};
use crate::polygon::PolyVert;

// ── PolyVert accessors ────────────────────────────────────────────────────────
impl PolyVert {
    pub fn get_pos(&self) -> Point2<f64> { self.pos }
    pub fn get_idx(&self) -> i32 { self.idx }
}

// ── MeshBoolError is_* ────────────────────────────────────────────────────────
impl MeshBoolError {
    pub fn is_no_error(&self) -> bool { *self == MeshBoolError::NoError }
    pub fn is_non_finite_vertex(&self) -> bool { *self == MeshBoolError::NonFiniteVertex }
    pub fn is_invalid_construction(&self) -> bool { *self == MeshBoolError::InvalidConstruction }
    pub fn is_result_too_large(&self) -> bool { *self == MeshBoolError::ResultTooLarge }
    pub fn is_not_manifold(&self) -> bool { *self == MeshBoolError::NotManifold }
    pub fn is_missing_position_properties(&self) -> bool { *self == MeshBoolError::MissingPositionProperties }
    pub fn is_merge_vectors_different_lengths(&self) -> bool { *self == MeshBoolError::MergeVectorsDifferentLengths }
    pub fn is_transform_wrong_length(&self) -> bool { *self == MeshBoolError::TransformWrongLength }
    pub fn is_run_index_wrong_length(&self) -> bool { *self == MeshBoolError::RunIndexWrongLength }
    pub fn is_face_id_wrong_length(&self) -> bool { *self == MeshBoolError::FaceIDWrongLength }
    pub fn is_merge_index_out_of_bounds(&self) -> bool { *self == MeshBoolError::MergeIndexOutOfBounds }
    pub fn is_vertex_out_of_bounds(&self) -> bool { *self == MeshBoolError::VertexOutOfBounds }
}

// ── MeshBool stubs ────────────────────────────────────────────────────────────
impl MeshBool {
    pub fn decompose(&self) -> Vec<Self> { vec![self.clone()] }
    pub fn warp_boxed(&self, f: Box<dyn Fn(&mut Point3<f64>)>) -> Self { self.warp(move |p| f(p)) }
    pub fn set_properties_boxed(
        &self, num_prop: i32,
        f: Option<Box<dyn Fn(&mut [f64], Point3<f64>, &[f64])>>,
    ) -> Self {
        self.set_properties(num_prop, f.map(|f| move |out: &mut [f64], pos: Point3<f64>, old: &[f64]| f(out, pos, old)))
    }
    pub fn refine(&self, _n: i32) -> Self { self.clone() }
    pub fn refine_to_length(&self, _l: f64) -> Self { self.clone() }
    pub fn refine_to_tolerance(&self, _t: f64) -> Self { self.clone() }
    pub fn from_meshgl_32(m: &MeshGL32) -> Self { Self::from_meshgl(m) }
    pub fn from_meshgl_64(m: &MeshGL64) -> Self { Self::from_meshgl(m) }
}

// ── Nalgebra coordinate accessors via separate traits per type ────────────────
// Each type gets its own trait to avoid ambiguity.
pub trait Vector3i32Coords { fn get_x(&self) -> i32; fn get_y(&self) -> i32; fn get_z(&self) -> i32; }
pub trait Vector4f64Coords { fn get_x(&self) -> f64; fn get_y(&self) -> f64; fn get_z(&self) -> f64; fn get_w(&self) -> f64; }
pub trait Vector4i32Coords { fn get_x(&self) -> i32; fn get_y(&self) -> i32; fn get_z(&self) -> i32; fn get_w(&self) -> i32; }
pub trait Point2f64Coords  { fn get_x(&self) -> f64; fn get_y(&self) -> f64; }
pub trait Point3f64Coords  { fn get_x(&self) -> f64; fn get_y(&self) -> f64; fn get_z(&self) -> f64; }

impl Vector3i32Coords for Vector3<i32> { fn get_x(&self) -> i32 { self.x } fn get_y(&self) -> i32 { self.y } fn get_z(&self) -> i32 { self.z } }
impl Vector4f64Coords for Vector4<f64> { fn get_x(&self) -> f64 { self.x } fn get_y(&self) -> f64 { self.y } fn get_z(&self) -> f64 { self.z } fn get_w(&self) -> f64 { self.w } }
impl Vector4i32Coords for Vector4<i32> { fn get_x(&self) -> i32 { self.x } fn get_y(&self) -> i32 { self.y } fn get_z(&self) -> i32 { self.z } fn get_w(&self) -> i32 { self.w } }
impl Point2f64Coords  for Point2<f64>  { fn get_x(&self) -> f64 { self.x } fn get_y(&self) -> f64 { self.y } }
impl Point3f64Coords  for Point3<f64>  { fn get_x(&self) -> f64 { self.x } fn get_y(&self) -> f64 { self.y } fn get_z(&self) -> f64 { self.z } }
