use std::f64;

///@brief These static properties control how circular shapes are quantized by
///default on construction.
///
///If circularSegments is specified, it takes
///precedence. If it is zero, then instead the minimum is used of the segments
///calculated based on edge length and angle, rounded up to the nearest
///multiple of four. To get numbers not divisible by four, circularSegments
///must be specified.
#[derive(Copy, Clone, PartialEq)]
pub struct SegmentResolution {
	///default number of circular segments for the
	///CrossSection::Circle(), Manifold::Cylinder(), Manifold::Sphere(), and
	///Manifold::Revolve() constructors. Overrides the edge length and angle
	///constraints and sets the number of segments to exactly this value.
	///
	///@param number Number of circular segments. Default is 0, meaning no
	///constraint is applied.
	pub circular_segments: u32,

	///angle constraint the default number of circular segments for the
	///CrossSection::Circle(), Manifold::Cylinder(), Manifold::Sphere(), and
	///Manifold::Revolve() constructors. The number of segments will be rounded up
	///to the nearest factor of four.
	///
	///@param angle The minimum angle in degrees between consecutive segments. The
	///angle will increase if the the segments hit the minimum edge length.
	///Default is 10 degrees.
	pub min_circular_angle: f64,

	///length constraint the default number of circular segments for the
	///CrossSection::Circle(), Manifold::Cylinder(), Manifold::Sphere(), and
	///Manifold::Revolve() constructors. The number of segments will be rounded up
	///to the nearest factor of four.
	///
	///@param length The minimum length of segments. The length will
	///increase if the the segments hit the minimum angle. Default is 1.0.
	pub min_circular_edge_length: f64,
}

impl Default for SegmentResolution {
	fn default() -> Self {
		Self {
			circular_segments: 0,
			min_circular_angle: 10.0,
			min_circular_edge_length: 1.0,
		}
	}
}

impl SegmentResolution {
	///Determine the result of the SetMinCircularAngle(),
	///SetMinCircularEdgeLength(), and SetCircularSegments() defaults.
	///
	///@param radius For a given radius of circle, determine how many default
	///segments there will be.
	pub fn get_circular_segments(&self, radius: f64) -> u32 {
		if self.circular_segments > 0 {
			return self.circular_segments;
		}
		let n_seg_a = 360.0 / self.min_circular_angle;
		// Keep nSegL a double so the truncating cast happens after fmin bounds it by
		// nSegA; a raw int cast is undefined for non-finite or huge radius.
		let n_seg_l = 2.0 * radius.abs() * f64::consts::PI / self.min_circular_edge_length;
		let mut n_seg = (n_seg_a.min(n_seg_l) + 3.0) as u32;
		n_seg -= n_seg % 4;
		n_seg.max(4)
	}
}
