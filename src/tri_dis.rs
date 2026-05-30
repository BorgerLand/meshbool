use nalgebra::Vector3;
// From NVIDIA-Omniverse PhysX - BSD 3-Clause "New" or "Revised" License
// https://github.com/NVIDIA-Omniverse/PhysX/blob/main/LICENSE.md
// https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/source/geomutils/src/sweep/GuSweepCapsuleCapsule.cpp
// With minor modifications

///Returns the distance between two line segments.
///
///@param[out] x Closest point on line segment pa.
///@param[out] y Closest point on line segment qb.
///@param[in]  p  One endpoint of the first line segment.
///@param[in]  a  Other endpoint of the first line segment.
///@param[in]  p  One endpoint of the second line segment.
///@param[in]  b  Other endpoint of the second line segment.
#[inline]
fn edge_edge_dist(
	x: &mut Vector3<f64>,
	y: &mut Vector3<f64>, // closest points
	p: &Vector3<f64>,
	a: &Vector3<f64>, // seg 1 origin, vector
	q: &Vector3<f64>,
	b: &Vector3<f64>,
) // seg 2 origin, vector
{
	let t: Vector3<f64> = q - p;
	let a_dot_a = a.dot(a);
	let b_dot_b = b.dot(b);
	let a_dot_b = a.dot(b);
	let a_dot_t = a.dot(&t);
	let b_dot_t = b.dot(&t);

	// t parameterizes ray (p, a)
	// u parameterizes ray (q, b)

	// Compute t for the closest point on ray (p, a) to ray (q, b)
	let denom = a_dot_a * b_dot_b - a_dot_b * a_dot_b;

	let mut t: f64; // We will clamp result so t is on the segment (p, a)
	t = if denom != 0.0 {
		f64::clamp((a_dot_t * b_dot_b - b_dot_t * a_dot_b) / denom, 0.0, 1.0)
	} else {
		0.0
	};

	// find u for point on ray (q, b) closest to point at t
	let mut u: f64;
	if b_dot_b != 0.0 {
		u = (t * a_dot_b - b_dot_t) / b_dot_b;

		// if u is on segment (q, b), t and u correspond to closest points,
		// otherwise, clamp u, recompute and clamp t
		if u < 0.0 {
			u = 0.0;
			t = if a_dot_a != 0.0 {
				f64::clamp(a_dot_t / a_dot_a, 0.0, 1.0)
			} else {
				0.0
			};
		} else if u > 1.0 {
			u = 1.0;
			t = if a_dot_a != 0.0 {
				f64::clamp((a_dot_b + a_dot_t) / a_dot_a, 0.0, 1.0)
			} else {
				0.0
			};
		}
	} else {
		u = 0.0;
		t = if a_dot_a != 0.0 {
			f64::clamp(a_dot_t / a_dot_a, 0.0, 1.0)
		} else {
			0.0
		};
	}
	*x = p + a * t;
	*y = q + b * u;
}

// From NVIDIA-Omniverse PhysX - BSD 3-Clause "New" or "Revised" License
// https://github.com/NVIDIA-Omniverse/PhysX/blob/main/LICENSE.md
// https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/source/geomutils/src/distance/GuDistanceTriangleTriangle.cpp
// With minor modifications

///Returns the minimum squared distance between two triangles.
///
///@param  p  First  triangle.
///@param  q  Second triangle.
#[inline]
pub fn distance_triangle_triangle_squared(p: &[Vector3<f64>; 3], q: &[Vector3<f64>; 3]) -> f64 {
	let s_v: [Vector3<f64>; 3] = [p[1] - p[0], p[2] - p[1], p[0] - p[2]];

	let t_v: [Vector3<f64>; 3] = [q[1] - q[0], q[2] - q[1], q[0] - q[2]];

	let mut shown_disjoint = false;

	let mut mindd = f64::MAX;

	for i in 0..3 {
		for j in 0..3 {
			let mut cp: Vector3<f64> = Default::default();
			let mut cq: Vector3<f64> = Default::default();
			edge_edge_dist(&mut cp, &mut cq, &p[i], &s_v[i], &q[j], &t_v[j]);
			let v: Vector3<f64> = cq - cp;
			let dd = v.dot(&v);

			if dd <= mindd {
				mindd = dd;

				let mut id = i + 2;
				if id >= 3 {
					id -= 3;
				}
				let mut z: Vector3<f64> = p[id] - cp;
				let mut a = z.dot(&v);
				id = j + 2;
				if id >= 3 {
					id -= 3;
				}
				z = q[id] - cq;
				let mut b = z.dot(&v);

				if (a <= 0.0) && (b >= 0.0) {
					return v.dot(&v);
				};

				if a <= 0.0 {
					a = 0.0;
				} else if b > 0.0 {
					b = 0.0;
				}

				if (mindd - a + b) > 0.0 {
					shown_disjoint = true;
				}
			}
		}
	}

	let s_n: Vector3<f64> = s_v[0].cross(&s_v[1]);
	let s_nl = s_n.dot(&s_n);

	if s_nl > 1e-15 {
		let t_p = Vector3::<f64>::new(
			(p[0] - q[0]).dot(&s_n),
			(p[0] - q[1]).dot(&s_n),
			(p[0] - q[2]).dot(&s_n),
		);

		let mut index: i32 = -1;
		if (t_p[0] > 0.0) && (t_p[1] > 0.0) && (t_p[2] > 0.0) {
			index = if t_p[0] < t_p[1] { 0 } else { 1 };
			if t_p[2] < t_p[index as usize] {
				index = 2;
			}
		} else if (t_p[0] < 0.0) && (t_p[1] < 0.0) && (t_p[2] < 0.0) {
			index = if t_p[0] > t_p[1] { 0 } else { 1 };
			if t_p[2] > t_p[index as usize] {
				index = 2;
			}
		}

		if index >= 0 {
			shown_disjoint = true;

			let q_index: &Vector3<f64> = &q[index as usize];

			let mut v: Vector3<f64> = q_index - p[0];
			let mut z: Vector3<f64> = s_n.cross(&s_v[0]);
			if v.dot(&z) > 0.0 {
				v = q_index - p[1];
				z = s_n.cross(&s_v[1]);
				if v.dot(&z) > 0.0 {
					v = q_index - p[2];
					z = s_n.cross(&s_v[2]);
					if v.dot(&z) > 0.0 {
						let cp: Vector3<f64> = q_index + s_n * t_p[index as usize] / s_nl;
						let cq: Vector3<f64> = *q_index;
						return (cp - cq).dot(&(cp - cq));
					}
				}
			}
		}
	}

	let t_n: Vector3<f64> = t_v[0].cross(&t_v[1]);
	let t_nl = t_n.dot(&t_n);

	if t_nl > 1e-15 {
		let s_p = Vector3::<f64>::new(
			(q[0] - p[0]).dot(&t_n),
			(q[0] - p[1]).dot(&t_n),
			(q[0] - p[2]).dot(&t_n),
		);

		let mut index: i32 = -1;
		if (s_p[0] > 0.0) && (s_p[1] > 0.0) && (s_p[2] > 0.0) {
			index = if s_p[0] < s_p[1] { 0 } else { 1 };
			if s_p[2] < s_p[index as usize] {
				index = 2;
			}
		} else if (s_p[0] < 0.0) && (s_p[1] < 0.0) && (s_p[2] < 0.0) {
			index = if s_p[0] > s_p[1] { 0 } else { 1 };
			if s_p[2] > s_p[index as usize] {
				index = 2;
			}
		}

		if index >= 0 {
			shown_disjoint = true;

			let p_index = &p[index as usize];

			let mut v: Vector3<f64> = p_index - q[0];
			let mut z: Vector3<f64> = t_n.cross(&t_v[0]);
			if v.dot(&z) > 0.0 {
				v = p_index - q[1];
				z = t_n.cross(&t_v[1]);
				if v.dot(&z) > 0.0 {
					v = p_index - q[2];
					z = t_n.cross(&t_v[2]);
					if v.dot(&z) > 0.0 {
						let cp: Vector3<f64> = *p_index;
						let cq: Vector3<f64> = p_index + t_n * s_p[index as usize] / t_nl;
						return (cp - cq).dot(&(cp - cq));
					}
				}
			}
		}
	}

	return if shown_disjoint { mindd } else { 0.0 };
}
