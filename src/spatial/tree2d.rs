use crate::triangulation::PolyVert;
use crate::spatial::aabb::Box2D;
use crate::util::num_convert::OrderedF64;
use nalgebra::Point2;

///Not really a proper KD-tree, but a kd tree with k = 2 and alternating x/y
///partition.
///Recursive sorting is not the most efficient, but simple and guaranteed to
///result in a balanced tree.
pub fn build_2d_tree(points: &mut [PolyVert]) {
	// don't even bother...
	if points.len() <= 8 {
		return;
	}
	build_2d_tree_impl(points, true);
}

fn build_2d_tree_impl(points: &mut [PolyVert], sort_x: bool) {
	if sort_x {
		points.sort_by_key(|vert| OrderedF64(vert.pos.x));
	} else {
		points.sort_by_key(|vert| OrderedF64(vert.pos.y));
	}

	let len = points.len();
	if len < 2 {
		return;
	}
	build_2d_tree_impl(&mut points[..len / 2], !sort_x);
	build_2d_tree_impl(&mut points[len / 2 + 1..], !sort_x);
}

pub fn query_2d_tree(points: &[PolyVert], r: Box2D, mut f: impl FnMut(&PolyVert)) {
	if points.len() <= 8 {
		for p in points {
			if r.contains(&p.pos) {
				f(p);
			}
		}

		return;
	}

	let mut current = Box2D::default();
	current.min = Point2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
	current.max = Point2::new(f64::INFINITY, f64::INFINITY);

	let mut level = 0;
	let mut current_view = points;
	let mut rect_stack = [Box2D::default(); 64];
	let mut view_stack: [&[PolyVert]; 64] = [&[]; 64];
	let mut level_stack = [0; 64];
	let mut stack_pointer = 0;

	loop {
		if current_view.len() <= 8 {
			for p in current_view {
				if r.contains(&p.pos) {
					f(p);
				}
			}

			if stack_pointer == 0 {
				break;
			}
			stack_pointer -= 1;

			level = level_stack[stack_pointer];
			current_view = view_stack[stack_pointer];
			current = rect_stack[stack_pointer];
			continue;
		}

		// these are conceptual left/right trees
		let mut left = current;
		let mut right = current;
		let middle = &current_view[current_view.len() / 2];
		if level % 2 == 0 {
			right.min.x = middle.pos.x;
			left.max.x = middle.pos.x;
		} else {
			right.min.y = middle.pos.y;
			left.max.y = middle.pos.y;
		}

		if r.contains(&middle.pos) {
			f(&middle);
		}
		if left.does_overlap(&r) {
			if right.does_overlap(&r) {
				debug_assert!(stack_pointer < 64, "Stack overflow");
				rect_stack[stack_pointer] = right;
				view_stack[stack_pointer] = &current_view[current_view.len() / 2 + 1..];
				level_stack[stack_pointer] = level + 1;
				stack_pointer += 1;
			}

			current = left;
			current_view = &current_view[0..current_view.len() / 2];
			level += 1;
		} else {
			current = right;
			current_view = &current_view[current_view.len() / 2 + 1..];
			level += 1;
		}
	}
}
