#[cfg(feature = "test")]
use {
	std::{fs, path},
	zngur::Zngur,
};

fn main() {
	#[cfg(feature = "test")]
	{
		build_rs::output::rerun_if_changed("Cargo.toml");
		build_rs::output::rerun_if_changed("test/test.zng");

		let crate_dir = build_rs::input::cargo_manifest_dir();
		let generated_dir = path::PathBuf::from("test/generated");
		let out_dir = build_rs::input::out_dir();

		fs::create_dir_all(&generated_dir).unwrap();
		let rs_file = out_dir.join("test.rs");

		Zngur::from_zng_file(crate_dir.join("test/test.zng"))
			.with_cpp_file(generated_dir.join("test.cpp"))
			.with_h_file(generated_dir.join("test.h"))
			.with_rs_file(rs_file.clone())
			.with_zng_header(generated_dir.join("zngur.h"))
			.generate();
	}
}
