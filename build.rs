#[cfg(feature = "test")]
use {
	std::{fs, path},
	zngur::Zngur,
};

fn main() {
	#[cfg(feature = "test")]
	{
		build_rs::output::rerun_if_changed("Cargo.toml");
		let crate_dir = build_rs::input::cargo_manifest_dir();
		let out_dir = build_rs::input::out_dir();

		build_rs::output::rerun_if_changed("test/test.zng");

		let generated_cpp = path::PathBuf::from("test/generated/meshbool/");
		let _ = fs::create_dir_all(&generated_cpp);
		let rs_file = out_dir.join("generated.rs");
		let h_file = generated_cpp.join("meshbool.h");

		Zngur::from_zng_file(crate_dir.join("test/test.zng"))
			.with_cpp_file(generated_cpp.join("generated.cpp"))
			.with_h_file(h_file)
			.with_rs_file(rs_file.clone())
			.generate();
		let s = fs::read_to_string(&rs_file).expect("File should exist");
		let new = s.replace("#[no_mangle]", "#[unsafe(no_mangle)]");
		fs::write(rs_file, new).expect("Failed to write file");
	}
}
