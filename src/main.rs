const EXECUTABLE: &[u8] = include_bytes!("../resources/survarium.exe");
const DEBUG_SYMBOLS: &[u8] = include_bytes!("../resources/survarium.pdb");

fn main() {
    let pdb = pdb2::PDB::open(std::io::Cursor::new(DEBUG_SYMBOLS)).unwrap();
    process_executable(pdb);
}

fn process_executable<S>(pdb: pdb2::PDB<'static, S>) {}
