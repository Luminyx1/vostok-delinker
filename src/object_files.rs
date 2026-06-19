use crate::pdb_symbols::PdbSymbols;
use crate::relocs::RelocKind;
use crate::symbol_matcher::{canonical_name, SymbolMatcher};
use crate::utils::{leak, ToU64, ToUsize};
use crate::Env;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use iced_x86::{Decoder, DecoderOptions, FlowControl, Instruction, OpKind};

use pdb2::{FallibleIterator, RawString};

use object::SectionKind;

pub struct ObjectFiles<'a> {
    pub objects: HashMap<&'a [u8], ObjectFile>,
}

pub struct ObjectFile {
    pub object: object::write::Object<'static>,
    pub data_section_id: object::write::SectionId,
    pub rdata_section_id: object::write::SectionId,
    pub text_section_id: object::write::SectionId,
}

#[derive(Copy, Clone)]
pub struct ObjectOffset {
    offset: u64,
    section_id: object::write::SectionId,
}

#[derive(Copy, Clone)]
pub enum ObjectLocation {
    Offset(ObjectOffset),
    Extern,
}

impl ObjectFiles<'_> {
    pub fn parse<'s, S>(
        env: &Env,
        pdb: &mut pdb2::PDB<'static, S>,

        symbols: &'s PdbSymbols,
        coff_data: &[u8],
        mut relocs_rva: BTreeMap<usize, RelocKind<'s>>,

        pad_empty_rdata: bool,
        matcher: &SymbolMatcher,
    ) -> anyhow::Result<Self>
    where
        S: pdb2::Source<'static> + 'static,
    {
        let mut lib_sources: BTreeMap<Vec<u8>, BTreeSet<Vec<u8>>> = BTreeMap::new();
        let mut module_libs: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

        {
            let mut modules = env.dbi.modules()?;
            while let Some(module) = modules.next()? {
                let lib = lib_name_from_bytes(module.object_file_name().as_bytes());

                let Ok(Some(module_info)) = pdb.module_info(&module) else {
                    module_libs.push((lib, Vec::new()));
                    continue;
                };
                let module_info = leak(module_info);
                let program = module_info.line_program()?;
                let mut iter = module_info.symbols()?;

                while let Some(symbol) = iter.next()? {
                    let (_, fun_offset) = match symbol.parse() {
                        Ok(pdb2::SymbolData::Procedure(p)) => (p.name, p.offset),
                        Ok(pdb2::SymbolData::Thunk(t)) => (t.name, t.offset),
                        _ => continue,
                    };
                    if let Some(src) = get_source_file(&program, env.string_table, fun_offset)?
                    {
                        lib_sources.entry(lib.clone()).or_default().insert(normalise(src));
                    }
                }

                module_libs.push((lib, Vec::new()));
            }
        }

        let lib_roots: HashMap<Vec<u8>, Vec<u8>> = lib_sources
            .iter()
            .map(|(lib, sources)| {
                let root = common_project_root(sources);
                (lib.clone(), root)
            })
            .collect();

        let collapse_depths: HashMap<Vec<u8>, usize> = lib_sources
            .iter()
            .map(|(lib, sources)| {
                let root = lib_roots.get(lib).unwrap();
                let relative_paths: BTreeSet<Vec<u8>> = sources
                    .iter()
                    .filter(|src| is_translation_unit(src))
                    .filter(|src| !is_likely_primary_source(src, lib))
                    .filter_map(|src| {
                        let r = src.strip_prefix(root.as_slice())?;
                        if r.is_empty() { None } else { Some(r.to_vec()) }
                    })
                    .collect();
                (lib.clone(), path_chain_depth(lib, &relative_paths))
            })
            .collect();

        for (lib, root) in &mut module_libs {
            if let Some(r) = lib_roots.get(lib) {
                *root = r.clone();
            }
        }

        let mut this = Self {
            objects: HashMap::new(),
        };

        let mut module_idx = 0usize;
        let mut modules = env.dbi.modules()?;
        while let Some(module) = modules.next()? {
            let (lib_name, project_root) = &module_libs[module_idx];
            module_idx += 1;

            if is_system_lib(lib_name) {
                continue;
            }

            let Some(module_info) = pdb.module_info(&module)? else {
                continue;
            };
            let module_info = leak(module_info);

            let program = module_info.line_program()?;
            let mut iter = module_info.symbols()?;

            while let Some(symbol) = iter.next()? {
                let (fun_name, fun_offset, fun_size) = match symbol.parse() {
                    Ok(pdb2::SymbolData::Procedure(pdb2::ProcedureSymbol {
                        name,
                        offset,
                        len,
                        ..
                    })) => (name, offset, len.to_usize()),
                    Ok(pdb2::SymbolData::Thunk(pdb2::ThunkSymbol {
                        name, offset, len, ..
                    })) => (name, offset, len.to_usize()),
                    _ => continue,
                };

                let source_file = get_source_file(
                    &program,
                    env.string_table,
                    fun_offset,
                )?;

                let object_key: &'static [u8] = match source_file {
                    Some(src) => {
                        let normalised = normalise(src);
                        if !is_translation_unit(&normalised) {
                            continue;
                        }
                        let relative: Vec<u8> = {
                            let r = normalised
                                .as_slice()
                                .strip_prefix(project_root.as_slice())
                                .unwrap_or(normalised.as_slice());
                            if r.is_empty() {
                                normalised
                                    .rsplit(|&b| b == b'\\')
                                    .next()
                                    .unwrap_or(&normalised)
                                    .to_vec()
                            } else if let Some(&depth) = collapse_depths.get(lib_name) {
                                if depth > 0 {
                                    let components: Vec<&[u8]> = r.split(|&b| b == b'\\').collect();
                                    if depth < components.len() {
                                        components[depth..].join(&b'\\')
                                    } else {
                                        r.to_vec()
                                    }
                                } else {
                                    r.to_vec()
                                }
                            } else {
                                r.to_vec()
                            }
                        };
                        let is_primary = is_translation_unit_match(&normalised, lib_name);
                        let mut key = if is_primary {
                            let name = normalised.rsplit(|&b| b == b'\\').next().unwrap_or(&normalised);
                            Vec::with_capacity(name.len())
                        } else {
                            Vec::with_capacity(lib_name.len() + 1 + relative.len())
                        };
                        if is_primary {
                            let name = normalised.rsplit(|&b| b == b'\\').next().unwrap_or(&normalised);
                            key.extend_from_slice(name);
                        } else {
                            key.extend_from_slice(lib_name);
                            key.push(b'\\');
                            key.extend_from_slice(&relative);
                        }
                        key.leak()
                    }
                    None => continue,
                };

                let fun_rva = env.text.rva + fun_offset.offset.to_usize();
                let fun_bytes = resolve_relative_relocations(
                    env,
                    fun_rva,
                    fun_size,
                    symbols,
                    coff_data,
                    &mut relocs_rva,
                )?;

                let object_file = this
                    .objects
                    .entry(object_key)
                    .or_insert_with(|| ObjectFile::empty(pad_empty_rdata));

                let fun_name = match symbols.functions.get(&fun_rva) {
                    Some(overloads) => matcher.pick(overloads, canonical_name(overloads)),
                    _ => fun_name,
                };

                let fun_offset_in_coff_text = object_file.add_function(fun_name, &fun_bytes);

                for (reloc_rva, reloc_kind) in relocs_rva.range(fun_rva..fun_rva + fun_size) {
                    let reloc_rva = *reloc_rva;
                    let reloc_kind = *reloc_kind;

                    let reloc_offset_in_fun = reloc_rva - fun_rva;
                    let reloc_offset_in_coff_text = fun_offset_in_coff_text + reloc_offset_in_fun;

                    // Fresh per top-level reloc (each pointer chain is independent).
                    let mut visited = HashSet::new();
                    object_file.add_relocation_at(
                        reloc_kind,
                        reloc_offset_in_coff_text,
                        matcher,
                        coff_data,
                        &relocs_rva,
                        &mut visited,
                    )?;
                }
            }
        }

        Ok(this)
    }

    pub fn write(self, base: &std::path::Path) -> anyhow::Result<()> {
        let base_len = base.as_os_str().as_encoded_bytes().len();
        let mut path = base.to_path_buf();

        for (prefix, object_file) in self.objects {
            path.as_mut_os_string().truncate(base_len);

            let prefix = prefix
                .iter()
                .map(|&c| match c {
                    b'\\' => '/',
                    _ => char::from(c),
                })
                .collect::<String>();
            path.as_mut_os_string().push("/");
            path.as_mut_os_string().push(&prefix);
            path.as_mut_os_string().push(".obj");

            std::fs::create_dir_all(path.parent().unwrap())?;
            std::fs::write(&path, object_file.object.write()?)?;
        }
        Ok(())
    }
}

impl ObjectFile {
    fn empty(pad_rdata: bool) -> Self {
        let mut object = object::write::Object::new(
            object::BinaryFormat::Coff,
            object::Architecture::I386,
            object::Endianness::Little,
        );
        object.set_mangling(object::write::Mangling::None);

        let data_section_id = object.add_section(vec![], b".data".into(), SectionKind::Data);
        let rdata_section_id =
            object.add_section(vec![], b".rdata".into(), SectionKind::ReadOnlyData);
        let text_section_id = object.add_section(vec![], b".text".into(), SectionKind::Text);

        // objdiff considers allocations to match if name is equal OR(!) offset
        // into reloc table is the same.
        //
        // This makes different relocations with different data and different names
        // to match, if they offsets match. These 4 bytes prevent that.
        if pad_rdata {
            object.append_section_data(rdata_section_id, &0_u32.to_le_bytes(), 4);
        }

        Self {
            object,
            data_section_id,
            rdata_section_id,
            text_section_id,
        }
    }
}

/// Returns the raw source file path for the line containing a given function.
fn get_source_file(
    program: &pdb2::LineProgram<'static>,
    string_table: &'static pdb2::StringTable<'static>,
    fun_offset: pdb2::PdbInternalSectionOffset,
) -> anyhow::Result<Option<&'static [u8]>> {
    let mut lines = program.lines_for_symbol(fun_offset);
    // Extracting only a single line should be enough to find a source file.
    if let Some(line_info) = lines.next()? {
        let file_info = program.get_file_info(line_info.file_index)?;
        let filename = string_table.get(file_info.name)?;
        return Ok(Some(filename.as_bytes()));
    }
    Ok(None)
}

/// Resolve external relative jumps in the function as relocations.
///
/// And return the final resolved function assembly.
// @NOTE: There is no reason to grow `relocs_rva`, since these allocations
// are specific to the current function and don't need to be kept alive after
// the function is processed.
//
// At the same time `relocs_rva` sorts automatically!
fn resolve_relative_relocations<'s>(
    env: &Env,

    fun_rva: usize,
    fun_size: usize,

    symbols: &'s PdbSymbols,

    coff_data: &[u8],
    relocs_rva: &mut BTreeMap<usize, RelocKind<'s>>,
) -> anyhow::Result<Vec<u8>> {
    let fun_va = env.image_base.to_usize() + fun_rva;

    // @NOTE: Requires a new allocation, since capstone cannot borrow function code mutably.
    let mut fun_bytes = coff_data[fun_rva..fun_rva + fun_size].to_vec();

    let code = &coff_data[fun_rva..fun_rva + fun_size];
    let mut decoder = Decoder::with_ip(32, code, fun_va as u64, DecoderOptions::NONE);
    let mut ix = Instruction::default();

    while decoder.can_decode() {
        decoder.decode_out(&mut ix);

        let offset_in_fun = (ix.ip() - fun_va as u64) as usize + ix.len();

        match ix.flow_control() {
            FlowControl::ConditionalBranch
            | FlowControl::UnconditionalBranch
            | FlowControl::Call => {}
            _ => continue,
        }

        let target_va = match ix.op0_kind() {
            OpKind::NearBranch16 => ix.near_branch16() as u64,
            OpKind::NearBranch32 => ix.near_branch32() as u64,
            OpKind::NearBranch64 => unreachable!(),
            _ => continue,
        };

        let target_rva = target_va - u64::from(env.image_base);

        let internal_branch = (fun_rva..fun_rva + fun_size).contains(&(target_rva.to_usize()));
        if internal_branch {
            continue;
        }

        if ix.len() <= 4 {
            // Read data as code. Which is jump tables stored inline.
            continue;
        }

        let Some(overloads) = symbols.functions.get(&target_rva.to_usize()) else {
            // Read data as code. Which is jump tables stored inline.
            continue;
        };

        let overloads = overloads.as_slice();

        fun_bytes[offset_in_fun - 4..offset_in_fun].copy_from_slice(&0_u32.to_le_bytes());
        let old_reloc = relocs_rva.insert(
            fun_rva + offset_in_fun - 4,
            RelocKind::Function { overloads },
        );

        if let Some(old_reloc) = old_reloc {
            let RelocKind::Function {
                overloads: old_overloads,
            } = old_reloc
            else {
                unreachable!();
            };
            assert_eq!(overloads.as_ptr(), old_overloads.as_ptr());
        }
    }

    Ok(fun_bytes)
}

impl ObjectFile {
    fn add_relocation_at(
        &mut self,
        reloc_kind: RelocKind,
        reloc_offset: ObjectOffset,
        //
        matcher: &SymbolMatcher,
        coff_data: &[u8],
        relocs_rva: &BTreeMap<usize, RelocKind>,
        // Target RVAs already expanded on this pointer chain (cycle guard).
        visited: &mut HashSet<usize>,
    ) -> anyhow::Result<()> {
        let reloc_name = reloc_kind.get_name(matcher);
        let reloc_name = reloc_name.as_raw_string();

        match reloc_kind {
            RelocKind::Function { overloads: _ } => {
                self.add_relocation(reloc_name, ObjectLocation::Extern, reloc_offset)?;
            }

            RelocKind::ConstantString { symbol: _, data } => {
                let const_offset_in_coff_rdata =
                    self.append_section_data(self.rdata_section_id, data, 0x00);

                self.add_relocation(
                    reloc_name,
                    ObjectLocation::Offset(const_offset_in_coff_rdata),
                    reloc_offset,
                )?;
            }

            RelocKind::Constant {
                symbol: _,
                target_rva,
            } => {
                let new_data =
                    bytemuck::pod_read_unaligned::<[u8; 4]>(&coff_data[target_rva..target_rva + 4]);
                let const_offset_in_coff_rdata =
                    self.append_section_data(self.rdata_section_id, &new_data, 0x00);
                self.add_relocation(
                    reloc_name,
                    ObjectLocation::Offset(const_offset_in_coff_rdata),
                    reloc_offset,
                )?;

                // Cycle guard for self-referential RVAs.
                if let Some(reloc_kind) = relocs_rva.get(&target_rva) {
                    if visited.insert(target_rva) {
                        self.add_relocation_at(
                            *reloc_kind,
                            const_offset_in_coff_rdata,
                            matcher,
                            coff_data,
                            relocs_rva,
                            visited,
                        )?;
                    }
                }
            }

            RelocKind::Static {
                symbol: _,
                target_rva,
            } => {
                let new_data =
                    bytemuck::pod_read_unaligned::<[u8; 4]>(&coff_data[target_rva..target_rva + 4]);
                let static_offset_in_coff_data =
                    self.append_section_data(self.data_section_id, &new_data, 0x00);
                self.add_relocation(
                    reloc_name,
                    ObjectLocation::Offset(static_offset_in_coff_data),
                    reloc_offset,
                )?;

                // Same cycle guard as the Constant arm above.
                if let Some(reloc_kind) = relocs_rva.get(&target_rva) {
                    if visited.insert(target_rva) {
                        self.add_relocation_at(
                            *reloc_kind,
                            static_offset_in_coff_data,
                            matcher,
                            coff_data,
                            relocs_rva,
                            visited,
                        )?;
                    }
                }
            }
        }

        Ok(())
    }
}

impl ObjectFile {
    fn append_section_data(
        &mut self,
        section_id: object::write::SectionId,
        data: &[u8],
        pad: u8,
    ) -> ObjectOffset {
        let offset = append_with_padding(&mut self.object, section_id, data, pad);
        ObjectOffset { offset, section_id }
    }

    fn add_relocation(
        &mut self,
        name: RawString,
        location: ObjectLocation,
        offset: ObjectOffset,
    ) -> anyhow::Result<()> {
        let (value, kind, section) = match location {
            ObjectLocation::Extern => (
                0,
                object::SymbolKind::Unknown,
                object::write::SymbolSection::Undefined,
            ),
            ObjectLocation::Offset(ObjectOffset { offset, section_id }) => {
                let kind = if section_id == self.text_section_id {
                    object::SymbolKind::Text
                } else {
                    object::SymbolKind::Data
                };
                (
                    offset,
                    kind,
                    object::write::SymbolSection::Section(section_id),
                )
            }
        };

        let symbol = self.object.add_symbol(object::write::Symbol {
            name: name.as_bytes().to_vec(),
            value,
            size: u64::MAX,
            kind,
            scope: object::SymbolScope::Linkage,
            weak: false,
            section,
            flags: object::SymbolFlags::None,
        });

        self.object.add_relocation(
            offset.section_id,
            object::write::Relocation {
                offset: offset.offset,
                symbol,
                addend: -4,
                flags: object::RelocationFlags::Generic {
                    kind: object::RelocationKind::Relative,
                    encoding: object::RelocationEncoding::Generic,
                    size: 32,
                },
            },
        )?;

        Ok(())
    }

    fn add_function(&mut self, name: RawString, body: &[u8]) -> ObjectOffset {
        let fun_offset_in_coff_text = self.append_section_data(self.text_section_id, body, 0x90);

        self.object.add_symbol(object::write::Symbol {
            name: name.as_bytes().to_vec(),
            value: fun_offset_in_coff_text.offset,
            size: u64::MAX,
            kind: object::SymbolKind::Text,
            scope: object::SymbolScope::Linkage,
            weak: false,
            section: object::write::SymbolSection::Section(fun_offset_in_coff_text.section_id),
            flags: object::SymbolFlags::None,
        });

        fun_offset_in_coff_text
    }
}

// Parse PDB symbols by iterating through symbol table and then through all modules

enum Name<'a> {
    Borrowed(RawString<'a>),
    Owned(Vec<u8>),
}

impl<'a> RelocKind<'a> {
    fn get_name(self, matcher: &SymbolMatcher) -> Name<'a> {
        match self {
            Self::Function { overloads } => {
                Name::Borrowed(matcher.pick(overloads, canonical_name(overloads)))
            }
            Self::ConstantString { symbol, data } => {
                let reloc_name = get_constant_name(symbol, data);
                Name::Owned(reloc_name)
            }
            Self::Constant {
                symbol: reloc_name,
                target_rva: _,
            } => Name::Borrowed(reloc_name),
            Self::Static {
                symbol: reloc_name,
                target_rva: _,
            } => Name::Borrowed(reloc_name),
        }
    }
}

impl Name<'_> {
    fn as_raw_string(&self) -> RawString<'_> {
        match self {
            Self::Owned(name) => RawString::from(name.as_slice()),
            Self::Borrowed(name) => *name,
        }
    }
}

impl std::ops::Add<usize> for ObjectOffset {
    type Output = Self;

    fn add(self, rhs: usize) -> Self::Output {
        Self {
            offset: self.offset + rhs.to_u64(),
            section_id: self.section_id,
        }
    }
}

// Always pads to 4
fn append_with_padding(
    object: &mut object::write::Object,
    section_id: object::write::SectionId,
    data: &[u8],
    pad: u8,
) -> u64 {
    let offset = object.append_section_data(section_id, data, 1);

    // sushi@NOTE: `object` crate doesn't(?) allow specifying auxiliary symbols.
    // Because of that 1-3 bytes of garbage are generated which objdiff doesn't like.
    // We replace those bytes with `nop`s and pad all of the functions ourselves,
    // which fixes the problem, but this is a hack, which needs to be fixed at some point.
    let padding: &[u8] = match 4 - data.len() % 4 {
        1 => &[pad],
        2 => &[pad, pad],
        3 => &[pad, pad, pad],
        _ => &[],
    };
    if !padding.is_empty() {
        _ = object.append_section_data(section_id, padding, 1);
    }

    offset
}

fn is_system_lib(name: &[u8]) -> bool {
    matches!(
        name,
        b"kernel32"
            | b"libcmtd"
            | b"libcpmtd"
            | b"shell32"
            | b"user32"
            | b"gdi32"
            | b"advapi32"
            | b"msvcrt"
            | b"msvcrtd"
            | b"oldnames"
            | b"libcmt"
            | b"libcpmt"
            | b"msvcprt"
            | b"msvcprtd"
    )
}

fn is_translation_unit(path: &[u8]) -> bool {
    path.ends_with(b".c")
        || path.ends_with(b".cpp")
        || path.ends_with(b".cc")
        || path.ends_with(b".cxx")
        || path.ends_with(b".asm")
        || path.ends_with(b".s")
}

/// Returns true if the source path's filename stem (without extension) matches
/// the lib_name, indicating a primary translation unit that should be placed
/// directly in the output directory rather than nested under source tree paths.
fn is_translation_unit_match(path: &[u8], lib_name: &[u8]) -> bool {
    let stem = path
        .rsplit(|&b| b == b'\\')
        .next()
        .and_then(|f| {
            let dot = f.iter().rposition(|&b| b == b'.')?;
            Some(&f[..dot])
        })
        .unwrap_or(path);
    if stem.eq_ignore_ascii_case(lib_name) {
        return true;
    }
    // Debug libraries in MSVC get a 'd' suffix (e.g., gx2windowsd). Primary
    // translation units are named without the 'd' (e.g., gx2windows.cpp).
    if let Some(stripped) = lib_name.strip_suffix(b"d").or_else(|| lib_name.strip_suffix(b"D")) {
        stem.eq_ignore_ascii_case(stripped)
    } else {
        false
    }
}

/// Same logic as `is_translation_unit_match` — used to exclude primary-TU
/// paths from per-library chain‑depth computation so they don't pollute it.
fn is_likely_primary_source(path: &[u8], lib_name: &[u8]) -> bool {
    is_translation_unit_match(path, lib_name)
}

/// Count leading path-component levels that form a "chain" (each level has
/// exactly one unique subdirectory across all paths and no paths end there),
/// or 1 if the first component duplicates the lib name.
///
/// A chain of >= 3 gets collapsed to the branch point; a duplicate gets
/// collapsed to eliminate the repeated directory.
fn path_chain_depth(lib_name: &[u8], relative_paths: &BTreeSet<Vec<u8>>) -> usize {
    if relative_paths.is_empty() {
        return 0;
    }

    let components: Vec<Vec<&[u8]>> = relative_paths
        .iter()
        .map(|p| p.split(|&b| b == b'\\').collect())
        .collect();

    let min_depth = components.iter().map(|c| c.len()).min().unwrap_or(0);

    let mut chain_depth = 0usize;
    for depth in 0..min_depth {
        let first = components[0][depth];
        let all_same = components.iter().all(|c| c[depth] == first);
        let all_continue = components.iter().all(|c| c.len() > depth + 1);
        if !all_same || !all_continue {
            break;
        }
        chain_depth += 1;
    }

    // Duplicate: relative starts with the lib name itself.
    let has_duplicate = chain_depth >= 1 && components[0][0] == lib_name;

    if chain_depth >= 3 {
        chain_depth
    } else if has_duplicate {
        1
    } else {
        0
    }
}

fn normalise(path: &[u8]) -> Vec<u8> {
    path.iter()
        .map(|&b| if b == b'/' { b'\\' } else { b.to_ascii_lowercase() })
        .collect()
}

fn lib_name_from_bytes(raw: &[u8]) -> Vec<u8> {
    let raw = if raw.len() > 4
        && (raw[raw.len() - 4..].eq_ignore_ascii_case(b".obj")
            || raw[raw.len() - 4..].eq_ignore_ascii_case(b".lib"))
    {
        &raw[..raw.len() - 4]
    } else {
        raw
    };
    let stem = raw.rsplit(|&b| b == b'\\' || b == b'/').next().unwrap_or(raw);
    stem.to_ascii_lowercase()
}

/// Find longest common directory prefix among a set of normalised paths.
/// Excludes system/MSVC include paths from the common prefix calculation
/// so they don't pollute the project root.
fn common_project_root(paths: &BTreeSet<Vec<u8>>) -> Vec<u8> {
    if paths.is_empty() {
        return Vec::new();
    }
    // Filter to only "project" paths (not system include paths)
    let project_paths: Vec<&[u8]> = paths
        .iter()
        .map(|p| p.as_slice())
        .filter(|p| {
            !p.starts_with(b"c:\\program files")
                && !p.starts_with(b"f:\\dd\\vctools")
                && !p.starts_with(b"c:\\dd\\vctools")
        })
        .collect();

    if project_paths.is_empty() {
        // All are system paths; use the full set
        let first: &[u8] = paths.iter().next().unwrap();
        let mut prefix_end = first.len();
        for p in paths.iter().skip(1) {
            let max = prefix_end.min(p.len());
            let mut split = 0usize;
            for i in 0..max {
                if first[i] != p[i] {
                    break;
                }
                if first[i] == b'\\' {
                    split = i + 1;
                }
            }
            prefix_end = split;
            if prefix_end == 0 {
                break;
            }
        }
        return first[..prefix_end].to_vec();
    }

    let first = project_paths[0];
    let mut prefix_end = first.len();
    for p in &project_paths[1..] {
        let max = prefix_end.min(p.len());
        let mut split = 0usize;
        for i in 0..max {
            if first[i] != p[i] {
                break;
            }
            if first[i] == b'\\' {
                split = i + 1;
            }
        }
        prefix_end = split;
        if prefix_end == 0 {
            break;
        }
    }
    first[..prefix_end].to_vec()
}

//
//
//

fn get_constant_name(symbol: RawString, data: &[u8]) -> Vec<u8> {
    match () {
        () if symbol.as_bytes().starts_with(b"??_C@_0") => data
            .iter()
            .copied()
            .map(|c| match c.is_ascii_alphanumeric() {
                true => c,
                false => b'_',
            })
            .collect::<Vec<_>>(),
        () if symbol.as_bytes().starts_with(b"??_C@_1") => data
            .windows(2)
            .map(|c| match c[0] == b'\0' && c[1].is_ascii_alphanumeric() {
                true => c[1],
                false => b'_',
            })
            .collect::<Vec<_>>(),
        () => unreachable!(),
    }
}
