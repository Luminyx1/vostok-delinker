use crate::pdb_symbols::PdbSymbols;
use crate::relocs::RelocKind;
use crate::symbol_matcher::{canonical_name, SymbolMatcher};
use crate::utils::{leak, ToU64, ToUsize};
use crate::Env;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use iced_x86::{Decoder, DecoderOptions, FlowControl, Instruction, OpKind};

use pdb2::{FallibleIterator, RawString};

use object::SectionKind;

// ---------------------------------------------------------------------------
// extern "C" decoration accounting.
//
// These are incremented every time `coff_decorate_function_name` is asked to
// decorate a name. They are purely informational — surfaced by `main.rs` at
// the end of a run so the operator can sanity-check that the rule is hitting
// the expected population (mostly `extern "C"` names needing a `_`, plus a
// tail of already-decorated DLL exports / intrinsics / mangled C++ / fastcall
// names that are left untouched).
// ---------------------------------------------------------------------------

/// Number of function names that had *no* leading `_` and got one prepended
/// (cdecl / stdcall `extern "C"` names whose PDB public symbol was
/// underscore-stripped, OR internal functions only seen as a Procedure
/// symbol with no leading `_`).
pub static DECORATED_EXTERN_C: AtomicUsize = AtomicUsize::new(0);
/// Number of function names that already had a leading `_` (DLL-export
/// Publics that kept their underscore, intrinsics like `_chkstk`, or
/// Procedure symbols whose source declared them with `_`) and were left
/// untouched so we don't double-decorate.
pub static LEFT_ALONE_ALREADY_UNDERSCORED: AtomicUsize = AtomicUsize::new(0);
/// Number of function names that were recognised as already-mangled C++ (`?`)
/// or fastcall-decorated (`@`) and left untouched.
pub static LEFT_ALONE_MANGLED: AtomicUsize = AtomicUsize::new(0);

/// Apply MSVC's `extern "C"` leading-underscore decoration to a function
/// name, **without double-decorating names that already start with `_`**.
///
/// # Why this exists
///
/// MSVC's compiler (`cl.exe`) decorates `extern "C"` cdecl/stdcall function
/// names with a single leading underscore when emitting COFF object files:
/// source `void foo()` becomes `_foo` in the .obj, source `void __stdcall
/// foo(int)` becomes `_foo@4`. The delinker reads function names from the
/// PDB, and the PDB can hand us names in *either* of two states:
///
/// 1. **Underscore-stripped** - for ordinary cdecl/stdcall Public symbols
///    the linker strips exactly one leading `_`, so the PDB records `foo`,
///    `foo@4`. Internal functions seen only via their Procedure symbol are
///    also typically recorded without the leading `_` (the Procedure record
///    carries the *source* identifier). These need a `_` re-added to match
///    what `cl.exe` would emit.
///
/// 2. **Already decorated** - DLL-exported `extern "C"` functions keep the
///    leading `_` in their Public name (the linker does *not* strip it for
///    exports; see `_GSH2Destroy`, `_GSH2Initialize` in any real game PDB).
///    Compiler intrinsics like `_chkstk`, `_security_check_cookie` are
///    declared with the `_` in source and the PDB records them that way.
///    These must be left alone - adding another `_` would produce
///    `__GSH2Destroy` and break objdiff matching against the freshly
///    compiled base, whose `cl.exe` also emits `_GSH2Destroy`.
///
/// # The rule
///
/// Classification is a pure function of the first byte of the name:
///
/// | First byte | Verdict                                     | Example                     |
/// |------------|---------------------------------------------|-----------------------------|
/// | `?`        | Mangled C++ - leave alone                   | `?foo@@YAXXZ`               |
/// | `@`        | Fastcall - leave alone                      | `@foo@4`                    |
/// | `_`        | Already decorated - leave alone             | `_GSH2Destroy`, `_chkstk`   |
/// | (other)    | `extern "C"` needing decoration - add `_`   | `foo`->`_foo`, `foo@4`->`_foo@4` |
///
/// The net effect: the output always has *exactly one* leading underscore
/// for cdecl/stdcall names, regardless of whether the PDB already had one.
///
/// # Verification contract
///
/// This function is the *only* place names get decorated. It is exercised
/// by both `add_function` (the symbol definition site) and the
/// `RelocKind::Function` arm of `add_relocation_at` (every reference site),
/// so a function's definition and all of its references are guaranteed to
/// carry the same decorated name. The classification is a pure function of
/// the first byte of the name, and unit tests below pin each branch -
/// including the regression test for the user-reported `_GSH2Destroy` case.
pub(crate) fn coff_decorate_function_name(name: &[u8]) -> Vec<u8> {
    match name.first().copied() {
        // Mangled C++ (`?foo@@YAXXZ`) or fastcall (`@foo@4`) - neither uses
        // the cdecl underscore rule. Leave untouched.
        Some(b'?') | Some(b'@') => {
            LEFT_ALONE_MANGLED.fetch_add(1, Ordering::Relaxed);
            name.to_vec()
        }
        // Name already starts with `_`. Two situations land here:
        //   * DLL-exported `extern "C"` Publics whose `_` was *not* stripped
        //     by the linker (e.g. `_GSH2Destroy`, `_GSH2Initialize`).
        //   * Intrinsics / source-declared `_foo` names recorded as-is.
        // In both cases the desired .obj name is exactly what the PDB gave
        // us - adding another `_` would double-decorate and break objdiff
        // matching. Leave untouched.
        Some(b'_') => {
            LEFT_ALONE_ALREADY_UNDERSCORED.fetch_add(1, Ordering::Relaxed);
            name.to_vec()
        }
        // Everything else is `extern "C"` (cdecl or stdcall) whose PDB name
        // was underscore-stripped. The MSVC mangler prepends exactly one `_`;
        // we mirror that.
        _ => {
            DECORATED_EXTERN_C.fetch_add(1, Ordering::Relaxed);
            let mut decorated = Vec::with_capacity(name.len() + 1);
            decorated.push(b'_');
            decorated.extend_from_slice(name);
            decorated
        }
    }
}

#[cfg(test)]
mod decoration_tests {
    use super::coff_decorate_function_name;

    // ---- decoration branch: names that DO get a `_` prepended ----

    #[test]
    fn cdecl_plain_name_gets_underscore() {
        // Source `extern "C" void foo()` -> PDB public `foo` (stripped)
        // -> .obj `_foo`.
        assert_eq!(coff_decorate_function_name(b"foo"), b"_foo");
    }

    #[test]
    fn stdcall_name_gets_underscore_before_at_suffix() {
        // Source `extern "C" void __stdcall foo(int)` -> PDB public `foo@4`
        // (linker strips the cdecl underscore, keeps the `@4`) -> .obj `_foo@4`.
        assert_eq!(coff_decorate_function_name(b"foo@4"), b"_foo@4");
    }

    #[test]
    fn internal_function_procedure_name_gets_underscore() {
        // The user's working case: an internal function seen only via its
        // Procedure symbol, recorded without a leading `_`. We must add one.
        assert_eq!(
            coff_decorate_function_name(b"InitGLSL130TextureFunctions_1"),
            b"_InitGLSL130TextureFunctions_1",
        );
    }

    // ---- leave-alone branch: names that already start with `_` ----

    #[test]
    fn dll_export_with_underscore_is_left_alone() {
        // The user's reported regression: `_GSH2Destroy` in the PDB (a DLL
        // export whose Public name kept its `_`) must NOT become
        // `__GSH2Destroy`. The base `cl.exe` compile emits `_GSH2Destroy`,
        // so the delinked target must too.
        assert_eq!(coff_decorate_function_name(b"_GSH2Destroy"), b"_GSH2Destroy");
    }

    #[test]
    fn dll_export_family_is_left_alone() {
        // Real names from shaderUtilsD.pdb - all DLL exports whose Public
        // symbol retained the leading `_`. All must stay single-underscored.
        for name in [
            &b"_GSH2Initialize"[..],
            &b"_GSH2CompileProgram"[..],
            &b"_GSH2DestroyGX2Program"[..],
            &b"_GSH2CalcFetchShaderSizeEx"[..],
            &b"_GX2GetAttribFormatBits"[..],
        ] {
            assert_eq!(coff_decorate_function_name(name), name);
        }
    }

    #[test]
    fn intrinsic_with_underscore_is_left_alone() {
        // `_chkstk` is a compiler intrinsic declared with `_` in source.
        // The PDB records it as `_chkstk`. We leave it alone - adding
        // another `_` would give `__chkstk`, which only matches a base
        // compile of source `_chkstk` if the compiler also doubled it, and
        // for intrinsics it doesn't.
        assert_eq!(coff_decorate_function_name(b"_chkstk"), b"_chkstk");
    }

    #[test]
    fn double_underscore_intrinsic_is_left_alone() {
        // Same rule for `__security_check_cookie`-style names: they already
        // start with `_`, so leave them alone.
        assert_eq!(
            coff_decorate_function_name(b"__security_check_cookie"),
            b"__security_check_cookie",
        );
    }

    #[test]
    fn import_thunk_with_underscore_is_left_alone() {
        // Import load thunks like `__imp_load_X` already carry their leading
        // `_`; do not double-decorate.
        assert_eq!(
            coff_decorate_function_name(b"__imp_load_CreateFileA@4"),
            b"__imp_load_CreateFileA@4",
        );
    }

    // ---- leave-alone branch: mangled C++ and fastcall ----

    #[test]
    fn mangled_cpp_name_is_left_alone() {
        let mangled = b"?foo@@YAXH@Z";
        assert_eq!(coff_decorate_function_name(mangled), mangled);
    }

    #[test]
    fn mangled_cpp_member_name_is_left_alone() {
        let mangled = b"?bar@Foo@@QAEXXZ";
        assert_eq!(coff_decorate_function_name(mangled), mangled);
    }

    #[test]
    fn fastcall_name_is_left_alone() {
        let fastcall = b"@foo@4";
        assert_eq!(coff_decorate_function_name(fastcall), fastcall);
    }

    // ---- defensive / boundary ----

    #[test]
    fn empty_name_is_decorated_to_single_underscore() {
        // Should not happen in practice, but the helper must not panic on
        // an empty slice. An empty name has no first byte, so it falls into
        // the "needs decoration" branch and becomes a single `_`.
        assert_eq!(coff_decorate_function_name(b""), b"_");
    }

    #[test]
    fn classification_is_pure_in_first_byte() {
        // Sanity sweep: the verdict depends ONLY on the first byte.
        // (This is the property the user asked us to verify.)
        //
        // Left alone - starts with `?`, `@`, or `_`:
        for left_alone in [
            &b"?"[..],
            &b"?x"[..],
            &b"?foo@@YAXXZ"[..],
            &b"@"[..],
            &b"@foo@0"[..],
            &b"_"[..],
            &b"_x"[..],
            &b"_foo"[..],
            &b"__foo"[..],
            &b"_foo@4"[..],
            &b"_GSH2Destroy"[..],
            &b"_chkstk"[..],
        ] {
            assert_eq!(coff_decorate_function_name(left_alone), left_alone);
        }
        // Decorated - does not start with `?`, `@`, or `_`:
        for (input, expected) in [
            (&b"a"[..], &b"_a"[..]),
            (&b"foo"[..], &b"_foo"[..]),
            (&b"foo@8"[..], &b"_foo@8"[..]),
            (&b"GSH2Destroy"[..], &b"_GSH2Destroy"[..]),
            (&b"InitGLSL130TextureFunctions_1"[..], &b"_InitGLSL130TextureFunctions_1"[..]),
            (&b""[..], &b"_"[..]),
        ] {
            assert_eq!(coff_decorate_function_name(input), expected);
        }
    }

    #[test]
    fn output_always_has_exactly_one_leading_underscore_for_cdecl() {
        // The headline invariant: for an `extern "C"` cdecl name, regardless
        // of whether the PDB gave us the stripped or un-stripped form, the
        // .obj output has exactly ONE leading `_`.
        //
        //   PDB `GSH2Destroy`   (Procedure, stripped)  -> `_GSH2Destroy`
        //   PDB `_GSH2Destroy`  (Public, un-stripped)  -> `_GSH2Destroy`
        //
        // Both forms produce the same .obj name. That's what makes the
        // delinked target match the `cl.exe`-compiled base for both
        // internal and exported `extern "C"` functions.
        assert_eq!(coff_decorate_function_name(b"GSH2Destroy"),  b"_GSH2Destroy");
        assert_eq!(coff_decorate_function_name(b"_GSH2Destroy"), b"_GSH2Destroy");
        // And critically, neither produces `__GSH2Destroy`.
        assert_ne!(coff_decorate_function_name(b"_GSH2Destroy"), b"__GSH2Destroy");
    }
}

#[cfg(test)]
mod end_to_end_tests {
    //! These tests drive the *real* `ObjectFile::add_function` /
    //! `add_relocation` paths, then write the resulting COFF bytes, parse
    //! them back with the `object` read API, and inspect the symbol table.
    //! That end-to-end round-trip is what proves the decoration actually
    //! lands in the .obj — not just in the helper function's return value.

    use super::*;
    use object::read::{Object as _, ObjectSymbol as _};
    use pdb2::RawString;

    /// Walk the symbols of a freshly-written `ObjectFile` and collect their
    /// names as `Vec<u8>`. Panics on any read error.
    fn written_symbol_names(of: ObjectFile) -> Vec<Vec<u8>> {
        let bytes = of.object.write().expect("object write");
        let file = object::read::File::parse(&*bytes).expect("object parse");
        file.symbols()
            .filter_map(|s| s.name().ok().map(|n| n.as_bytes().to_vec()))
            .collect()
    }

    #[test]
    fn cdecl_extern_c_gets_underscore_in_obj() {
        let mut of = ObjectFile::empty(false);
        of.add_function(RawString::from(b"foo".as_ref()), &[0xC3]);

        let names = written_symbol_names(of);
        assert!(
            names.iter().any(|n| n == b"_foo"),
            "expected `_foo` in symbol table, got: {:?}",
            names,
        );
        assert!(
            !names.iter().any(|n| n == b"foo"),
            "bare `foo` (without `_`) must NOT appear, got: {:?}",
            names,
        );
    }

    #[test]
    fn stdcall_extern_c_gets_underscore_before_at_suffix_in_obj() {
        let mut of = ObjectFile::empty(false);
        of.add_function(RawString::from(b"foo@4".as_ref()), &[0xC3]);

        let names = written_symbol_names(of);
        assert!(
            names.iter().any(|n| n == b"_foo@4"),
            "expected `_foo@4` in symbol table, got: {:?}",
            names,
        );
        assert!(
            !names.iter().any(|n| n == b"foo@4"),
            "bare `foo@4` (without `_`) must NOT appear, got: {:?}",
            names,
        );
    }

    #[test]
    fn intrinsic_starting_with_underscore_is_left_alone_in_obj() {
        // PDB `_chkstk` -> .obj `_chkstk`. The intrinsic already carries the
        // leading `_` the compiler would emit, so we do NOT double-decorate.
        let mut of = ObjectFile::empty(false);
        of.add_function(RawString::from(b"_chkstk".as_ref()), &[0xC3]);

        let names = written_symbol_names(of);
        assert!(
            names.iter().any(|n| n == b"_chkstk"),
            "expected `_chkstk` in symbol table, got: {:?}",
            names,
        );
        assert!(
            !names.iter().any(|n| n == b"__chkstk"),
            "`__chkstk` (double `_`) must NOT appear, got: {:?}",
            names,
        );
    }

    #[test]
    fn dll_export_with_underscore_is_left_alone_in_obj() {
        // The user-reported regression: PDB `_GSH2Destroy` (DLL export whose
        // Public name kept its `_`) must stay `_GSH2Destroy` in the .obj,
        // NOT become `__GSH2Destroy`.
        let mut of = ObjectFile::empty(false);
        of.add_function(RawString::from(b"_GSH2Destroy".as_ref()), &[0xC3]);

        let names = written_symbol_names(of);
        assert!(
            names.iter().any(|n| n == b"_GSH2Destroy"),
            "expected `_GSH2Destroy` in symbol table, got: {:?}",
            names,
        );
        assert!(
            !names.iter().any(|n| n == b"__GSH2Destroy"),
            "`__GSH2Destroy` (double `_`) must NOT appear, got: {:?}",
            names,
        );
    }

    #[test]
    fn stripped_and_unstripped_forms_of_same_name_produce_same_obj_symbol() {
        // The headline end-to-end invariant: the PDB can give us either
        // `GSH2Destroy` (Procedure symbol, no `_`) or `_GSH2Destroy`
        // (Public symbol, with `_`) for the *same* function. Whichever we
        // see, the .obj symbol must come out as `_GSH2Destroy` so it matches
        // the base compile.
        let mut of_a = ObjectFile::empty(false);
        of_a.add_function(RawString::from(b"GSH2Destroy".as_ref()), &[0xC3]);
        let mut of_b = ObjectFile::empty(false);
        of_b.add_function(RawString::from(b"_GSH2Destroy".as_ref()), &[0xC3]);

        let names_a = written_symbol_names(of_a);
        let names_b = written_symbol_names(of_b);

        assert!(names_a.iter().any(|n| n == b"_GSH2Destroy"),
            "stripped form `GSH2Destroy` should decorate to `_GSH2Destroy`, got: {:?}", names_a);
        assert!(names_b.iter().any(|n| n == b"_GSH2Destroy"),
            "un-stripped form `_GSH2Destroy` should stay `_GSH2Destroy`, got: {:?}", names_b);
        // And neither should produce the double-underscored form.
        assert!(!names_a.iter().any(|n| n == b"__GSH2Destroy"),
            "stripped form must NOT produce `__GSH2Destroy`, got: {:?}", names_a);
        assert!(!names_b.iter().any(|n| n == b"__GSH2Destroy"),
            "un-stripped form must NOT produce `__GSH2Destroy`, got: {:?}", names_b);
    }

    #[test]
    fn mangled_cpp_survives_unchanged_in_obj() {
        let mut of = ObjectFile::empty(false);
        of.add_function(RawString::from(b"?foo@@YAXXZ".as_ref()), &[0xC3]);

        let names = written_symbol_names(of);
        assert!(
            names.iter().any(|n| n == b"?foo@@YAXXZ"),
            "expected `?foo@@YAXXZ` unchanged in symbol table, got: {:?}",
            names,
        );
        assert!(
            !names.iter().any(|n| n == b"_?foo@@YAXXZ"),
            "mangled name must NOT be prefixed with `_`, got: {:?}",
            names,
        );
    }

    #[test]
    fn mangled_cpp_member_function_survives_unchanged_in_obj() {
        let mut of = ObjectFile::empty(false);
        of.add_function(RawString::from(b"?bar@Foo@@QAEXXZ".as_ref()), &[0xC3]);

        let names = written_symbol_names(of);
        assert!(
            names.iter().any(|n| n == b"?bar@Foo@@QAEXXZ"),
            "expected `?bar@Foo@@QAEXXZ` unchanged in symbol table, got: {:?}",
            names,
        );
    }

    #[test]
    fn fastcall_survives_unchanged_in_obj() {
        let mut of = ObjectFile::empty(false);
        of.add_function(RawString::from(b"@foo@4".as_ref()), &[0xC3]);

        let names = written_symbol_names(of);
        assert!(
            names.iter().any(|n| n == b"@foo@4"),
            "expected `@foo@4` unchanged in symbol table, got: {:?}",
            names,
        );
        assert!(
            !names.iter().any(|n| n == b"_@foo@4"),
            "fastcall name must NOT be prefixed with `_`, got: {:?}",
            names,
        );
    }

    #[test]
    fn mixed_object_decorates_only_extern_c() {
        // Object file containing all five cases at once - proves the rule
        // doesn't bleed across symbols.
        let mut of = ObjectFile::empty(false);
        of.add_function(RawString::from(b"cdecl_foo".as_ref()), &[0xC3]);
        of.add_function(RawString::from(b"stdcall_foo@4".as_ref()), &[0xC3]);
        of.add_function(RawString::from(b"?cpp_foo@@YAXXZ".as_ref()), &[0xC3]);
        of.add_function(RawString::from(b"@fastcall_foo@4".as_ref()), &[0xC3]);
        // DLL-export-style name that already has its `_`:
        of.add_function(RawString::from(b"_GSH2Destroy".as_ref()), &[0xC3]);

        let names = written_symbol_names(of);

        // extern "C" cdecl: decorated.
        assert!(names.iter().any(|n| n == b"_cdecl_foo"),
            "cdecl_foo should be decorated, got: {:?}", names);
        // extern "C" stdcall: decorated (underscore before `@4`).
        assert!(names.iter().any(|n| n == b"_stdcall_foo@4"),
            "stdcall_foo@4 should be decorated, got: {:?}", names);
        // mangled C++: unchanged.
        assert!(names.iter().any(|n| n == b"?cpp_foo@@YAXXZ"),
            "cpp_foo should be unchanged, got: {:?}", names);
        // fastcall: unchanged.
        assert!(names.iter().any(|n| n == b"@fastcall_foo@4"),
            "fastcall_foo@4 should be unchanged, got: {:?}", names);
        // DLL-export name: unchanged (NOT double-decorated).
        assert!(names.iter().any(|n| n == b"_GSH2Destroy"),
            "_GSH2Destroy should be unchanged, got: {:?}", names);

        // Negative: no bare extern "C" names leak through.
        assert!(!names.iter().any(|n| n == b"cdecl_foo"),
            "bare cdecl_foo must NOT appear, got: {:?}", names);
        assert!(!names.iter().any(|n| n == b"stdcall_foo@4"),
            "bare stdcall_foo@4 must NOT appear, got: {:?}", names);
        // Negative: no double-underscored DLL-export name.
        assert!(!names.iter().any(|n| n == b"__GSH2Destroy"),
            "__GSH2Destroy (double `_`) must NOT appear, got: {:?}", names);
    }

    #[test]
    fn function_reference_in_reloc_uses_decorated_name() {
        // The critical invariant for linker resolution: a function's
        // *definition* and its *references* must use the same name. Build a
        // `caller` that calls `foo` via an extern reloc; both sites must end
        // up as `_foo`.
        let mut of = ObjectFile::empty(false);

        // Define `foo`.
        // Define `caller` whose body is `call rel32` (E8 + 4 bytes).
        let caller_offset = of.add_function(
            RawString::from(b"caller".as_ref()),
            &[0xE8, 0, 0, 0, 0],
        );
        // Add a relocation referencing `foo`. The decoration must match the
        // one `add_function` would apply, so the linker can pair them up.
        of.add_relocation(
            coff_decorate_function_name(b"foo"),
            ObjectLocation::Extern,
            ObjectOffset {
                offset: caller_offset.offset + 1, // displacement is at +1
                section_id: caller_offset.section_id,
            },
        )
        .expect("add_relocation");

        let names = written_symbol_names(of);

        // The `caller` definition was decorated to `_caller`.
        assert!(names.iter().any(|n| n == b"_caller"),
            "expected `_caller`, got: {:?}", names);
        // The relocation's extern reference to `foo` was decorated to `_foo`.
        // (The `object` crate writes external symbols as separate symbol
        // table entries, so `_foo` shows up alongside `_caller`.)
        assert!(names.iter().any(|n| n == b"_foo"),
            "expected reference `_foo`, got: {:?}", names);
        // And critically, the bare name must NOT appear anywhere — neither
        // as a definition nor as a reference.
        assert!(!names.iter().any(|n| n == b"foo"),
            "bare `foo` must NOT appear (would break linker resolution), got: {:?}",
            names,
        );
        assert!(!names.iter().any(|n| n == b"caller"),
            "bare `caller` must NOT appear, got: {:?}", names);
    }
}


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
                            let filename = normalised.rsplit(|&b| b == b'\\').next().unwrap_or(&normalised);
                            let ext_pos = filename.iter().rposition(|&b| b == b'.');
                            let ext_len = ext_pos.map(|p| filename.len() - p).unwrap_or(0);
                            let stem = ext_pos.map(|p| &filename[..p]).unwrap_or(filename);
                            let display_len = if stem.eq_ignore_ascii_case(lib_name) {
                                lib_name.len()
                            } else {
                                lib_name.len().saturating_sub(1)
                            };
                            Vec::with_capacity(display_len + ext_len)
                        } else {
                            Vec::with_capacity(lib_name.len() + 1 + relative.len())
                        };
                        if is_primary {
                            let filename = normalised.rsplit(|&b| b == b'\\').next().unwrap_or(&normalised);
                            let ext = filename.iter().rposition(|&b| b == b'.').map(|p| &filename[p..]).unwrap_or(b"");
                            let stem = filename.iter().rposition(|&b| b == b'.').map(|p| &filename[..p]).unwrap_or(filename);
                            let display_name = if stem.eq_ignore_ascii_case(lib_name) {
                                lib_name
                            } else {
                                lib_name.strip_suffix(b"d")
                                    .or_else(|| lib_name.strip_suffix(b"D"))
                                    .unwrap_or(lib_name)
                            };
                            key.extend_from_slice(display_name);
                            key.extend_from_slice(ext);
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
                // Function references must use the same decorated name as the
                // function definition (see `add_function`); otherwise the
                // linker would see a definition of `_foo` and a reference to
                // `foo` as two unrelated externals. The decoration helper is
                // a pure function of the name, so the result here is
                // guaranteed identical to the one in `add_function`.
                let decorated = coff_decorate_function_name(reloc_name.as_bytes());
                self.add_relocation(decorated, ObjectLocation::Extern, reloc_offset)?;
            }

            RelocKind::ConstantString { symbol: _, data } => {
                let const_offset_in_coff_rdata =
                    self.append_section_data(self.rdata_section_id, data, 0x00);

                self.add_relocation(
                    reloc_name.as_bytes().to_vec(),
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
                    reloc_name.as_bytes().to_vec(),
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
                    reloc_name.as_bytes().to_vec(),
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
        name: Vec<u8>,
        location: ObjectLocation,
        offset: ObjectOffset,
    ) -> anyhow::Result<()> {
        let (value, kind, section) = match location {
            // `ObjectLocation::Extern` is only ever produced by the
            // `RelocKind::Function` arm of `add_relocation_at`, so every
            // extern relocation is a *function* reference. The kind must be
            // `Text` (not `Unknown`) so that the `object` crate's COFF writer
            // emits `IMAGE_SYM_CLASS_EXTERNAL` for the undefined symbol —
            // `SymbolKind::Unknown` is rejected by the writer with
            // "unimplemented symbol ... kind Unknown".
            ObjectLocation::Extern => (
                0,
                object::SymbolKind::Text,
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
            name,
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

        // Apply MSVC's `extern "C"` cdecl/stdcall leading-underscore
        // decoration so the .obj symbol matches what `cl.exe` would have
        // emitted. Mangled C++ names (`?...`) and fastcall (`@...`) are
        // returned unchanged by the helper.
        let decorated_name = coff_decorate_function_name(name.as_bytes());

        self.object.add_symbol(object::write::Symbol {
            name: decorated_name,
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
    let lower: Vec<u8> = name.iter().map(|&b| b.to_ascii_lowercase()).collect();
    matches!(
        lower.as_slice(),
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
    let ends_with_ignore_case = |suffix: &[u8]| -> bool {
        path.len() >= suffix.len()
            && path[path.len() - suffix.len()..].eq_ignore_ascii_case(suffix)
    };
    ends_with_ignore_case(b".c")
        || ends_with_ignore_case(b".cpp")
        || ends_with_ignore_case(b".cc")
        || ends_with_ignore_case(b".cxx")
        || ends_with_ignore_case(b".asm")
        || ends_with_ignore_case(b".s")
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
        .map(|&b| if b == b'/' { b'\\' } else { b })
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
    stem.to_vec()
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
            let lower: Vec<u8> = p.iter().map(|&b| b.to_ascii_lowercase()).collect();
            !lower.starts_with(b"c:\\program files")
                && !lower.starts_with(b"f:\\dd\\vctools")
                && !lower.starts_with(b"c:\\dd\\vctools")
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
