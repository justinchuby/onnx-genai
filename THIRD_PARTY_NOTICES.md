# Third-party source notices

The Rust dependency graph and exact package checksums are recorded in
`Cargo.lock`. Licenses of dependencies remain their own; the repository's MIT
license does not replace them.

## Symphonia Opus

The admitted Ogg/WebM Opus decoder uses the unmodified published `symphonia-opus` **0.1.3**
package under the **Mozilla Public License 2.0**. It is a pure-Rust dependency
with no bundled C library or external decoding executable.

- Complete corresponding source, including original notices and license:
  <https://crates.io/api/v1/crates/symphonia-opus/0.1.3/download>
- Upstream repository: <https://github.com/jiztastamablastamarang/symphonia-opus>
- MPL 2.0: <https://www.mozilla.org/en-US/MPL/2.0/>
- Package SHA-256: `5af6bc6d58f41f4f5107f0b58c2f128fd24531b104bc3ef13e1e50785994c63c`

Binary redistributors must retain this source-availability notice and the
dependency's applicable notices. Changes to covered source files, if made in
future distributions, must also be made available under MPL 2.0. The runtime's
separate integration module uses the dependency's public API; it does not copy
the decoder implementation or alter its license.

Symphonia 0.6.1 and its selected codec/container packages are also MPL-2.0;
their exact versions, source URLs and checksums are available through the
`Cargo.lock` crates.io package records and <https://github.com/pdeljanov/Symphonia>.

## Maintained Matroska reader

`onnx-genai-symphonia-mkv` is a maintained **MPL-2.0** format-only derivative of
`symphonia-format-mkv` 0.6.1, copyright 2019–2026 The Project Symphonia Developers.
Original file headers and LICENSE remain in `crates/onnx-genai-symphonia-mkv`.
Its exact upstream source archive is included in the package under `upstream/`,
SHA-256 `d015c5c0558864665894b3f4cbd95e10abb01b9c868e751c72670f326a56360e`;
`UPSTREAM.json` records copied-file hashes and revision. `CHANGES.md` identifies
local modifications, and all modified covered source is distributed in the crate.

- Original archive: <https://crates.io/api/v1/crates/symphonia-format-mkv/0.6.1/download>
- Corresponding maintained source: <https://github.com/justinchuby/onnx-genai/tree/main/crates/onnx-genai-symphonia-mkv>

Binary redistributors must preserve these notices and make the covered maintained
source available under MPL-2.0, including their changes to covered source files.
The repository's MIT license does not replace the reader's MPL-2.0 license.

GenAI Python and CLI wheels carry this notice and the reader's complete maintained
covered source under `<python-package>/_third_party/onnx-genai-symphonia-mkv-<version>/`.
That versioned snapshot includes the license, original source archive/provenance,
change notes and build script. Source distributions carry the canonical reader
crate, including an identical `NOTICE.md`; wheel staging never maintains a second
copy of the reader source. The bundled snapshot identifies the exact covered
source used by that artifact; the repository URL above is additional discovery.
