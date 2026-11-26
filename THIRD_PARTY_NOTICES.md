# Third-party source notices

The Rust dependency graph and exact package checksums are recorded in
`Cargo.lock`. Licenses of dependencies remain their own; the repository's MIT
license does not replace them.

## Symphonia Opus

The Ogg Opus decoder uses the unmodified published `symphonia-opus` **0.1.3**
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
