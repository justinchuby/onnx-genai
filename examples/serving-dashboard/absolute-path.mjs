/**
 * Absolute-filesystem-path detection for disclosure guards.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * The disclosure guards used to ask `text.includes('/Users/')`. That is a
 * macOS-shaped question, and it was written on a macOS desk. Measured at
 * b63f0a82, three genuine disclosures walked through it while the suite
 * reported 5/5 green:
 *
 *     /home/presenter/models/qwen2.5-0.5b     -> NOT DETECTED
 *     C:\Users\presenter\models\qwen          -> NOT DETECTED
 *     /var/lib/onnx/models/qwen               -> NOT DETECTED
 *
 * A guard that encodes its author's operating system is not guarding the
 * product, it is guarding the desk it was written on.
 *
 * THE PROPERTY BEING MEASURED
 * ---------------------------
 * The threat is an ABSOLUTE path: it discloses the layout of the machine the
 * server runs on. A RELATIVE, namespaced identifier is not a threat and must
 * not be flagged -- `Qwen/Qwen2.5-0.5B-Instruct` is a legal `--model-id` for
 * this very repository, and a guard that reddens on a legal configuration gets
 * loosened by whoever hits it. The false positive is the delivery mechanism for
 * the regression.
 *
 * So: absoluteness is the predicate. Not "contains a slash".
 */

/**
 * Filesystem roots that make a POSIX path unambiguously absolute AND
 * machine-revealing.
 *
 * This is deliberately a bounded denylist rather than "starts with /", because
 * these helpers also scan RENDERED TEXT, where a bare leading slash matches
 * every URL path and every `req/s` axis label. On a discrete field VALUE the
 * complete anchored predicate is available and is used instead -- see
 * `isAbsolutePathValue`.
 */
const POSIX_ROOTS = Object.freeze([
  'Users', 'home', 'root', 'var', 'opt', 'srv', 'mnt',
  'media', 'tmp', 'private', 'usr', 'Volumes', 'data',
]);

/** Where a path stops when it is embedded in a sentence or an aria-label. */
const PATH_TAIL = "[^\\s\"'<>,;)]*";

const POSIX_IN_TEXT = new RegExp(`/(?:${POSIX_ROOTS.join('|')})/${PATH_TAIL}`, 'g');

/** `C:\dir`, `C:/dir`, and UNC `\\server\share`. */
const WINDOWS_IN_TEXT = new RegExp(`(?:[A-Za-z]:[\\\\/]|\\\\\\\\)${PATH_TAIL}`, 'g');

/**
 * Is this whole value an absolute path?
 *
 * Use on a discrete field value, where the anchored question is answerable
 * completely and no denylist is needed.
 *
 * @param {unknown} value
 * @returns {boolean} false for anything that is not a string.
 */
export function isAbsolutePathValue(value) {
  if (typeof value !== 'string') return false;
  // POSIX absolute, Windows drive-absolute, or UNC.
  return /^(?:\/|[A-Za-z]:[\\/]|\\\\)/.test(value.trim());
}

/**
 * Every absolute path appearing anywhere inside a block of rendered text.
 *
 * Use on concatenated render output -- text nodes AND attribute values, which
 * are two different bugs: an aria-label disclosure changes zero pixels and is
 * invisible to screenshot diffing by construction.
 *
 * @param {string} text
 * @returns {string[]} the matches, in order, for use in a failure message.
 */
export function findAbsolutePaths(text) {
  if (typeof text !== 'string' || text === '') return [];
  return [
    ...(text.match(POSIX_IN_TEXT) ?? []),
    ...(text.match(WINDOWS_IN_TEXT) ?? []),
  ];
}

/**
 * The stand-in left where an absolute path was removed.
 *
 * Deliberately says something. An empty replacement leaves a sentence that
 * reads as though the server sent nothing at all, which is a different and
 * equally false claim; a truncated prefix leaks the interesting half. Naming
 * the CLASS of the removed thing keeps the sentence diagnostic without keeping
 * any of its bytes.
 */
export const WITHHELD_PATH_TEXT = '[absolute path withheld]';

/**
 * Remove every absolute path from a block of text, preserving the prose.
 *
 * For diagnostics that quote what a server sent. A provenance warning has to
 * explain WHICH field disagreed with the table and WHY; it never needs the
 * operator's directory layout to do that, and a warning is a string that gets
 * logged, stored, rendered and put into aria-labels, so any path inside one is
 * disclosed on several surfaces at once.
 *
 * Uses the same two patterns as {@link findAbsolutePaths}, deliberately: a
 * redactor with its own copy of the path vocabulary drifts from the detector,
 * and the direction it drifts is redacting less than the guard flags.
 *
 * @param {unknown} text
 * @returns {unknown} the redacted string, or the input unchanged if not a string.
 */
export function redactAbsolutePaths(text) {
  if (typeof text !== 'string' || text === '') return text;
  return text.replace(POSIX_IN_TEXT, WITHHELD_PATH_TEXT).replace(WINDOWS_IN_TEXT, WITHHELD_PATH_TEXT);
}

/**
 * The roots this scanner knows about, so a test can prove the list is non-empty
 * and report it rather than trusting it.
 *
 * @returns {readonly string[]}
 */
export function knownPosixRoots() {
  return POSIX_ROOTS;
}
