//! A readable picture of the KV page pool.
//!
//! Page counters answer "what happened"; this answers "what is here now". The
//! question it exists for is the one you ask when a pool is filling up: is that
//! one runaway conversation, or many small ones that should be sharing and are
//! not?
//!
//! Sharing is the number worth looking at. Paged KV earns its complexity by
//! letting conversations that open the same way hold *one* copy of that opening
//! between them, so a pool where nothing is shared is a pool doing bookkeeping
//! for no benefit.

use std::fmt::Write as _;

use onnx_genai::kv::PageUsage;

/// Cells in the occupancy bar.
const BAR_WIDTH: usize = 32;

/// Render `usage` as a short report.
pub(crate) fn render(usage: &PageUsage) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "kv pages   {}", bar(usage));
    let _ = writeln!(
        out,
        "           {} of {} pages held · {} tokens/page",
        usage.in_use,
        usage.capacity.max(usage.in_use),
        usage.page_size
    );

    // Partially filled pages are the standing cost of paging, so the gap
    // between slots held and slots used is worth naming rather than leaving to
    // be inferred from two other numbers.
    if usage.slot_capacity > 0 {
        let _ = writeln!(
            out,
            "           {} of {} token slots used ({}% of held pages)",
            usage.filled_slots,
            usage.slot_capacity,
            percent(usage.filled_slots, usage.slot_capacity)
        );
    }

    if usage.in_use > 0 {
        let _ = writeln!(
            out,
            "           {} shared ({}%) — pages more than one sequence or cached prefix holds",
            usage.shared,
            percent(usage.shared, usage.in_use)
        );
    }

    if usage.references.len() > 1 {
        let _ = writeln!(out, "\nreferences per page");
        for (count, pages) in &usage.references {
            let _ = writeln!(out, "  {count:>3}x  {pages:>6} pages");
        }
    }

    if !usage.sequences.is_empty() {
        let _ = writeln!(out, "\nlive sequences");
        let _ = writeln!(
            out,
            "  {:<10} {:>7} {:>8} {:>8}",
            "sequence", "pages", "tokens", "shared"
        );
        for sequence in &usage.sequences {
            let _ = writeln!(
                out,
                "  {:<10} {:>7} {:>8} {:>8}",
                sequence.sequence, sequence.pages, sequence.tokens, sequence.shared
            );
        }
    }

    if usage.tiers.len() > 1 {
        let _ = writeln!(out, "\npages per tier");
        for (device, pages) in &usage.tiers {
            let _ = writeln!(out, "  {:<12} {pages:>6}", tier_name(*device));
        }
    }
    out
}

/// Occupancy as a bar: shared pages, then privately held, then free.
///
/// Shared is drawn first and distinctly because it is the part that would
/// otherwise have been duplicated — the reason the pool fits what it fits.
fn bar(usage: &PageUsage) -> String {
    let capacity = usage.capacity.max(usage.in_use);
    if capacity == 0 {
        return "(no pages)".to_string();
    }
    let shared = cells(usage.shared, capacity);
    let owned = cells(usage.in_use.saturating_sub(usage.shared), capacity);
    let used = (shared + owned).min(BAR_WIDTH);
    format!(
        "{}{}{}  {}%",
        "▓".repeat(shared.min(BAR_WIDTH)),
        "█".repeat(used - shared.min(BAR_WIDTH)),
        "·".repeat(BAR_WIDTH - used),
        percent(usage.in_use, capacity)
    )
}

/// Bar cells for `part` of `whole`, never rounding a non-zero part away.
///
/// A single held page in a large pool should still show, or the bar would say
/// "empty" about a pool that is not.
fn cells(part: usize, whole: usize) -> usize {
    if part == 0 || whole == 0 {
        return 0;
    }
    ((part * BAR_WIDTH).div_ceil(whole)).clamp(1, BAR_WIDTH)
}

/// A tier's name. The device enum is a runtime concept with no display form,
/// and inventing one here keeps that decision out of the storage layer.
fn tier_name(device: onnx_genai::kv::Device) -> String {
    match device {
        onnx_genai::kv::Device::Gpu(index) => format!("gpu{index}"),
        onnx_genai::kv::Device::Cpu => "cpu".to_string(),
        onnx_genai::kv::Device::Disk => "disk".to_string(),
    }
}

fn percent(part: usize, whole: usize) -> usize {
    part.saturating_mul(100).checked_div(whole).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_genai::kv::SequenceUsage;

    fn usage() -> PageUsage {
        PageUsage {
            page_size: 16,
            capacity: 100,
            in_use: 40,
            free: 60,
            filled_slots: 500,
            slot_capacity: 640,
            shared: 10,
            references: vec![(1, 30), (2, 8), (3, 2)],
            sequences: vec![SequenceUsage {
                sequence: 7,
                pages: 12,
                tokens: 190,
                shared: 5,
            }],
            tiers: Vec::new(),
        }
    }

    #[test]
    fn the_report_names_what_is_held_shared_and_free() {
        let report = render(&usage());
        assert!(report.contains("40 of 100 pages held"), "{report}");
        assert!(report.contains("10 shared (25%)"), "{report}");
        assert!(report.contains("500 of 640 token slots"), "{report}");
    }

    #[test]
    fn sequences_are_listed_with_their_share() {
        let report = render(&usage());
        assert!(report.contains("live sequences"), "{report}");
        assert!(report.contains("190"), "token count: {report}");
    }

    #[test]
    fn an_empty_pool_says_so_rather_than_dividing_by_zero() {
        let empty = PageUsage {
            page_size: 16,
            capacity: 0,
            in_use: 0,
            free: 0,
            filled_slots: 0,
            slot_capacity: 0,
            shared: 0,
            references: Vec::new(),
            sequences: Vec::new(),
            tiers: Vec::new(),
        };
        let report = render(&empty);
        assert!(report.contains("(no pages)"), "{report}");
    }

    /// A pool holding something must never draw as empty, however large it is.
    #[test]
    fn one_page_in_a_huge_pool_still_shows() {
        let mut sparse = usage();
        sparse.capacity = 100_000;
        sparse.in_use = 1;
        sparse.shared = 0;
        assert!(bar(&sparse).contains('█'), "{}", bar(&sparse));
    }

    #[test]
    fn the_bar_never_overflows_its_width() {
        let mut full = usage();
        full.capacity = 40;
        full.in_use = 40;
        full.shared = 40;
        let drawn = bar(&full);
        let cells = drawn.chars().filter(|c| "▓█·".contains(*c)).count();
        assert_eq!(cells, BAR_WIDTH, "{drawn}");
    }

    #[test]
    fn shared_and_owned_together_never_exceed_the_bar() {
        // Both round up, so a pool that is exactly full must not produce more
        // cells than the bar has.
        let mut awkward = usage();
        awkward.capacity = 7;
        awkward.in_use = 7;
        awkward.shared = 3;
        let drawn = bar(&awkward);
        assert_eq!(
            drawn.chars().filter(|c| "▓█·".contains(*c)).count(),
            BAR_WIDTH,
            "{drawn}"
        );
    }
}
