/**
 * @module validation
 *
 * Input validation functions for reference sentence data.
 *
 * All functions in this module are **pure** and **synchronous**.
 * They either return a validated result or throw a descriptive error.
 * This makes them trivial to unit-test: pass bad data, assert the error.
 *
 * ## Correspondence to Python
 *
 * | Python `response_rater.py`                              | This module                          |
 * | ------------------------------------------------------- | ------------------------------------ |
 * | `_assert_reference_sentence_dataframe_structure(df, …)` | {@link validateReferenceSentences}   |
 */

import type { ReferenceSentence, ProcessedReferenceSet } from "./types";

/**
 * The set of field names that every {@link ReferenceSentence} must have.
 */
const REQUIRED_FIELDS: readonly (keyof ReferenceSentence)[] = [
  "id",
  "intResponse",
  "sentence",
];

/**
 * Validate and group an array of reference sentences into processed reference sets.
 *
 * This function enforces the following invariants:
 *
 * 1. Every entry has `id`, `intResponse`, and `sentence` fields.
 * 2. If `requireEmbeddings` is true, every entry must also have an `embedding`.
 * 3. No reference set may use the reserved id `"mean"`.
 * 4. Each reference set must contain at least 2 sentences.
 * 5. The `intResponse` values within each set must form a contiguous sequence of integers with step 1.
 *
 * ### Correspondence to Python
 * Port of `_assert_reference_sentence_dataframe_structure(df, embeddings_column)`.
 *
 * @param sentences          - Array of reference sentence objects to validate.
 * @param requireEmbeddings  - If `true`, every sentence must have a non-undefined
 *                             `embedding` field. Set this to `true` when operating
 *                             in embedding mode.
 * @returns A map from set id → {@link ProcessedReferenceSet} (without `embeddingMatrix`
 *          populated — that is done by the caller after computing/loading embeddings).
 *          The sentences within each set are sorted by `intResponse` ascending.
 *
 * @throws {Error} With a descriptive message if any invariant is violated.
 *
 * @example
 * ```ts
 * const refs: ReferenceSentence[] = [
 *   { id: "set1", intResponse: 1, sentence: "Strongly disagree" },
 *   { id: "set1", intResponse: 2, sentence: "Disagree" },
 *   { id: "set1", intResponse: 3, sentence: "Neutral" },
 *   { id: "set1", intResponse: 4, sentence: "Agree" },
 *   { id: "set1", intResponse: 5, sentence: "Strongly agree" },
 * ];
 * const sets = validateReferenceSentences(refs);
 * // sets has one entry: sets.get("set1")
 * ```
 *
 * @pure This function has no side effects.
 */
export function validateReferenceSentences(
  sentences: ReferenceSentence[],
  requireEmbeddings: boolean = false
): Map<string, ProcessedReferenceSet> {
  // --- Check required fields ---
  for (let i = 0; i < sentences.length; i++) {
    const entry = sentences[i];
    for (const field of REQUIRED_FIELDS) {
      if (entry[field] === undefined || entry[field] === null) {
        throw new Error(
          `Reference sentence at index ${i} is missing required field "${field}". ` +
            `Required fields: ${REQUIRED_FIELDS.join(", ")}.`
        );
      }
    }
    if (requireEmbeddings && (entry.embedding === undefined || entry.embedding === null)) {
      throw new Error(
        `Reference sentence at index ${i} (id="${entry.id}", intResponse=${entry.intResponse}) ` +
          `is missing an "embedding" field, but embedding mode requires all entries to have embeddings.`
      );
    }
  }

  // --- Group by id ---
  const groups = new Map<string, ReferenceSentence[]>();
  for (const entry of sentences) {
    if (!groups.has(entry.id)) {
      groups.set(entry.id, []);
    }
    groups.get(entry.id)!.push(entry);
  }

  // --- Validate each group ---
  const result = new Map<string, ProcessedReferenceSet>();

  for (const [id, group] of groups) {
    // No set may use the reserved "mean" id
    if (id === "mean") {
      throw new Error(
        `Reference set id "mean" is reserved for computing the average across all sets. ` +
          `Please choose a different id.`
      );
    }

    // Must have at least 2 entries (minimum for a distribution)
    if (group.length < 2) {
      throw new Error(
        `Reference set "${id}" has ${group.length} sentence(s), but at least 2 are required ` +
          `(a Likert scale needs at least 2 points).`
      );
    }

    // Sort by intResponse ascending
    const sorted = [...group].sort((a, b) => a.intResponse - b.intResponse);

    // intResponse values must be integers
    const actualResponses = sorted.map((s) => s.intResponse);
    for (const val of actualResponses) {
      if (!Number.isInteger(val)) {
        throw new Error(
          `Reference set "${id}" has a non-integer intResponse value ${val}. ` +
            `All intResponse values must be integers.`
        );
      }
    }
    for (let i = 1; i < actualResponses.length; i++) {
      if (actualResponses[i] - actualResponses[i - 1] !== 1) {
        throw new Error(
          `Reference set "${id}" has intResponse values [${actualResponses.join(", ")}], ` +
            `but they must form a contiguous sequence of integers with step 1 ` +
            `(e.g. [1, 2, 3] or [0, 1, 2, 3, 4]).`
        );
      }
    }

    result.set(id, {
      id,
      sentences: sorted.map((s) => s.sentence),
      embeddingMatrix: [], // Populated by the caller after embedding computation
    });
  }

  return result;
}
