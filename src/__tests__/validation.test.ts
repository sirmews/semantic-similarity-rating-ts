/**
 * Tests for validation module — reference sentence validation.
 *
 * Port of the validation-related tests from Python `test_response_rater.py`.
 */

import { validateReferenceSentences } from "../validation";
import type { ReferenceSentence } from "../types";
import { SAMPLE_TEXTS } from "./helpers";

// ---------------------------------------------------------------------------
// Valid input
// ---------------------------------------------------------------------------

describe("validateReferenceSentences — valid input", () => {
  it("accepts a well-formed 5-point reference set", () => {
    const refs: ReferenceSentence[] = [
      { id: "set1", intResponse: 1, sentence: "Strongly disagree" },
      { id: "set1", intResponse: 2, sentence: "Disagree" },
      { id: "set1", intResponse: 3, sentence: "Neutral" },
      { id: "set1", intResponse: 4, sentence: "Agree" },
      { id: "set1", intResponse: 5, sentence: "Strongly agree" },
    ];

    const result = validateReferenceSentences(refs);

    expect(result.size).toBe(1);
    expect(result.has("set1")).toBe(true);
    expect(result.get("set1")!.sentences).toEqual([
      "Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree",
    ]);
  });

  it("accepts multiple reference sets", () => {
    const refs: ReferenceSentence[] = [];
    for (const setId of ["set1", "set2"]) {
      for (let i = 1; i <= 5; i++) {
        refs.push({ id: setId, intResponse: i, sentence: SAMPLE_TEXTS[i - 1] });
      }
    }

    const result = validateReferenceSentences(refs);

    expect(result.size).toBe(2);
    expect(result.has("set1")).toBe(true);
    expect(result.has("set2")).toBe(true);
  });

  it("sorts sentences by intResponse within each set", () => {
    // Provide in reverse order
    const refs: ReferenceSentence[] = [
      { id: "set1", intResponse: 5, sentence: "five" },
      { id: "set1", intResponse: 3, sentence: "three" },
      { id: "set1", intResponse: 1, sentence: "one" },
      { id: "set1", intResponse: 4, sentence: "four" },
      { id: "set1", intResponse: 2, sentence: "two" },
    ];

    const result = validateReferenceSentences(refs);

    expect(result.get("set1")!.sentences).toEqual([
      "one", "two", "three", "four", "five",
    ]);
  });

  it("accepts references with embeddings when requireEmbeddings=true", () => {
    const refs: ReferenceSentence[] = [];
    for (let i = 1; i <= 5; i++) {
      refs.push({
        id: "set1",
        intResponse: i,
        sentence: SAMPLE_TEXTS[i - 1],
        embedding: Array(10).fill(i * 0.1),
      });
    }

    const result = validateReferenceSentences(refs, true);
    expect(result.size).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Invalid input — missing fields
// ---------------------------------------------------------------------------

describe("validateReferenceSentences — missing fields", () => {
  it("rejects entries missing the 'sentence' field", () => {
    const refs = [
      { id: "set1", intResponse: 1 },
      { id: "set1", intResponse: 2, sentence: "Disagree" },
      { id: "set1", intResponse: 3, sentence: "Neutral" },
      { id: "set1", intResponse: 4, sentence: "Agree" },
      { id: "set1", intResponse: 5, sentence: "Strongly agree" },
    ] as ReferenceSentence[];

    expect(() => validateReferenceSentences(refs)).toThrow(/missing required field "sentence"/i);
  });

  it("rejects entries missing the 'id' field", () => {
    const refs = [
      { intResponse: 1, sentence: "A" },
    ] as unknown as ReferenceSentence[];

    expect(() => validateReferenceSentences(refs)).toThrow(/missing required field "id"/i);
  });

  it("rejects missing embeddings when requireEmbeddings=true", () => {
    const refs: ReferenceSentence[] = [];
    for (let i = 1; i <= 5; i++) {
      refs.push({ id: "set1", intResponse: i, sentence: SAMPLE_TEXTS[i - 1] });
    }

    expect(() => validateReferenceSentences(refs, true)).toThrow(/missing an "embedding" field/i);
  });
});

// ---------------------------------------------------------------------------
// Invalid input — structural
// ---------------------------------------------------------------------------

describe("validateReferenceSentences — structural errors", () => {
  it("rejects the reserved id 'mean'", () => {
    const refs: ReferenceSentence[] = [];
    for (let i = 1; i <= 5; i++) {
      refs.push({ id: "mean", intResponse: i, sentence: SAMPLE_TEXTS[i - 1] });
    }

    expect(() => validateReferenceSentences(refs)).toThrow(/reserved/i);
  });

  it("rejects sets with fewer than 5 entries", () => {
    const refs: ReferenceSentence[] = [
      { id: "set1", intResponse: 1, sentence: "A" },
      { id: "set1", intResponse: 2, sentence: "B" },
      { id: "set1", intResponse: 3, sentence: "C" },
      { id: "set1", intResponse: 4, sentence: "D" },
    ];

    expect(() => validateReferenceSentences(refs)).toThrow(/4 sentences.*exactly 5/i);
  });

  it("rejects sets with more than 5 entries", () => {
    const refs: ReferenceSentence[] = [];
    for (let i = 1; i <= 6; i++) {
      refs.push({ id: "set1", intResponse: i, sentence: `S${i}` });
    }

    expect(() => validateReferenceSentences(refs)).toThrow(/6 sentences.*exactly 5/i);
  });

  it("rejects sets with non-consecutive intResponse values", () => {
    const refs: ReferenceSentence[] = [
      { id: "set1", intResponse: 1, sentence: "A" },
      { id: "set1", intResponse: 2, sentence: "B" },
      { id: "set1", intResponse: 3, sentence: "C" },
      { id: "set1", intResponse: 4, sentence: "D" },
      { id: "set1", intResponse: 6, sentence: "F" }, // skips 5
    ];

    expect(() => validateReferenceSentences(refs)).toThrow(/expected exactly \[1, 2, 3, 4, 5\]/i);
  });

  it("rejects sets with duplicate intResponse values", () => {
    const refs: ReferenceSentence[] = [
      { id: "set1", intResponse: 1, sentence: "A" },
      { id: "set1", intResponse: 2, sentence: "B" },
      { id: "set1", intResponse: 3, sentence: "C" },
      { id: "set1", intResponse: 3, sentence: "D" }, // duplicate
      { id: "set1", intResponse: 5, sentence: "E" },
    ];

    expect(() => validateReferenceSentences(refs)).toThrow(/expected exactly \[1, 2, 3, 4, 5\]/i);
  });
});
