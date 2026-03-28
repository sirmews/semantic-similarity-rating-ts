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

  it("accepts a 3-point reference set", () => {
    const refs: ReferenceSentence[] = [
      { id: "set1", intResponse: 1, sentence: "Disagree" },
      { id: "set1", intResponse: 2, sentence: "Neutral" },
      { id: "set1", intResponse: 3, sentence: "Agree" },
    ];

    const result = validateReferenceSentences(refs);

    expect(result.size).toBe(1);
    expect(result.get("set1")!.sentences).toEqual(["Disagree", "Neutral", "Agree"]);
  });

  it("accepts a 7-point reference set", () => {
    const refs: ReferenceSentence[] = [];
    for (let i = 1; i <= 7; i++) {
      refs.push({ id: "set1", intResponse: i, sentence: `Point ${i}` });
    }

    const result = validateReferenceSentences(refs);

    expect(result.size).toBe(1);
    expect(result.get("set1")!.sentences).toHaveLength(7);
  });

  it("accepts a 2-point (minimum) reference set", () => {
    const refs: ReferenceSentence[] = [
      { id: "set1", intResponse: 1, sentence: "No" },
      { id: "set1", intResponse: 2, sentence: "Yes" },
    ];

    const result = validateReferenceSentences(refs);

    expect(result.size).toBe(1);
    expect(result.get("set1")!.sentences).toEqual(["No", "Yes"]);
  });

  it("accepts non-1-based contiguous sequences (e.g. 0-indexed)", () => {
    const refs: ReferenceSentence[] = [
      { id: "set1", intResponse: 0, sentence: "Zero" },
      { id: "set1", intResponse: 1, sentence: "One" },
      { id: "set1", intResponse: 2, sentence: "Two" },
    ];

    const result = validateReferenceSentences(refs);

    expect(result.get("set1")!.sentences).toEqual(["Zero", "One", "Two"]);
  });

  it("accepts negative-start contiguous sequences (e.g. -2 to 2)", () => {
    const refs: ReferenceSentence[] = [
      { id: "set1", intResponse: -2, sentence: "Strongly disagree" },
      { id: "set1", intResponse: -1, sentence: "Disagree" },
      { id: "set1", intResponse: 0, sentence: "Neutral" },
      { id: "set1", intResponse: 1, sentence: "Agree" },
      { id: "set1", intResponse: 2, sentence: "Strongly agree" },
    ];

    const result = validateReferenceSentences(refs);

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

  it("accepts multiple reference sets with different N sizes", () => {
    const refs: ReferenceSentence[] = [
      // 3-point set
      { id: "small", intResponse: 1, sentence: "Low" },
      { id: "small", intResponse: 2, sentence: "Mid" },
      { id: "small", intResponse: 3, sentence: "High" },
      // 7-point set
      ...Array.from({ length: 7 }, (_, i) => ({
        id: "large",
        intResponse: i + 1,
        sentence: `Point ${i + 1}`,
      })),
    ];

    const result = validateReferenceSentences(refs);

    expect(result.size).toBe(2);
    expect(result.get("small")!.sentences).toHaveLength(3);
    expect(result.get("large")!.sentences).toHaveLength(7);
  });

  it("accepts an empty input array", () => {
    const result = validateReferenceSentences([]);
    expect(result.size).toBe(0);
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

  it("rejects sets with only 1 entry", () => {
    const refs: ReferenceSentence[] = [
      { id: "set1", intResponse: 1, sentence: "A" },
    ];

    expect(() => validateReferenceSentences(refs)).toThrow(/1 sentence.*at least 2/i);
  });

  it("rejects sets with non-consecutive intResponse values", () => {
    const refs: ReferenceSentence[] = [
      { id: "set1", intResponse: 1, sentence: "A" },
      { id: "set1", intResponse: 2, sentence: "B" },
      { id: "set1", intResponse: 4, sentence: "D" }, // skips 3
    ];

    expect(() => validateReferenceSentences(refs)).toThrow(/contiguous sequence/i);
  });

  it("rejects sets with duplicate intResponse values", () => {
    const refs: ReferenceSentence[] = [
      { id: "set1", intResponse: 1, sentence: "A" },
      { id: "set1", intResponse: 2, sentence: "B" },
      { id: "set1", intResponse: 2, sentence: "C" }, // duplicate
    ];

    expect(() => validateReferenceSentences(refs)).toThrow(/contiguous sequence/i);
  });

  it("rejects sets with non-integer intResponse values", () => {
    const refs: ReferenceSentence[] = [
      { id: "set1", intResponse: 1.5, sentence: "A" },
      { id: "set1", intResponse: 2.5, sentence: "B" },
      { id: "set1", intResponse: 3.5, sentence: "C" },
    ];

    expect(() => validateReferenceSentences(refs)).toThrow(/integer/i);
  });

  it("rejects sets with gaps in a larger scale", () => {
    const refs: ReferenceSentence[] = [
      { id: "set1", intResponse: 1, sentence: "A" },
      { id: "set1", intResponse: 2, sentence: "B" },
      { id: "set1", intResponse: 3, sentence: "C" },
      { id: "set1", intResponse: 4, sentence: "D" },
      { id: "set1", intResponse: 6, sentence: "F" }, // skips 5
    ];

    expect(() => validateReferenceSentences(refs)).toThrow(/contiguous sequence/i);
  });
});
