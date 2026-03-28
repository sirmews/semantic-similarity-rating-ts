/**
 * Tests for ResponseRater — high-level SSR interface.
 *
 * Port of Python `tests/test_response_rater.py`.
 *
 * All tests use a mock embedding provider (no real model loading)
 * so they run instantly and deterministically.
 */

import { ResponseRater } from "../response-rater";
import type { ReferenceSentence, Matrix } from "../types";
import {
  EMBEDDING_DIM_SMALL,
  LIKERT_SIZE,
  SAMPLE_TEXTS,
  TEST_RESPONSES,
  createTestReferences,
  createTestEmbeddings,
  createMockEmbeddingProvider,
  assertValidPmfVector,
  assertValidPmfMatrix,
  allClose,
  entropy,
} from "./helpers";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Create a rater in embedding mode (no model required). */
async function createEmbeddingModeRater(numSets: number = 1) {
  const refs = createTestReferences(true, numSets, EMBEDDING_DIM_SMALL);
  return ResponseRater.create(refs);
}

/** Create a rater in text mode with a mock provider. */
async function createTextModeRater(numSets: number = 1) {
  const refs = createTestReferences(false, numSets);
  return ResponseRater.create(refs, {
    embeddingProvider: createMockEmbeddingProvider(EMBEDDING_DIM_SMALL),
  });
}

// ===================================================================
// Core functionality (port of TestResponseRaterCore)
// ===================================================================

describe("ResponseRater core", () => {
  it("text mode produces valid PMFs for text inputs", async () => {
    const rater = await createTextModeRater();

    const pmfs = await rater.getResponsePmfs("set1", TEST_RESPONSES);

    assertValidPmfMatrix(pmfs, TEST_RESPONSES.length);
  });

  it("embedding mode produces valid PMFs for embedding inputs", async () => {
    const rater = await createEmbeddingModeRater();

    const testEmbeddings = createTestEmbeddings(
      TEST_RESPONSES.length,
      EMBEDDING_DIM_SMALL,
      99
    );
    const pmfs = await rater.getResponsePmfs("set1", testEmbeddings);

    assertValidPmfMatrix(pmfs, TEST_RESPONSES.length);
  });

  it("survey PMF aggregates individual responses correctly", async () => {
    const rater = await createTextModeRater();

    const individualPmfs = await rater.getResponsePmfs("set1", TEST_RESPONSES);
    const surveyPmf = rater.getSurveyResponsePmf(individualPmfs);

    assertValidPmfVector(surveyPmf);

    // Convenience method should give same result
    const surveyPmfConv = await rater.getSurveyResponsePmfByReferenceSetId(
      "set1",
      TEST_RESPONSES
    );
    expect(allClose(surveyPmf, surveyPmfConv)).toBe(true);
  });
});

// ===================================================================
// Behavioral properties (port of TestResponseRaterBehavior)
// ===================================================================

describe("ResponseRater behavior", () => {
  it("temperature affects distribution sharpness", async () => {
    const rater = await createTextModeRater();
    const response = ["This is great"];

    const sharpPmfs = await rater.getResponsePmfs("set1", response, 0.1);
    const smoothPmfs = await rater.getResponsePmfs("set1", response, 5.0);

    assertValidPmfMatrix(sharpPmfs, 1);
    assertValidPmfMatrix(smoothPmfs, 1);

    expect(entropy(smoothPmfs[0])).toBeGreaterThan(entropy(sharpPmfs[0]));
  });

  it("epsilon regularization changes results", async () => {
    const rater = await createTextModeRater();
    const response = ["Test response"];

    const noEps = await rater.getResponsePmfs("set1", response, 1.0, 0.0);
    const withEps = await rater.getResponsePmfs("set1", response, 1.0, 0.1);

    assertValidPmfMatrix(noEps, 1);
    assertValidPmfMatrix(withEps, 1);
    expect(allClose(noEps[0], withEps[0])).toBe(false);
  });

  it("mean reference aggregates across multiple sets", async () => {
    const rater = await createTextModeRater(2);

    const pmfs = await rater.getResponsePmfs("mean", TEST_RESPONSES);

    assertValidPmfMatrix(pmfs, TEST_RESPONSES.length);
  });
});

// ===================================================================
// Edge cases (port of TestResponseRaterEdgeCases)
// ===================================================================

describe("ResponseRater edge cases", () => {
  it("handles empty text input gracefully", async () => {
    const rater = await createTextModeRater();

    const pmfs = await rater.getResponsePmfs("set1", [] as string[]);

    expect(pmfs).toEqual([]);
  });

  it("handles empty embedding input gracefully", async () => {
    const rater = await createEmbeddingModeRater();

    const pmfs = await rater.getResponsePmfs("set1", [] as Matrix);

    expect(pmfs).toEqual([]);
  });

  it("single text response produces valid PMF", async () => {
    const rater = await createTextModeRater();

    const pmfs = await rater.getResponsePmfs("set1", ["single response"]);

    assertValidPmfMatrix(pmfs, 1);
  });

  it("single embedding response produces valid PMF", async () => {
    const rater = await createEmbeddingModeRater();

    const embedding = createTestEmbeddings(1, EMBEDDING_DIM_SMALL, 77);
    const pmfs = await rater.getResponsePmfs("set1", embedding);

    assertValidPmfMatrix(pmfs, 1);
  });
});

// ===================================================================
// Mode validation (port of TestResponseRaterEdgeCases.test_mode_validation)
// ===================================================================

describe("ResponseRater mode validation", () => {
  it("text mode rejects embedding input", async () => {
    const rater = await createTextModeRater();

    const embeddings = createTestEmbeddings(2, EMBEDDING_DIM_SMALL);
    await expect(
      rater.getResponsePmfs("set1", embeddings)
    ).rejects.toThrow(/text mode/i);
  });

  it("embedding mode rejects text input", async () => {
    const rater = await createEmbeddingModeRater();

    await expect(
      rater.getResponsePmfs("set1", ["text response"] as any)
    ).rejects.toThrow(/embedding mode/i);
  });

  it("nonexistent reference set throws descriptive error", async () => {
    const rater = await createTextModeRater();

    await expect(
      rater.getResponsePmfs("nonexistent", ["test"])
    ).rejects.toThrow(/not found/i);
  });
});

// ===================================================================
// Input validation (port of TestResponseRaterValidation)
// ===================================================================

describe("ResponseRater input validation", () => {
  it("rejects references missing required fields", async () => {
    const badRefs = [
      { id: "set1", intResponse: 1 },
    ] as unknown as ReferenceSentence[];

    await expect(ResponseRater.create(badRefs)).rejects.toThrow(/missing required field/i);
  });

  it("rejects incomplete Likert scale (only 4 entries)", async () => {
    const refs: ReferenceSentence[] = [];
    for (let i = 1; i <= 4; i++) {
      refs.push({ id: "set1", intResponse: i, sentence: SAMPLE_TEXTS[i - 1] });
    }

    await expect(
      ResponseRater.create(refs, {
        embeddingProvider: createMockEmbeddingProvider(),
      })
    ).rejects.toThrow(/4 sentences.*exactly 5/i);
  });
});

// ===================================================================
// Utilities (port of TestResponseRaterUtilities)
// ===================================================================

describe("ResponseRater utilities", () => {
  it("lists available reference sets", async () => {
    const rater = await createTextModeRater(2);

    const sets = rater.availableReferenceSets;

    expect(sets).toContain("set1");
    expect(sets).toContain("set2");
    expect(sets.length).toBe(2);
  });

  it("returns reference sentences for a set", async () => {
    const rater = await createTextModeRater();

    const sentences = rater.getReferenceSentences("set1");

    expect(sentences).toEqual(SAMPLE_TEXTS);
  });

  it("throws for unknown reference set in getReferenceSentences", async () => {
    const rater = await createTextModeRater();

    expect(() => rater.getReferenceSentences("nope")).toThrow(/not found/i);
  });

  it("reports correct mode in modelInfo", async () => {
    const textRater = await createTextModeRater();
    expect(textRater.modelInfo.mode).toBe("text");

    const embedRater = await createEmbeddingModeRater();
    expect(embedRater.modelInfo.mode).toBe("embedding");
  });

  it("reports embedding dimension in modelInfo", async () => {
    const rater = await createEmbeddingModeRater();

    expect(rater.modelInfo.embeddingDimension).toBe(EMBEDDING_DIM_SMALL);
  });

  it("encodeTexts works in text mode", async () => {
    const rater = await createTextModeRater();

    const embeddings = await rater.encodeTexts(["hello", "world"]);

    expect(embeddings.length).toBe(2);
    expect(embeddings[0].length).toBe(EMBEDDING_DIM_SMALL);
  });

  it("encodeTexts throws in embedding mode", async () => {
    const rater = await createEmbeddingModeRater();

    await expect(rater.encodeTexts(["hello"])).rejects.toThrow(/embedding mode/i);
  });
});
