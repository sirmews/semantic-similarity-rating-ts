/**
 * Integration tests for OpenAIEmbeddingProvider against the real OpenAI API.
 *
 * Skipped unless OPENAI_API_KEY is set in the environment.
 *
 * Run with:
 *   OPENAI_API_KEY=sk-... npx jest openai-integration
 */

import { OpenAIEmbeddingProvider } from "../openai-embeddings";
import { ResponseRater } from "../response-rater";
import type { ReferenceSentence } from "../types";

const apiKey = process.env.OPENAI_API_KEY;

const describeIfKey = apiKey ? describe : describe.skip;

// Allow generous timeout for API calls
jest.setTimeout(30_000);

describeIfKey("OpenAIEmbeddingProvider integration", () => {
  const provider = new OpenAIEmbeddingProvider({ apiKey: apiKey! });

  it("encodes a single text and returns the correct shape", async () => {
    const result = await provider.encode(["Hello world"]);

    expect(result.length).toBe(1);
    // text-embedding-3-small returns 1536 dimensions
    expect(result[0].length).toBe(1536);
    // Values should be finite numbers
    expect(result[0].every((v) => Number.isFinite(v))).toBe(true);
  });

  it("encodes a batch and preserves input order", async () => {
    const texts = ["I love cats", "I love dogs", "The weather is nice"];
    const result = await provider.encode(texts);

    expect(result.length).toBe(3);
    expect(result.every((row) => row.length === 1536)).toBe(true);

    // "I love cats" and "I love dogs" should be more similar to each other
    // than either is to "The weather is nice"
    const cosineSim = (a: number[], b: number[]) => {
      let dot = 0, normA = 0, normB = 0;
      for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
      }
      return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    };

    const simCatsDogs = cosineSim(result[0], result[1]);
    const simCatsWeather = cosineSim(result[0], result[2]);
    expect(simCatsDogs).toBeGreaterThan(simCatsWeather);
  });

  it("works end-to-end with ResponseRater", async () => {
    const refs: ReferenceSentence[] = [
      { id: "set1", intResponse: 1, sentence: "Strongly disagree" },
      { id: "set1", intResponse: 2, sentence: "Disagree" },
      { id: "set1", intResponse: 3, sentence: "Neutral" },
      { id: "set1", intResponse: 4, sentence: "Agree" },
      { id: "set1", intResponse: 5, sentence: "Strongly agree" },
    ];

    const rater = await ResponseRater.create(refs, {
      embeddingProvider: provider,
    });

    const pmfs = await rater.getResponsePmfs("set1", [
      "I completely agree with this statement",
      "I have no opinion either way",
    ]);

    expect(pmfs.length).toBe(2);

    // Each PMF should sum to ~1
    for (const pmf of pmfs) {
      expect(pmf.length).toBe(5);
      const total = pmf.reduce((a, b) => a + b, 0);
      expect(total).toBeCloseTo(1.0, 5);
    }

    // "I completely agree" should have highest weight on points 4 or 5
    const agreeArgmax = pmfs[0].indexOf(Math.max(...pmfs[0]));
    expect(agreeArgmax).toBeGreaterThanOrEqual(3); // 0-indexed: 3=Agree, 4=Strongly agree
  });

  it("temperature affects distribution sharpness", async () => {
    const refs: ReferenceSentence[] = [
      { id: "set1", intResponse: 1, sentence: "Strongly disagree" },
      { id: "set1", intResponse: 2, sentence: "Disagree" },
      { id: "set1", intResponse: 3, sentence: "Neutral" },
      { id: "set1", intResponse: 4, sentence: "Agree" },
      { id: "set1", intResponse: 5, sentence: "Strongly agree" },
    ];

    const rater = await ResponseRater.create(refs, {
      embeddingProvider: provider,
    });

    const response = ["I mostly agree"];
    const sharpPmfs = await rater.getResponsePmfs("set1", response, 0.1);
    const smoothPmfs = await rater.getResponsePmfs("set1", response, 5.0);

    // Sharp distribution should have lower entropy than smooth
    const entropy = (p: number[]) =>
      -p.reduce((s, v) => s + (v > 0 ? v * Math.log(v) : 0), 0);

    expect(entropy(smoothPmfs[0])).toBeGreaterThan(entropy(sharpPmfs[0]));
  });

  it("epsilon regularization changes results", async () => {
    const refs: ReferenceSentence[] = [
      { id: "set1", intResponse: 1, sentence: "Strongly disagree" },
      { id: "set1", intResponse: 2, sentence: "Disagree" },
      { id: "set1", intResponse: 3, sentence: "Neutral" },
      { id: "set1", intResponse: 4, sentence: "Agree" },
      { id: "set1", intResponse: 5, sentence: "Strongly agree" },
    ];

    const rater = await ResponseRater.create(refs, {
      embeddingProvider: provider,
    });

    const response = ["Test response"];
    const noEps = await rater.getResponsePmfs("set1", response, 1.0, 0.0);
    const withEps = await rater.getResponsePmfs("set1", response, 1.0, 0.1);

    // Results should differ
    const differs = noEps[0].some(
      (v, i) => Math.abs(v - withEps[0][i]) > 1e-6
    );
    expect(differs).toBe(true);
  });

  it("survey PMF aggregates correctly", async () => {
    const refs: ReferenceSentence[] = [
      { id: "set1", intResponse: 1, sentence: "Strongly disagree" },
      { id: "set1", intResponse: 2, sentence: "Disagree" },
      { id: "set1", intResponse: 3, sentence: "Neutral" },
      { id: "set1", intResponse: 4, sentence: "Agree" },
      { id: "set1", intResponse: 5, sentence: "Strongly agree" },
    ];

    const rater = await ResponseRater.create(refs, {
      embeddingProvider: provider,
    });

    const responses = ["I agree", "I disagree", "Not sure"];
    const pmfs = await rater.getResponsePmfs("set1", responses);
    const surveyPmf = rater.getSurveyResponsePmf(pmfs);

    expect(surveyPmf.length).toBe(5);
    const total = surveyPmf.reduce((a, b) => a + b, 0);
    expect(total).toBeCloseTo(1.0, 5);

    // Convenience method should give same result
    const surveyPmfConv = await rater.getSurveyResponsePmfByReferenceSetId(
      "set1",
      responses
    );
    expect(surveyPmfConv.length).toBe(5);
    surveyPmf.forEach((v, i) => {
      expect(v).toBeCloseTo(surveyPmfConv[i], 10);
    });
  });

  it("mean reference aggregates across multiple sets", async () => {
    const refs: ReferenceSentence[] = [
      { id: "set1", intResponse: 1, sentence: "Strongly disagree" },
      { id: "set1", intResponse: 2, sentence: "Disagree" },
      { id: "set1", intResponse: 3, sentence: "Neutral" },
      { id: "set1", intResponse: 4, sentence: "Agree" },
      { id: "set1", intResponse: 5, sentence: "Strongly agree" },
      { id: "set2", intResponse: 1, sentence: "Terrible" },
      { id: "set2", intResponse: 2, sentence: "Bad" },
      { id: "set2", intResponse: 3, sentence: "Okay" },
      { id: "set2", intResponse: 4, sentence: "Good" },
      { id: "set2", intResponse: 5, sentence: "Excellent" },
    ];

    const rater = await ResponseRater.create(refs, {
      embeddingProvider: provider,
    });

    expect(rater.availableReferenceSets).toContain("set1");
    expect(rater.availableReferenceSets).toContain("set2");

    const pmfs = await rater.getResponsePmfs("mean", ["This is great"]);

    expect(pmfs.length).toBe(1);
    expect(pmfs[0].length).toBe(5);
    const total = pmfs[0].reduce((a, b) => a + b, 0);
    expect(total).toBeCloseTo(1.0, 5);
  });
});
