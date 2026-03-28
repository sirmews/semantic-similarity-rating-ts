/**
 * @module response-rater
 *
 * High-level interface for the SSR methodology.
 *
 * The {@link ResponseRater} class is the main entry point for consumers.
 * It mirrors the Python `ResponseRater` class with identical semantics:
 *
 * - **Text mode**: Provide reference sentences without embeddings. The rater
 *   automatically computes embeddings using an {@link EmbeddingProvider}.
 *   Call methods with plain text strings.
 *
 * - **Embedding mode**: Provide reference sentences with pre-computed embeddings.
 *   Call methods with pre-computed embedding matrices. No model loading required.
 *
 * ## Correspondence to Python
 *
 * | Python `response_rater.py`                                  | This module                                  |
 * | ----------------------------------------------------------- | -------------------------------------------- |
 * | `ResponseRater(df, embeddings_column, model_name, device)`  | {@link ResponseRater.create}                 |
 * | `rater.get_response_pmfs(ref_id, responses, T, eps)`        | {@link ResponseRater.getResponsePmfs}        |
 * | `rater.get_survey_response_pmf(pmfs)`                       | {@link ResponseRater.getSurveyResponsePmf}   |
 * | `rater.get_survey_response_pmf_by_reference_set_id(…)`      | {@link ResponseRater.getSurveyResponsePmfByReferenceSetId} |
 * | `rater.encode_texts(texts)`                                 | {@link ResponseRater.encodeTexts}            |
 * | `rater.get_reference_sentences(ref_id)`                     | {@link ResponseRater.getReferenceSentences}  |
 * | `rater.available_reference_sets`                            | {@link ResponseRater.availableReferenceSets} |
 * | `rater.model_info`                                          | {@link ResponseRater.modelInfo}              |
 *
 * ## Testability
 *
 * The rater accepts an {@link EmbeddingProvider} via dependency injection,
 * so tests can supply a deterministic mock instead of loading a real model:
 *
 * ```ts
 * const mockProvider: EmbeddingProvider = {
 *   async encode(texts) {
 *     return texts.map(() => Array(384).fill(0.1));
 *   },
 * };
 * const rater = await ResponseRater.create(refs, { embeddingProvider: mockProvider });
 * ```
 */

import type {
  Vector,
  Matrix,
  ReferenceSentence,
  ProcessedReferenceSet,
  EmbeddingProvider,
  ResponseRaterOptions,
} from "./types";
import { transpose } from "./linalg";
import { responseEmbeddingsToPmf, scalePmf } from "./compute";
import { validateReferenceSentences } from "./validation";
import { TransformersEmbeddingProvider } from "./embeddings";

// ---------------------------------------------------------------------------
// Extended options (includes DI for testing)
// ---------------------------------------------------------------------------

/**
 * Full construction options, extending the public {@link ResponseRaterOptions}
 * with an optional embedding provider for dependency injection.
 */
interface CreateOptions extends ResponseRaterOptions {
  /**
   * Custom embedding provider. When supplied, this is used instead of
   * automatically creating a {@link TransformersEmbeddingProvider}.
   *
   * Primarily used for testing with deterministic mock embeddings.
   */
  embeddingProvider?: EmbeddingProvider;
}

// ---------------------------------------------------------------------------
// ResponseRater
// ---------------------------------------------------------------------------

/**
 * Convert text strings or embeddings into Likert-scale probability distributions
 * using the SSR methodology.
 *
 * ### Construction
 *
 * Use the async factory method {@link ResponseRater.create} instead of `new`:
 *
 * ```ts
 * // Text mode (auto-computes embeddings)
 * const rater = await ResponseRater.create(referenceSentences);
 *
 * // Embedding mode (uses pre-computed embeddings)
 * const rater = await ResponseRater.create(referenceSentencesWithEmbeddings);
 * ```
 *
 * ### Usage
 *
 * ```ts
 * // Get PMFs for LLM responses
 * const pmfs = await rater.getResponsePmfs("set1", [
 *   "I totally agree",
 *   "Not sure about this",
 * ]);
 * // pmfs[0] → [0.02, 0.05, 0.10, 0.35, 0.48]  (agrees)
 * // pmfs[1] → [0.10, 0.15, 0.50, 0.15, 0.10]  (neutral)
 * ```
 */
export class ResponseRater {
  /** Whether the rater is operating in embedding mode (true) or text mode (false). */
  readonly embeddingMode: boolean;

  /** Processed reference sets keyed by their id. */
  private readonly referenceSets: Map<string, ProcessedReferenceSet>;

  /** Reference sentences keyed by set id (includes the synthetic "mean" key). */
  private readonly referenceSentencesMap: Map<string, string[]>;

  /** Embedding provider (null in embedding mode). */
  private readonly provider: EmbeddingProvider | null;

  /**
   * Private constructor — use {@link ResponseRater.create} instead.
   * @internal
   */
  private constructor(
    referenceSets: Map<string, ProcessedReferenceSet>,
    referenceSentencesMap: Map<string, string[]>,
    embeddingMode: boolean,
    provider: EmbeddingProvider | null
  ) {
    this.referenceSets = referenceSets;
    this.referenceSentencesMap = referenceSentencesMap;
    this.embeddingMode = embeddingMode;
    this.provider = provider;
  }

  /**
   * Create and initialise a new ResponseRater.
   *
   * This is an async factory because text mode requires loading an embedding
   * model and computing reference embeddings (both async operations).
   *
   * ### Correspondence to Python
   * Port of `ResponseRater.__init__(df, embeddings_column, model_name, device)`.
   *
   * @param referenceSentences - Array of reference sentences. If any entry has
   *   an `embedding` field, **all** must have one and the rater enters embedding mode.
   *   Otherwise it enters text mode and computes embeddings automatically.
   * @param options - Optional configuration. See {@link CreateOptions}.
   * @returns An initialised `ResponseRater` ready for use.
   *
   * @throws {Error} If reference sentence validation fails (see {@link validateReferenceSentences}).
   *
   * @example
   * ```ts
   * // Text mode
   * const rater = await ResponseRater.create([
   *   { id: "set1", intResponse: 1, sentence: "Strongly disagree" },
   *   { id: "set1", intResponse: 2, sentence: "Disagree" },
   *   { id: "set1", intResponse: 3, sentence: "Neutral" },
   *   { id: "set1", intResponse: 4, sentence: "Agree" },
   *   { id: "set1", intResponse: 5, sentence: "Strongly agree" },
   * ]);
   * ```
   */
  static async create(
    referenceSentences: ReferenceSentence[],
    options: CreateOptions = {}
  ): Promise<ResponseRater> {
    const {
      modelName = "Xenova/all-MiniLM-L6-v2",
      embeddingProvider,
    } = options;

    // Detect mode: if any sentence has an embedding, require all to have one
    const hasAnyEmbedding = referenceSentences.some(
      (s) => s.embedding !== undefined
    );
    const embeddingMode = hasAnyEmbedding;

    // Validate structure
    const validatedSets = validateReferenceSentences(
      referenceSentences,
      embeddingMode
    );

    // Build or use the embedding provider
    let provider: EmbeddingProvider | null = null;
    if (!embeddingMode) {
      provider =
        embeddingProvider ?? new TransformersEmbeddingProvider(modelName);
    }

    // Build the reference sentences map (with the synthetic "mean" key)
    const referenceSentencesMap = new Map<string, string[]>();
    referenceSentencesMap.set("mean", ["1", "2", "3", "4", "5"]);

    // Compute/load embedding matrices for each set
    for (const [id, set] of validatedSets) {
      referenceSentencesMap.set(id, set.sentences);

      let embeddingMatrix: Matrix;

      if (embeddingMode) {
        // Use pre-computed embeddings from the input data
        const setEntries = referenceSentences
          .filter((s) => s.id === id)
          .sort((a, b) => a.intResponse - b.intResponse);
        const embeddings = setEntries.map((s) => s.embedding!);
        // Transpose: [numPoints, embeddingDim] → [embeddingDim, numPoints]
        embeddingMatrix = transpose(embeddings);
      } else {
        // Compute embeddings via the provider
        const embeddings = await provider!.encode(set.sentences);
        // Transpose: [numPoints, embeddingDim] → [embeddingDim, numPoints]
        embeddingMatrix = transpose(embeddings);
      }

      set.embeddingMatrix = embeddingMatrix;
    }

    return new ResponseRater(
      validatedSets,
      referenceSentencesMap,
      embeddingMode,
      provider
    );
  }

  /**
   * Convert LLM responses into probability mass functions over the Likert scale.
   *
   * ### Correspondence to Python
   * Port of `ResponseRater.get_response_pmfs(reference_set_id, llm_responses, temperature, epsilon)`.
   *
   * @param referenceSetId - Which reference set to score against:
   *   - A set id (e.g. `"set1"`) → uses that specific reference set.
   *   - `"mean"` → averages PMFs across all reference sets.
   * @param llmResponses   - The responses to convert:
   *   - **Text mode**: `string[]` of LLM response texts.
   *   - **Embedding mode**: `number[][]` matrix of pre-computed response embeddings,
   *     shape `[numResponses, embeddingDim]`.
   * @param temperature    - Temperature for PMF scaling. Default `1.0` (no scaling).
   *   See {@link scalePmf} for details.
   * @param epsilon        - Regularisation parameter. Default `0.0` (no regularisation).
   *   See {@link responseEmbeddingsToPmf} for details.
   * @returns Matrix of PMFs, shape `[numResponses, numLikertPoints]`.
   *   Each row sums to 1.
   *
   * @throws {Error} If input type doesn't match the rater's mode.
   * @throws {Error} If `referenceSetId` doesn't exist (unless `"mean"`).
   *
   * @example
   * ```ts
   * // Text mode
   * const pmfs = await rater.getResponsePmfs("set1", [
   *   "I totally agree",
   *   "Completely disagree",
   * ]);
   *
   * // With temperature and epsilon
   * const smoothPmfs = await rater.getResponsePmfs(
   *   "set1",
   *   ["Somewhat agree"],
   *   2.0,   // temperature: smoother distribution
   *   0.01   // epsilon: small regularisation
   * );
   * ```
   */
  async getResponsePmfs(
    referenceSetId: string,
    llmResponses: string[] | Matrix,
    temperature: number = 1.0,
    epsilon: number = 0.0
  ): Promise<Matrix> {
    // --- Resolve the response embedding matrix ---
    let llmResponseMatrix: Matrix;

    if (this.embeddingMode) {
      // Embedding mode: expect number[][]
      if (!Array.isArray(llmResponses) || (llmResponses.length > 0 && typeof llmResponses[0] === "string")) {
        throw new Error(
          "ResponseRater is in embedding mode (reference data contains embeddings). " +
            "Expected number[][] of embeddings, got string[]. " +
            "Provide pre-computed embedding vectors instead of text."
        );
      }
      llmResponseMatrix = llmResponses as Matrix;
    } else {
      // Text mode: expect string[]
      if (!Array.isArray(llmResponses) || (llmResponses.length > 0 && typeof llmResponses[0] !== "string")) {
        throw new Error(
          "ResponseRater is in text mode (no embeddings in reference data). " +
            "Expected string[] of text responses, got number[][]. " +
            "Provide text strings instead of embedding vectors."
        );
      }
      const texts = llmResponses as string[];
      if (texts.length === 0) {
        llmResponseMatrix = [];
      } else {
        llmResponseMatrix = await this.provider!.encode(texts);
      }
    }

    // --- Compute PMFs ---
    let pmfs: Matrix;

    if (referenceSetId.toLowerCase() === "mean") {
      // Average PMFs across all reference sets
      const allSets = Array.from(this.referenceSets.values());
      const allPmfs = allSets.map((set) =>
        responseEmbeddingsToPmf(llmResponseMatrix, set.embeddingMatrix, epsilon)
      );

      if (allPmfs.length === 0 || llmResponseMatrix.length === 0) {
        return [];
      }

      // Element-wise mean across all sets
      const numResponses = allPmfs[0].length;
      const numPoints = allPmfs[0][0].length;
      pmfs = Array.from({ length: numResponses }, (_, i) =>
        Array.from({ length: numPoints }, (_, j) => {
          let total = 0;
          for (const setPmfs of allPmfs) {
            total += setPmfs[i][j];
          }
          return total / allPmfs.length;
        })
      );
    } else {
      // Use specific reference set
      const refSet = this.referenceSets.get(referenceSetId);
      if (!refSet) {
        throw new Error(
          `Reference set "${referenceSetId}" not found. ` +
            `Available sets: ${Array.from(this.referenceSets.keys()).join(", ")}.`
        );
      }
      pmfs = responseEmbeddingsToPmf(
        llmResponseMatrix,
        refSet.embeddingMatrix,
        epsilon
      );
    }

    // --- Apply temperature scaling ---
    if (temperature !== 1.0) {
      pmfs = pmfs.map((row) => scalePmf(row, temperature));
    }

    return pmfs;
  }

  /**
   * Aggregate individual response PMFs into a single survey-level PMF
   * by averaging across all responses.
   *
   * ### Correspondence to Python
   * Port of `ResponseRater.get_survey_response_pmf(response_pmfs)`.
   *
   * @param responsePmfs - Matrix of individual response PMFs.
   *   Shape: `[numResponses, numLikertPoints]`.
   * @returns A single PMF vector of length `numLikertPoints`, where each
   *   element is the mean probability across all responses.
   *
   * @example
   * ```ts
   * const pmfs = await rater.getResponsePmfs("set1", responses);
   * const surveyPmf = rater.getSurveyResponsePmf(pmfs);
   * // surveyPmf is a single Vector summing to 1
   * ```
   *
   * @pure This method has no side effects.
   */
  getSurveyResponsePmf(responsePmfs: Matrix): Vector {
    if (responsePmfs.length === 0) return [];

    const numPoints = responsePmfs[0].length;
    const result = new Array(numPoints).fill(0);

    for (const row of responsePmfs) {
      for (let j = 0; j < numPoints; j++) {
        result[j] += row[j];
      }
    }

    const n = responsePmfs.length;
    return result.map((v) => v / n);
  }

  /**
   * Convenience method: compute PMFs for responses and aggregate into
   * a single survey-level PMF in one call.
   *
   * Equivalent to calling {@link getResponsePmfs} followed by {@link getSurveyResponsePmf}.
   *
   * ### Correspondence to Python
   * Port of `ResponseRater.get_survey_response_pmf_by_reference_set_id(…)`.
   *
   * @param referenceSetId - Reference set id or `"mean"`.
   * @param llmResponses   - Text strings (text mode) or embeddings (embedding mode).
   * @param temperature    - Temperature for PMF scaling. Default `1.0`.
   * @param epsilon        - Regularisation parameter. Default `0.0`.
   * @returns A single PMF vector representing the aggregated survey response.
   */
  async getSurveyResponsePmfByReferenceSetId(
    referenceSetId: string,
    llmResponses: string[] | Matrix,
    temperature: number = 1.0,
    epsilon: number = 0.0
  ): Promise<Vector> {
    const pmfs = await this.getResponsePmfs(
      referenceSetId,
      llmResponses,
      temperature,
      epsilon
    );
    return this.getSurveyResponsePmf(pmfs);
  }

  /**
   * Compute embeddings for a list of texts using the loaded model.
   *
   * Only available in **text mode**. In embedding mode, embeddings should
   * be pre-computed externally.
   *
   * ### Correspondence to Python
   * Port of `ResponseRater.encode_texts(texts)`.
   *
   * @param texts - Array of text strings to encode.
   * @returns Matrix of embeddings, shape `[texts.length, embeddingDim]`.
   * @throws {Error} If called in embedding mode.
   */
  async encodeTexts(texts: string[]): Promise<Matrix> {
    if (this.embeddingMode) {
      throw new Error(
        "encodeTexts() is not available in embedding mode. " +
          "Embeddings should be pre-computed and provided directly."
      );
    }
    return this.provider!.encode(texts);
  }

  /**
   * Get the reference sentences for a specific reference set.
   *
   * @param referenceSetId - The reference set id.
   * @returns Array of sentence strings, ordered by Likert point (1→5).
   * @throws {Error} If the reference set id doesn't exist.
   */
  getReferenceSentences(referenceSetId: string): string[] {
    const sentences = this.referenceSentencesMap.get(referenceSetId);
    if (!sentences) {
      throw new Error(
        `Reference set "${referenceSetId}" not found. ` +
          `Available sets: ${Array.from(this.referenceSentencesMap.keys()).join(", ")}.`
      );
    }
    return sentences;
  }

  /**
   * List all available reference set ids.
   *
   * Does **not** include the synthetic `"mean"` key.
   *
   * @returns Array of reference set id strings.
   */
  get availableReferenceSets(): string[] {
    return Array.from(this.referenceSets.keys());
  }

  /**
   * Get metadata about the rater's configuration.
   *
   * @returns Object with mode and dimension information.
   */
  get modelInfo(): Record<string, string | number> {
    const firstSet = Array.from(this.referenceSets.values())[0];
    const embeddingDim = firstSet ? firstSet.embeddingMatrix.length : 0;

    const info: Record<string, string | number> = {
      mode: this.embeddingMode ? "embedding" : "text",
      embeddingDimension: embeddingDim,
    };

    return info;
  }
}
