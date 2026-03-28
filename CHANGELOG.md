# Changelog


## v1.2.0

### Features

- Support **N-point Likert scales** — reference sets now accept any scale with 2 or more points (e.g. 3-point, 7-point, 11-point) instead of requiring exactly 5
- `intResponse` values may start at any integer and must form a contiguous sequence with step 1 (e.g. `[1,2,3]`, `[-2,-1,0,1,2]`)

### Tests

- Add validation tests for 2-point, 3-point, 7-point, 0-indexed, and negative-start Likert scales
- Update structural error tests to match new N-point validation rules

## v1.1.0

### Features

- Add `OpenAIEmbeddingProvider` — call the OpenAI embeddings API using native `fetch` (zero extra dependencies)
- Supports custom model (`text-embedding-3-small` default), custom `baseURL` for OpenAI-compatible APIs
- Export `OpenAIEmbeddingProviderOptions` type

### Tests

- Add unit tests for `OpenAIEmbeddingProvider` (mocked fetch)
- Add integration test suite (`npm run test:integration`) for end-to-end validation with real API
- Separate `test` and `test:integration` npm scripts

## v1.0.0

- Initial release
