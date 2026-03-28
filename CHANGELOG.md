# Changelog


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
