.DEFAULT_GOAL := help

## — Dependencies —

.PHONY: install
install: ## Install dependencies
	npm install

## — Development —

.PHONY: build
build: ## Compile TypeScript to lib/
	npm run build

.PHONY: test
test: ## Run all tests
	npx jest

.PHONY: typecheck
typecheck: ## Type-check without emitting
	npx tsc --noEmit

.PHONY: clean
clean: ## Remove build artifacts
	rm -rf lib

## — Release —

.PHONY: version-patch
version-patch: ## Bump patch version (1.0.0 → 1.0.1), commit & tag
	npm version patch

.PHONY: version-minor
version-minor: ## Bump minor version (1.0.0 → 1.1.0), commit & tag
	npm version minor

.PHONY: version-major
version-major: ## Bump major version (1.0.0 → 2.0.0), commit & tag
	npm version major

.PHONY: pack
pack: ## Preview the npm tarball contents (dry run)
	npm pack --dry-run

.PHONY: publish
publish: preflight ## Publish to npm (runs tests and build first)
	npm publish --access public

.PHONY: publish-dry
publish-dry: preflight ## Dry-run publish (verify everything without uploading)
	npm publish --access public --dry-run

## — CI / Checks —

.PHONY: preflight
preflight: clean install typecheck test build ## Run all checks before publishing
	@echo "✅ All preflight checks passed"

## — Help —

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
