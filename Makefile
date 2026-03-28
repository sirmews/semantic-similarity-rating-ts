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

.PHONY: changelog
changelog: ## Generate CHANGELOG.md from conventional commits
	npx changelogen --output CHANGELOG.md

.PHONY: release-patch
release-patch: preflight changelog ## Release patch (1.0.0 → 1.0.1): changelog, bump, commit & tag
	git add CHANGELOG.md
	npm version patch -m "release: %s"

.PHONY: release-minor
release-minor: preflight changelog ## Release minor (1.0.0 → 1.1.0): changelog, bump, commit & tag
	git add CHANGELOG.md
	npm version minor -m "release: %s"

.PHONY: release-major
release-major: preflight changelog ## Release major (1.0.0 → 2.0.0): changelog, bump, commit & tag
	git add CHANGELOG.md
	npm version major -m "release: %s"

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
