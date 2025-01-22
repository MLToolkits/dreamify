change=patch
message=RELEASE

publish:
	@poetry version $(change)
	@version=$$(poetry version -s) && \
	poetry build && \
	git tag v$$version && \
	git add . && \
    git commit -m "$(message)" && \
	git push origin v$$version