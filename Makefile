.PHONY: test test-watch test-debug

# Run all tests
test:
	PYTHONPATH=./ pytest -v tests/

# Watch for file changes and run tests
test-watch:
	PYTHONPATH=./ ptw --now . -- tests/

# Debug test failures
test-debug:
	PYTHONPATH=./ pytest -v --pdb tests/

# Run only note detection tests
note-tests:
	PYTHONPATH=./ pytest -v tests/note_detection/
