.PHONY: test test-watch test-debug

# Run all tests
test:
	PYTHONPATH=./ pytest -s

# Watch for file changes and run tests
test.watch:
	PYTHONPATH=./ ptw --now . --

# Debug test failures
test.logs:
	PYTHONPATH=./ pytest -v -o log_cli=true -o log_cli_level=DEBUG

# Debug test failures
test.pdb:
	PYTHONPATH=./ pytest -v --pdb -o log_cli=true -o log_cli_level=DEBUG

test.realnotes.logs:
	PYTHONPATH=./ pytest -v -o log_cli=true -o log_cli_level=DEBUG tonal_recall/tests/test_audio_processing.py

device.list:
	lsusb

device.rocksmith:
	lsusb | grep Rocksmith
