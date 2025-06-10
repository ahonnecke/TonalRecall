# Refactor

This game, currently works with this entrypoint:
uv run python ./tonal_recall/main.py --ui pygame --difficulty 3 --duration 60

it succesfully ingests audio, detects notes and allows the user to play a
"flashcard" game.

The current goal is to retain the functionality, but to refactor such that the
detection module is seperated from the flashcard frontend.

Long term the goal is for this backend note detection to be able to be re-used
for different frontend games.

You absolutely cannot just jump into this and go bashing about changing things,
there needs to be a meticulous and safe approach.

Because testing with an instrument is time consuming, craft a plan for very
carefully and safely extracting the note detection.
