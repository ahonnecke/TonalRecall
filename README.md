# Tonal Recall

![Demo Video](https://github.com/user-attachments/assets/2f598bf9-fe6c-48e3-9457-852a3537ba37)

## How to Run

Run the following command to start the game with the Pygame UI at difficulty level 3:

```bash
uv run python ./tonal_recall/main.py --ui pygame --difficulty 3
```

- `--ui pygame`: Launches the graphical interface using Pygame
- `--difficulty 3`: Sets the game difficulty to level 3 (see below for difficulty details)

## About Difficulty Levels

The game supports multiple difficulty levels. Higher difficulty levels may introduce more notes, faster sequences, or more complex patterns. Example breakdown:

- **Difficulty 1:** Basic note recognition, slow tempo, fewer notes
- **Difficulty 2:** Moderate difficulty, more notes, slightly faster
- **Difficulty 3:** Advanced, full set of notes, increased speed and complexity

Adjust the `--difficulty` parameter to match your skill or training needs!
