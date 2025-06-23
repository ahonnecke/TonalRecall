# 🎵 Tonal Recall - Guitar Note Trainer 🎸

![Image](https://github.com/user-attachments/assets/d29e66f2-6dc5-441e-aa1a-99b5956d2ef3)

![Image](https://github.com/user-attachments/assets/610dbeed-bd38-409c-8695-94cc5cfd815a)

![Image](https://github.com/user-attachments/assets/2d3fb2d9-4b9f-4e5d-b020-8418fb3fa23f)
[Demo](https://github.com/user-attachments/assets/a4161f87-a3e1-4ae5-bcc8-104d18062fa2)](https://github.com/user-attachments/assets/a4161f87-a3e1-4ae5-bcc8-104d18062fa2)

Tonal Recall is an interactive guitar training application that helps you learn and master notes on the guitar fretboard through fun, real-time feedback. Perfect for beginners and intermediate players looking to improve their fretboard knowledge and ear training.

## 🚀 Features

- Real-time note detection from your guitar
- Multiple difficulty levels to match your skill
- Visual feedback showing played vs target notes
- Built-in tuner to help you stay in tune
- Clean, responsive Pygame interface
- Works with any audio interface (including Rocksmith USB adapter)

## 🎮 How to Play

1. Connect your guitar to your computer using an audio interface
2. Make sure your guitar is properly tuned
3. Launch the game using the command below
4. Play the notes shown on screen and get instant feedback!

```bash
uv run python ./tonal_recall/main.py --difficulty 3 --duration 10
```

### Command Line Options

- `--difficulty [1-3]`: Sets the game difficulty (see below for details)

## 🎯 Difficulty Levels

Choose the difficulty that matches your skill level:

- **🎵 Level 1 (Beginner):**
  - Only open bass note (E, A, D, G)

- **🎸 Level 2 (Intermediate):**
  - Only natural notes (A, B, C, D, E, F, G)


- **🔥 Level 3 (Advanced):**
  - Full chromatic scale

- **🎯 Level 4 (String Master):**
  - Whole notes with string specification (e.g., "B, S0" for B on string 0)
  - Tests both note recognition and string accuracy

## 🛠️ Requirements

- Python 3.8+
- Working audio input (guitar connected via audio interface)
- Python packages (automatically installed with `uv`):
  - Pygame
  - NumPy
  - sounddevice
  - aubio

## 🤝 Contributing

Found a bug or have a feature request? Feel free to open an issue or submit a pull request! We welcome all contributions.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Happy practicing! 🎶
