### `README.md`

**Sentient Dialog** is a lightweight NPC personality engine using DialoGPT fine-tuning. This repo includes tools to generate engaging, character-driven dialogue for games and narrative experiences.

#### Features

* Prepares datasets for fine-tuning DialoGPT.
* Customizable personalities via `npc_dialog_data.txt`.
* Example NPC: A flirtatious bard and a Kind Autority figure, Just remove what you want from the comment

#### Files

* `train_dialog.py`
* `npc_dialog_data.txt`

#### Usage

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Run training prep:

   ```bash
   python train_dialog.py
   ```

#### Example Personality

> **Lyric**, flirtatious bard
> “Trust me to lie beautifully, kiss honestly, and disappear poetically.”

#### Model

Base: `microsoft/DialoGPT-small`
Ready for integration into Unity, Unreal, or chatbot frameworks.

---
