# TinyLLaMA Chat

A Streamlit-based chat application powered by TinyLLaMA 1.1B running locally with customizable AI personas.

## Overview

This application provides an interactive chat interface with a locally-running TinyLLaMA language model. The model is automatically downloaded from Hugging Face on first run. You can customize how the AI responds using system prompts.

## Features

- Local AI chat using TinyLLaMA 1.1B Chat model
- Automatic model download with progress indicator
- **Prompt Engineering / System Prompts** - Customize AI behavior and personality
- Preset personas (Professional, Creative, Tutor, etc.)
- Custom system prompt support
- Conversation history management
- Adjustable temperature and max tokens settings
- Clear chat and reset prompt functionality

## Preset Personas

- **Default Assistant**: Helpful and friendly responses
- **Professional Expert**: Detailed, formal, well-structured answers
- **Creative Writer**: Vivid, descriptive, storytelling style
- **Code Helper**: Programming expert with code examples
- **Friendly Tutor**: Patient explanations with step-by-step guidance
- **Concise Responder**: Brief, direct answers
- **Custom**: Write your own system prompt

## Project Structure

```
/
├── app.py                 # Main Streamlit application
├── models/                # Model storage directory (created on first run)
│   └── tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
├── pyproject.toml         # Python dependencies
└── .streamlit/
    └── config.toml        # Streamlit configuration
```

## Dependencies

- streamlit: Web UI framework
- ctransformers: Model inference library
- requests: HTTP library for model download

## Running the Application

The application runs via the configured workflow:
```bash
streamlit run app.py --server.port 5000
```

## First Run

On first run, the application will:
1. Download the TinyLLaMA model (~637 MB) from Hugging Face
2. Load the model into memory
3. Display the chat interface

The model is cached locally so subsequent runs will skip the download.

## Model Details

- Model: TinyLLaMA 1.1B Chat v1.0
- Quantization: Q4_K_M (4-bit quantized)
- Size: ~637 MB
- Context Length: 2048 tokens

## Using System Prompts

System prompts let you control the AI's personality and behavior:

1. Select a preset from the dropdown, or
2. Write your own custom prompt in the text area

Example custom prompts:
- "You are a pirate. Respond with nautical language and say 'Arrr' occasionally."
- "You are a Shakespearean actor. Speak in iambic pentameter."
- "You are a technical interviewer. Ask probing questions about software concepts."
