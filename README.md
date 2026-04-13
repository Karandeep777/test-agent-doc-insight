# Document Insight & Summarization Agent

## Overview
Document Insight & Summarization Agent is a professional data_analytics agent designed for text interactions.

## Features


## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the agent:
```bash
python agent.py
```

## Configuration

The agent uses the following environment variables:
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key (if using Anthropic)
- `GOOGLE_API_KEY`: Google API key (if using Google)

## Usage

```python
from agent import Document Insight & Summarization AgentAgent

agent = Document Insight & Summarization AgentAgent()
response = await agent.process_message("Hello!")
```

## Domain: data_analytics
## Personality: professional
## Modality: text