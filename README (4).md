# AI Literature Review Agent
Author: Ayush Kumar  
University: Indian Institute of Technology Kanpur  
Department: Electrical Engineering

A functional AI agent that automates the academic literature review process by searching, analyzing, synthesizing, and evaluating research papers into structured outputs.

## Overview

This system streamlines the literature review workflow by:
- Retrieving relevant research papers
- Extracting key insights from academic texts
- Generating coherent literature summaries
- Evaluating review quality using established metrics
- Exporting results for further analysis

It is designed for researchers, students, and data practitioners who require rapid aggregation of academic knowledge.

## Features

- Automated paper search with fallback sources
- Key information extraction from research texts
- Literature review synthesis
- Output evaluation using standard metrics
- Export of structured results to JSON
- Command line interface for interactive or batch use
- Fine-tuning support for specific academic domains
- Error handling for unavailable or incomplete data sources

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Interactive Mode

```bash
python main.py
```

### Single Query Mode

```bash
python main.py --query "transformer neural networks" --papers 5
```

This will:
- Search for relevant academic sources
- Analyze research text
- Produce a literature review
- Save results to JSON format

## Fine-Tuning

The system supports fine-tuning to enhance:
- Academic writing style
- Domain-specific terminology
- Coherence of synthesized reviews

Run the example:

```bash
python fine_tuning_example.py
```

## Folder Structure

```
├── main.py
├── fine_tuning_example.py
├── requirements.txt
├── src/
│   ├── search.py
│   ├── analyzer.py
│   ├── synthesizer.py
│   └── evaluator.py
├── outputs/
```

## Architecture

The system is composed of four modules:

- Search Module: Retrieves relevant papers and metadata
- Analysis Module: Extracts key information and insights
- Synthesis Module: Produces coherent literature summaries
- Evaluation Module: Scores output using established metrics

This modular design ensures maintainability and scalability.

## Example Output (Truncated)

```json
{
  "query": "transformer neural networks",
  "papers_analyzed": 5,
  "summary": "Transformer architectures have significantly advanced natural language processing by enabling parallel sequence processing...",
  "evaluation_score": {
    "rouge_1": 0.87,
    "coherence": 0.91
  }
}
```

## Error Handling

The system manages:
- Network failures
- Missing metadata
- Limited dataset availability
- Incomplete academic references

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

Contributions are encouraged.


