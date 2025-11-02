# main.py
import os
import asyncio
import json
import requests
from typing import List, Dict, Any
from datetime import datetime
import argparse
from dataclasses import dataclass
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

@dataclass
class ResearchPaper:
    title: str
    authors: List[str]
    abstract: str
    year: int
    citations: int
    url: str
    keywords: List[str]

class SearchAgent:
    def __init__(self):
        self.arxiv_base_url = "http://export.arxiv.org/api/query?"
        
    def search_papers(self, query: str, max_results: int = 10) -> List[ResearchPaper]:
        """Search arXiv for relevant papers"""
        try:
            search_query = f"search_query=all:{query.replace(' ', '+')}"
            url = f"{self.arxiv_base_url}{search_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
            
            response = requests.get(url)
            if response.status_code != 200:
                return self._get_sample_papers(query)
                
            return self._parse_arxiv_response(response.text)
        except:
            return self._get_sample_papers(query)
    
    def _parse_arxiv_response(self, xml_response: str) -> List[ResearchPaper]:
        """Parse arXiv API response"""
        papers = []
        try:
            from xml.etree import ElementTree as ET
            root = ET.fromstring(xml_response)
            
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
                summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
                authors = [author.find('{http://www.w3.org/2005/Atom}name').text 
                          for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
                published = entry.find('{http://www.w3.org/2005/Atom}published').text
                year = int(published[:4])
                url = entry.find('{http://www.w3.org/2005/Atom}id').text
                
                papers.append(ResearchPaper(
                    title=title,
                    authors=authors,
                    abstract=summary,
                    year=year,
                    citations=0,  # arXiv doesn't provide citation count
                    url=url,
                    keywords=[]
                ))
        except:
            # Fallback to sample data
            papers = self._get_sample_papers("machine learning")
            
        return papers
    
    def _get_sample_papers(self, query: str) -> List[ResearchPaper]:
        """Provide sample papers when API fails"""
        sample_papers = [
            ResearchPaper(
                title="Attention Is All You Need",
                authors=["Vaswani", "Shazeer", "Parmar", "Uszkoreit", "Jones", "Gomez", "Kaiser", "Polosukhin"],
                abstract="We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
                year=2017,
                citations=80000,
                url="https://arxiv.org/abs/1706.03762",
                keywords=["transformer", "attention", "neural networks"]
            ),
            ResearchPaper(
                title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                authors=["Devlin", "Chang", "Lee", "Toutanova"],
                abstract="We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.",
                year=2018,
                citations=50000,
                url="https://arxiv.org/abs/1810.04805",
                keywords=["bert", "nlp", "transformer"]
            ),
            ResearchPaper(
                title="Language Models are Few-Shot Learners",
                authors=["Brown", "Mann", "Ryder", "Subbiah", "Kaplan"],
                abstract="We demonstrate that scaling up language models greatly improves task-agnostic, few-shot performance.",
                year=2020,
                citations=12000,
                url="https://arxiv.org/abs/2005.14165",
                keywords=["gpt", "language models", "few-shot learning"]
            )
        ]
        return sample_papers

class AnalysisAgent:
    def __init__(self):
        # Use a smaller model that's more likely to work without GPU
        try:
            self.summarizer = pipeline("summarization", 
                                     model="facebook/bart-large-cnn",
                                     tokenizer="facebook/bart-large-cnn",
                                     framework="pt")
        except:
            self.summarizer = None
            
        self.keyword_patterns = {
            "methodology": ["method", "approach", "technique", "framework", "architecture"],
            "results": ["results", "findings", "experiments", "evaluation", "performance"],
            "contribution": ["contribution", "novel", "propose", "introduce", "new"],
            "limitations": ["limitation", "challenge", "future work", "drawback"]
        }
    
    def analyze_paper(self, paper: ResearchPaper) -> Dict[str, Any]:
        """Analyze a paper and extract key information"""
        analysis = {
            "key_findings": "",
            "methodology": "",
            "contributions": "",
            "limitations": "",
            "relevance_score": 0,
            "summary": "",
            "extracted_keywords": []
        }
        
        # Generate summary
        if self.summarizer and len(paper.abstract) > 50:
            try:
                summary = self.summarizer(paper.abstract, 
                                        max_length=150, 
                                        min_length=30, 
                                        do_sample=False)
                analysis["summary"] = summary[0]['summary_text']
            except:
                analysis["summary"] = paper.abstract[:200] + "..."
        else:
            analysis["summary"] = paper.abstract[:200] + "..."
        
        # Extract key sections
        analysis.update(self._extract_key_sections(paper.abstract))
        
        # Calculate relevance score
        analysis["relevance_score"] = self._calculate_relevance_score(paper)
        
        # Extract keywords
        analysis["extracted_keywords"] = self._extract_keywords(paper.abstract)
        
        return analysis
    
    def _extract_key_sections(self, text: str) -> Dict[str, str]:
        """Extract key sections from paper text"""
        sections = {}
        text_lower = text.lower()
        
        for section, patterns in self.keyword_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    # Find sentence containing the pattern
                    sentences = text.split('.')
                    for sentence in sentences:
                        if pattern in sentence.lower():
                            sections[section] = sentence.strip()
                            break
                    if section not in sections:
                        sections[section] = f"Discusses {section}"
                    break
            if section not in sections:
                sections[section] = "Not explicitly mentioned"
                
        return sections
    
    def _calculate_relevance_score(self, paper: ResearchPaper) -> float:
        """Calculate relevance score based on various factors"""
        score = 0.0
        
        # Recent papers get higher score
        current_year = datetime.now().year
        year_diff = current_year - paper.year
        if year_diff <= 2:
            score += 0.3
        elif year_diff <= 5:
            score += 0.2
        else:
            score += 0.1
            
        # High citation count indicates importance
        if paper.citations > 1000:
            score += 0.4
        elif paper.citations > 100:
            score += 0.3
        elif paper.citations > 10:
            score += 0.2
            
        # Longer abstracts might contain more information
        abstract_length = len(paper.abstract.split())
        if abstract_length > 150:
            score += 0.2
        elif abstract_length > 100:
            score += 0.1
            
        return min(score, 1.0)
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract important keywords from text"""
        words = text.lower().split()
        # Simple frequency-based keyword extraction
        from collections import Counter
        common_words = set(['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'])
        filtered_words = [word for word in words if word not in common_words and len(word) > 3]
        word_freq = Counter(filtered_words)
        return [word for word, count in word_freq.most_common(max_keywords)]

class SynthesisAgent:
    def __init__(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
            self.model.eval()
        except:
            self.tokenizer = None
            self.model = None
    
    def generate_literature_review(self, papers: List[ResearchPaper], analyses: List[Dict]) -> Dict[str, str]:
        """Generate comprehensive literature review"""
        
        if not self.model:
            return self._generate_fallback_review(papers, analyses)
        
        # Prepare context from papers
        context = self._prepare_context(papers, analyses)
        
        try:
            # Generate review using the model
            prompt = f"Write a comprehensive literature review about recent advances in the field. Focus on summarizing key contributions and identifying research gaps.\n\nContext:\n{context}\n\nLiterature Review:"
            
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=1500,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            review = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated part (after the prompt)
            if prompt in review:
                review = review.split(prompt)[-1].strip()
                
        except Exception as e:
            review = self._generate_fallback_review(papers, analyses)["review"]
        
        return {
            "review": review,
            "research_gaps": self._identify_research_gaps(papers, analyses),
            "key_trends": self._identify_trends(papers),
            "future_directions": self._suggest_future_directions(papers, analyses)
        }
    
    def _prepare_context(self, papers: List[ResearchPaper], analyses: List[Dict]) -> str:
        """Prepare context for the literature review generation"""
        context = []
        for i, (paper, analysis) in enumerate(zip(papers, analyses)):
            context.append(f"Paper {i+1}: {paper.title} ({paper.year})")
            context.append(f"Summary: {analysis['summary']}")
            context.append(f"Key Findings: {analysis.get('key_findings', 'N/A')}")
            context.append("---")
        
        return "\n".join(context)
    
    def _identify_research_gaps(self, papers: List[ResearchPaper], analyses: List[Dict]) -> List[str]:
        """Identify research gaps from analyzed papers"""
        gaps = []
        
        # Simple gap identification logic
        methodologies = set()
        domains = set()
        
        for paper, analysis in zip(papers, analyses):
            # Check for methodology diversity
            if "methodology" in analysis:
                methodologies.add(analysis["methodology"].lower())
            
            # Simple domain extraction from title
            title_lower = paper.title.lower()
            if "transformer" in title_lower:
                domains.add("transformer_architectures")
            if "bert" in title_lower:
                domains.add("bert_variants")
            if "gpt" in title_lower:
                domains.add("gpt_models")
        
        if len(methodologies) < 2:
            gaps.append("Limited diversity in methodological approaches")
        if len(domains) < 3:
            gaps.append("Narrow focus on specific model architectures")
        
        gaps.extend([
            "Lack of comprehensive comparative studies across different architectures",
            "Limited research on computational efficiency and scalability",
            "Need for more real-world application studies"
        ])
        
        return gaps[:3]  # Return top 3 gaps
    
    def _identify_trends(self, papers: List[ResearchPaper]) -> List[str]:
        """Identify research trends from papers"""
        trends = []
        years = [paper.year for paper in papers]
        recent_papers = [p for p in papers if p.year >= 2020]
        
        if len(recent_papers) > len(papers) * 0.6:
            trends.append("Growing research interest in recent years")
        
        # Analyze titles for trends
        titles = " ".join([p.title.lower() for p in papers])
        if "transformer" in titles:
            trends.append("Dominance of transformer-based architectures")
        if "pre-training" in titles or "pre-trained" in titles:
            trends.append("Focus on pre-training and transfer learning")
        if "efficient" in titles or "lightweight" in titles:
            trends.append("Emerging interest in efficient model design")
        
        return trends
    
    def _suggest_future_directions(self, papers: List[ResearchPaper], analyses: List[Dict]) -> List[str]:
        """Suggest future research directions"""
        return [
            "Explore hybrid architectures combining different neural network paradigms",
            "Investigate more efficient training methods for large-scale models",
            "Develop better evaluation metrics beyond benchmark performance",
            "Study ethical implications and mitigation strategies",
            "Research domain adaptation for specialized applications"
        ]
    
    def _generate_fallback_review(self, papers: List[ResearchPaper], analyses: List[Dict]) -> Dict[str, str]:
        """Generate fallback review when model is unavailable"""
        review_parts = ["LITERATURE REVIEW\n"]
        
        review_parts.append("INTRODUCTION")
        review_parts.append("This review synthesizes recent advances in the field based on analyzed research papers.")
        
        review_parts.append("\nKEY CONTRIBUTIONS:")
        for i, (paper, analysis) in enumerate(zip(papers, analyses)):
            review_parts.append(f"{i+1}. {paper.title} ({paper.year})")
            review_parts.append(f"   - {analysis['summary']}")
        
        review_parts.append("\nRESEARCH GAPS:")
        gaps = self._identify_research_gaps(papers, analyses)
        for gap in gaps:
            review_parts.append(f"- {gap}")
        
        review_parts.append("\nFUTURE DIRECTIONS:")
        future_dirs = self._suggest_future_directions(papers, analyses)
        for direction in future_dirs[:3]:
            review_parts.append(f"- {direction}")
        
        return {
            "review": "\n".join(review_parts),
            "research_gaps": gaps,
            "key_trends": self._identify_trends(papers),
            "future_directions": future_dirs[:3]
        }

class Orchestrator:
    def __init__(self):
        self.search_agent = SearchAgent()
        self.analysis_agent = AnalysisAgent()
        self.synthesis_agent = SynthesisAgent()
        self.evaluation_results = {}
    
    async def process_literature_review(self, query: str, max_papers: int = 5) -> Dict[str, Any]:
        """Orchestrate the complete literature review process"""
        print(f" Starting literature review for: {query}")
        
        # Step 1: Search for papers
        print(" Searching for relevant papers...")
        papers = self.search_agent.search_papers(query, max_papers)
        print(f" Found {len(papers)} papers")
        
        # Step 2: Analyze papers
        print(" Analyzing papers...")
        analyses = []
        for i, paper in enumerate(papers):
            print(f"   Analyzing paper {i+1}/{len(papers)}: {paper.title[:50]}...")
            analysis = self.analysis_agent.analyze_paper(paper)
            analyses.append(analysis)
        
        # Step 3: Generate literature review
        print(" Synthesizing literature review...")
        literature_review = self.synthesis_agent.generate_literature_review(papers, analyses)
        
        # Step 4: Evaluate results
        print(" Evaluating results...")
        self.evaluation_results = self._evaluate_results(papers, analyses, literature_review)
        
        return {
            "query": query,
            "papers": papers,
            "analyses": analyses,
            "literature_review": literature_review,
            "evaluation": self.evaluation_results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _evaluate_results(self, papers: List[ResearchPaper], analyses: List[Dict], review: Dict) -> Dict[str, float]:
        """Evaluate the quality of the literature review"""
        scores = {}
        
        # Coverage score - based on number of papers analyzed
        scores["coverage"] = min(len(papers) / 10, 1.0)
        
        # Diversity score - based on publication years
        years = [paper.year for paper in papers]
        year_range = max(years) - min(years) if years else 0
        scores["temporal_diversity"] = min(year_range / 10, 1.0)
        
        # Relevance score - average of paper relevance scores
        relevance_scores = [analysis["relevance_score"] for analysis in analyses]
        scores["average_relevance"] = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        # Content quality - based on review length and structure
        review_text = review["review"]
        scores["content_quality"] = min(len(review_text) / 1000, 1.0)
        
        # Gap identification - based on number of identified gaps
        scores["gap_analysis"] = min(len(review["research_gaps"]) / 5, 1.0)
        
        # Overall score
        scores["overall_score"] = sum(scores.values()) / len(scores)
        
        return scores

class CLIInterface:
    def __init__(self):
        self.orchestrator = Orchestrator()
    
    def display_results(self, results: Dict[str, Any]):
        """Display results in a formatted way"""
        print("\n" + "="*80)
        print(" LITERATURE REVIEW RESULTS")
        print("="*80)
        
        print(f"\n Query: {results['query']}")
        print(f" Generated: {results['timestamp']}")
        
        print(f"\n EVALUATION SCORES:")
        for metric, score in results['evaluation'].items():
            print(f"   {metric.replace('_', ' ').title()}: {score:.2f}")
        
        print(f"\n ANALYZED PAPERS:")
        for i, (paper, analysis) in enumerate(zip(results['papers'], results['analyses'])):
            print(f"\n{i+1}. {paper.title}")
            print(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
            print(f"   Year: {paper.year} | Citations: {paper.citations}")
            print(f"   Relevance Score: {analysis['relevance_score']:.2f}")
            print(f"   Summary: {analysis['summary'][:100]}...")
        
        print(f"\n LITERATURE REVIEW:")
        print("-" * 40)
        print(results['literature_review']['review'])
        
        print(f"\n RESEARCH GAPS IDENTIFIED:")
        for gap in results['literature_review']['research_gaps']:
            print(f"   • {gap}")
        
        print(f"\n KEY TRENDS:")
        for trend in results['literature_review']['key_trends']:
            print(f"   • {trend}")
        
        print(f"\n FUTURE DIRECTIONS:")
        for direction in results['literature_review']['future_directions']:
            print(f"   • {direction}")
    
    async def run(self):
        """Run the CLI interface"""
        print(" AI Literature Review Agent")
        print("="*40)
        
        while True:
            print("\nEnter your research topic (or 'quit' to exit):")
            query = input("> ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            try:
                results = await self.orchestrator.process_literature_review(query)
                self.display_results(results)
                
                # Save results to file
                self.save_results(results, query)
                
            except Exception as e:
                print(f" Error: {e}")
                print("Please try again with a different query.")
    
    def save_results(self, results: Dict[str, Any], query: str):
        """Save results to JSON file"""
        filename = f"literature_review_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert dataclasses to dictionaries for JSON serialization
        serializable_results = {
            "query": results["query"],
            "timestamp": results["timestamp"],
            "evaluation": results["evaluation"],
            "literature_review": results["literature_review"],
            "papers": [
                {
                    "title": paper.title,
                    "authors": paper.authors,
                    "abstract": paper.abstract,
                    "year": paper.year,
                    "citations": paper.citations,
                    "url": paper.url,
                    "keywords": paper.keywords
                }
                for paper in results["papers"]
            ],
            "analyses": results["analyses"]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n Results saved to: {filename}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='AI Literature Review Agent')
    parser.add_argument('--query', type=str, help='Research topic to analyze')
    parser.add_argument('--papers', type=int, default=5, help='Number of papers to analyze')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    if args.query:
        # Run single query
        orchestrator = Orchestrator()
        results = asyncio.run(orchestrator.process_literature_review(args.query, args.papers))
        
        cli = CLIInterface()
        cli.display_results(results)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        # Run interactive mode
        cli = CLIInterface()
        asyncio.run(cli.run())

if __name__ == "__main__":
    main()