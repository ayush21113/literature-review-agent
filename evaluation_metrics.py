# evaluation_metrics.py
import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics.precision_recall_fscore_support import precision_recall_fscore_support
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from typing import List, Dict, Any
import re
from dataclasses import dataclass
from collections import Counter

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

@dataclass
class EvaluationResult:
    overall_score: float
    content_coverage: float
    coherence_score: float
    gap_analysis_quality: float
    structure_quality: float
    academic_tone_score: float
    details: Dict[str, Any]

class EvaluationMetrics:
    def __init__(self):
        try:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.stop_words = set(nltk.corpus.stopwords.words('english'))
        except:
            self.rouge_scorer = None
            self.stop_words = set(['the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but'])
    
    def calculate_rouge_scores(self, generated_text: str, reference_text: str) -> Dict[str, float]:
        """Calculate ROUGE scores for text generation quality"""
        if not self.rouge_scorer:
            return {'rouge1': 0.5, 'rouge2': 0.3, 'rougeL': 0.4}
        
        try:
            scores = self.rouge_scorer.score(reference_text, generated_text)
            return {key: scores[key].fmeasure for key in scores}
        except:
            return {'rouge1': 0.5, 'rouge2': 0.3, 'rougeL': 0.4}
    
    def calculate_content_coverage(self, papers: List[Dict], review: str) -> float:
        """Calculate how well the review covers the paper content"""
        if not papers:
            return 0.0
            
        all_key_terms = set()
        mentioned_terms = set()
        
        # Extract key terms from papers (titles + abstracts)
        for paper in papers:
            # Use title words as key terms
            if 'title' in paper:
                title_terms = self._extract_key_terms(paper['title'])
                all_key_terms.update(title_terms)
            
            # Use abstract words as key terms
            if 'abstract' in paper:
                abstract_terms = self._extract_key_terms(paper['abstract'])[:10]  # Top 10 terms
                all_key_terms.update(abstract_terms)
        
        # Extract terms from review
        review_terms = self._extract_key_terms(review)
        
        # Calculate coverage
        mentioned_terms = all_key_terms.intersection(review_terms)
        
        if not all_key_terms:
            return 0.0
            
        coverage = len(mentioned_terms) / len(all_key_terms)
        return min(coverage, 1.0)
    
    def _extract_key_terms(self, text: str, max_terms: int = 20) -> List[str]:
        """Extract key terms from text"""
        if not text:
            return []
        
        # Convert to lowercase and tokenize
        words = word_tokenize(text.lower())
        
        # Remove stopwords and short words
        filtered_words = [
            word for word in words 
            if word not in self.stop_words 
            and len(word) > 3 
            and word.isalpha()
        ]
        
        # Get most frequent terms
        word_freq = Counter(filtered_words)
        return [word for word, count in word_freq.most_common(max_terms)]
    
    def calculate_coherence_score(self, review_text: str) -> float:
        """Calculate coherence score based on text structure and flow"""
        if not review_text:
            return 0.0
        
        sentences = nltk.sent_tokenize(review_text)
        if len(sentences) < 3:
            return 0.3
        
        # Check for transition words (indicates good flow)
        transition_words = [
            'however', 'therefore', 'moreover', 'furthermore', 'consequently',
            'additionally', 'similarly', 'conversely', 'nevertheless', 'thus'
        ]
        
        transition_count = 0
        for sentence in sentences:
            if any(transition in sentence.lower() for transition in transition_words):
                transition_count += 1
        
        transition_score = transition_count / len(sentences)
        
        # Check sentence length variation (indicates good writing style)
        sentence_lengths = [len(word_tokenize(sent)) for sent in sentences]
        length_variance = np.var(sentence_lengths)
        length_score = min(length_variance / 100, 1.0)  # Normalize
        
        # Check paragraph structure (rough estimate)
        paragraph_count = review_text.count('\n\n') + 1
        paragraph_score = min(paragraph_count / 5, 1.0)
        
        return (transition_score * 0.4 + length_score * 0.3 + paragraph_score * 0.3)
    
    def calculate_gap_analysis_quality(self, research_gaps: List[str]) -> float:
        """Evaluate the quality of identified research gaps"""
        if not research_gaps:
            return 0.0
        
        gap_keywords = [
            'lack', 'limited', 'missing', 'gap', 'challenge', 'future', 
            'need', 'required', 'improve', 'enhance', 'address'
        ]
        
        quality_scores = []
        for gap in research_gaps:
            gap_lower = gap.lower()
            
            # Score based on presence of gap indicators
            keyword_score = sum(1 for keyword in gap_keywords if keyword in gap_lower) / len(gap_keywords)
            
            # Score based on length (substantial gaps are usually longer)
            length_score = min(len(gap.split()) / 20, 1.0)
            
            # Score based on specificity (contains technical terms)
            technical_terms = len([word for word in gap.split() if len(word) > 6])
            technical_score = min(technical_terms / 5, 1.0)
            
            gap_score = (keyword_score * 0.4 + length_score * 0.3 + technical_score * 0.3)
            quality_scores.append(gap_score)
        
        return sum(quality_scores) / len(quality_scores)
    
    def calculate_structure_quality(self, review: Dict[str, Any]) -> float:
        """Evaluate the structure and organization of the review"""
        score = 0.0
        required_sections = ['review', 'research_gaps', 'key_trends', 'future_directions']
        
        # Check presence of required sections
        present_sections = 0
        for section in required_sections:
            if section in review and review[section]:
                content = review[section]
                if isinstance(content, list):
                    if len(content) > 0:
                        present_sections += 1
                elif isinstance(content, str):
                    if len(content.strip()) > 0:
                        present_sections += 1
                else:
                    present_sections += 1
        
        score += (present_sections / len(required_sections)) * 0.4
        
        # Check review length
        review_text = review.get('review', '')
        if len(review_text) > 1000:
            score += 0.3
        elif len(review_text) > 500:
            score += 0.2
        elif len(review_text) > 200:
            score += 0.1
            
        # Check if review has multiple paragraphs
        paragraph_count = review_text.count('\n\n') + 1
        if paragraph_count >= 3:
            score += 0.2
        elif paragraph_count >= 2:
            score += 0.1
            
        # Check for section headings (indicates good structure)
        heading_indicators = ['introduction', 'method', 'result', 'conclusion', 'discussion']
        heading_score = 0.0
        for heading in heading_indicators:
            if heading in review_text.lower():
                heading_score += 0.1
        score += min(heading_score, 0.1)
        
        return min(score, 1.0)
    
    def calculate_academic_tone_score(self, text: str) -> float:
        """Evaluate how academic the writing tone is"""
        if not text:
            return 0.0
        
        # Academic writing indicators
        academic_phrases = [
            'this paper', 'we propose', 'our results', 'experimental results',
            'in this study', 'the findings', 'research shows', 'literature suggests',
            'methodology', 'framework', 'evaluation', 'analysis'
        ]
        
        formal_words = ['however', 'therefore', 'moreover', 'furthermore', 'consequently']
        informal_words = ['awesome', 'terrible', 'huge', 'tiny', 'stuff', 'things', 'got', 'get']
        
        text_lower = text.lower()
        
        # Positive indicators
        academic_count = sum(1 for phrase in academic_phrases if phrase in text_lower)
        formal_count = sum(1 for word in formal_words if word in text_lower)
        
        # Negative indicators (informal language)
        informal_count = sum(1 for word in informal_words if word in text_lower)
        
        # Calculate scores
        academic_score = min(academic_count / 5, 1.0)
        formal_score = min(formal_count / 3, 1.0)
        informal_penalty = min(informal_count / 3, 1.0)
        
        # Sentence length analysis (academic writing tends to have longer sentences)
        sentences = nltk.sent_tokenize(text)
        if sentences:
            avg_sentence_length = np.mean([len(word_tokenize(sent)) for sent in sentences])
            sentence_score = min(avg_sentence_length / 25, 1.0)
        else:
            sentence_score = 0.5
        
        final_score = (academic_score * 0.4 + formal_score * 0.3 + sentence_score * 0.3) * (1 - informal_penalty * 0.5)
        return max(0.0, min(final_score, 1.0))
    
    def evaluate_complete_review(self, papers: List[Dict], analyses: List[Dict], review: Dict[str, Any]) -> EvaluationResult:
        """Comprehensive evaluation of the literature review"""
        
        # Content coverage
        content_coverage = self.calculate_content_coverage(papers, review.get('review', ''))
        
        # Coherence score
        coherence_score = self.calculate_coherence_score(review.get('review', ''))
        
        # Gap analysis quality
        gap_analysis_quality = self.calculate_gap_analysis_quality(review.get('research_gaps', []))
        
        # Structure quality
        structure_quality = self.calculate_structure_quality(review)
        
        # Academic tone
        academic_tone_score = self.calculate_academic_tone_score(review.get('review', ''))
        
        # Calculate overall score (weighted average)
        weights = {
            'content_coverage': 0.3,
            'coherence_score': 0.2,
            'gap_analysis_quality': 0.2,
            'structure_quality': 0.15,
            'academic_tone_score': 0.15
        }
        
        overall_score = (
            content_coverage * weights['content_coverage'] +
            coherence_score * weights['coherence_score'] +
            gap_analysis_quality * weights['gap_analysis_quality'] +
            structure_quality * weights['structure_quality'] +
            academic_tone_score * weights['academic_tone_score']
        )
        
        # Additional detailed metrics
        details = {
            'paper_count': len(papers),
            'review_length': len(review.get('review', '')),
            'gap_count': len(review.get('research_gaps', [])),
            'trend_count': len(review.get('key_trends', [])),
            'avg_relevance_score': np.mean([analysis.get('relevance_score', 0) for analysis in analyses]) if analyses else 0
        }
        
        return EvaluationResult(
            overall_score=overall_score,
            content_coverage=content_coverage,
            coherence_score=coherence_score,
            gap_analysis_quality=gap_analysis_quality,
            structure_quality=structure_quality,
            academic_tone_score=academic_tone_score,
            details=details
        )
    
    def generate_evaluation_report(self, result: EvaluationResult) -> str:
        """Generate a human-readable evaluation report"""
        report = []
        report.append(" LITERATURE REVIEW EVALUATION REPORT")
        report.append("=" * 50)
        
        report.append(f"\nOverall Quality Score: {result.overall_score:.2f}/1.00")
        
        # Score breakdown
        report.append("\n Detailed Breakdown:")
        report.append(f"  • Content Coverage: {result.content_coverage:.2f}/1.00")
        report.append(f"  • Coherence & Flow: {result.coherence_score:.2f}/1.00")
        report.append(f"  • Gap Analysis: {result.gap_analysis_quality:.2f}/1.00")
        report.append(f"  • Structure Quality: {result.structure_quality:.2f}/1.00")
        report.append(f"  • Academic Tone: {result.academic_tone_score:.2f}/1.00")
        
        # Additional details
        report.append("\n Additional Metrics:")
        report.append(f"  • Papers Analyzed: {result.details['paper_count']}")
        report.append(f"  • Review Length: {result.details['review_length']} characters")
        report.append(f"  • Research Gaps Identified: {result.details['gap_count']}")
        report.append(f"  • Trends Identified: {result.details['trend_count']}")
        report.append(f"  • Average Paper Relevance: {result.details['avg_relevance_score']:.2f}")
        
        # Quality assessment
        report.append("\n Quality Assessment:")
        if result.overall_score >= 0.8:
            report.append("   EXCELLENT - High-quality literature review")
        elif result.overall_score >= 0.6:
            report.append("   GOOD - Solid review with minor improvements possible")
        elif result.overall_score >= 0.4:
            report.append("   FAIR - Acceptable but needs significant improvements")
        else:
            report.append("    POOR - Major revisions required")
        
        # Recommendations
        report.append("\n Recommendations for Improvement:")
        if result.content_coverage < 0.6:
            report.append("  • Include more content from the analyzed papers")
        if result.coherence_score < 0.5:
            report.append("  • Improve flow between sentences and paragraphs")
        if result.gap_analysis_quality < 0.4:
            report.append("  • Provide more specific and substantial research gaps")
        if result.structure_quality < 0.5:
            report.append("  • Better organize sections with clear headings")
        if result.academic_tone_score < 0.5:
            report.append("  • Use more formal academic language")
        
        return "\n".join(report)

# Example usage and testing
if __name__ == "__main__":
    # Test the evaluation metrics
    evaluator = EvaluationMetrics()
    
    # Sample test data
    sample_papers = [
        {
            "title": "Attention Is All You Need",
            "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms."
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract": "We introduce a new language representation model called BERT for language understanding."
        }
    ]
    
    sample_analyses = [
        {"relevance_score": 0.8},
        {"relevance_score": 0.9}
    ]
    
    sample_review = {
        "review": "This literature review examines transformer architectures in NLP. The paper 'Attention Is All You Need' introduced the transformer model using self-attention mechanisms. BERT advanced this further with bidirectional pre-training. However, there are limitations in computational efficiency.",
        "research_gaps": ["Limited research on efficient transformer variants", "Need for better evaluation metrics"],
        "key_trends": ["Growing use of pre-trained models", "Focus on larger architectures"],
        "future_directions": ["Develop more efficient models", "Explore new attention mechanisms"]
    }
    
    # Run evaluation
    result = evaluator.evaluate_complete_review(sample_papers, sample_analyses, sample_review)
    
    # Generate report
    report = evaluator.generate_evaluation_report(result)
    print(report)