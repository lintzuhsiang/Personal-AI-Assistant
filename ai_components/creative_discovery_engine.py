# ai_components/creative_discovery_engine.py
"""
Enhanced Creative Discovery Engine for Scenario 1:
Upload notes and discuss to echo and inspire thoughts or potential fields
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sqlalchemy.orm import Session
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import google.generativeai as genai

logger = logging.getLogger(__name__)

@dataclass
class ConceptConnection:
    """Represents a connection between two concepts"""
    source_concept: str
    target_concept: str
    connection_type: str  # 'analogical', 'causal', 'metaphorical', 'domain_transfer'
    strength: float
    explanation: str
    supporting_evidence: List[str]

@dataclass
class InsightGeneration:
    """Represents a generated insight"""
    insight_text: str
    confidence_score: float
    source_concepts: List[str]
    novelty_score: float
    actionability_score: float
    related_fields: List[str]

class ConceptGraph:
    """Knowledge graph for concept relationships and analogical reasoning"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.concept_embeddings = {}
        self.domain_mappings = {}
        
    def add_concept(self, concept: str, domain: str, embedding: np.ndarray, metadata: Dict = None):
        """Add a concept to the knowledge graph"""
        self.graph.add_node(concept, domain=domain, metadata=metadata or {})
        self.concept_embeddings[concept] = embedding
        
        if domain not in self.domain_mappings:
            self.domain_mappings[domain] = []
        self.domain_mappings[domain].append(concept)
    
    def find_analogical_connections(self, source_concept: str, threshold: float = 0.7) -> List[ConceptConnection]:
        """Find analogical connections to other domains"""
        if source_concept not in self.concept_embeddings:
            return []
        
        source_embedding = self.concept_embeddings[source_concept]
        source_domain = self.graph.nodes[source_concept].get('domain')
        
        connections = []
        
        for concept, embedding in self.concept_embeddings.items():
            if concept == source_concept:
                continue
                
            target_domain = self.graph.nodes[concept].get('domain')
            
            # Calculate similarity
            similarity = cosine_similarity([source_embedding], [embedding])[0][0]
            
            if similarity > threshold and target_domain != source_domain:
                connection = ConceptConnection(
                    source_concept=source_concept,
                    target_concept=concept,
                    connection_type='analogical',
                    strength=similarity,
                    explanation=f"Cross-domain analogy between {source_domain} and {target_domain}",
                    supporting_evidence=[]
                )
                connections.append(connection)
        
        return sorted(connections, key=lambda x: x.strength, reverse=True)

class AnalogyDetector:
    """Advanced analogy detection using transformer models"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.analogy_patterns = [
            "is to", "as", "like", "similar to", "reminds me of",
            "parallels", "corresponds to", "mirrors", "echoes"
        ]
        
    def detect_analogies(self, text: str) -> List[Dict[str, Any]]:
        """Detect potential analogical relationships in text"""
        analogies = []
        
        # Simple pattern matching for explicit analogies
        for pattern in self.analogy_patterns:
            if pattern in text.lower():
                # Extract context around pattern
                sentences = text.split('.')
                for sentence in sentences:
                    if pattern in sentence.lower():
                        analogies.append({
                            "pattern": pattern,
                            "context": sentence.strip(),
                            "type": "explicit"
                        })
        
        return analogies
    
    def generate_creative_analogies(self, concept: str, target_domains: List[str]) -> List[str]:
        """Generate creative analogies across different domains"""
        analogies = []
        
        for domain in target_domains:
            # This would use a more sophisticated model in production
            prompt = f"Create an analogy connecting '{concept}' to the field of {domain}:"
            # Placeholder for actual LLM generation
            analogy = f"{concept} is like [concept in {domain}] because both involve [shared principle]"
            analogies.append(analogy)
        
        return analogies

class CreativeDiscoveryEngine:
    """
    Main engine for creative discovery and inspiration from user notes
    
    Features:
    - Cross-domain concept mapping
    - Analogical reasoning
    - Unexpected connection discovery
    - Socratic question generation
    - Field exploration suggestions
    """
    
    def __init__(self, vector_store: Chroma, embeddings: OpenAIEmbeddings):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.concept_graph = ConceptGraph()
        self.analogy_detector = AnalogyDetector()
        
        # Initialize topic modeling
        self.topic_model = LatentDirichletAllocation(n_components=10, random_state=42)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        
        # Domain knowledge base
        self.domain_knowledge = {
            "technology": ["AI", "algorithms", "data structures", "networks", "cybersecurity"],
            "biology": ["evolution", "genetics", "ecosystems", "cellular processes", "adaptation"],
            "psychology": ["cognition", "behavior", "learning", "motivation", "perception"],
            "physics": ["energy", "forces", "waves", "particles", "thermodynamics"],
            "economics": ["markets", "incentives", "optimization", "game theory", "resource allocation"],
            "art": ["creativity", "composition", "aesthetics", "expression", "symbolism"],
            "philosophy": ["ethics", "logic", "metaphysics", "epistemology", "consciousness"]
        }
        
    async def analyze_notes_for_insights(self, user_notes: List[str], user_id: str) -> Dict[str, Any]:
        """Comprehensive analysis of user notes to generate insights and connections"""
        try:
            # Extract key concepts
            concepts = await self._extract_key_concepts(user_notes)
            
            # Build concept graph
            await self._build_concept_graph(concepts, user_notes)
            
            # Find cross-domain connections
            cross_domain_connections = await self._find_cross_domain_connections(concepts)
            
            # Generate thought-provoking questions
            questions = await self._generate_socratic_questions(user_notes, concepts)
            
            # Suggest unexplored fields
            field_suggestions = await self._suggest_related_fields(concepts)
            
            # Generate creative insights
            insights = await self._generate_creative_insights(concepts, cross_domain_connections)
            
            return {
                "key_concepts": concepts,
                "cross_domain_connections": cross_domain_connections,
                "thought_provoking_questions": questions,
                "field_suggestions": field_suggestions,
                "creative_insights": insights,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze notes for insights: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _extract_key_concepts(self, notes: List[str]) -> List[Dict[str, Any]]:
        """Extract and rank key concepts from notes"""
        # Combine all notes
        combined_text = " ".join(notes)
        
        # TF-IDF analysis
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([combined_text])
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        importance_scores = tfidf_matrix.toarray()[0]
        
        # Get top concepts
        concept_scores = list(zip(feature_names, importance_scores))
        concept_scores.sort(key=lambda x: x[1], reverse=True)
        
        concepts = []
        for term, score in concept_scores[:20]:  # Top 20 concepts
            # Determine domain
            domain = self._classify_concept_domain(term)
            
            concepts.append({
                "term": term,
                "importance_score": float(score),
                "domain": domain,
                "frequency": combined_text.lower().count(term.lower())
            })
        
        return concepts
    
    def _classify_concept_domain(self, concept: str) -> str:
        """Classify concept into domain categories"""
        concept_lower = concept.lower()
        
        for domain, keywords in self.domain_knowledge.items():
            for keyword in keywords:
                if keyword.lower() in concept_lower or concept_lower in keyword.lower():
                    return domain
        
        return "general"
    
    async def _build_concept_graph(self, concepts: List[Dict], notes: List[str]) -> None:
        """Build concept graph with embeddings"""
        for concept_data in concepts:
            concept = concept_data["term"]
            domain = concept_data["domain"]
            
            # Generate embedding
            embedding = await self._get_concept_embedding(concept)
            
            # Add to graph
            self.concept_graph.add_concept(
                concept=concept,
                domain=domain,
                embedding=embedding,
                metadata=concept_data
            )
    
    async def _get_concept_embedding(self, concept: str) -> np.ndarray:
        """Get embedding for a concept"""
        # In production, use your embedding model
        # For now, return random embedding as placeholder
        return np.random.rand(384)  # Typical embedding dimension
    
    async def _find_cross_domain_connections(self, concepts: List[Dict]) -> List[ConceptConnection]:
        """Find unexpected connections across different domains"""
        connections = []
        
        for concept_data in concepts:
            concept = concept_data["term"]
            analogical_connections = self.concept_graph.find_analogical_connections(concept)
            connections.extend(analogical_connections)
        
        # Remove duplicates and sort by strength
        unique_connections = {}
        for conn in connections:
            key = (conn.source_concept, conn.target_concept)
            if key not in unique_connections or conn.strength > unique_connections[key].strength:
                unique_connections[key] = conn
        
        return sorted(unique_connections.values(), key=lambda x: x.strength, reverse=True)[:10]
    
    async def _generate_socratic_questions(self, notes: List[str], concepts: List[Dict]) -> List[str]:
        """Generate thought-provoking Socratic questions"""
        questions = []
        
        # Question templates based on Socratic method
        question_templates = [
            "What assumptions are you making about {concept}?",
            "How does {concept} relate to {other_concept}?",
            "What would happen if {concept} didn't exist?",
            "What are the implications of {concept} for {domain}?",
            "How might someone from {field} view {concept} differently?",
            "What evidence supports your understanding of {concept}?",
            "What are the potential negative consequences of {concept}?",
            "How has your thinking about {concept} evolved?",
            "What would {concept} look like in 50 years?",
            "What underlying principles govern {concept}?"
        ]
        
        # Generate questions for top concepts
        top_concepts = [c["term"] for c in concepts[:5]]
        
        for i, concept in enumerate(top_concepts):
            # Select appropriate templates
            for template in question_templates[:2]:  # Limit questions per concept
                if "{other_concept}" in template and i < len(top_concepts) - 1:
                    other_concept = top_concepts[i + 1]
                    question = template.format(concept=concept, other_concept=other_concept)
                elif "{domain}" in template:
                    domain = concepts[i]["domain"]
                    question = template.format(concept=concept, domain=domain)
                elif "{field}" in template:
                    # Pick a random different field
                    fields = list(self.domain_knowledge.keys())
                    current_domain = concepts[i]["domain"]
                    other_fields = [f for f in fields if f != current_domain]
                    if other_fields:
                        field = np.random.choice(other_fields)
                        question = template.format(concept=concept, field=field)
                    else:
                        continue
                else:
                    question = template.format(concept=concept)
                
                questions.append(question)
        
        return questions[:8]  # Return top 8 questions
    
    async def _suggest_related_fields(self, concepts: List[Dict]) -> List[Dict[str, Any]]:
        """Suggest related fields based on concept analysis"""
        # Count concepts per domain
        domain_counts = {}
        for concept in concepts:
            domain = concept["domain"]
            domain_counts[domain] = domain_counts.get(domain, 0) + concept["importance_score"]
        
        # Find underrepresented but potentially relevant domains
        all_domains = set(self.domain_knowledge.keys())
        represented_domains = set(domain_counts.keys())
        underrepresented = all_domains - represented_domains
        
        suggestions = []
        
        # Suggest based on analogical connections
        for domain in underrepresented:
            relevance_score = self._calculate_domain_relevance(concepts, domain)
            if relevance_score > 0.3:  # Threshold for relevance
                suggestions.append({
                    "field": domain,
                    "relevance_score": relevance_score,
                    "reasoning": f"Concepts in your notes have strong analogical connections to {domain}",
                    "example_connections": self.domain_knowledge[domain][:3]
                })
        
        # Also suggest adjacent fields to represented domains
        for domain in represented_domains:
            adjacent_fields = self._get_adjacent_fields(domain)
            for adj_field in adjacent_fields:
                if adj_field not in represented_domains:
                    suggestions.append({
                        "field": adj_field,
                        "relevance_score": 0.6,
                        "reasoning": f"Adjacent field to your interest in {domain}",
                        "example_connections": self.domain_knowledge[adj_field][:3]
                    })
        
        return sorted(suggestions, key=lambda x: x["relevance_score"], reverse=True)[:5]
    
    def _calculate_domain_relevance(self, concepts: List[Dict], domain: str) -> float:
        """Calculate relevance score for a domain based on concepts"""
        domain_keywords = self.domain_knowledge.get(domain, [])
        if not domain_keywords:
            return 0.0
        
        relevance_score = 0.0
        total_importance = sum(c["importance_score"] for c in concepts)
        
        for concept in concepts:
            concept_term = concept["term"].lower()
            for keyword in domain_keywords:
                # Check for semantic similarity (simplified)
                if any(word in concept_term for word in keyword.lower().split()) or \
                   any(word in keyword.lower() for word in concept_term.split()):
                    relevance_score += concept["importance_score"] / total_importance
        
        return min(1.0, relevance_score)
    
    def _get_adjacent_fields(self, domain: str) -> List[str]:
        """Get fields adjacent/related to given domain"""
        adjacency_map = {
            "technology": ["psychology", "economics", "physics"],
            "biology": ["psychology", "physics", "philosophy"],
            "psychology": ["biology", "technology", "philosophy"],
            "physics": ["technology", "philosophy", "art"],
            "economics": ["technology", "psychology", "philosophy"],
            "art": ["psychology", "philosophy", "physics"],
            "philosophy": ["psychology", "physics", "biology", "art"]
        }
        return adjacency_map.get(domain, [])
    
    async def _generate_creative_insights(self, concepts: List[Dict], connections: List[ConceptConnection]) -> List[InsightGeneration]:
        """Generate creative insights based on concept analysis"""
        insights = []
        
        # Insight generation strategies
        strategies = [
            self._generate_synthesis_insights,
            self._generate_transfer_insights,
            self._generate_contradiction_insights,
            self._generate_evolution_insights
        ]
        
        for strategy in strategies:
            try:
                strategy_insights = await strategy(concepts, connections)
                insights.extend(strategy_insights)
            except Exception as e:
                logger.error(f"Failed to generate insights with strategy {strategy.__name__}: {e}")
        
        # Score and rank insights
        for insight in insights:
            insight.confidence_score = self._calculate_insight_confidence(insight, concepts)
            insight.novelty_score = self._calculate_novelty_score(insight, concepts)
            insight.actionability_score = self._calculate_actionability_score(insight)
        
        # Sort by overall quality score
        insights.sort(key=lambda x: (x.confidence_score + x.novelty_score + x.actionability_score) / 3, reverse=True)
        
        return insights[:5]  # Return top 5 insights
    
    async def _generate_synthesis_insights(self, concepts: List[Dict], connections: List[ConceptConnection]) -> List[InsightGeneration]:
        """Generate insights by synthesizing multiple concepts"""
        insights = []
        
        # Find concepts that can be meaningfully combined
        high_importance_concepts = [c for c in concepts if c["importance_score"] > 0.1]
        
        for i in range(len(high_importance_concepts)):
            for j in range(i + 1, min(i + 3, len(high_importance_concepts))):  # Limit combinations
                concept_a = high_importance_concepts[i]
                concept_b = high_importance_concepts[j]
                
                # Generate synthesis insight
                insight_text = f"The intersection of {concept_a['term']} and {concept_b['term']} " \
                             f"could lead to innovations in {concept_a['domain']} by applying " \
                             f"principles from {concept_b['domain']}."
                
                insight = InsightGeneration(
                    insight_text=insight_text,
                    confidence_score=0.0,  # Will be calculated later
                    source_concepts=[concept_a['term'], concept_b['term']],
                    novelty_score=0.0,
                    actionability_score=0.0,
                    related_fields=[concept_a['domain'], concept_b['domain']]
                )
                insights.append(insight)
        
        return insights
    
    async def _generate_transfer_insights(self, concepts: List[Dict], connections: List[ConceptConnection]) -> List[InsightGeneration]:
        """Generate insights by transferring concepts across domains"""
        insights = []
        
        for connection in connections[:3]:  # Top 3 connections
            insight_text = f"Consider applying the concept of {connection.source_concept} " \
                         f"from its traditional context to {connection.target_concept}. " \
                         f"This cross-pollination could reveal new perspectives."
            
            insight = InsightGeneration(
                insight_text=insight_text,
                confidence_score=connection.strength,
                source_concepts=[connection.source_concept, connection.target_concept],
                novelty_score=0.0,
                actionability_score=0.0,
                related_fields=[]
            )
            insights.append(insight)
        
        return insights
    
    async def _generate_contradiction_insights(self, concepts: List[Dict], connections: List[ConceptConnection]) -> List[InsightGeneration]:
        """Generate insights by exploring contradictions and tensions"""
        insights = []
        
        # Look for concepts that might be in tension
        for i, concept_a in enumerate(concepts[:5]):
            for concept_b in concepts[i+1:6]:
                if concept_a['domain'] != concept_b['domain']:
                    insight_text = f"Explore the tension between {concept_a['term']} and {concept_b['term']}. " \
                                 f"What happens when these seemingly contradictory ideas coexist?"
                    
                    insight = InsightGeneration(
                        insight_text=insight_text,
                        confidence_score=0.0,
                        source_concepts=[concept_a['term'], concept_b['term']],
                        novelty_score=0.0,
                        actionability_score=0.0,
                        related_fields=[concept_a['domain'], concept_b['domain']]
                    )
                    insights.append(insight)
        
        return insights[:2]  # Limit contradiction insights
    
    async def _generate_evolution_insights(self, concepts: List[Dict], connections: List[ConceptConnection]) -> List[InsightGeneration]:
        """Generate insights about how concepts might evolve"""
        insights = []
        
        for concept in concepts[:3]:  # Top 3 concepts
            insight_text = f"How might {concept['term']} evolve over the next decade? " \
                         f"Consider emerging trends in {concept['domain']} and adjacent fields."
            
            insight = InsightGeneration(
                insight_text=insight_text,
                confidence_score=0.0,
                source_concepts=[concept['term']],
                novelty_score=0.0,
                actionability_score=0.0,
                related_fields=[concept['domain']]
            )
            insights.append(insight)
        
        return insights
    
    def _calculate_insight_confidence(self, insight: InsightGeneration, concepts: List[Dict]) -> float:
        """Calculate confidence score for an insight"""
        # Base confidence on source concept importance
        total_importance = 0.0
        for concept_name in insight.source_concepts:
            for concept in concepts:
                if concept['term'] == concept_name:
                    total_importance += concept['importance_score']
                    break
        
        return min(1.0, total_importance / len(insight.source_concepts))
    
    def _calculate_novelty_score(self, insight: InsightGeneration, concepts: List[Dict]) -> float:
        """Calculate novelty score based on concept diversity"""
        unique_domains = set()
        for concept_name in insight.source_concepts:
            for concept in concepts:
                if concept['term'] == concept_name:
                    unique_domains.add(concept['domain'])
                    break
        
        # Higher novelty for cross-domain insights
        return len(unique_domains) / max(1, len(insight.source_concepts))
    
    def _calculate_actionability_score(self, insight: InsightGeneration) -> float:
        """Calculate how actionable an insight is"""
        # Simple heuristic: shorter insights with concrete terms are more actionable
        word_count = len(insight.insight_text.split())
        
        # Check for action words
        action_words = ['apply', 'explore', 'consider', 'develop', 'create', 'investigate']
        action_score = sum(1 for word in action_words if word in insight.insight_text.lower())
        
        # Normalize scores
        length_score = max(0.0, 1.0 - (word_count - 20) / 50)  # Prefer 20-word insights
        action_score = min(1.0, action_score / 3)  # Up to 3 action words
        
        return (length_score + action_score) / 2
    
    async def generate_exploratory_discussion(self, user_input: str, user_notes: List[str]) -> Dict[str, Any]:
        """Generate an exploratory discussion response that echoes and expands user thoughts"""
        try:
            # Analyze user input for key themes
            input_concepts = await self._extract_key_concepts([user_input])
            
            # Find connections to existing notes
            note_concepts = await self._extract_key_concepts(user_notes)
            
            # Generate discussion points
            discussion_points = await self._generate_discussion_points(user_input, input_concepts, note_concepts)
            
            # Suggest related questions
            follow_up_questions = await self._generate_follow_up_questions(user_input, input_concepts)
            
            # Find unexpected angles
            unexpected_angles = await self._find_unexpected_angles(input_concepts, note_concepts)
            
            return {
                "echo_response": self._generate_echo_response(user_input),
                "discussion_points": discussion_points,
                "follow_up_questions": follow_up_questions,
                "unexpected_angles": unexpected_angles,
                "suggested_explorations": await self._suggest_explorations(input_concepts)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate exploratory discussion: {e}")
            return {"status": "error", "message": str(e)}
    
    def _generate_echo_response(self, user_input: str) -> str:
        """Generate an echoing response that shows understanding"""
        # Extract key themes
        sentences = user_input.split('.')
        key_sentence = max(sentences, key=len) if sentences else user_input
        
        echo_phrases = [
            f"I hear you reflecting on {key_sentence.strip().lower()}",
            f"Your thoughts on {key_sentence.strip().lower()} are intriguing",
            f"You're exploring some fascinating territory with {key_sentence.strip().lower()}"
        ]
        
        return np.random.choice(echo_phrases)
    
    async def _generate_discussion_points(self, user_input: str, input_concepts: List[Dict], note_concepts: List[Dict]) -> List[str]:
        """Generate relevant discussion points"""
        points = []
        
        # Find overlapping concepts
        input_terms = {c['term'] for c in input_concepts}
        note_terms = {c['term'] for c in note_concepts}
        overlap = input_terms.intersection(note_terms)
        
        for term in list(overlap)[:3]:
            points.append(f"This connects to your earlier thoughts on {term} - how has your perspective evolved?")
        
        # Add domain-bridging points
        input_domains = {c['domain'] for c in input_concepts}
        note_domains = {c['domain'] for c in note_concepts}
        
        for domain in input_domains:
            if domain in note_domains:
                points.append(f"Your interest in {domain} seems to be a recurring theme - what draws you to this field?")
        
        return points[:4]
    
    async def _generate_follow_up_questions(self, user_input: str, concepts: List[Dict]) -> List[str]:
        """Generate thoughtful follow-up questions"""
        questions = []
        
        for concept in concepts[:2]:
            term = concept['term']
            domain = concept['domain']
            
            question_templates = [
                f"What if {term} were approached from the perspective of {self._get_random_other_domain(domain)}?",
                f"What assumptions about {term} might you be taking for granted?",
                f"How might {term} look different in 10 years?",
                f"What would someone who disagrees with your view on {term} say?"
            ]
            
            questions.append(np.random.choice(question_templates))
        
        return questions
    
    def _get_random_other_domain(self, current_domain: str) -> str:
        """Get a random domain different from current"""
        all_domains = list(self.domain_knowledge.keys())
        other_domains = [d for d in all_domains if d != current_domain]
        return np.random.choice(other_domains) if other_domains else "art"
    
    async def _find_unexpected_angles(self, input_concepts: List[Dict], note_concepts: List[Dict]) -> List[str]:
        """Find unexpected angles to explore"""
        angles = []
        
        # Cross-domain combinations
        for input_concept in input_concepts[:2]:
            for note_concept in note_concepts[:3]:
                if input_concept['domain'] != note_concept['domain']:
                    angle = f"What if you combined insights from {input_concept['term']} " \
                           f"with your understanding of {note_concept['term']}?"
                    angles.append(angle)
        
        return angles[:3]
    
    async def _suggest_explorations(self, concepts: List[Dict]) -> List[str]:
        """Suggest concrete explorations"""
        explorations = []
        
        for concept in concepts[:2]:
            term = concept['term']
            domain = concept['domain']
            
            exploration_templates = [
                f"Research how {term} is being reimagined in {self._get_random_other_domain(domain)}",
                f"Interview someone who works with {term} in a different context",
                f"Write a brief essay arguing against your current view of {term}",
                f"Create a visual map connecting {term} to seemingly unrelated concepts"
            ]
            
            explorations.append(np.random.choice(exploration_templates))
        
        return explorations