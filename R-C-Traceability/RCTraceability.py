#!/usr/bin/env python3
"""
Enhanced Requirements-Code Traceability System
- Uses FAISS vector search and knowledge graph
- Integrates LLM for semantic analysis
- Implements Bayesian reasoning for confidence scoring
- Supports multi-language code analysis
"""

import os
import json
import numpy as np
import faiss
from dashscope import TextEmbedding, Generation
import dashscope
from typing import List, Dict, Set, Tuple, Optional, Any, Union, DefaultDict
from collections import defaultdict
import re
from pathlib import Path
import sys
import traceback
from math import exp, log, gamma
import hashlib
import time
import random
from dataclasses import dataclass
from enum import Enum
import statistics
from functools import lru_cache
from collections import deque
import logging

logger = logging.getLogger(__name__)

# --- Configuration paths ---
KNOWLEDGE_GRAPH_PATH = r""
METADATAS_JSON_PATH = r""
FAISS_INDEX_PATH = r""

TOP_K = 10
MAX_HOPS = 3
BAYESIAN_TOP_K = 5
LLM_JUDGE_THRESHOLD = 3
SIMILARITY_THRESHOLD = 0.5 

# --- New configuration parameters ---
TRACE_THRESHOLD = 0.6
MIN_CONFIDENCE = 0.4
EARLY_STOP_AFTER_CONFIDENT = 2
ENABLE_INFORMATIVE_PRIOR = True
LLM_SAMPLING_COUNT = 3
LLM_TEMPERATURE_RANGE = [0.1, 0.3, 0.5]

# Hierarchical analysis thresholds
HIGH_SIMILARITY_THRESHOLD = 0.7
MEDIUM_SIMILARITY_THRESHOLD = 0.6

# --- API Key ---
dashscope.api_key = ""


def clean_func_id(func_name: str) -> str:
    """Standardize function name format, remove 'Function:' or 'Method:' prefix"""
    if not func_name:
        return ""
    if isinstance(func_name, str):
        if func_name.startswith("Function:"):
            return func_name.replace("Function:", "")
        if func_name.startswith("Method:"):
            return func_name.replace("Method:", "")
    return func_name


@dataclass
class LLMDimensionScores:
    """LLM multi-dimensional scores"""
    semantic_relevance: float
    logical_completeness: float
    dataflow_continuity: float
    call_rationality: float
    overall: float
    uncertainty: float


class TextSplitter:
    """Text splitter for requirements documents"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_split(self, file_path: str) -> List[str]:
        if not os.path.exists(file_path):
            print(f"Error: Requirements file not found: {file_path}")
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(para) > self.chunk_size:
                    sub_chunks = self._split_long_text(para)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = para
            else:
                current_chunk += (" " + para) if current_chunk else para
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _split_long_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size // 5
            chunk_words = words[start:end]
            if not chunk_words:
                break
            chunks.append(" ".join(chunk_words))
            start = end - self.chunk_overlap // 5
        return chunks


class CallChainSummarizer:
    """Call chain semantic summarizer"""

    def __init__(self, summary_map: Dict[str, str]):
        self.summary_map = summary_map

    def summarize_call_chain(self, call_chain: List[str], max_length: int = 300) -> Tuple[str, List[str]]:
        """Convert call chain to semantic summary"""

        if not call_chain:
            return "", []

        # Collect node semantic information
        node_summaries = []
        key_nodes = []

        for i, func_name in enumerate(call_chain):
            clean_name = clean_func_id(func_name)
            summary = self.summary_map.get(clean_name, "") or self.summary_map.get(func_name, "")

            if summary:
                # Extract key information (remove generic descriptions)
                concise_summary = self._extract_concise_info(summary)
                if concise_summary and len(concise_summary) > 3:
                    node_summaries.append(f"{i + 1}. {concise_summary}")

                    # Identify key nodes (with actual functional descriptions)
                    if not self._is_common_utility(concise_summary):
                        key_nodes.append(func_name)

        # Build semantic summary
        if node_summaries:
            # Method 1: Concise version (suitable for short chains)
            if len(node_summaries) <= 5:
                semantic_summary = " → ".join([s.split('. ', 1)[1] for s in node_summaries])
            # Method 2: Detailed version (suitable for long chains)
            else:
                semantic_summary = f"Functional chain contains {len(node_summaries)} main steps:\n" + "\n".join(node_summaries[:8])

                if len(node_summaries) > 8:
                    semantic_summary += f"\n... and {len(node_summaries) - 8} more steps"
        else:
            # Fallback: Use function names
            semantic_summary = " → ".join([clean_func_id(f) for f in call_chain])

        # Truncate length
        if len(semantic_summary) > max_length:
            semantic_summary = semantic_summary[:max_length - 3] + "..."

        return semantic_summary, key_nodes

    def _extract_concise_info(self, summary: str) -> str:
        """Extract concise information from function description"""
        # Remove common redundant prefixes
        prefixes = ["This function", "This method", "The function", "The method",
                    "Function to", "Method to", "This procedure", "This routine"]

        concise = summary.strip()
        for prefix in prefixes:
            if concise.lower().startswith(prefix.lower()):
                concise = concise[len(prefix):].lstrip()
                break

        # Extract core action
        if ' ' in concise:
            words = concise.split()
            if len(words) > 8:
                # Keep verb phrase
                concise = ' '.join(words[:8]) + "..."

        return concise.strip(" .,")

    def _is_common_utility(self, summary: str) -> bool:
        """Check if it's a common utility function"""
        utility_keywords = ['get', 'set', 'is', 'has', 'check', 'validate',
                            'create', 'delete', 'update', 'find', 'search']

        lower_summary = summary.lower()
        for keyword in utility_keywords:
            if lower_summary.startswith(keyword + ' '):
                return True
        return False


class EnhancedLLMJudge:
    """Enhanced LLM call chain discriminator"""

    def __init__(self):
        self.chain_summary_template = self._create_chain_summary_template()

    def _create_chain_summary_template(self) -> str:
        """Create chain-level semantic analysis prompt template"""
        return """Please act as a software architect and analyze whether the following functional chain implements the specified requirement.

## Requirement Description:
{requirement}

## Functional Chain Semantic Description:
{semantic_summary}

## Key Function Details:
{key_node_details}

## Analysis Dimensions (0.0-1.0 score):
1. Functional Completeness: Whether the chain completely implements the core functionality of the requirement
2. Logical Coherence: Whether the logical connections between steps are natural and reasonable
3. Implementation Efficiency: Whether the chain design is efficient and non-redundant
4. Extensibility: Whether the chain design is easy to extend and maintain

## Scoring Standards:
- 0.90-1.00: Perfect implementation, excellent architecture
- 0.80-0.89: Good implementation, minor optimization space
- 0.70-0.79: Basic implementation, has improvement points
- 0.60-0.69: Partial implementation, has defects
- 0.50-0.59: Marginally related, implementation incomplete
- 0.30-0.49: Basically unrelated
- 0.00-0.29: Completely unrelated

## Output Format:
Overall Score: [0.00-1.00]
Functional Completeness: [0.00-1.00]
Logical Coherence: [0.00-1.00]
Implementation Efficiency: [0.00-1.00]
Extensibility: [0.00-1.00]
Analysis Conclusion: [Detailed explanation of this functional chain's advantages and shortcomings, at least 80 words]"""

    def judge_call_chain_by_semantic_summary(self, requirement: str,
                                             semantic_summary: str,
                                             key_nodes: List[str],
                                             node_details: Dict[str, str]) -> Tuple[LLMDimensionScores, str]:
        """Analyze call chain based on semantic summary"""

        # Collect key node details
        key_details_text = ""
        for i, node in enumerate(key_nodes[:4]):  # Show at most 4 key nodes
            detail = node_details.get(node, "No detailed description")
            if len(detail) > 100:
                detail = detail[:97] + "..."
            key_details_text += f"{i + 1}. {node}: {detail}\n"

        if not key_details_text:
            key_details_text = "No detailed function information"

        prompt = self.chain_summary_template.format(
            requirement=requirement[:400],
            semantic_summary=semantic_summary,
            key_node_details=key_details_text
        )

        try:
            response = Generation.call(
                model="qwen-max",
                prompt=prompt,
                max_tokens=250,
                temperature=0.1
            )

            if response.status_code == 200:
                parsed = self._parse_enhanced_response(response.output.text)
                if parsed is not None:
                    return parsed
        except Exception as e:
            print(f"Chain-level semantic analysis exception: {e}")

        # Fallback solution
        return self._fallback_dimension_scores("Chain analysis failed")

    def _fallback_dimension_scores(self, reason: str) -> Tuple[LLMDimensionScores, str]:
        """Neutral fallback used when LLM parsing fails or returns nothing;
        always returns a (scores, reason) tuple callers can unpack."""
        return LLMDimensionScores(
            semantic_relevance=0.5,
            logical_completeness=0.5,
            dataflow_continuity=0.5,
            call_rationality=0.5,
            overall=0.5,
            uncertainty=0.3
        ), reason

    def _parse_enhanced_response(self, response_text: str) -> Optional[Tuple[LLMDimensionScores, str]]:
        """Parse enhanced LLM response. Returns None only on truly empty/unreadable input;
        on regex/parse failures returns a fallback (LLMDimensionScores, reason) tuple so
        callers can safely unpack."""
        if not response_text:
            return None
        try:
            # Initialize default values
            scores = {
                'overall': 0.5,
                'semantic': 0.5,
                'completeness': 0.5,
                'dataflow': 0.5,
                'rationality': 0.5
            }

            # Multiple matching patterns
            patterns = [
                (r'Overall Score\s*[:：]\s*([0-9]\.[0-9]{1,2})', 'overall'),
                (r'Semantic Relevance\s*[:：]\s*([0-9]\.[0-9]{1,2})', 'semantic'),
                (r'Logical Completeness\s*[:：]\s*([0-9]\.[0-9]{1,2})', 'completeness'),
                (r'Dataflow Continuity\s*[:：]\s*([0-9]\.[0-9]{1,2})', 'dataflow'),
                (r'Call Rationality\s*[:：]\s*([0-9]\.[0-9]{1,2})', 'rationality'),
                (r'Functional Completeness\s*[:：]\s*([0-9]\.[0-9]{1,2})', 'completeness'),
                (r'Logical Coherence\s*[:：]\s*([0-9]\.[0-9]{1,2})', 'completeness'),
                (r'Implementation Efficiency\s*[:：]\s*([0-9]\.[0-9]{1,2})', 'rationality'),
                (r'Extensibility\s*[:：]\s*([0-9]\.[0-9]{1,2})', 'semantic'),
            ]

            for pattern, key in patterns:
                match = re.search(pattern, response_text)
                if match:
                    try:
                        scores[key] = float(match.group(1))
                        # Ensure within 0-1 range
                        scores[key] = max(0.0, min(1.0, scores[key]))
                    except Exception:
                        pass

            # If no overall score found, calculate dimension average
            if scores['overall'] == 0.5 and any(v != 0.5 for v in scores.values() if v != 'overall'):
                dim_scores = [scores[k] for k in ['semantic', 'completeness', 'dataflow', 'rationality']]
                scores['overall'] = float(np.mean(dim_scores))

            # Extract reasoning
            reason = "No detailed reasoning"
            reason_patterns = [
                r'Judgment Reason\s*[:：]\s*(.+?)(?:\n\n|\n[A-Z]|$)',
                r'Reason\s*[:：]\s*(.+?)(?:\n\n|\n[A-Z]|$)',
                r'Analysis Conclusion\s*[:：]\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            ]

            for pattern in reason_patterns:
                match = re.search(pattern, response_text, re.DOTALL)
                if match:
                    reason = match.group(1).strip()
                    # Clean line breaks
                    reason = ' '.join(reason.split())
                    break

            # Create dimension score object
            dim_scores = LLMDimensionScores(
                semantic_relevance=scores['semantic'],
                logical_completeness=scores['completeness'],
                dataflow_continuity=scores['dataflow'],
                call_rationality=scores['rationality'],
                overall=scores['overall'],
                uncertainty=0.1
            )

            return dim_scores, reason

        except Exception:
            return self._fallback_dimension_scores("Response parse failed; using neutral fallback")


class InformativeBayesianEngine:
    """Informative prior Bayesian fusion engine"""

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode

        # Beta distribution parameters
        self.semantic_params = {'alpha': 3, 'beta': 3}
        self.chain_summary_params = {'alpha': 6, 'beta': 2}
        self.llm_params = {'alpha': 5, 'beta': 1}

        # Dimension weights
        self.dimension_weights = {
            'semantic': 0.4,
            'completeness': 0.3,
            'dataflow': 0.2,
            'rationality': 0.1
        }

    def beta_pdf(self, x: float, alpha: float, beta: float) -> float:
        """Beta distribution probability density function"""
        if x <= 0 or x >= 1:
            return 1e-10

        # Use log to avoid numerical overflow
        log_pdf = (alpha - 1) * log(x) + (beta - 1) * log(1 - x) - log(gamma(alpha)) - log(gamma(beta)) + log(
            gamma(alpha + beta))
        return exp(log_pdf)

    def sigmoid(self, x: float) -> float:
        """Sigmoid function"""
        return 1 / (1 + exp(-x))

    def _compute_function_centrality(self, function_name: str, graph_structure: Dict) -> float:
        """Calculate function centrality in the graph"""
        clean_name = clean_func_id(function_name)

        if not graph_structure:
            return 0.5

        # Calculate out-degree and in-degree
        out_degree = 0
        in_degree = 0

        if clean_name in graph_structure:
            out_degree = len(graph_structure[clean_name].get("calls", []))

        # Count in-degree
        for _, node_info in graph_structure.items():
            if clean_name in node_info.get("calls", []):
                in_degree += 1

        # Normalize centrality score
        total_nodes = len(graph_structure)
        if total_nodes > 0:
            max_possible_degree = min(10, total_nodes - 1)
            degree_centrality = (out_degree + in_degree) / (2 * max_possible_degree)
            return min(degree_centrality, 1.0)

        return 0.5

    def _calculate_structure_score(self, chain: List[str], knowledge_graph: Dict,
                                   dataflow_graph: Dict) -> float:
        """Calculate call chain structure score"""

        if len(chain) < 2:
            return 0.3  # Single node chain has lower structure score

        structure_score = 0.0

        # 1. Data flow integrity score (0-0.4)
        dataflow_score = self._calculate_dataflow_integrity(chain, dataflow_graph)

        # 2. Call rationality score (0-0.3)
        call_rationality_score = self._calculate_call_rationality(chain, knowledge_graph)

        # 3. Graph topology score (0-0.3)
        topology_score = self._calculate_topology_score(chain, knowledge_graph)

        # Calculate total score
        structure_score = dataflow_score + call_rationality_score + topology_score

        return min(structure_score, 1.0)

    def _calculate_dataflow_integrity(self, chain: List[str], dataflow_graph: Dict) -> float:
        """Calculate data flow integrity"""
        if len(chain) < 2:
            return 0.2

        score = 0.0
        dataflow_count = 0
        param_return_count = 0

        for i in range(len(chain) - 1):
            source = clean_func_id(chain[i])
            target = clean_func_id(chain[i + 1])

            # Check if data flow edge exists
            if source in dataflow_graph:
                edges = dataflow_graph[source].get("dataflow_edges", [])
                for edge in edges:
                    if edge.get("target") == target:
                        dataflow_count += 1

                        # Parameter passing and return values are stronger data flows
                        if edge.get("type") in ["PARAMETER", "RETURN"]:
                            param_return_count += 2

        # Calculate data flow integrity score
        if len(chain) > 1:
            # Basic data flow existence
            flow_density = dataflow_count / (len(chain) - 1)
            param_return_density = param_return_count / (len(chain) - 1)

            # Weight allocation
            score = flow_density * 0.6 + param_return_density * 0.4

        return score * 0.4  # Data flow accounts for at most 0.4 points

    def _calculate_call_rationality(self, chain: List[str], knowledge_graph: Dict) -> float:
        """Calculate call rationality"""
        if len(chain) < 2:
            return 0.2

        valid_calls = 0
        for i in range(len(chain) - 1):
            caller = clean_func_id(chain[i])
            callee = clean_func_id(chain[i + 1])

            # Check if call relationship exists
            if (caller in knowledge_graph and
                    callee in knowledge_graph[caller].get("calls", [])):
                valid_calls += 1

        # Call rationality score
        call_ratio = valid_calls / (len(chain) - 1) if len(chain) > 1 else 0

        # Consider call pattern rationality
        # 1. Deep call chains (>3 levels) may be more reasonable
        depth_bonus = 0.0
        if len(chain) >= 3:
            depth_bonus = min((len(chain) - 2) * 0.1, 0.2)

        return (call_ratio * 0.7 + depth_bonus * 0.3) * 0.3  # Call rationality accounts for at most 0.3 points

    def _calculate_topology_score(self, chain: List[str], knowledge_graph: Dict) -> float:
        """Calculate graph topology score"""
        if len(chain) < 2:
            return 0.1

        score = 0.0

        # 1. Centrality distribution
        centrality_scores = []
        for func in chain:
            centrality = self._compute_function_centrality(func, knowledge_graph)
            centrality_scores.append(centrality)

        # Ideally, chain should contain high-centrality nodes
        max_centrality = max(centrality_scores) if centrality_scores else 0
        avg_centrality = sum(centrality_scores) / len(centrality_scores) if centrality_scores else 0

        # 2. In-degree out-degree analysis
        in_out_balance = 0.0
        for i, func in enumerate(chain):
            clean_func = clean_func_id(func)
            if clean_func in knowledge_graph:
                out_degree = len(knowledge_graph[clean_func].get("calls", []))

                # Calculate in-degree
                in_degree = 0
                for _, node_info in knowledge_graph.items():
                    if clean_func in node_info.get("calls", []):
                        in_degree += 1

                # Ideal situation: intermediate nodes have both in-degree and out-degree
                if 0 < i < len(chain) - 1:
                    if in_degree > 0 and out_degree > 0:
                        in_out_balance += 0.1

        # Normalize
        in_out_balance = min(in_out_balance, 0.3) / 0.3 if in_out_balance > 0 else 0

        # 3. Calculate topology score
        score = (max_centrality * 0.4 + avg_centrality * 0.3 + in_out_balance * 0.3) * 0.3

        return min(score, 0.3)

    def _calculate_chain_prior(self, chain: List[str], avg_semantic: float,
                               summary_score: float, knowledge_graph: Dict) -> float:
        """Calculate call chain prior probability"""

        # Base prior
        base_prior = 0.05

        # 1. Semantic similarity contribution
        semantic_prior = 0.0
        if avg_semantic >= 0.8:
            semantic_prior = avg_semantic * 0.6
        elif avg_semantic >= 0.6:
            semantic_prior = avg_semantic * 0.4
        elif avg_semantic >= 0.4:
            semantic_prior = avg_semantic * 0.2

        # 2. Chain length contribution (appropriate length is better)
        length_prior = 0.0
        if 3 <= len(chain) <= 7:  # Chains with 3-7 nodes are usually more complete
            length_prior = 0.2
        elif len(chain) > 7:
            length_prior = 0.1

        # 3. Key node contribution
        key_node_prior = 0.0
        for func in chain:
            centrality = self._compute_function_centrality(func, knowledge_graph)
            if centrality >= 0.7:  # High centrality node
                key_node_prior += 0.05

        key_node_prior = min(key_node_prior, 0.15)

        # 4. LLM summary score contribution
        llm_prior = 0.0
        if summary_score >= 0.8:
            llm_prior = summary_score * 0.5
        elif summary_score >= 0.6:
            llm_prior = summary_score * 0.3

        # Calculate total prior
        total_prior = base_prior + semantic_prior + length_prior + key_node_prior + llm_prior

        return min(max(total_prior, 0.05), 0.5)

    def compute_likelihood_semantic(self, semantic_score: float) -> float:
        """Semantic similarity likelihood"""
        normalized_score = max(0.0, min(1.0, semantic_score))
        likelihood = self.beta_pdf(normalized_score,
                                   self.semantic_params['alpha'],
                                   self.semantic_params['beta'])
        return max(likelihood, 1e-10)

    def compute_likelihood_llm(self, llm_score: float) -> float:
        """LLM score likelihood"""
        normalized_score = max(0.0, min(1.0, llm_score))
        likelihood = self.beta_pdf(normalized_score,
                                   self.llm_params['alpha'],
                                   self.llm_params['beta'])
        return max(likelihood, 1e-10)

    def compute_likelihood_chain_summary(self, summary_score: float) -> float:
        """Chain-level semantic summary score likelihood"""
        normalized_score = max(0.0, min(1.0, summary_score))
        likelihood = self.beta_pdf(normalized_score,
                                   self.chain_summary_params['alpha'],
                                   self.chain_summary_params['beta'])
        return max(likelihood, 1e-10)

    def compute_likelihood_structure(self, structure_score: float) -> float:
        """Chain structure likelihood. Symmetric Beta(3,3) is intentionally
        neutral so a moderate structure score (around 0.5) is treated as
        ordinary evidence rather than penalised or rewarded."""
        normalized_score = max(0.0, min(1.0, structure_score))
        likelihood = self.beta_pdf(normalized_score, 3, 3)
        return max(likelihood, 1e-10)

    def compute_dimension_likelihood(self, dim_scores: LLMDimensionScores) -> Dict[str, float]:
        """Calculate multi-dimensional likelihood"""
        return {
            'semantic': self.compute_likelihood_llm(dim_scores.semantic_relevance),
            'completeness': self.compute_likelihood_llm(dim_scores.logical_completeness),
            'dataflow': self.compute_likelihood_llm(dim_scores.dataflow_continuity),
            'rationality': self.compute_likelihood_llm(dim_scores.call_rationality)
        }

    def compute_chain_posterior(self, prior: float, summary_score: float,
                                avg_semantic: float, structure_score: float,
                                llm_scores: LLMDimensionScores,
                                chain: List[str] = None,
                                knowledge_graph: Dict = None,
                                dataflow_graph: Dict = None) -> float:
        """Calculate functional chain posterior probability"""

        # Calculate real structure score (if chain and knowledge graph provided)
        real_structure_score = structure_score
        if chain and knowledge_graph and dataflow_graph:
            real_structure_score = self._calculate_structure_score(
                chain, knowledge_graph, dataflow_graph
            )

        # Dimension likelihoods
        l_summary = self.compute_likelihood_chain_summary(summary_score)
        l_semantic = self.compute_likelihood_semantic(avg_semantic)
        l_structure = self.compute_likelihood_structure(real_structure_score)

        # LLM multi-dimensional likelihood
        dim_likelihoods = self.compute_dimension_likelihood(llm_scores)

        # Calculate weighted likelihood
        chain_likelihood = l_summary * l_semantic * l_structure

        # Dimension weighting
        dim_weighted = 1.0
        for dim, weight in self.dimension_weights.items():
            dim_weighted *= (dim_likelihoods[dim] ** weight)

        # Combine chain-level and dimension-level
        combined_likelihood = (chain_likelihood * 0.6) + (dim_weighted * 0.4)

        # Posterior calculation
        posterior = prior * combined_likelihood

        # Consider uncertainty
        if llm_scores.uncertainty > 0.1:
            posterior *= (1.0 - llm_scores.uncertainty * 0.3)

        # Ensure within reasonable range
        posterior = min(max(posterior, 0.0), 1.0)

        return posterior


class EnhancedExplainableReasoningEngine:
    """Enhanced explainable reasoning engine"""

    def __init__(self, index_path: str, metadata_path: str, graph_path: str):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Vector index file not found: {index_path}")

        self.index = faiss.read_index(index_path)
        self.index_vectors = None
        self.function_embeddings_cache = {}
        print(f"Initializing FAISS index: dimension={self.index.d}, vectors={self.index.ntotal}")

        # Initialize enhanced LLM discriminator
        self.llm_judge = EnhancedLLMJudge()

        # Initialize informative prior Bayesian engine
        self.bayesian_engine = InformativeBayesianEngine(debug_mode=False)

        # Initialize call chain summarizer
        self.chain_summarizer = None  # Will be initialized after summary_map is loaded

        # Metadata and graph structure loading
        self.meta_map = {}
        self.summary_map = {}
        self.id_to_meta_index = {}

        with open(metadata_path, 'r', encoding='utf-8') as f:
            raw_metadata = json.load(f)

        vectors_list = []
        if isinstance(raw_metadata, dict):
            vectors_list = raw_metadata.get("vectors", raw_metadata.get("Vectors", []))
        elif isinstance(raw_metadata, list):
            vectors_list = raw_metadata

        # id_list aligns raw_id with FAISS index position: id_list[i] is the raw_id
        # whose embedding is stored at FAISS index i. _search_similar_functions
        # uses self.id_list[idx] to map FAISS hits back to a function name.
        self.id_list: List[str] = []

        for idx, meta in enumerate(vectors_list):
            raw_id = meta.get('node_id') or meta.get('id') or meta.get('function_name') or meta.get('name')
            if raw_id:
                clean_id = clean_func_id(raw_id)

                self.meta_map[raw_id] = meta
                self.meta_map[clean_id] = meta
                self.id_to_meta_index[raw_id] = idx
                self.id_to_meta_index[clean_id] = idx
                # Pad id_list to match the FAISS index layout (index i must
                # align with vectors_list[i]).
                while len(self.id_list) <= idx:
                    self.id_list.append(None)
                self.id_list[idx] = raw_id

                summary = meta.get('summary', '') or meta.get('docstring', '')
                if summary:
                    self.summary_map[raw_id] = summary
                    self.summary_map[clean_id] = summary

        # Drop any None slots (the metadata list may have had entries missing
        # an id) so id_list is strictly parallel to the FAISS vectors.
        self.id_list = [x for x in self.id_list if x is not None]

        # Initialize call chain summarizer
        self.chain_summarizer = CallChainSummarizer(self.summary_map)

        with open(graph_path, 'r', encoding='utf-8') as f:
            raw_graph = json.load(f)

        self.knowledge_graph = {}
        self.reverse_graph = {}
        self.dataflow_graph = {}

        edges = []
        if isinstance(raw_graph, dict) and "edges" in raw_graph:
            edges = raw_graph["edges"]
        elif isinstance(raw_graph, list):
            edges = raw_graph

        call_edges_count = 0
        dataflow_edges_count = 0

        for edge in edges:
            edge_type = edge.get("type", "")
            raw_source = edge.get("source_id")
            raw_target = edge.get("target_id")

            if not raw_source or not raw_target:
                continue

            source = clean_func_id(raw_source)
            target = clean_func_id(raw_target)

            if edge_type == "CALLS":
                if source not in self.knowledge_graph:
                    self.knowledge_graph[source] = {"calls": []}
                if target not in self.knowledge_graph[source]["calls"]:
                    self.knowledge_graph[source]["calls"].append(target)

                if target not in self.reverse_graph:
                    self.reverse_graph[target] = []
                if source not in self.reverse_graph[target]:
                    self.reverse_graph[target].append(source)

                call_edges_count += 1

            elif edge_type in ["DATAFLOW", "PARAMETER", "RETURN"]:
                if source not in self.dataflow_graph:
                    self.dataflow_graph[source] = {
                        "has_params": False,
                        "has_returns": False,
                        "dataflow_edges": []
                    }

                self.dataflow_graph[source]["dataflow_edges"].append({
                    "type": edge_type,
                    "target": target
                })

                if edge_type == "PARAMETER":
                    self.dataflow_graph[source]["has_params"] = True
                elif edge_type == "RETURN":
                    self.dataflow_graph[source]["has_returns"] = True

                dataflow_edges_count += 1

        self.embedder = TextEmbedding
        self.stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'to', 'of', 'in', 'for', 'a', 'an', 'be', 'by',
                           'from', 'or', 'as', 'with', 'it', 'this', 'that', 'c', 'cpp', 'java', 'void', 'int', 'char',
                           'return'}

        self.embedding_cache = {}
        self._load_index_vectors()

    def _load_index_vectors(self):
        """Extract all vectors from FAISS index"""
        try:
            if hasattr(self.index, 'reconstruct') and self.index.ntotal > 0:
                vectors = []
                for i in range(self.index.ntotal):
                    vec = self.index.reconstruct(i)
                    vectors.append(vec)
                self.index_vectors = np.array(vectors, dtype='float32')
        except Exception as e:
            logger.warning(f"_load_index_vectors failed (ntotal={getattr(self.index, 'ntotal', '?')}): {e}")
            self.index_vectors = None

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get text embedding vector with caching"""
        if not text:
            return np.zeros(self.index.d, dtype='float32')

        cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()

        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        try:
            resp = self.embedder.call(model="text-embedding-v4", input=text, parameters={"text_type": "query"})
            if resp.status_code == 200:
                embedding = np.array(resp.output['embeddings'][0]['embedding'], dtype='float32')
                self.embedding_cache[cache_key] = embedding
                return embedding
            else:
                raise Exception(f"Embedding Error: {resp.message}")
        except Exception as e:
            logger.warning(f"_get_embedding failed (text_len={len(text)}): {e}")
            return np.zeros(self.index.d, dtype='float32')

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        if vec1 is None or vec2 is None or np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _calculate_semantic_similarity(self, query_text: str, function_name: str) -> Tuple[float, str]:
        """Calculate dual semantic similarity"""
        candidate_names = []
        candidate_names.append(function_name)
        candidate_names.append(clean_func_id(function_name))

        if function_name.startswith("Function:"):
            candidate_names.append(function_name.replace("Function:", ""))

        valid_func_name = None
        for name in candidate_names:
            if name in self.id_to_meta_index:
                valid_func_name = name
                break

        if not valid_func_name:
            return 0.0, f"Function not in vector database"

        function_name = valid_func_name

        function_summary = self.summary_map.get(function_name, "")

        query_embedding = self._get_embedding(query_text)

        vector_similarity = 0.0
        if self.index_vectors is not None and function_name in self.id_to_meta_index:
            idx = self.id_to_meta_index[function_name]
            if idx < len(self.index_vectors):
                func_embedding = self.index_vectors[idx]
                vector_similarity = self._cosine_similarity(query_embedding, func_embedding)

        summary_similarity = 0.0
        if function_summary:
            summary_embedding = self._get_embedding(function_summary)
            summary_similarity = self._cosine_similarity(query_embedding, summary_embedding)

        if vector_similarity > 0 and summary_similarity > 0:
            final_similarity = 0.6 * vector_similarity + 0.4 * summary_similarity
            detail = f"Vector DB:{vector_similarity:.3f}+Summary:{summary_similarity:.3f}"
        elif vector_similarity > 0:
            final_similarity = vector_similarity
            detail = f"Vector DB:{vector_similarity:.3f}"
        elif summary_similarity > 0:
            final_similarity = summary_similarity
            detail = f"Summary:{summary_similarity:.3f}"
        else:
            final_similarity = 0.0
            detail = "No semantic information"

        return max(0.0, min(1.0, final_similarity)), detail

    def _search_similar_functions(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, float, str]]:
        """Search for similar functions"""
        query_vec = self._get_embedding(query)
        if len(query_vec) != self.index.d:
            return []

        query_array = np.array([query_vec], dtype='float32')
        try:
            distances, indices = self.index.search(query_array, top_k)
            results = []

            for idx, dist in zip(indices[0], distances[0]):
                if idx != -1 and idx < len(self.id_list):
                    raw_func_name = self.id_list[idx]

                    possible_names = [raw_func_name]

                    clean_name = clean_func_id(raw_func_name)
                    if clean_name != raw_func_name:
                        possible_names.append(clean_name)

                    if raw_func_name.startswith("Function:"):
                        possible_names.append(raw_func_name.replace("Function:", ""))

                    func_name = None
                    for name in possible_names:
                        if name in self.id_to_meta_index:
                            func_name = name
                            break

                    if not func_name:
                        func_name = raw_func_name

                    similarity, detail = self._calculate_semantic_similarity(query, func_name)

                    if similarity > SIMILARITY_THRESHOLD:
                        results.append((func_name, similarity, detail))

            return results
        except Exception as e:
            logger.warning(f"_search_similar_functions failed (query_len={len(query)}): {e}")
            return []

    def _extract_function_info_from_metadata(self, node_id: str) -> Dict:
        """Extract function information from metadata"""
        clean_id = clean_func_id(node_id)

        if clean_id in self.meta_map:
            meta = self.meta_map[clean_id]
        elif node_id in self.meta_map:
            meta = self.meta_map[node_id]
        else:
            return {
                "function_name": node_id,
                "file_path": "Unknown",
                "language": "unknown",
                "summary": "",
                "signature": ""
            }

        return {
            "function_name": node_id,
            "file_path": meta.get("file_path", meta.get("path", "Unknown")),
            "language": meta.get("language", "unknown"),
            "summary": meta.get("summary", "") or meta.get("docstring", ""),
            "signature": meta.get("signature", "")
        }

    def _get_keywords(self, text: str) -> Set[str]:
        """Extract keywords"""
        words = re.findall(r'\w+', text.lower())
        return set([w for w in words if w not in self.stop_words and len(w) > 1])

    def _traverse_with_semantic_summary(self, start_func_name: str, query_text: str,
                                        analysis_mode: str) -> Tuple[List[Dict], Dict]:
        """Traversal analysis based on semantic summary"""

        analyzed_summaries = set()
        evidence_chains = []
        stats = {
            "total_chains_found": 0,
            "unique_chains_analyzed": 0,
            "llm_judgments_made": 0,
            "analysis_mode": analysis_mode
        }

        def dfs_collect_chains(current_func, depth, current_chain, all_chains, seen, max_depth=MAX_HOPS):
            """Collect all possible call chains (not analyzed immediately)"""
            if depth >= max_depth or len(current_chain) > 10:
                return

            current_clean = clean_func_id(current_func)

            # Record current chain. `seen` is a per-call set scoped to this traversal;
            # a fresh `_traverse_with_semantic_summary` call gets its own set
            # so the same chain shape can be re-analysed across requirements.
            if len(current_chain) >= 2:  # At least 2 nodes to be considered valid chain
                chain_key = "->".join(current_chain)
                if chain_key not in seen:
                    all_chains.append(current_chain.copy())
                    seen.add(chain_key)

            # Continue traversal
            if current_clean in self.knowledge_graph:
                for callee in self.knowledge_graph[current_clean].get("calls", []):
                    if len(current_chain) < 10:
                        dfs_collect_chains(callee, depth + 1,
                                           current_chain + [callee],
                                           all_chains, seen, max_depth)

        # Step 1: Collect all possible call chains (per-call set, not instance state)
        all_chains = []
        seen_chains: Set[str] = set()
        dfs_collect_chains(start_func_name, 0, [start_func_name], all_chains,
                           seen_chains, max_depth=MAX_HOPS)

        stats["total_chains_found"] = len(all_chains)

        if not all_chains:
            return [], stats

        # Step 2: Perform semantic summarization and deduplication on collected chains
        unique_chains = []

        for chain in all_chains:
            # Generate semantic summary
            semantic_summary, key_nodes = self.chain_summarizer.summarize_call_chain(chain)

            if not semantic_summary or len(semantic_summary) < 10:
                continue

            # Create hash of summary for deduplication
            summary_hash = hashlib.md5(semantic_summary.encode()).hexdigest()[:16]

            if summary_hash in analyzed_summaries:
                continue  # Skip semantically similar chains

            analyzed_summaries.add(summary_hash)
            unique_chains.append({
                "chain": chain,
                "semantic_summary": semantic_summary,
                "key_nodes": key_nodes,
                "summary_hash": summary_hash
            })

        stats["unique_chains_analyzed"] = len(unique_chains)

        if not unique_chains:
            return [], stats

        # Step 3: Analyze unique semantic chains. "simplified" caps the chain
        # count at 1 to keep the analysis cheap; "longest_only" is wider,
        # "full" is widest.
        if analysis_mode == "full":
            max_chains_to_analyze = 20
        elif analysis_mode == "longest_only":
            max_chains_to_analyze = 10
        else:  # simplified / fallback
            max_chains_to_analyze = 1

        for i, chain_info in enumerate(unique_chains[:max_chains_to_analyze]):
            chain = chain_info["chain"]
            semantic_summary = chain_info["semantic_summary"]
            key_nodes = chain_info["key_nodes"]

            # Collect key node details
            node_details = {}
            for node in key_nodes[:5]:  # Collect at most 5 key nodes
                clean_node = clean_func_id(node)
                detail = self.summary_map.get(clean_node, "")
                if detail:
                    node_details[node] = detail

            # Use LLM to analyze semantic summary
            llm_scores, llm_reason = self.llm_judge.judge_call_chain_by_semantic_summary(
                requirement=query_text[:500],
                semantic_summary=semantic_summary,
                key_nodes=key_nodes,
                node_details=node_details
            )

            stats["llm_judgments_made"] += 1

            # Calculate chain comprehensive score
            # 1. LLM score for semantic summary
            summary_score = llm_scores.overall

            # 2. Average semantic similarity of nodes in chain
            chain_semantic_scores = []
            for node in chain:
                score, _ = self._calculate_semantic_similarity(query_text, node)
                chain_semantic_scores.append(score)
            avg_semantic = sum(chain_semantic_scores) / len(chain_semantic_scores) if chain_semantic_scores else 0

            # 3. Initial structure score of chain
            structure_score = min(len(chain) / 8, 1.0)  # Initial structure score

            # 4. Calculate prior probability
            prior = self.bayesian_engine._calculate_chain_prior(
                chain=chain,
                avg_semantic=avg_semantic,
                summary_score=summary_score,
                knowledge_graph=self.knowledge_graph
            )

            # 5. Calculate comprehensive posterior probability
            posterior = self.bayesian_engine.compute_chain_posterior(
                prior=prior,
                summary_score=summary_score,
                avg_semantic=avg_semantic,
                structure_score=structure_score,
                llm_scores=llm_scores,
                chain=chain,
                knowledge_graph=self.knowledge_graph,
                dataflow_graph=self.dataflow_graph
            )

            # Create evidence entry
            evidence = {
                "chain": chain,
                "semantic_summary": semantic_summary,
                "key_nodes": key_nodes,
                "chain_length": len(chain),
                "avg_semantic_score": avg_semantic,
                "summary_llm_score": summary_score,
                "bayesian_posterior": posterior,
                "llm_reason": llm_reason,
                "llm_scores": {
                    "overall": llm_scores.overall,
                    "completeness": llm_scores.logical_completeness,
                    "coherence": llm_scores.dataflow_continuity,
                    "efficiency": llm_scores.call_rationality
                },
                "analysis_mode": analysis_mode,
                "prior_probability": prior
            }

            evidence_chains.append(evidence)

        return evidence_chains, stats

    def _select_best_chain_and_generate_report(self, all_chains: List[Dict],
                                               query_text: str) -> Dict:
        """Select best call chain and generate explanatory report"""

        if not all_chains:
            return {
                "selected_chain": None,
                "explanatory_report": "No relevant call chain found",
                "selection_reason": "No available call chains"
            }

        # Strategy 1: Prioritize long chains with high posterior probability
        candidate_chains = []

        for chain_info in all_chains:
            # Basic filtering conditions
            if (chain_info.get("bayesian_posterior", 0) >= MIN_CONFIDENCE and
                    chain_info.get("summary_llm_score", 0) >= 0.6 and
                    len(chain_info.get("chain", [])) >= 2):
                candidate_chains.append(chain_info)

        if not candidate_chains:
            # If no chains meet conditions, use all chains
            candidate_chains = all_chains

        # Scoring function: comprehensively consider length, posterior probability, LLM score
        def score_chain(chain_info):
            length = len(chain_info.get("chain", []))
            posterior = chain_info.get("bayesian_posterior", 0)
            llm_score = chain_info.get("summary_llm_score", 0)
            avg_semantic = chain_info.get("avg_semantic_score", 0)

            # Normalize length into [0, 1] using MAX_HOPS+1 as the maximum expected
            # chain depth. Capping prevents very long chains from drowning out
            # the other signal sources.
            max_expected_length = MAX_HOPS + 1
            normalized_length = min(length, max_expected_length) / max_expected_length

            # Weights: length(0.4) + posterior probability(0.3) + LLM score(0.2) + semantic similarity(0.1)
            score = (
                    normalized_length * 0.4 +
                    posterior * 0.3 +
                    llm_score * 0.2 +
                    avg_semantic * 0.1
            )

            return score

        # Sort by comprehensive score
        candidate_chains.sort(key=score_chain, reverse=True)

        # Select best chain
        best_chain = candidate_chains[0] if candidate_chains else all_chains[0]

        # Generate explanatory report
        explanatory_report = self._generate_explanatory_report(best_chain, query_text)

        return {
            "selected_chain": best_chain,
            "explanatory_report": explanatory_report,
            "selection_reason": self._explain_selection_reason(best_chain, candidate_chains),
            "alternative_chains": candidate_chains[1:4] if len(candidate_chains) > 1 else []
        }

    def _generate_explanatory_report(self, chain_info: Dict, query_text: str) -> str:
        """Generate explanatory report"""

        report_parts = []

        # 1. Report title
        report_parts.append("=" * 80)
        report_parts.append("Requirements-Code Traceability Explanatory Report")
        report_parts.append("=" * 80)

        # 2. Requirement overview
        report_parts.append("\n## 1. Requirement Overview")
        report_parts.append(f"Requirement Description: {query_text[:200]}...")

        # 3. Selected call chain information
        report_parts.append("\n## 2. Selected Call Chain Information")

        chain = chain_info.get("chain", [])
        semantic_summary = chain_info.get("semantic_summary", "")
        key_nodes = chain_info.get("key_nodes", [])

        report_parts.append(f"### 2.1 Functional Chain Semantic Description")
        report_parts.append(f"{semantic_summary}")

        report_parts.append(f"\n### 2.2 Original Call Chain ({len(chain)} functions)")
        report_parts.append(f"Call Path: {' → '.join(chain)}")

        if key_nodes:
            report_parts.append(f"\n### 2.3 Key Function Nodes ({len(key_nodes)} nodes)")
            for i, node in enumerate(key_nodes[:5]):  # Show at most 5 key nodes
                node_info = self._extract_function_info_from_metadata(node)
                summary = node_info.get("summary", "")
                if summary:
                    report_parts.append(f"{i + 1}. {node}: {summary[:100]}" +
                                        ("..." if len(summary) > 100 else ""))

        # 4. Quality assessment scores
        report_parts.append("\n## 3. Quality Assessment Scores")

        scores_table = [
            ["Assessment Dimension", "Score", "Evaluation"],
            ["Posterior Probability", f"{chain_info.get('bayesian_posterior', 0):.4f}",
             "Excellent" if chain_info.get('bayesian_posterior', 0) >= 0.7 else
             "Good" if chain_info.get('bayesian_posterior', 0) >= 0.5 else "Average"],
            ["LLM Chain Score", f"{chain_info.get('summary_llm_score', 0):.3f}",
             "Excellent" if chain_info.get('summary_llm_score', 0) >= 0.8 else
             "Good" if chain_info.get('summary_llm_score', 0) >= 0.6 else "Average"],
            ["Average Semantic Similarity", f"{chain_info.get('avg_semantic_score', 0):.3f}",
             "Excellent" if chain_info.get('avg_semantic_score', 0) >= 0.7 else
             "Good" if chain_info.get('avg_semantic_score', 0) >= 0.5 else "Average"],
            ["Chain Length", f"{len(chain)}",
             "Complete" if len(chain) >= 4 else "Moderate" if len(chain) >= 3 else "Short"]
        ]

        # Format table
        col_widths = [20, 10, 10]
        for row in scores_table:
            formatted_row = "".join([f"{cell:<{width}}" for cell, width in zip(row, col_widths)])
            report_parts.append(formatted_row)

        # 5. LLM analysis conclusion
        if chain_info.get("llm_reason"):
            report_parts.append("\n## 4. LLM Analysis Conclusion")
            report_parts.append(chain_info.get("llm_reason"))

        # 6. Traceability confidence summary
        report_parts.append("\n## 5. Traceability Confidence Summary")

        posterior = chain_info.get("bayesian_posterior", 0)
        if posterior >= 0.8:
            confidence = "High Confidence"
            explanation = "This call chain is highly likely to completely implement the requirement functionality"
        elif posterior >= 0.6:
            confidence = "Medium Confidence"
            explanation = "This call chain likely implements the core functionality of the requirement"
        elif posterior >= 0.4:
            confidence = "Low Confidence"
            explanation = "This call chain may be partially related to the requirement, further verification recommended"
        else:
            confidence = "Very Low Confidence"
            explanation = "This call chain has weak relevance to the requirement"

        report_parts.append(f"**Confidence Level**: {confidence}")
        report_parts.append(f"**Explanation**: {explanation}")
        report_parts.append(f"**Main Basis**: ")
        report_parts.append(f"  - Posterior Probability: {posterior:.4f}")
        report_parts.append(f"  - LLM Score: {chain_info.get('summary_llm_score', 0):.3f}")
        report_parts.append(f"  - Chain Length: {len(chain)} functions")

        # 7. Recommendations and next steps
        report_parts.append("\n## 6. Recommendations and Next Steps")

        if posterior >= 0.7:
            report_parts.append("**Recommendation**: This call chain has high quality and can be directly used for code implementation reference")
        elif posterior >= 0.5:
            report_parts.append("**Recommendation**: This call chain is basically usable, recommend further verification with other code contexts")
        else:
            report_parts.append("**Recommendation**: This call chain has low quality, recommend re-analyzing requirements or checking code base")

        report_parts.append("\n**Recommended Next Steps**:")
        report_parts.append("1. Review key function implementations in the selected call chain")
        report_parts.append("2. Verify data flow transmission in the chain is correct")
        report_parts.append("3. Test if this call chain meets all requirement boundary conditions")

        report_parts.append("\n" + "=" * 80)
        report_parts.append("Report generation completed")
        report_parts.append("=" * 80)

        return "\n".join(report_parts)

    def _explain_selection_reason(self, selected_chain: Dict, candidate_chains: List[Dict]) -> str:
        """Explain reason for selecting this chain"""

        reasons = []
        chain_length = len(selected_chain.get("chain", []))
        posterior = selected_chain.get("bayesian_posterior", 0)
        llm_score = selected_chain.get("summary_llm_score", 0)

        # Reason 1: Chain length
        if chain_length >= 4:
            reasons.append(f"Long chain length ({chain_length} functions), more likely to form complete functionality")
        elif chain_length >= 3:
            reasons.append(f"Moderate chain length ({chain_length} functions)")
        else:
            reasons.append(f"Short chain length ({chain_length} functions), but other scores are high")

        # Reason 2: Posterior probability
        if posterior >= 0.7:
            reasons.append(f"High posterior probability ({posterior:.3f}), strong relevance to requirement")
        elif posterior >= 0.5:
            reasons.append(f"Medium posterior probability ({posterior:.3f}), some relevance to requirement")

        # Reason 3: LLM score
        if llm_score >= 0.8:
            reasons.append(f"High LLM chain score ({llm_score:.3f}), good functional completeness")
        elif llm_score >= 0.6:
            reasons.append(f"Medium LLM chain score ({llm_score:.3f}), basically complete functionality")

        # Compare with alternative chains
        if len(candidate_chains) > 1:
            second_chain = candidate_chains[1] if len(candidate_chains) > 1 else None
            if second_chain:
                second_posterior = second_chain.get("bayesian_posterior", 0)
                second_length = len(second_chain.get("chain", []))

                if posterior > second_posterior + 0.1:
                    reasons.append(f"Posterior probability significantly higher than second place ({second_posterior:.3f})")
                if chain_length > second_length:
                    reasons.append(f"Chain length longer than second place ({second_length} functions)")

        return "; ".join(reasons) if reasons else "Highest comprehensive score"

    def explain_file_enhanced_strategy(self, file_path: str) -> Dict:
        """Enhanced strategy requirement explanation function"""
        print(f"Reading requirement document: {file_path}")

        splitter = TextSplitter(chunk_size=600, chunk_overlap=50)
        chunks = splitter.load_and_split(file_path)

        if not chunks:
            return {"error": "Document is empty"}

        # Phase 1: All chunk vector retrieval aggregation
        all_candidates = []
        candidate_details = {}

        for i, chunk in enumerate(chunks):
            similar_results = self._search_similar_functions(chunk, top_k=5)

            for func_name, similarity, detail in similar_results:
                if func_name not in candidate_details:
                    candidate_details[func_name] = {
                        "total_score": 0.0,
                        "chunks": [],
                        "best_similarity": 0.0,
                        "best_detail": "",
                        "best_chunk_index": i
                    }

                candidate_details[func_name]["total_score"] += similarity
                candidate_details[func_name]["chunks"].append({
                    "chunk_index": i,
                    "similarity": similarity,
                    "detail": detail
                })

                if similarity > candidate_details[func_name]["best_similarity"]:
                    candidate_details[func_name]["best_similarity"] = similarity
                    candidate_details[func_name]["best_detail"] = detail
                    candidate_details[func_name]["best_chunk_index"] = i

        for func_name, details in candidate_details.items():
            all_candidates.append({
                "function": func_name,
                "total_score": details["total_score"],
                "chunk_count": len(details["chunks"]),
                "best_similarity": details["best_similarity"],
                "best_detail": details["best_detail"],
                "best_chunk_index": details["best_chunk_index"],
                "chunks": details["chunks"]
            })

        all_candidates.sort(key=lambda x: x["total_score"], reverse=True)

        # Phase 2: Intelligent hierarchical analysis
        if not all_candidates:
            return {
                "total_chunks": len(chunks),
                "total_candidates": 0,
                "analyzed_candidates": 0,
                "analysis_stats": {},
                "selected_result": {
                    "chain": [],
                    "semantic_summary": "",
                    "bayesian_posterior": 0,
                    "summary_llm_score": 0,
                    "selection_reason": "No candidate functions",
                    "explanatory_report": "No relevant call chain found"
                }
            }

        top_k_candidates = min(5, len(all_candidates))
        selected_candidates = all_candidates[:top_k_candidates]

        # Check if there are high similarity candidates
        similarity_scores = [c["best_similarity"] for c in selected_candidates]
        max_similarity = max(similarity_scores) if similarity_scores else 0

        has_high_similarity = max_similarity >= HIGH_SIMILARITY_THRESHOLD

        if not has_high_similarity:
            backup_analysis_modes = ["full", "longest_only", "simplified", "simplified", "simplified"]
        else:
            backup_analysis_modes = None

        all_evidence = []
        all_stats = []

        for i, candidate in enumerate(selected_candidates):
            func_name = candidate["function"]
            base_similarity = candidate["best_similarity"]

            # Select analysis mode based on whether there are high similarity candidates
            if not has_high_similarity:
                # Fallback strategy: use predefined analysis modes
                if i < len(backup_analysis_modes):
                    analysis_mode = backup_analysis_modes[i]
                else:
                    analysis_mode = "simplified"
            else:
                # Standard hierarchical strategy
                if base_similarity >= HIGH_SIMILARITY_THRESHOLD:
                    analysis_mode = "full"
                elif base_similarity >= MEDIUM_SIMILARITY_THRESHOLD:
                    analysis_mode = "longest_only"
                else:
                    analysis_mode = "simplified"

            merged_keywords = set()
            for chunk_info in candidate["chunks"]:
                chunk_idx = chunk_info["chunk_index"]
                if chunk_idx < len(chunks):
                    chunk_text = chunks[chunk_idx]
                    keywords = self._get_keywords(chunk_text)
                    merged_keywords.update(keywords)

            best_chunk_idx = candidate.get("best_chunk_index", candidate["chunks"][0]["chunk_index"])
            query_text = chunks[best_chunk_idx] if 0 <= best_chunk_idx < len(chunks) else chunks[0]

            evidence_chains, stats = self._traverse_with_semantic_summary(
                start_func_name=func_name,
                query_text=query_text,
                analysis_mode=analysis_mode
            )

            all_evidence.extend(evidence_chains)
            all_stats.append(stats)

        # Phase 3: Deduplication and sorting
        seen_summaries = set()
        unique_evidence = []

        for item in all_evidence:
            summary_hash = hashlib.md5(item["semantic_summary"].encode()).hexdigest()[:16]
            if summary_hash not in seen_summaries:
                seen_summaries.add(summary_hash)
                unique_evidence.append(item)

        unique_evidence.sort(key=lambda x: x["bayesian_posterior"], reverse=True)

        # Combine statistics
        combined_stats = {
            "total_chains_found": sum(s.get("total_chains_found", 0) for s in all_stats),
            "unique_chains_analyzed": sum(s.get("unique_chains_analyzed", 0) for s in all_stats),
            "total_llm_judgments": sum(s.get("llm_judgments_made", 0) for s in all_stats)
        }

        # Select best chain and generate report
        query_text = chunks[0] if chunks else ""
        result_info = self._select_best_chain_and_generate_report(unique_evidence, query_text)

        return {
            "total_chunks": len(chunks),
            "total_candidates": len(all_candidates),
            "analyzed_candidates": len(selected_candidates),
            "analysis_stats": combined_stats,
            "selected_result": {
                "chain": result_info["selected_chain"]["chain"] if result_info["selected_chain"] else [],
                "semantic_summary": result_info["selected_chain"].get("semantic_summary", "") if result_info[
                    "selected_chain"] else "",
                "bayesian_posterior": result_info["selected_chain"].get("bayesian_posterior", 0) if result_info[
                    "selected_chain"] else 0,
                "summary_llm_score": result_info["selected_chain"].get("summary_llm_score", 0) if result_info[
                    "selected_chain"] else 0,
                "selection_reason": result_info["selection_reason"],
                "explanatory_report": result_info["explanatory_report"]
            },
            "alternative_chains": [
                {
                    "chain": alt["chain"],
                    "semantic_summary": alt.get("semantic_summary", ""),
                    "bayesian_posterior": alt.get("bayesian_posterior", 0),
                    "summary_llm_score": alt.get("summary_llm_score", 0)
                }
                for alt in result_info.get("alternative_chains", [])
            ]
        }


def _build_faiss_from_metadata(metadata_path: str, output_path: str) -> bool:
    """Rebuild a FAISS index by embedding each entry in metadatas.json via DashScope.
    Used as a fallback when no pre-built FAISS index is available alongside the knowledge base.
    Returns True on success, False on failure. Requires dashscope.api_key to be configured."""
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            entries = raw.get("vectors", raw.get("Vectors", []))
        elif isinstance(raw, list):
            entries = raw
        else:
            print(f"Unsupported metadata schema in {metadata_path}")
            return False

        if not entries:
            print("No entries in metadata; cannot build FAISS index.")
            return False

        vectors = []
        for entry in entries:
            text = (entry.get("summary") or entry.get("docstring") or
                    entry.get("signature") or entry.get("name") or "")
            if not text.strip():
                continue
            try:
                resp = TextEmbedding.call(model="text-embedding-v4", input=text)
                if resp.status_code == 200:
                    emb = np.array(resp.output['embeddings'][0]['embedding'], dtype='float32')
                    vectors.append(emb)
            except Exception as e:
                raw_id = entry.get("node_id") or entry.get("id") or entry.get("name")
                print(f"Embedding failed for {raw_id}: {e}")
                continue

        if not vectors:
            print("No embeddings produced; check dashscope.api_key configuration.")
            return False

        mat = np.vstack(vectors)
        dim = mat.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(mat)
        faiss.write_index(index, output_path)
        print(f"Built FAISS index from metadata: {len(vectors)} vectors, dim={dim}, saved to {output_path}")
        return True
    except Exception as e:
        print(f"FAISS rebuild failed: {e}")
        return False


def main():
    """Main function"""
    # If FAISS_INDEX_PATH or KNOWLEDGE_GRAPH_PATH is left blank, derive it from
    # the directory that holds METADATAS_JSON_PATH (KB writes
    # metadatas.json / vectors.faiss / edges.json side-by-side).
    global FAISS_INDEX_PATH, KNOWLEDGE_GRAPH_PATH
    metadata_dir = os.path.dirname(METADATAS_JSON_PATH) if METADATAS_JSON_PATH else ""
    if not FAISS_INDEX_PATH and metadata_dir:
        FAISS_INDEX_PATH = os.path.join(metadata_dir, "vectors.faiss")
        print(f"FAISS_INDEX_PATH not set; using {FAISS_INDEX_PATH}")
    if not KNOWLEDGE_GRAPH_PATH and metadata_dir:
        # Prefer knowledge_graph.json (RC-shaped sidecar), fall back to edges.json (KB-native).
        candidate = os.path.join(metadata_dir, "knowledge_graph.json")
        if not os.path.exists(candidate):
            candidate = os.path.join(metadata_dir, "edges.json")
        KNOWLEDGE_GRAPH_PATH = candidate
        print(f"KNOWLEDGE_GRAPH_PATH not set; using {KNOWLEDGE_GRAPH_PATH}")

    if not os.path.exists(FAISS_INDEX_PATH):
        # Fallback: rebuild from metadata so the system can run without a pre-built index.
        if os.path.exists(METADATAS_JSON_PATH):
            print(f"FAISS index not found at {FAISS_INDEX_PATH}; rebuilding from {METADATAS_JSON_PATH}...")
            if not _build_faiss_from_metadata(METADATAS_JSON_PATH, FAISS_INDEX_PATH):
                print("Error: Failed to build FAISS index. Check dashscope.api_key configuration.")
                return
        else:
            print(f"Error: Vector index file not found: {FAISS_INDEX_PATH}")
            return

    try:
        print("Initializing enhanced reasoning engine...")
        engine = EnhancedExplainableReasoningEngine(
            index_path=FAISS_INDEX_PATH,
            metadata_path=METADATAS_JSON_PATH,
            graph_path=KNOWLEDGE_GRAPH_PATH
        )

        print("System initialization completed")

        while True:
            user_input = input("\nPlease enter requirement file path (or enter 'quit' to exit): ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input:
                continue

            if os.path.exists(user_input) and user_input.endswith('.txt'):
                result = engine.explain_file_enhanced_strategy(user_input)
            else:
                print("Please provide a valid .txt file path")
                continue

            if "error" in result:
                print(f"Error: {result['error']}")
                continue

            # Display analysis statistics
            stats = result.get("analysis_stats", {})
            print(f"\nAnalysis Statistics:")
            print(f"  Requirement Chunks: {result['total_chunks']}")
            print(f"  Candidate Functions: {result['total_candidates']}")
            print(f"  Analyzed Functions: {result['analyzed_candidates']}")
            print(f"  Analyzed Call Chains: {stats.get('unique_chains_analyzed', 0)}")
            print(f"  LLM Analysis Count: {stats.get('total_llm_judgments', 0)}")

            print(f"\n{'=' * 60} Final Traceability Result {'=' * 60}")

            if "selected_result" in result and result["selected_result"]["chain"]:
                selected = result["selected_result"]

                # Display selected call chain
                print(f"\nSelected Functional Chain:")
                print(f"   Functional Description: {selected.get('semantic_summary', '')[:100]}...")
                print(f"   Call Chain: {' → '.join(selected['chain'][:5])}" +
                      (f" ... ({len(selected['chain'])} functions)" if len(selected['chain']) > 5 else ""))
                print(f"   Posterior Probability: {selected.get('bayesian_posterior', 0):.4f}")
                print(f"   LLM Chain Score: {selected.get('summary_llm_score', 0):.3f}")
                print(f"   Selection Reason: {selected.get('selection_reason', '')}")

                # Display explanatory report summary
                print(f"\nExplanatory Report Summary:")
                report_lines = selected.get("explanatory_report", "").split('\n')
                for line in report_lines[:30]:  # Show first 30 lines
                    print(line)

                # Ask if full report should be displayed
                if len(report_lines) > 30:
                    user_choice = input(f"\n📄 Explanatory report has {len(report_lines)} lines, display full report? (y/n): ")
                    if user_choice.lower() == 'y':
                        for line in report_lines:
                            print(line)

                # Display alternative chains
                if result.get("alternative_chains"):
                    print(f"\nAlternative Call Chains:")
                    for i, alt in enumerate(result["alternative_chains"][:2]):  # Show at most 2 alternatives
                        print(f"[Alternative {i + 1}] {alt.get('semantic_summary', '')[:80]}...")
                        print(f"    Chain: {' → '.join(alt['chain'][:3])}" +
                              (f" ..." if len(alt['chain']) > 3 else ""))
                        print(
                            f"    Posterior: {alt.get('bayesian_posterior', 0):.4f} | LLM Score: {alt.get('summary_llm_score', 0):.3f}")
                        print("-" * 40)
            else:
                print("No qualified call chain found")

    except Exception as e:
        print(f"System runtime error: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()