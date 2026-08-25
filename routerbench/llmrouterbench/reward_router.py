"""
Reward-based Routing for LLM Selection
-----------------------------------------
Implements the reward-based routing strategy from report Section 4.2.4:
Instead of using an LLM as the decision agent, we directly compute rewards
from retrieved similar queries and select the LLM with the highest reward.

Architecture (follows agenticrouter_normalizedcost.py):
    1. For each test query, use an LLM to generate difficulty analysis
    # 2. Use difficulty analysis text to retrieve top-k similar response analyses
    #    from each model's FAISS Response DB
    # 3. For each model, compute reward from the retrieved historical samples:
    #    Reward = γ * Mean(Score) - (1-γ) * Normalized_Mean(Cost)
    2. Use difficulty analysis text to retrieve top-k similar difficulty analyses
       from the FAISS Difficulty DB (getting a unified set of similar queries).
    3. For each model, compute reward on this EXACT SAME set of queries:
       Reward = γ * Mean(Score) - (1-γ) * Normalized_Mean(Cost)
    4. Select the model with the highest reward

Usage:
    python reward_router.py \
        --k 10 --gamma 0.8 \
        --model-config qwen2.5 \
        --routing-data data/routing_data_train.jsonl \
        --input data/query_data_test.jsonl \
        --output data/results_rewardrouter_g08.jsonl
"""

import argparse
import json
import os
import sys
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from tqdm.auto import tqdm
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

TARGET_MODELS = [
    'claude-sonnet-4', 'deepseek-r1-0528', 'deepseek-v3-0324', 'deepseek-v3.1-terminus',
    'gemini-2.5-flash', 'gemini-2.5-pro', 'glm-4.6', 'gpt-5', 'intern-s1', 'kimi-k2-0905',
    'qwen3-235b-a22b-2507'
]

# LLM configurations for the difficulty analysis agent
CONFIG_LIST = [
    {
        "name": "qwen3.5",
        "model": "Qwen/Qwen3.5-35B-A3B",
        "api_key": "NULL",
        "base_url": "http://0.0.0.0:8000/v1",
    },
    {
        "name": "gemma4",
        "model": "google/gemma-4-31b-it",
        "api_key": os.getenv('NVIDIA_API_KEY'),
        "base_url": "https://integrate.api.nvidia.com/v1",
    }
    # {
    #     "name": "gpt-4o-mini",
    #     "model": "gpt-4o-mini",
    #     "api_key": os.getenv("OPENAI_API_KEY1"),
    # },
    # {
    #     "name": "gemini-2.5-flash-lite",
    #     "model": "gemini-2.5-flash-lite-preview-06-17",
    #     "api_key": os.getenv('GEMINI_API_KEY'),
    #     "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    # },
]


class DifficultyAnalystAgent:
    """Reused from agenticrouter_normalizedcost.py — generates difficulty analysis for queries."""
    
    def __init__(self, client: OpenAI, model_name: str, temperature: float = 0.5):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.system_message = (
            "Your role as an assistant is to analyze the difficulty of a given query for a large "
            "language model through a systematic long thinking process analysis. You will be "
            "provided with the user query and some context from past similar analyses. You need "
            "to evaluate the incoming query on several key dimensions: reasoning, comprehension, "
            "instruction following, agentic, knowledge retrieval, coding, multilingual. For each "
            "dimension, elaborate on the specific challenges and required capabilities. Now, try "
            "to analyze the following query through the above guidelines:\n"
        )
    
    def analyze(self, query: str, relevant_analyses: List[Any]) -> Dict[str, Any]:
        formatted_analyses = "\n".join([f"- {doc.page_content}" for doc in relevant_analyses])
        
        prompt = f"""
        **Context from similar past analyses:**
        {formatted_analyses if formatted_analyses else "No relevant past analyses found."}

        ---

        **Query to Analyze:**
        "{query}"
        """
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": self.system_message + prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=2000,
                )
                if response.choices[0].message.content is not None:
                    difficulty_assessment = response.choices[0].message.content.strip().lower()
                    return {
                        "query": query,
                        "difficulty": difficulty_assessment,
                    }
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"  Warning: DifficultyAnalyst failed after {max_retries} retries: {e}")
                    return {
                        "query": query,
                        "difficulty": f"Failed to analyze: {query[:100]}",
                    }
        
        return {"query": query, "difficulty": ""}


class Retriever:
    """Reused from agenticrouter_normalizedcost.py — retrieves from FAISS DBs."""
    
    def __init__(self, difficulty_db: FAISS):
        self.difficulty_db = difficulty_db

    def retrieve_difficulty_analyses(self, query: str, k: int = 3) -> List[Any]:
        """Return list of (Document, similarity_weight) tuples.
        FAISS returns L2 distance (lower = more similar), so we convert to
        similarity weight = 1 / (1 + distance).
        """
        if self.difficulty_db:
            results_with_scores = self.difficulty_db.similarity_search_with_score(query, k=k)
            # Convert L2 distance to similarity weight
            return [(doc, 1.0 / (1.0 + score)) for doc, score in results_with_scores]
        return []


def load_routing_data(path: str) -> dict:
    """
    Load routing_data_train.jsonl and build lookup:
        {query_text: {model_name: {"performance": ..., "cost": ...}}}
    """
    lookup = defaultdict(dict)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            q = rec['query']
            m = rec['model_name']
            lookup[q][m] = {
                'performance': float(rec.get('performance', 0.0)),
                'cost': float(rec.get('cost', 0.0)),
            }
    return dict(lookup)


def compute_max_total_cost(routing_lookup: dict) -> float:
    """
    Compute max_total_cost for normalization, same as evaluate_routing.py:
    Average of the maximum cost per query across all models.
    """
    if not routing_lookup:
        return 1.0
    total = sum(
        max(m['cost'] for m in q_models.values())
        for q_models in routing_lookup.values()
    )
    return total / len(routing_lookup)


def compute_reward_routing(
    relevant_training_docs_with_scores: List[Any],
    routing_lookup: dict,
    max_total_cost: float,
    gamma: float,
    target_models: list,
) -> str:
    """
    Compute reward for each model based on the EXACT SAME set of retrieved training queries,
    using similarity weights for weighted averaging.
    """
    model_rewards = {}
    
    # 1. 取出所有檢索到的 training queries 及其 similarity weights
    training_entries = []  # list of (query, weight)
    for doc, sim_weight in relevant_training_docs_with_scores:
        query = doc.metadata.get('query', '')
        if query and query in routing_lookup:
            training_entries.append((query, sim_weight))
            
    # 2. 針對所有 target_models，用 similarity-weighted 平均計算表現與成本
    for model_name in target_models:
        scores = []
        costs = []
        weights = []
        
        for query, w in training_entries:
            if model_name in routing_lookup[query]:
                perf = routing_lookup[query][model_name]['performance']
                cost = routing_lookup[query][model_name]['cost']
                scores.append(perf)
                costs.append(cost)
                weights.append(w)
                
        if not scores:
            model_rewards[model_name] = float('-inf')
            continue
        
        weights_arr = np.array(weights)
        # ----- 原始寫法: simple mean (註解掉) -----
        # mean_score = np.mean(scores)
        # mean_cost = np.mean(costs)
        
        # ----- 新寫法: similarity-weighted mean -----
        mean_score = np.average(scores, weights=weights_arr)
        mean_cost = np.average(costs, weights=weights_arr)
        
        normalized_cost = mean_cost / max_total_cost if max_total_cost > 0 else 0
        reward = gamma * mean_score - (1 - gamma) * normalized_cost
        model_rewards[model_name] = reward
    
    # Select the model with the highest reward
    if not model_rewards or all(v == float('-inf') for v in model_rewards.values()):
        # Fallback if no valid reward computed
        return target_models[0]
        
    best_model = max(model_rewards, key=model_rewards.get)
    return best_model


def main():
    parser = argparse.ArgumentParser(description="Reward-based LLM Routing")
    parser.add_argument('--k', type=int, default=10,
                        help='Top-k retrieval count per model')
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='Reward weight γ (higher = more weight on performance)')
    parser.add_argument('--model-config', type=str, default='qwen2.5',
                        choices=[c['name'] for c in CONFIG_LIST],
                        help='LLM config for difficulty analysis agent')
    parser.add_argument('--difficulty-db', type=str,
                        default=os.path.join(DATA_DIR, 'faiss_difficulty_db'),
                        help='Path to FAISS Difficulty DB')
    parser.add_argument('--routing-data', type=str,
                        default=os.path.join(DATA_DIR, 'routing_data_train.jsonl'),
                        help='Training routing data (performance/cost ground truth)')
    parser.add_argument('--input', type=str,
                        default=os.path.join(DATA_DIR, 'query_data_test.jsonl'),
                        help='Test queries input file')
    parser.add_argument('--output', type=str,
                        default=os.path.join(DATA_DIR, 'results_rewardrouter.jsonl'),
                        help='Output results file')
    parser.add_argument('--embedding-device', type=str, default='cuda:1',
                        help='Device for the embedding model')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of parallel threads for routing')
    args = parser.parse_args()

    print("=" * 60)
    print("  Reward-based LLM Router")
    print("=" * 60)
    print(f"  k={args.k}, γ={args.gamma}")
    print(f"  Model config: {args.model_config}")
    print(f"  Difficulty DB: {args.difficulty_db}")
    print(f"  Routing data: {args.routing_data}")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print()

    # ---------- Initialize LLM client for difficulty analysis ----------
    selected_config = next((c for c in CONFIG_LIST if c['name'] == args.model_config), None)
    if not selected_config:
        print(f"Error: Config '{args.model_config}' not found.")
        sys.exit(1)
    
    client = OpenAI(
        api_key=selected_config.get('api_key'),
        base_url=selected_config.get('base_url'),
    )
    agent_model = selected_config['model']
    print(f"Difficulty analysis agent: {agent_model}")

    # ---------- Initialize embedding model ----------
    print("Loading embedding model (Qwen/Qwen3-Embedding-0.6B)...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={'device': args.embedding_device}
    )
    print(f"Embedding model loaded on {args.embedding_device}.")

    # ---------- Load FAISS databases ----------
    print("Loading FAISS databases...")
    
    difficulty_db = None
    if os.path.exists(args.difficulty_db):
        difficulty_db = FAISS.load_local(
            args.difficulty_db, embedding_model,
            allow_dangerous_deserialization=True
        )
        print(f"  Loaded Difficulty DB from {args.difficulty_db}")
    else:
        print(f"  WARNING: Difficulty DB not found at {args.difficulty_db}")

    if not difficulty_db:
        print("\nError: Could not load the difficulty database.")
        sys.exit(1)

    # ---------- Load routing ground truth ----------
    print(f"\nLoading routing data from {args.routing_data}...")
    routing_lookup = load_routing_data(args.routing_data)
    print(f"  Loaded {len(routing_lookup)} queries with routing data")

    max_total_cost = compute_max_total_cost(routing_lookup)
    print(f"  max_total_cost for normalization: {max_total_cost:.6f}")

    # ---------- Load test queries ----------
    print(f"\nLoading test queries from {args.input}...")
    test_queries = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                test_queries.append(rec['query'])
    print(f"  Total test queries: {len(test_queries)}")

    # ---------- Initialize components ----------
    retriever = Retriever(difficulty_db)
    analyst = DifficultyAnalystAgent(client, agent_model)

    # ---------- Run routing ----------
    print(f"\n{'='*60}")
    print("Starting Reward-based Routing...")
    print(f"{'='*60}\n")

    # Load cache
    cache_dir = os.path.join(DATA_DIR, "difficulty_analysis_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_filepath = os.path.join(cache_dir, f"difficulty_cache_{args.model_config}.json")
    
    difficulty_cache = {}
    if os.path.exists(cache_filepath):
        with open(cache_filepath, 'r', encoding='utf-8') as f:
            difficulty_cache = json.load(f)
        print(f"  Loaded {len(difficulty_cache)} cached difficulty analyses")
    else:
        print(f"  No existing cache found. Will generate and save to {cache_filepath}")

    results = []
    model_counts = defaultdict(int)
    cache_hits = 0
    cache_misses = 0
    cache_lock = threading.Lock()
    
    def process_query(query):
        # Cache check
        with cache_lock:
            cached_diff = difficulty_cache.get(query)
        is_cache_hit = cached_diff is not None
        
        if is_cache_hit:
            difficulty_text = cached_diff
        else:
            relevant_difficulty_with_scores = retriever.retrieve_difficulty_analyses(query, k=3)
            relevant_difficulty = [doc for doc, _ in relevant_difficulty_with_scores]
            difficulty_analysis = analyst.analyze(query, relevant_difficulty)
            difficulty_text = difficulty_analysis.get('difficulty', '')
            with cache_lock:
                difficulty_cache[query] = difficulty_text
                
        # ----- 原始寫法 (註解掉) -----
        # relevant_responses = retriever.retrieve_model_responses(difficulty_text, k=args.k)
        # best_model = compute_reward_routing(
        #     relevant_responses, routing_lookup, max_total_cost,
        #     args.gamma, TARGET_MODELS
        # )
        
        # ----- 新寫法 (修正方向 2) -----
        # 用 difficulty analysis 去檢索 difficulty DB (統一的 query DB)
        relevant_training_docs = retriever.retrieve_difficulty_analyses(difficulty_text, k=args.k)
        best_model = compute_reward_routing(
            relevant_training_docs, routing_lookup, max_total_cost,
            args.gamma, TARGET_MODELS
        )
        
        return {
            "query": query,
            "model_name": best_model,
            "success": True,
        }, is_cache_hit

    print(f"  Using {args.threads} parallel threads.")
    with open(args.output, 'w', encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(process_query, q) for q in test_queries]
            
            for future in tqdm(as_completed(futures), total=len(test_queries), desc="Routing"):
                result, is_cache_hit = future.result()
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush()
                
                results.append(result)
                model_counts[result['model_name']] += 1
                if is_cache_hit:
                    cache_hits += 1
                else:
                    cache_misses += 1

    # Save cache
    with open(cache_filepath, 'w', encoding='utf-8') as f:
        json.dump(difficulty_cache, f, ensure_ascii=False)
    print(f"\n  Difficulty analysis cache saved to {cache_filepath} ({len(difficulty_cache)} entries)")
    print(f"  Cache stats: {cache_hits} hits, {cache_misses} misses")

    # ---------- Summary ----------
    print(f"\n{'='*60}")
    print("  Routing Complete!")
    print(f"{'='*60}")
    print(f"  Total queries routed: {len(results)}")
    print(f"  Output: {args.output}")
    print(f"  γ={args.gamma}, k={args.k}")
    print(f"\n  Model distribution:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        bar = "█" * int(pct / 2)
        print(f"    {model:30s} {count:5d} ({pct:5.1f}%) {bar}")
    print()


if __name__ == '__main__':
    main()
