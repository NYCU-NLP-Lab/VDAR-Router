"""
Reward-based Routing Baseline for LLM Selection
---------------------------------------------------
This is a training-free baseline that replaces the LLM decision-maker
in agenticrouter_normalizedcost.py with a mathematical reward computation.

Architecture:
    1. For each test query, use an LLM to generate difficulty analysis
    2. Use difficulty analysis text to retrieve top-k similar response analyses
       from each model's FAISS Response DB
    3. For each model, compute reward from the retrieved historical samples:
       Reward = γ * Mean(Score) - (1-γ) * Normalized_Mean(Cost)
    4. Select the model with the highest reward

All data loading, FAISS DB, and train/test split follow
agenticrouter_normalizedcost.py exactly.

Usage:
    python reward_router.py --model-config qwen2.5 --gamma 0.8 --k 10
    python reward_router.py --model-config gpt-4o-mini --gamma 0.9 --k 5 --ood
"""

import numpy as np
import pandas as pd
import pickle
import json
import sys
import os
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Resolve paths relative to the routerbench/ directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROUTERBENCH_DIR = os.path.dirname(SCRIPT_DIR)  # routerbench/

DIFFICULTY_DB_PATH = os.path.join(ROUTERBENCH_DIR, "faiss_dbs", "difficulty_db")
OOD_DIFFICULTY_DB_PATH = os.path.join(ROUTERBENCH_DIR, "faiss_dbs", "difficulty_db_ood")
DIFFICULTY_CACHE_DIR = os.path.join(ROUTERBENCH_DIR, "cache", "difficulty_analysis_cache")

LLM_LIST = {
    'WizardLM/WizardLM-13B-V1.2': 1.03, 
    'claude-instant-v1': 1.23, 
    'claude-v1': 3.59, 
    'claude-v2': 3.93, 
    'gpt-3.5-turbo-1106': 1.24, 
    'gpt-4-1106-preview': 5.0, 
    'meta/code-llama-instruct-34b-chat': 1.16, 
    'meta/llama-2-70b-chat': 1.19, 
    'mistralai/mistral-7b-chat': 1.0, 
    'mistralai/mixtral-8x7b-chat': 1.11, 
    'zero-one-ai/Yi-34B-Chat': 1.17
}

config_list = [
    {
        "name": "gemma4",
        "model": "google/gemma-4-31b-it",
        "api_key": os.getenv('NVIDIA_API_KEY'),
        "base_url": "https://integrate.api.nvidia.com/v1",
    },
    # {
    #     "name": "qwen3.5",
    #     "model": "qwen/qwen3.5-flash-02-23",
    #     "api_key": os.getenv("OPENROUTER_API_KEY"),
    #     "base_url": "https://openrouter.ai/api/v1",
    # },
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

llm_config = {
    "config_list": config_list,
    "cache_seed": 42,
    "temperature": 0.5,
}

# Global data loaded at import time (same as agenticrouter_normalizedcost.py)
try:
    with open(os.path.join(ROUTERBENCH_DIR, 'data', 'llm_analyses_results.pkl'), 'rb') as f:
        response_data = pickle.load(f)
except (FileNotFoundError, EOFError):
    print("Warning: 'llm_analyses_results.pkl' not found. Response retrieval will be disabled.")
    response_data = {}

try:
    difficulty_df = pd.read_csv(os.path.join(ROUTERBENCH_DIR, 'data', 'sampled_router_bench_with_difficulty_analysis.csv'))
    difficulty_data = difficulty_df.set_index('sample_id')['difficulty_analysis_summary'].to_dict()
    resp_sample_ids = difficulty_df['sample_id'].tolist()
except (FileNotFoundError, KeyError):
    print("Warning: 'sampled_router_bench_with_difficulty_analysis.csv' not found. Difficulty retrieval will be disabled.")
    difficulty_data = {}


# ─────────────────────────────────────────────────────────────
# DifficultyAnalystAgent  (identical to agenticrouter_normalizedcost.py)
# ─────────────────────────────────────────────────────────────
class DifficultyAnalystAgent:
    def __init__(self, client: OpenAI, model_name: str, temperature: float):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.system_message = """Your role as an assistant is to analyze the difficulty of a given query for a large language model through a systematic long thinking process analysis. You will be provided with the user query and some context from past similar analyses. You need to evaluate the incoming query on several key dimensions: reasoning, comprehension, instruction following, agentic, knowledge retrieval, coding, multilingual. For each dimension, elaborate on the specific challenges and required capabilities. Now, try to analyze the following query through the above guidelines:\n"""
        
    def analyze(self, query: str, relevant_analyses: List[Any]) -> Dict[str, Any]:
        formatted_analyses = "\n".join([f"- {doc.page_content}" for doc in relevant_analyses])

        prompt = f"""
        **Context from similar past analyses:**
        {formatted_analyses if formatted_analyses else "No relevant past analyses found."}

        ---

        **Query to Analyze:**
        "{query}"
        """
        
        while True:
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
                    "analysis": f"The query is classified as '{difficulty_assessment}' based on LLM analysis."
                }


# ─────────────────────────────────────────────────────────────
# Retriever  (identical to agenticrouter_normalizedcost.py)
# ─────────────────────────────────────────────────────────────
class Retriever:
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


# ─────────────────────────────────────────────────────────────
# RewardRoutingDecisionMaker  (replaces LLM-based RoutingDecisionMakerAgent)
# ─────────────────────────────────────────────────────────────
class RewardRoutingDecisionMaker:
    """
    Compute a reward score per model from the retrieved historical samples
    and select the model with the highest reward.

    Reward = γ * Mean(Score) - (1-γ) * Normalized_Mean(Cost)

    Where:
        - Score is the correctness (0/1) of each retrieved sample for that model
        - Cost is the actual cost from the cost column in the dataframe
        - Normalized cost = Mean(Cost) / max_cost_across_all_models
    """
    def __init__(self, gamma: float = 0.8):
        self.gamma = gamma

    # ----- 原始寫法 (註解掉) -----
    # def decide(
    #     self,
    #     relevant_responses: Dict[str, List[Any]],
    #     llm_list: Dict[str, float],
    #     train_df: pd.DataFrame,
    # ) -> Tuple[str, Dict[str, float]]:
    #     """
    #     Args:
    #         relevant_responses: {model_name: [Document, ...]} from FAISS retrieval
    #         llm_list: {model_name: normalized_cost_weight} — the static cost info
    #         train_df: training dataframe with sample_id, model columns (scores),
    #                   and model|total_cost columns (costs)
    #     
    #     Returns:
    #         (best_model_name, model_rewards_dict)
    #     """
    #     llm_names = list(llm_list.keys())
    #     train_df_indexed = train_df.set_index('sample_id')
    # 
    #     model_rewards = {}
    #     model_details = {}
    # 
    #     # Pre-compute max cost across all models in the training set for normalization
    #     cost_columns = [f"{m}|total_cost" for m in llm_names if f"{m}|total_cost" in train_df.columns]
    #     if cost_columns:
    #         max_cost = train_df[cost_columns].values.max()
    #     else:
    #         max_cost = 1.0  # fallback
    # 
    #     for model_name in llm_names:
    #         docs = relevant_responses.get(model_name, [])
    #         if not docs:
    #             model_rewards[model_name] = float('-inf')
    #             continue
    # 
    #         scores = []
    #         costs = []
    #         cost_col = f"{model_name}|total_cost"
    # 
    #         for doc in docs:
    #             sample_id = doc.metadata.get('sample_id')
    #             if sample_id is not None and sample_id in train_df_indexed.index:
    #                 try:
    #                     score = float(train_df_indexed.loc[sample_id, model_name])
    #                     scores.append(score)
    #                 except (KeyError, ValueError):
    #                     pass
    #                 
    #                 # Get actual cost for this sample
    #                 if cost_col in train_df_indexed.columns:
    #                     try:
    #                         cost = float(train_df_indexed.loc[sample_id, cost_col])
    #                         costs.append(cost)
    #                     except (KeyError, ValueError):
    #                         pass
    # 
    #         if not scores:
    #             model_rewards[model_name] = float('-inf')
    #             continue
    # 
    #         mean_score = np.mean(scores)
    #         mean_cost = np.mean(costs) if costs else 0.0
    #         normalized_cost = mean_cost / max_cost if max_cost > 0 else 0.0
    # 
    #         reward = self.gamma * mean_score - (1 - self.gamma) * normalized_cost
    #         model_rewards[model_name] = reward
    #         model_details[model_name] = {
    #             'mean_score': mean_score,
    #             'mean_cost': mean_cost,
    #             'normalized_cost': normalized_cost,
    #             'reward': reward,
    #             'n_retrieved': len(scores),
    #         }
    # 
    #     # Select the model with the highest reward
    #     best_model = max(model_rewards, key=model_rewards.get)
    #     return best_model, model_rewards

    # ----- 新寫法 (修正方向 2) -----
    def decide(
        self,
        relevant_training_docs_with_scores: List[Any],
        llm_list: Dict[str, float],
        train_df: pd.DataFrame,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Args:
            relevant_training_docs_with_scores: [(Document, similarity_weight), ...] from FAISS difficulty DB
            llm_list: {model_name: normalized_cost_weight}
            train_df: training dataframe with sample_id, model columns (scores),
                      and model|total_cost columns (costs)
        
        Returns:
            (best_model_name, model_rewards_dict)
        """
        llm_names = list(llm_list.keys())
        train_df_indexed = train_df.set_index('sample_id')

        model_rewards = {}
        model_details = {}

        # Pre-compute max cost across all models in the training set for normalization
        cost_columns = [f"{m}|total_cost" for m in llm_names if f"{m}|total_cost" in train_df.columns]
        if cost_columns:
            max_cost = train_df[cost_columns].values.max()
        else:
            max_cost = 1.0  # fallback
            
        # 1. 取出所有檢索到的 training query sample_ids 及其 similarity weights
        training_entries = []  # list of (sample_id, weight)
        for doc, sim_weight in relevant_training_docs_with_scores:
            sample_id = doc.metadata.get('sample_id')
            if sample_id is not None and sample_id in train_df_indexed.index:
                training_entries.append((sample_id, sim_weight))

        # 2. 針對所有 model_names，用 similarity-weighted 平均計算表現與成本
        for model_name in llm_names:
            scores = []
            costs = []
            weights = []
            cost_col = f"{model_name}|total_cost"

            for sample_id, w in training_entries:
                # 取得分數
                try:
                    score = float(train_df_indexed.loc[sample_id, model_name])
                    scores.append(score)
                except (KeyError, ValueError):
                    continue
                
                # 取得成本
                if cost_col in train_df_indexed.columns:
                    try:
                        cost = float(train_df_indexed.loc[sample_id, cost_col])
                        costs.append(cost)
                    except (KeyError, ValueError):
                        costs.append(0.0)
                else:
                    costs.append(0.0)
                
                weights.append(w)

            if not scores:
                model_rewards[model_name] = float('-inf')
                continue

            weights_arr = np.array(weights)
            # ----- 原始寫法: simple mean (註解掉) -----
            # mean_score = np.mean(scores)
            # mean_cost = np.mean(costs) if costs else 0.0
            
            # ----- 新寫法: similarity-weighted mean -----
            mean_score = np.average(scores, weights=weights_arr)
            mean_cost = np.average(costs, weights=weights_arr) if costs else 0.0
            
            normalized_cost = mean_cost / max_cost if max_cost > 0 else 0.0

            reward = self.gamma * mean_score - (1 - self.gamma) * normalized_cost
            model_rewards[model_name] = reward
            model_details[model_name] = {
                'mean_score': mean_score,
                'mean_cost': mean_cost,
                'normalized_cost': normalized_cost,
                'reward': reward,
                'n_retrieved': len(scores),
            }

        # Select the model with the highest reward
        if not model_rewards or all(v == float('-inf') for v in model_rewards.values()):
            return llm_names[0], model_rewards
            
        best_model = max(model_rewards, key=model_rewards.get)
        return best_model, model_rewards


# ─────────────────────────────────────────────────────────────
# RewardRouter  (replaces AgenticRouter)
# ─────────────────────────────────────────────────────────────
class RewardRouter:
    def __init__(
        self,
        retriever: Retriever,
        llm_list: Dict[str, float],
        client: OpenAI,
        model_name: str,
        temperature: float,
        train_df: pd.DataFrame,
        gamma: float = 0.8,
        k: int = 10,
    ):
        self.difficulty_analyst = DifficultyAnalystAgent(client, model_name, temperature)
        self.retriever = retriever
        self.decision_maker = RewardRoutingDecisionMaker(gamma=gamma)
        self.llm_list = llm_list
        self.train_df = train_df
        self.k = k

    def route(self, query: str, cached_difficulty: str = None) -> Tuple[str, Dict[str, Any], Dict[str, float]]:
        # Step 1: Retrieve relevant difficulty analyses to provide context to the analyst.
        relevant_analyses_with_scores = self.retriever.retrieve_difficulty_analyses(query)
        relevant_analyses = [doc for doc, _ in relevant_analyses_with_scores]

        # Step 2: Analyze query difficulty (use cache if available, otherwise call LLM).
        if cached_difficulty is not None:
            difficulty_analysis = {
                "query": query,
                "difficulty": cached_difficulty,
                "analysis": f"The query is classified as '{cached_difficulty[:50]}...' (loaded from cache)."
            }
        else:
            difficulty_analysis = self.difficulty_analyst.analyze(query, relevant_analyses)
        
        # ----- 原始寫法 (註解掉) -----
        # Step 3: Retrieve relevant model responses using the difficulty analysis text.
        # relevant_responses = self.retriever.retrieve_model_responses(
        #     difficulty_analysis['difficulty'], k=self.k
        # )
        # 
        # Step 4: Compute reward for each model and select the best one.
        # decision, model_rewards = self.decision_maker.decide(
        #     relevant_responses,
        #     self.llm_list,
        #     self.train_df,
        # )
        
        # ----- 新寫法 (修正方向 2) -----
        # Step 3: 用 difficulty analysis 去檢索 difficulty DB (統一的 query DB)
        relevant_training_docs_with_scores = self.retriever.retrieve_difficulty_analyses(
            difficulty_analysis['difficulty'], k=self.k
        )
        
        # Step 4: Compute reward for each model and select the best one based on the unified queries.
        decision, model_rewards = self.decision_maker.decide(
            relevant_training_docs_with_scores,
            self.llm_list,
            self.train_df,
        )

        return decision, difficulty_analysis, model_rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Reward-based Router Benchmark")
    parser.add_argument(
        "--model-config",
        type=str,
        default="qwen2.5",
        choices=[c["name"] for c in config_list],
        help="The name of the model configuration to use for the difficulty analysis agent."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.8,
        help="Reward weight γ: Reward = γ * Mean(Score) - (1-γ) * Normalized_Mean(Cost). "
             "Higher γ favors performance over cost. (default: 0.8)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of top-k similar responses to retrieve per model. (default: 10)"
    )
    parser.add_argument(
        "--ood",
        action="store_true",
        help="Enable OOD testing mode. Uses top 1-10 eval_names for DBs and 11-15 for testing."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of parallel threads for routing. (default: 8)"
    )
    args = parser.parse_args()

    # Set DB paths based on OOD flag
    if args.ood:
        DIFFICULTY_DB_PATH = OOD_DIFFICULTY_DB_PATH
        print("OOD testing mode enabled. Using OOD database paths.")

    # Initialize OpenAI from selected config
    selected_config = next((c for c in config_list if c["name"] == args.model_config), None)
    if not selected_config:
        print(f"Error: Model configuration '{args.model_config}' not found.")
        sys.exit(1)
    
    print(f"Using model configuration: '{selected_config['name']}'")
    print(f"Reward parameters: γ={args.gamma}, k={args.k}")
    client = OpenAI(
        api_key=selected_config.get("api_key"),
        base_url=selected_config.get("base_url")
    )
    agent_model_name = selected_config["model"]
    temperature = llm_config["temperature"]

    print("Initializing embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={'device': 'cuda:0'}
    )
    print("Embedding model initialized on cuda:0.")

    # --- Load data and split into train/test for benchmark ---
    print("Loading and splitting data for benchmark...")
    try:
        full_df = pd.read_csv(os.path.join(ROUTERBENCH_DIR, 'data', 'router_bench_with_keywords.csv'))
        original_train_df_for_responses = pd.read_csv(os.path.join(ROUTERBENCH_DIR, 'data', 'sampled_router_bench_with_difficulty_analysis.csv'))
        original_resp_sample_ids = original_train_df_for_responses['sample_id'].tolist()

        if args.ood:
            print("OOD Mode: Splitting data based on 'eval_name' categories.")
            eval_name_counts = full_df['eval_name'].value_counts()
            id_eval_names = eval_name_counts.index[:10]
            ood_eval_names = eval_name_counts.index[10:15]

            print(f"In-distribution 'eval_name's (top 10) for DB creation: {list(id_eval_names)}")
            print(f"Out-of-distribution 'eval_name's (11-15) for testing: {list(ood_eval_names)}")

            # Filter original training data to create in-distribution training set
            train_df = original_train_df_for_responses[original_train_df_for_responses['eval_name'].isin(id_eval_names)].copy()
            
            # Create OOD test set
            test_df = full_df[full_df['eval_name'].isin(ood_eval_names)].copy()
            print(f"Loaded {len(train_df)} samples for OOD training DB and {len(test_df)} for OOD testing.")
            total_test_samples = len(test_df)
        else:
            train_df = pd.read_csv(os.path.join(ROUTERBENCH_DIR, 'data', 'sampled_router_bench_with_difficulty_analysis.csv'))
            
            train_ids = set(train_df['sample_id'])
            test_df = full_df[~full_df['sample_id'].isin(train_ids)].copy()
            
            print(f"Loaded {len(train_df)} samples for training and {len(test_df)} for testing.")
            
            # --- Stratified sampling of test_df ---
            total_test_samples = 2000
            n_per_category = (test_df['eval_name'].value_counts(normalize=True) * total_test_samples).round().astype(int)
            diff = total_test_samples - n_per_category.sum()
            if diff != 0:
                n_per_category[n_per_category.idxmax()] += diff
            test_df = test_df.groupby('eval_name', group_keys=False).apply(
                lambda x: x.sample(n=int(n_per_category[x.name]), random_state=42)
            ).reset_index(drop=True)
            print(f"Final test set sampled to {len(test_df)} samples based on eval_name proportion.")

        # The training data for difficulty analysis is from train_df
        difficulty_data = train_df.set_index('sample_id')['difficulty_analysis_summary'].to_dict()
        # resp_sample_ids is now used to filter responses, so it should be a set for efficient lookup
        resp_sample_ids_set = set(train_df['sample_id'].tolist())

    except FileNotFoundError as e:
        print(f"Error: Required data file not found: {e}. Aborting benchmark.")
        sys.exit(1)

    difficulty_db = None
    if os.path.exists(DIFFICULTY_DB_PATH):
        print(f"Loading difficulty DB from {DIFFICULTY_DB_PATH}...")
        difficulty_db = FAISS.load_local(
            DIFFICULTY_DB_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        print("Difficulty DB loaded.")
    elif difficulty_data:
        print("Creating difficulty analysis vector database from training data...")
        difficulty_documents = [
            Document(page_content=summary, metadata={"sample_id": sample_id})
            for sample_id, summary in difficulty_data.items()
        ]
        difficulty_db = FAISS.from_documents(difficulty_documents, embedding_model)
        difficulty_db.save_local(DIFFICULTY_DB_PATH)
        print(f"Difficulty DB saved to {DIFFICULTY_DB_PATH}.")

    if not difficulty_db:
        print("\nError: No data available to create vector database. Please check your data files.")
        sys.exit(1)

    print("\nInitializing Reward Router...")
    retriever = Retriever(difficulty_db)
    router = RewardRouter(
        retriever, LLM_LIST, client, agent_model_name, temperature,
        train_df, gamma=args.gamma, k=args.k
    )
    print(f"Reward Router initialized (γ={args.gamma}, k={args.k}).")

    print("\n" + "="*50 + "\n")
    print("Starting benchmark on the test set...")

    # --- Load or initialize difficulty analysis cache ---
    os.makedirs(DIFFICULTY_CACHE_DIR, exist_ok=True)
    cache_filename = f"difficulty_cache_{args.model_config}_{'ood' if args.ood else 'indomain'}.json"
    cache_filepath = os.path.join(DIFFICULTY_CACHE_DIR, cache_filename)
    difficulty_cache = {}  # key: query text -> value: difficulty analysis text
    if os.path.exists(cache_filepath):
        with open(cache_filepath, 'r', encoding='utf-8') as f:
            difficulty_cache = json.load(f)
        print(f"Loaded {len(difficulty_cache)} cached difficulty analyses from {cache_filepath}")
    else:
        print(f"No existing cache found at {cache_filepath}. Will generate and save.")

    total_correct = 0
    total_cost = 0.0
    num_test_samples = len(test_df)
    model_selection_counts = {m: 0 for m in LLM_LIST}
    cache_hits = 0
    cache_misses = 0
    cache_lock = threading.Lock()

    # Pre-build a list of (row_idx, query, row_data) for parallel processing
    test_items = [(i, row['prompt'], row) for i, (_, row) in enumerate(test_df.iterrows())]
    # Pre-allocate results arrays (indexed by position)
    results_by_idx = [None] * num_test_samples  # (difficulty_text, decision)

    def process_query(item):
        """Process a single query: cache lookup -> route -> return results."""
        idx, query, row = item

        # Check cache for pre-computed difficulty analysis
        cached_difficulty = difficulty_cache.get(query, None)
        is_cache_hit = cached_difficulty is not None

        decision, difficulty_analysis, model_rewards = router.route(query, cached_difficulty=cached_difficulty)

        # Thread-safe cache update for new analyses
        if not is_cache_hit:
            with cache_lock:
                difficulty_cache[query] = difficulty_analysis['difficulty']

        return idx, query, decision, difficulty_analysis['difficulty'], is_cache_hit, row

    if num_test_samples > 0:
        print(f"Using {args.threads} threads for parallel routing...")
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = {executor.submit(process_query, item): item[0] for item in test_items}

            with tqdm(total=num_test_samples, desc="Benchmarking") as pbar:
                for future in as_completed(futures):
                    idx, query, decision, difficulty_text, is_cache_hit, row = future.result()

                    results_by_idx[idx] = (difficulty_text, decision)

                    if is_cache_hit:
                        cache_hits += 1
                    else:
                        cache_misses += 1

                    model_selection_counts[decision] = model_selection_counts.get(decision, 0) + 1

                    # Check correctness
                    if decision in test_df.columns:
                        correctness = row[decision]
                        total_correct += correctness

                    # Calculate cost
                    cost_column = f"{decision}|total_cost"
                    if cost_column in test_df.columns:
                        cost = row[cost_column]
                        total_cost += cost

                    pbar.update(1)

        # Unpack results in order
        difficulty_analyses_results = [r[0] for r in results_by_idx]
        routing_decisions_results = [r[1] for r in results_by_idx]

        test_df['agent_difficulty_analysis'] = difficulty_analyses_results
        test_df['agent_routing_decision'] = routing_decisions_results
        
        # --- Save difficulty analysis cache ---
        with open(cache_filepath, 'w', encoding='utf-8') as f:
            json.dump(difficulty_cache, f, ensure_ascii=False)
        print(f"\nDifficulty analysis cache saved to {cache_filepath} ({len(difficulty_cache)} entries)")
        print(f"Cache stats: {cache_hits} hits, {cache_misses} misses")

        results_subdir = os.path.join(ROUTERBENCH_DIR, 'results', 'ood' if args.ood else 'indomain')
        os.makedirs(results_subdir, exist_ok=True)
        results_filename = os.path.join(results_subdir, f"reward_router_routerbenchsample{total_test_samples}_results_{args.model_config}_gamma{args.gamma}_k{args.k}_{'ood' if args.ood else 'indomain'}_no_resp_anlys.csv")
        test_df.to_csv(results_filename, index=False)
        print(f"Benchmark results with analyses and decisions saved to {results_filename}")

        print("\n" + "="*50)
        print("Benchmark Finished!")
        print(f"Reward Router Parameters: γ={args.gamma}, k={args.k}")
        print(f"Total test samples: {num_test_samples}")
        print(f"Total correct decisions: {total_correct}")
        print(f"Accuracy: {total_correct / num_test_samples:.4f}")
        print(f"Total cumulative cost (Reward Router): {total_cost}")

        # Model selection distribution
        print(f"\nModel Selection Distribution:")
        for model_name, count in sorted(model_selection_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = count / num_test_samples * 100
                bar = "█" * int(pct / 2)
                print(f"  {model_name:45s} {count:5d} ({pct:5.1f}%) {bar}")

        # Calculate and print cost if all queries were routed to gpt-4-1106-preview
        gpt4_cost_column = "gpt-4-1106-preview|total_cost"
        total_gpt4_cost = 0.0
        if gpt4_cost_column in test_df.columns:
            total_gpt4_cost = test_df[gpt4_cost_column].sum()
            print(f"\nTotal cumulative cost (All to gpt-4-1106-preview): {total_gpt4_cost}")
        else:
            print(f"\nWarning: Cost column '{gpt4_cost_column}' not found in test data. Cannot calculate baseline cost.")

        # Calculate and print accuracy if all queries were routed to gpt-4-1106-preview
        gpt4_accuracy_column = "gpt-4-1106-preview"
        total_gpt4_correct = 0
        if gpt4_accuracy_column in test_df.columns:
            total_gpt4_correct = test_df[gpt4_accuracy_column].sum()
            print(f"Total correct decisions (All to gpt-4-1106-preview): {total_gpt4_correct}")
        else:
            print(f"Warning: Accuracy column '{gpt4_accuracy_column}' not found in test data. Cannot calculate baseline accuracy.")

        # Calculate and print for random routing baseline
        np.random.seed(42) # for reproducibility
        random_total_correct = 0.0
        random_total_cost = 0.0
        llm_names = list(LLM_LIST.keys())
        for _, row in test_df.iterrows():
            random_decision = np.random.choice(llm_names)
            if random_decision in test_df.columns:
                random_total_correct += row[random_decision]
            
            random_cost_column = f"{random_decision}|total_cost"
            if random_cost_column in test_df.columns:
                random_total_cost += row[random_cost_column]
        
        print(f"\nTotal correct decisions (Random): {random_total_correct}")
        print(f"Total cumulative cost (Random): {random_total_cost}")

        print("="*50 + "\n")
    else:
        print("No test samples found to benchmark.")
