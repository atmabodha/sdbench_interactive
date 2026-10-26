import math
import os
import pickle
import re
import time
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, cast

import faiss  # pyright: ignore[reportMissingTypeStubs]
from fuzzywuzzy import fuzz  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from openai import OpenAI
import pandas as pd
from config import CostingConfig
from models import create_chat_completion

if TYPE_CHECKING:
    from costing.cost_types import CostMethodEstimate


class CPTSearchResult(TypedDict):
    """TypedDict for CPT search results with required core fields and optional specialized fields."""

    # Core fields (always present)
    cpt_code: str
    procedure_name: str
    gross_charge: float
    setting: str

    search_type: NotRequired[
        str
    ]  # "semantic", "keyword", "llm_keyword", "llm_semantic", "llm_sourced"

    # Semantic search fields (optional)
    similarity_score: NotRequired[float]

    # Keyword search fields (optional)
    keyword_score: NotRequired[float]
    fuzzy_ratio: NotRequired[int]
    partial_ratio: NotRequired[int]
    token_sort_ratio: NotRequired[int]
    token_set_ratio: NotRequired[int]
    contains_match: NotRequired[bool]
    word_overlap: NotRequired[float]

    # LLM expansion fields (optional)
    expanded_keyword: NotRequired[str]

    pre_merge_results: NotRequired[list["CPTSearchResult"]]  # When multiple results are merged


class CPTSelectionResult(TypedDict, total=False):
    """TypedDict for CPT selection method return values with different possible structures."""

    # Either selected_cpts or error is always present
    selected_cpts: list[str]
    error: str

    selected_results: list[CPTSearchResult]
    missing_cpts: list[str] | None

    reasoning: str
    raw_response: str

    query: str
    search_params: dict[str, Any]
    num_candidates: int
    candidates: list[CPTSearchResult]


def _merge_results(results: list[CPTSearchResult]) -> CPTSearchResult:
    """Merge multiple search results with the same CPT code by taking the median price."""
    if not results:
        raise ValueError("Cannot merge empty list of results")
    return {
        "cpt_code": results[0]["cpt_code"],
        "procedure_name": f"Merged: {','.join(r['procedure_name'] for r in results)}",
        "gross_charge": float(np.median([r["gross_charge"] for r in results])),
        "setting": f"Merged: {','.join(r['setting'] for r in results)}",
        "pre_merge_results": results,
    }


class CPTFuzzyMatcher:
    """A class for fuzzy matching medical procedure descriptions to CPT codes."""

    def __init__(self, config: CostingConfig) -> None:
        """Initialize the CPT Fuzzy Matcher.

        Args:
            config: SDBench configuration containing paths and settings.
        """
        # Prefer OpenRouter when its key is set, otherwise fall back to OpenAI.
        openrouter_key: str | None = os.environ.get("OPENROUTER_API_KEY")
        openai_key: str | None = os.environ.get("OPENAI_API_KEY")
        if openrouter_key:
            self._openai_client: OpenAI = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
            )
            # OpenRouter namespaces models by provider.
            self._embedding_model: str = "openai/text-embedding-3-large"
        elif openai_key:
            self._openai_client = OpenAI(api_key=openai_key)
            self._embedding_model = "text-embedding-3-large"
        else:
            raise ValueError(
                "Neither OPENROUTER_API_KEY nor OPENAI_API_KEY environment variable is set. "
                "One is required to build embeddings for CPT matching."
            )

        self._config = config

        # Load and process the charge master data (resolved under SDBENCH_DATA_PATH/costing_data)
        source_file_path: str = config.resolved_costing_source_file_path
        self._df: pd.DataFrame = self._load_and_process_charge_data(source_file_path)

        # Load or create embeddings and FAISS index (cached next to the charge master)
        embeddings_file: str = source_file_path + ".embeddings.pkl"
        faiss_index_file: str = source_file_path + ".faiss.index"
        self._faiss_index = self._load_or_create_embeddings_and_index(
            embeddings_file, faiss_index_file
        )

    @staticmethod
    def _load_and_process_charge_data(source_file_path: str) -> pd.DataFrame:
        """Load and clean the hospital charge master data."""
        print("Loading charge master data...")

        if not os.path.exists(source_file_path):
            raise FileNotFoundError(
                f"Grounded costing data not found at '{source_file_path}'. "
                "Download the hospital charge master spreadsheet first; see "
                "data/costing_data/README.md for instructions."
            )

        # Read the charge master directly from the source spreadsheet. The relevant
        # data lives on the "Charge Description Master" sheet beneath four header rows.
        df = pd.read_excel(  # pyright: ignore[reportUnknownMemberType]
            source_file_path,
            sheet_name="Charge Description Master",
            skiprows=4,
        )

        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()

        required_columns = [
            "Procedure Code (CPT / HCPCS)",
            "Procedure Name",
            "Gross Charge",
            "Hospital Inpatient / Outpatient / Both",
        ]

        # Validate that all required columns are present
        missing_columns: list[str] = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in charge master file: {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )

        # Keep only rows with valid procedure names
        df = df.dropna(subset=["Procedure Name"])  # pyright: ignore[reportUnknownMemberType]

        # Clean procedure names
        df["Procedure Name"] = df["Procedure Name"].str.strip()

        # Remove empty procedure names
        df = df[df["Procedure Name"] != ""]

        # Ensure gross charge is a valid float and positive
        df["Gross Charge"] = df["Gross Charge"].astype(float)  # Make sure we've got valid charges
        if not (df["Gross Charge"] > 0).all():
            raise ValueError("Data has invalid (zero) charges: " + source_file_path)

        print(f"Loaded {len(df)} valid procedures from charge master")
        return cast(pd.DataFrame, df)

    def median_price(self, cpt_code: str) -> float | None:
        """Get the median price from the charge data for a given CPT code."""
        matching_rows = self._df[self._df["Procedure Code (CPT / HCPCS)"] == cpt_code]
        if matching_rows.empty:
            return None
        if len(matching_rows) == 1:
            return float(matching_rows.iloc[0]["Gross Charge"])
        return float(cast(pd.Series, matching_rows["Gross Charge"]).median())

    def _clean_price(self, price_str: Any) -> float:
        """Clean a price string/value (e.g. a pandas cell) and convert to float."""
        if price_str is None:
            return 0.0
        if isinstance(price_str, float) and math.isnan(price_str):
            return 0.0

        # Remove $ and , and convert to float
        price_str_clean: str = str(price_str).replace("$", "").replace(",", "").strip()
        try:
            return float(price_str_clean)
        except ValueError:
            return 0.0

    def _get_embeddings(self, text: list[str]) -> list[list[float]] | None:
        """Get embedding(s) for text(s) using the configured embedding model."""
        try:
            response = self._openai_client.embeddings.create(
                model=self._embedding_model,
                input=text,
            )
            return [item.embedding for item in response.data]

        except Exception as e:
            if isinstance(text, str):
                print(f"Error getting embedding for text: {text[:50]}... Error: {e}")
            else:
                print(f"Error getting embeddings for batch of {len(text)} texts. Error: {e}")
            return None

    def _load_or_create_embeddings_and_index(
        self, embeddings_file: str, faiss_index_file: str
    ) -> Any:
        """Load embeddings and FAISS index from cache or create new ones."""
        if os.path.exists(faiss_index_file):
            print(f"Loading FAISS index from: {faiss_index_file}")
            return cast(Any, faiss.read_index(faiss_index_file))  # pyright: ignore[reportUnknownMemberType]

        embeddings_array: np.ndarray | None = None
        if os.path.exists(embeddings_file):
            print(f"Loading embeddings from file: {embeddings_file}")
            with open(embeddings_file, "rb") as f:
                cache_data: dict[str, Any] = pickle.load(f)

            # Verify cache is valid for current data
            if len(cache_data["embeddings"]) == len(self._df):
                embeddings_array = np.array(cache_data["embeddings"], dtype=np.float32)
            else:
                print(
                    f"Cache invalid: embeddings count {len(cache_data['embeddings'])} != dataframe rows {len(self._df)}. Rebuilding..."
                )

        if embeddings_array is None:
            print("Creating embeddings for all procedures...")
            all_embeddings: list[list[float] | None] = []

            # Process in batches to optimize API calls and avoid rate limits
            batch_size: int = 500  # OpenAI allows up to 2048, but 500 is a safe batch size
            total_batches: int = (len(self._df) - 1) // batch_size + 1

            for i in range(0, len(self._df), batch_size):
                batch = self._df.iloc[i : i + batch_size]
                batch_texts: list[str] = batch["Procedure Name"].tolist()

                print(
                    f"Processing batch {i // batch_size + 1}/{total_batches} ({len(batch_texts)} texts)"
                )

                # Get embeddings for the entire batch in a single API call
                batch_embeddings = self._get_embeddings(batch_texts)

                if batch_embeddings is not None:
                    all_embeddings.extend(batch_embeddings)  # type: ignore
                else:
                    raise RuntimeError(f"Failed to get embeddings for batch {i // batch_size + 1}")

            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            embeddings_data = {"embeddings": embeddings_array.tolist()}

            # Cache the embeddings
            with open(embeddings_file, "wb") as f:
                pickle.dump(embeddings_data, f)

        # Create FAISS index
        print("Creating FAISS index...")
        dimension: int = embeddings_array.shape[1]

        # Use IndexFlatIP for cosine similarity (inner product with normalized vectors)
        faiss_index = faiss.IndexFlatIP(dimension)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)  # pyright: ignore[reportUnknownMemberType]

        # Add embeddings to index
        faiss_index.add(embeddings_array)  # pyright: ignore[reportUnknownMemberType, reportCallIssue]

        # Cache the FAISS index
        faiss.write_index(faiss_index, faiss_index_file)  # pyright: ignore[reportUnknownMemberType]

        print(f"Created FAISS index with {faiss_index.ntotal} vectors")

        return faiss_index

    def _search_with_embedding(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[CPTSearchResult]:
        """Search using a pre-computed embedding for the top-k most similar CPT codes.

        Args:
            query_embedding: Pre-computed embedding vector
            top_k: Number of top results to return

        Returns:
            List of dictionaries containing the top matches with similarity scores
        """
        # Prepare query embedding for FAISS
        query_embedding_array: np.ndarray = np.array(query_embedding, dtype=np.float32).reshape(
            1, -1
        )

        # Normalize query embedding for cosine similarity
        faiss.normalize_L2(query_embedding_array)  # pyright: ignore[reportUnknownMemberType]

        # Search using FAISS
        similarities, top_indices = self._faiss_index.search(query_embedding_array, top_k)

        # Prepare results
        results: list[CPTSearchResult] = []
        for i in range(len(top_indices[0])):
            idx: int = top_indices[0][i]
            similarity_score: float = float(similarities[0][i])

            # Skip if similarity is too low (can happen with FAISS)
            if similarity_score <= 0:
                continue

            row = self._df.iloc[idx]

            # Clean the gross charge for display
            gross_charge: float = self._clean_price(row["Gross Charge"])

            result: CPTSearchResult = {
                "similarity_score": similarity_score,
                "cpt_code": str(row["Procedure Code (CPT / HCPCS)"]),
                "procedure_name": row["Procedure Name"],
                "gross_charge": gross_charge,
                "setting": row["Hospital Inpatient / Outpatient / Both"],
            }
            results.append(result)

        return results

    def embedding_search(self, query: str, top_k: int = 5) -> list[CPTSearchResult]:
        """Search using embeddings for the top-k most similar CPT codes for a given procedure description.

        Args:
            query: The procedure description to search for
            top_k: Number of top results to return

        Returns:
            List of dictionaries containing the top matches with similarity scores
        """
        # Only print for direct searches, not for LLM expanded searches
        if not hasattr(self, "_quiet_search"):
            print(f"Searching for: '{query}'")

        # Get embedding for the query
        query_embeddings = self._get_embeddings([query])
        if query_embeddings is None:
            print("Failed to get embedding for query")
            return []

        return self._search_with_embedding(query_embeddings[0], top_k)

    def keyword_search(self, query: str, top_k: int = 15) -> list[CPTSearchResult]:
        """Perform keyword-based search using fuzzy string matching and text contains.

        Args:
            query: The procedure description to search for
            top_k: Number of top results to return

        Returns:
            List of dictionaries containing the top matches with fuzzy scores
        """
        # Only print for direct searches, not for LLM expanded searches
        if not hasattr(self, "_quiet_search"):
            print(f"Performing keyword search for: '{query}'")

        query_lower: str = query.lower()
        query_words: set[str] = set(re.findall(r"\b\w+\b", query_lower))

        results: list[CPTSearchResult] = []

        for _, row in self._df.iterrows():
            procedure_name: str = str(row["Procedure Name"])
            procedure_lower: str = procedure_name.lower()

            # Calculate different matching scores
            fuzzy_ratio: int = fuzz.ratio(query_lower, procedure_lower)  # pyright: ignore[reportUnknownMemberType]
            partial_ratio: int = fuzz.partial_ratio(query_lower, procedure_lower)  # pyright: ignore[reportUnknownMemberType]
            token_sort_ratio: int = fuzz.token_sort_ratio(query_lower, procedure_lower)  # pyright: ignore[reportUnknownMemberType]
            token_set_ratio: int = fuzz.token_set_ratio(query_lower, procedure_lower)  # pyright: ignore[reportUnknownMemberType]

            # Check for exact substring match
            contains_match: bool = query_lower in procedure_lower

            # Check for word overlap
            procedure_words: set[str] = set(re.findall(r"\b\w+\b", procedure_lower))
            word_overlap: float = (
                len(query_words.intersection(procedure_words)) / len(query_words)
                if query_words
                else 0
            )

            # Calculate combined score (weighted combination of different metrics)
            combined_score: float = (
                fuzzy_ratio * 0.2
                + partial_ratio * 0.2
                + token_sort_ratio * 0.2
                + token_set_ratio * 0.2
                + (100 if contains_match else 0) * 0.1
                + word_overlap * 100 * 0.1
            )

            # Only include results above a threshold
            if combined_score > 30:  # Adjust threshold as needed
                gross_charge: float = self._clean_price(row["Gross Charge"])

                result: CPTSearchResult = {
                    "cpt_code": str(row["Procedure Code (CPT / HCPCS)"]),
                    "procedure_name": procedure_name,
                    "gross_charge": gross_charge,
                    "setting": str(row["Hospital Inpatient / Outpatient / Both"]),
                    "keyword_score": combined_score,
                    "fuzzy_ratio": fuzzy_ratio,
                    "partial_ratio": partial_ratio,
                    "token_sort_ratio": token_sort_ratio,
                    "token_set_ratio": token_set_ratio,
                    "contains_match": contains_match,
                    "word_overlap": word_overlap,
                }
                results.append(result)

        # Sort by combined score (descending)
        results.sort(key=lambda x: x.get("keyword_score", 0), reverse=True)

        return results[:top_k]

    def generate_expanded_keywords(self, query: str) -> list[str]:
        """Use LLM to generate expanded keywords and synonyms for the original query.

        Args:
            query: The original search term

        Returns:
            List of expanded keywords and synonyms
        """
        print(f"Generating expanded keywords for: '{query}'")

        prompt: str = f"""You are a medical terminology expert. Given a medical procedure or condition query, generate a list of related medical terms, synonyms, abbreviations, and alternative phrasings that a healthcare professional might use.

Original query: "{query}"

Please provide 10-15 related medical terms, one per line. Include:
- Medical synonyms and alternative names
- Common abbreviations
- Related procedures or tests
- Alternative phrasings (e.g., "chest x-ray" -> "chest radiograph", "thoracic imaging")
- Specific variations (e.g., "blood test" -> "complete blood count", "CBC", "hematology panel")

Only return the terms, one per line, no explanations or numbering:"""

        try:
            response = create_chat_completion(
                model=self._config.costing_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical terminology expert that provides concise lists of related medical terms.",
                    },
                    {"role": "user", "content": prompt},
                ],
                debug=self._config.debug,
                **self._config.costing_model_args,
            )

            keywords_text: str = (
                response.choices[0].message.content.strip()
                if response.choices[0].message.content
                else ""
            )
            keywords: list[str] = [kw.strip() for kw in keywords_text.split("\n") if kw.strip()]

            # Remove the original query if it appears in the list
            keywords = [kw for kw in keywords if kw.lower() != query.lower()]

            print(
                f"Generated {len(keywords)} expanded keywords: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}"
            )
            return keywords

        except Exception as e:
            print(f"Error generating expanded keywords: {e}")
            return []

    def llm_expanded_search(self, query: str, top_k: int = 60) -> list[CPTSearchResult]:
        """Perform search using LLM-generated expanded keywords.
        Optimized to batch all embedding API calls together.

        Args:
            query: The original procedure description
            top_k: Number of top results to return

        Returns:
            List of dictionaries containing search results from expanded keywords
        """
        # Generate expanded keywords
        expanded_keywords: list[str] = self.generate_expanded_keywords(query)

        if not expanded_keywords:
            print("No expanded keywords generated, skipping LLM expanded search")
            return []

        print(f"Fetching embeddings for {len(expanded_keywords)} keywords in one batch...")
        batch_embeddings = self._get_embeddings(expanded_keywords)

        if batch_embeddings is None or len(batch_embeddings) != len(expanded_keywords):
            print("Failed to get embeddings for expanded keywords, falling back to sequential")
            # Fallback to original sequential approach if batching fails
            batch_embeddings = []
            for keyword in expanded_keywords:
                emb = self._get_embeddings([keyword])
                if emb:
                    batch_embeddings.append(emb[0])
                else:
                    batch_embeddings.append(None)

        # Create mapping of keyword -> embedding
        keyword_embeddings: dict[str, list[float] | None] = {
            keyword: batch_embeddings[i] if i < len(batch_embeddings) else None
            for i, keyword in enumerate(expanded_keywords)
        }

        # Filter out keywords without embeddings
        valid_keywords = [kw for kw in expanded_keywords if keyword_embeddings.get(kw) is not None]
        valid_embeddings = [
            keyword_embeddings[kw] for kw in valid_keywords if keyword_embeddings[kw] is not None
        ]

        all_results: list[CPTSearchResult] = []
        results_per_keyword: int = max(
            1, top_k // len(expanded_keywords)
        )  # Distribute results across keywords

        if valid_embeddings:
            # Convert embeddings to numpy array for batch processing
            embeddings_array = np.array(valid_embeddings, dtype=np.float32)

            # Normalize all embeddings at once for cosine similarity
            faiss.normalize_L2(embeddings_array)  # pyright: ignore[reportUnknownMemberType]

            # Batch FAISS search for all keywords at once
            similarities, indices = self._faiss_index.search(embeddings_array, results_per_keyword)

            # Process results for each keyword
            for keyword_idx, keyword in enumerate(valid_keywords):
                for result_idx in range(results_per_keyword):
                    cpt_idx = indices[keyword_idx][result_idx]
                    similarity_score = float(similarities[keyword_idx][result_idx])

                    # Skip if similarity is too low
                    if similarity_score <= 0:
                        continue

                    row = self._df.iloc[cpt_idx]

                    result: CPTSearchResult = {
                        "similarity_score": similarity_score,
                        "cpt_code": str(row["Procedure Code (CPT / HCPCS)"]),
                        "procedure_name": row["Procedure Name"],
                        "gross_charge": self._clean_price(row["Gross Charge"]),
                        "setting": row["Hospital Inpatient / Outpatient / Both"],
                        "expanded_keyword": keyword,
                        "search_type": "llm_semantic",
                    }
                    all_results.append(result)

        # Set quiet mode to reduce verbose output during keyword search
        self._quiet_search = True

        for keyword in expanded_keywords:
            try:
                keyword_results: list[CPTSearchResult] = self.keyword_search(
                    keyword, top_k=results_per_keyword
                )
                for result in keyword_results:
                    result["expanded_keyword"] = keyword
                    result["search_type"] = "llm_keyword"
                    # Convert keyword_score to similarity_score for consistency
                    result["similarity_score"] = (result.get("keyword_score", 0)) / 100.0
                    all_results.append(result)
            except Exception as e:
                print(f"Error in keyword search for '{keyword}': {e}")
                continue

        # Remove quiet mode
        if hasattr(self, "_quiet_search"):
            delattr(self, "_quiet_search")

        # Remove duplicates based on CPT code and sort by score
        seen_codes: set[str] = set()
        unique_results: list[CPTSearchResult] = []

        # Sort all results by their scores first
        all_results.sort(
            key=lambda x: (x.get("similarity_score") or 0)
            if x.get("similarity_score") is not None
            else ((x.get("keyword_score") or 0) / 100.0),
            reverse=True,
        )

        for result in all_results:
            cpt_code: str = result["cpt_code"]
            if cpt_code not in seen_codes:
                unique_results.append(result)
                seen_codes.add(cpt_code)

        print(
            f"LLM expanded search found {len(unique_results)} unique results from {len(all_results)} total results"
        )
        return unique_results[:top_k]

    def combined_search(
        self, query: str, semantic_k: int = 20, keyword_k: int = 20, llm_expanded_k: int = 60
    ) -> list[CPTSearchResult]:
        """Perform semantic, keyword, and LLM-expanded search, combining results.

        Args:
            query: The procedure description to search for
            semantic_k: Number of results from direct semantic search
            keyword_k: Number of results from direct keyword search
            llm_expanded_k: Number of results from LLM-expanded keyword search

        Returns:
            List of dictionaries containing combined results with duplicates removed
        """
        print(
            f"Performing combined search for: '{query}' (semantic: {semantic_k}, keyword: {keyword_k}, llm-expanded: {llm_expanded_k})"
        )

        # Get direct semantic search results
        semantic_results: list[CPTSearchResult] = self.embedding_search(query, top_k=semantic_k)

        # Get direct keyword search results
        keyword_results: list[CPTSearchResult] = self.keyword_search(query, top_k=keyword_k)

        # Get LLM-expanded search results
        llm_expanded_results: list[CPTSearchResult] = self.llm_expanded_search(
            query, top_k=llm_expanded_k
        )

        # Combine results and remove duplicates based on CPT code
        seen_codes: set[str] = set()
        combined_results: list[CPTSearchResult] = []

        # Add direct semantic results first (these are typically highest quality)
        for result in semantic_results:
            cpt_code: str = result["cpt_code"]
            if cpt_code not in seen_codes:
                result["search_type"] = "semantic"
                combined_results.append(result)
                seen_codes.add(cpt_code)

        # Add direct keyword results that aren't already included
        for result in keyword_results:
            cpt_code = result["cpt_code"]
            if cpt_code not in seen_codes:
                result["search_type"] = "keyword"
                # Convert keyword_score to similarity_score format for consistency
                result["similarity_score"] = result.get("keyword_score", 0) / 100.0
                combined_results.append(result)
                seen_codes.add(cpt_code)

        # Add LLM-expanded results that aren't already included
        for result in llm_expanded_results:
            cpt_code = result["cpt_code"]
            if cpt_code not in seen_codes:
                combined_results.append(result)
                seen_codes.add(cpt_code)

        total_before_dedup: int = (
            len(semantic_results) + len(keyword_results) + len(llm_expanded_results)
        )
        duplicates_removed: int = total_before_dedup - len(combined_results)

        print(f"Combined search returned {len(combined_results)} unique results")
        print(f"  - Direct semantic: {len(semantic_results)}")
        print(f"  - Direct keyword: {len(keyword_results)}")
        print(f"  - LLM-expanded: {len(llm_expanded_results)}")
        print(f"  - Duplicates removed: {duplicates_removed}")

        # Sort all results by their best available score for better ranking
        def get_best_score(result: CPTSearchResult) -> float:
            if "similarity_score" in result:
                return result["similarity_score"]
            elif "keyword_score" in result:
                return result["keyword_score"] / 100.0  # Normalize to 0-1 range
            else:
                return 0.0

        combined_results.sort(key=get_best_score, reverse=True)

        return combined_results

    def select_best_cpt(self, query: str, results: list[CPTSearchResult]) -> CPTSelectionResult:
        """Use LLM to select the single best CPT code from all search results.

        Args:
            query: The original search query
            results: List of all search results

        Returns:
            A CPTSelectionResult for the selected CPTs
        """
        if not results:
            return {"error": "No results to choose from"}

        print(f"🤖 Using LLM to select best CPT code(s) from {len(results)} candidates...")

        # Format the results for the LLM
        candidates_text: str = ""
        for i, result in enumerate(results[:50], 1):  # Limit to top 50 to avoid token limits
            score_info: str = ""
            if "similarity_score" in result:
                score_info += f"Similarity: {result['similarity_score']:.3f}"
            if "keyword_score" in result:
                if score_info:
                    score_info += f", Keyword: {result['keyword_score']:.1f}"
                else:
                    score_info += f"Keyword: {result['keyword_score']:.1f}"

            search_type: str = result.get("search_type", "unknown")
            expanded_info: str = ""
            if "expanded_keyword" in result:
                expanded_info = f" (expanded from: '{result['expanded_keyword']}')"

            candidates_text += f"{i}. CPT: {result['cpt_code']}\n"
            candidates_text += f"   Procedure: {result['procedure_name']}\n"
            candidates_text += f"   Price: ${result['gross_charge']:,.2f}\n"
            candidates_text += f"   Setting: {result['setting']}\n"
            candidates_text += f"   Search: {search_type}{expanded_info}\n"
            if score_info:
                candidates_text += f"   Scores: {score_info}\n"
            candidates_text += "\n"

        prompt: str = f"""You are an expert medical coder specializing in CPT (Current Procedural Terminology) codes. Your task is to select the most appropriate CPT code(s) for a given medical procedure query.

Original Query: "{query}"

Below are {min(len(results), 50)} candidate CPT codes found through semantic search, keyword matching, and LLM-expanded terminology search:

{candidates_text}

Instructions:
1. Analyze the original query and determine what specific medical procedure(s) or service(s) are being requested
2. Review all candidate CPT codes and their descriptions
3. PRIORITIZE comprehensive panels and bundled procedures over individual component tests
4. Select the CPT code(s) that best match the original query using this hierarchy:
   a) First look for comprehensive panels that include multiple components (e.g., "renal function panel" vs separate electrolyte + urea tests)
   b) Then consider single comprehensive procedures
   c) Only select multiple individual codes if no comprehensive option exists. If a code should be used multiple times, then select the code multiple times.
   d) Don't suggest multiple codes if they would not typically be billed together.
5. Consider factors like:
   - Exact procedural match
   - Appropriate level of specificity
   - Standard medical coding practices
   - Whether multiple procedures/services are truly separate and unbundled

Common medical terminology:
- "U&E" = Urea and Electrolytes = typically covered by renal function panel
- "LFTs" = Liver Function Tests = hepatic function panel
- "CBC" = Complete Blood Count = single comprehensive test
- "CMP" = Comprehensive Metabolic Panel = single comprehensive test

Examples of when to prefer SINGLE comprehensive codes:
- "U&E" or "urea and electrolytes" → renal function panel (NOT separate electrolyte + urea codes)
- "liver function tests" → hepatic function panel (NOT separate ALT + AST + bilirubin codes)
- "blood work" → comprehensive metabolic panel or CBC (NOT individual glucose + sodium + potassium codes)
- "nerve conduction studies" → if the query does not specify the number of nerves tested, then select a reasonable median number of nerves tested (e.g. 6 nerves)

Examples of when to select MULTIPLE codes (truly separate procedures):
- "liver biopsy with immunohistochemistry" → biopsy code + immunohistochemistry code
- "colonoscopy with biopsy" → colonoscopy code + biopsy code
- "surgery with pathology" → surgical code + pathology code

Please respond in this exact format:
REASONING: [Brief explanation of why these CPT code(s) are the best match]
SELECTED_CPT: [CPT code 1]
SELECTED_CPT: [CPT code 2] (if multiple codes needed)
SELECTED_CPT: [CPT code 3] (if additional codes needed)

Example response format for single code:
REASONING: This code specifically covers "RADIOLOGIC EXAM, CHEST; 2 VIEWS" which is the standard chest x-ray procedure typically performed in clinical practice.
SELECTED_CPT: 71046

Example response format for comprehensive panel (preferred):
REASONING: "U&E" refers to urea and electrolytes testing. Rather than selecting separate codes for electrolyte panel and urea nitrogen, the renal function panel (80069) comprehensively covers both components in a single, more appropriate code.
SELECTED_CPT: 80069

Example response format for multiple codes (only when truly separate procedures):
REASONING: This query requires two separate codes because it involves two distinct procedures: 47000 for the liver biopsy procedure itself, and 88342 for the immunohistochemistry staining performed on the specimen. No single comprehensive code covers both procedures.
SELECTED_CPT: 47000
SELECTED_CPT: 88342
"""

        try:
            response = create_chat_completion(
                model=self._config.costing_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert medical coder who selects the most appropriate CPT codes for medical procedures.",
                    },
                    {"role": "user", "content": prompt},
                ],
                debug=self._config.debug,
                **self._config.costing_model_args,
            )

            response_text: str = (
                response.choices[0].message.content.strip()
                if response.choices[0].message.content
                else ""
            )

            # Parse the response for multiple CPT codes
            selected_cpts: list[str] = []
            reasoning: str = ""

            lines: list[str] = response_text.split("\n")
            for line in lines:
                if line.startswith("SELECTED_CPT:"):
                    cpt: str = line.replace("SELECTED_CPT:", "").strip()
                    if cpt:
                        selected_cpts.append(cpt)
                elif line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()

            if not selected_cpts:
                return {
                    "error": f"No CPTs selected by LLM from {len(results)} candidates",
                    "selected_results": [],
                    "selected_cpts": [],
                    "reasoning": reasoning,
                    "missing_cpts": None,
                    "raw_response": response_text,
                }

            # Find the full details of the selected CPT codes
            selected_results: list[CPTSearchResult] = []
            missing_cpts: list[str] = []

            for selected_cpt in selected_cpts:
                found: list[CPTSearchResult] = []
                for result in results:
                    if result["cpt_code"] == selected_cpt:
                        found.append(result)
                if not found:
                    # Try to find the CPT in the charges table directly
                    matching_rows = self._df[
                        self._df["Procedure Code (CPT / HCPCS)"] == selected_cpt
                    ]
                    if not matching_rows.empty:
                        # Use the first row found
                        row = matching_rows.iloc[0]
                        sourced_result: CPTSearchResult = {
                            "cpt_code": str(row["Procedure Code (CPT / HCPCS)"]),
                            "procedure_name": row["Procedure Name"],
                            "gross_charge": self._clean_price(row["Gross Charge"]),
                            "setting": row["Hospital Inpatient / Outpatient / Both"],
                            "search_type": "llm_sourced",
                        }
                        selected_results.append(sourced_result)
                    else:
                        missing_cpts.append(selected_cpt)
                elif len(found) == 1:
                    selected_results.append(found[0])
                else:
                    # Note: We don't actually get to this path as combined search already deduplicates,
                    # but that may change in the future.
                    selected_results.append(_merge_results(found))

            if missing_cpts:
                return {
                    "error": f"LLM selected CPTs {missing_cpts} not found in charge data",
                    "selected_results": selected_results,
                    "selected_cpts": selected_cpts,
                    "reasoning": reasoning,
                    "missing_cpts": missing_cpts,
                    "raw_response": response_text,
                }

            return {
                "selected_results": selected_results,
                "selected_cpts": selected_cpts,
                "reasoning": reasoning,
                "missing_cpts": None,
                "raw_response": response_text,
            }

        except Exception as e:
            return {
                "error": f"Error in LLM selection: {e}",
            }

    def print_selection(self, selection: CPTSelectionResult, query: str = "") -> None:
        """Pretty print the LLM selection result."""
        if query:
            print(f"\n🎯 Best CPT Selection for: '{query}'")
        print("=" * 80)

        if "error" in selection:
            print(f"❌ Error: {selection['error']}")
            return

        # Handle multiple CPT codes
        if "selected_results" in selection:
            selected_results = selection["selected_results"]
            print(f"🏆 Selected CPT Code(s): {len(selected_results)} code(s)")

            total_charge: float = 0
            for i, result in enumerate(selected_results, 1):
                print(f"\n{i}. CPT Code: {result['cpt_code']}")
                print(f"   📋 Procedure: {result['procedure_name']}")
                print(f"   💰 Gross Charge: ${result['gross_charge']:,.2f}")
                print(f"   🏥 Setting: {result['setting']}")

                if result.get("search_type"):
                    print(f"   🔍 Found via: {result.get('search_type')}")
                if result.get("expanded_keyword"):
                    print(f"   🤖 Expanded from: '{result.get('expanded_keyword')}'")

                # Show scores if available
                scores: list[str] = []
                if result.get("similarity_score"):
                    scores.append(f"Similarity: {result.get('similarity_score'):.3f}")
                if result.get("keyword_score"):
                    scores.append(f"Keyword: {result.get('keyword_score'):.1f}")
                if scores:
                    print(f"   📊 Scores: {', '.join(scores)}")

                total_charge += result["gross_charge"]

            print(f"\n💰 Total Combined Charge: ${total_charge:,.2f}")

            if "missing_cpts" in selection:
                print(
                    f"\n⚠️  Warning: Some selected CPTs not found in results: {selection['missing_cpts']}"
                )

        print(f"\n🧠 Reasoning: {selection.get('reasoning', 'No reasoning provided')}")
        print("=" * 80)

    def print_results(self, results: list[CPTSearchResult], query: str = "") -> None:
        """Pretty print the search results."""
        if query:
            print(f"\n🔍 Search Results for: '{query}'")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            print(f"\n{i}. CPT Code: {result['cpt_code']}")
            print(f"   Procedure: {result['procedure_name']}")

            # Handle different score types
            if "similarity_score" in result:
                print(f"   Similarity: {result['similarity_score']:.3f}")
            if "keyword_score" in result:
                print(f"   Keyword Score: {result['keyword_score']:.1f}")
            if "search_type" in result:
                print(f"   Search Type: {result['search_type']}")
            if "expanded_keyword" in result:
                print(f"   Expanded From: '{result['expanded_keyword']}'")

            print(f"   Gross Charge: ${result['gross_charge']:,.2f}")
            print(f"   Setting: {result['setting']}")
            print("-" * 60)

    def get_index_stats(self) -> dict[str, Any]:
        """Get statistics about the FAISS index and dataset."""
        return {
            "total_procedures": len(self._df),
            "faiss_index_size": self._faiss_index.ntotal,
            "faiss_index_type": type(self._faiss_index).__name__,
        }

    def find_best_cpt(
        self,
        query: str,
        search_params: dict[str, Any] | None = None,
        return_candidates: bool = False,
    ) -> CPTSelectionResult:
        """High-level method to find and select the best CPT code for a query.
        This is the recommended entry point for library usage.

        Args:
            query: The medical procedure description to search for
            search_params: Optional dict with search configuration:
                - semantic_k: Number of semantic search results (default: 20)
                - keyword_k: Number of keyword search results (default: 20)
                - llm_expanded_k: Number of LLM-expanded search results (default: 30)
                - search_method: 'combined' (default), 'semantic', 'keyword'
            return_candidates: If True, include all candidate results in response

        Returns:
            A CPTSelectionResult containing the selection results
        """
        # Set default search parameters
        # Can be moved into config if we need to adjust
        default_params: dict[str, Any] = {
            "semantic_k": 20,
            "keyword_k": 20,
            "llm_expanded_k": 30,
            "search_method": "combined",
        }

        if search_params:
            default_params.update(search_params)

        try:
            # Perform the search based on method
            results: list[CPTSearchResult]
            if default_params["search_method"] == "semantic":
                results = self.embedding_search(query, top_k=default_params["semantic_k"])
            elif default_params["search_method"] == "keyword":
                results = self.keyword_search(query, top_k=default_params["keyword_k"])
            else:  # combined (default)
                results = self.combined_search(
                    query,
                    semantic_k=default_params["semantic_k"],
                    keyword_k=default_params["keyword_k"],
                    llm_expanded_k=default_params["llm_expanded_k"],
                )

            if not results:
                return {
                    "error": f"No CPT codes found for query: '{query}'",
                    "query": query,
                    "search_params": default_params,
                }

            # Use LLM to select the best CPT code(s)
            selection: CPTSelectionResult = self.select_best_cpt(query, results)

            # Add query and search info
            selection["query"] = query
            selection["search_params"] = default_params
            selection["num_candidates"] = len(results)

            # Optionally include all candidates
            if return_candidates:
                selection["candidates"] = results

            return selection

        except Exception as e:
            return {
                "error": f"Error processing query '{query}': {e!s}",
                "query": query,
                "search_params": default_params,
            }

    def cost_test(self, test_name: str) -> "CostMethodEstimate":
        start_time = time.time()

        cpt_selection = self.find_best_cpt(test_name)
        cpts = cpt_selection.get("selected_cpts", [])

        elapsed_time = time.time() - start_time

        if "error" in cpt_selection:
            return {
                "error": f"Error finding best CPTs: {cpt_selection['error']}",
                "cpt_codes": cpts,
                "cpt_selection": cpt_selection,
                "time": elapsed_time,
            }
        if not cpts:
            return {
                "error": "No CPTs found",
                "cpt_codes": cpts,
                "cpt_selection": cpt_selection,
                "time": elapsed_time,
            }
        # Now, get the costs for each CPT, taking a median if multiple entries exist in the charges table
        grounded_costs = [self.median_price(code) for code in cpts]
        if any(cost is None for cost in grounded_costs):
            # Don't think this can actually happen as all CPTs come from the dataset
            return {
                "error": f"Some selected CPTs have no cost data: {[code for code, cost in zip(cpts, grounded_costs) if cost is None]}",
                "cpt_codes": cpts,
                "cpt_selection": cpt_selection,
                "time": elapsed_time,
            }
        return {
            "cost": sum(cast(list[float], grounded_costs)),
            "cpt_codes": cpts,
            "cpt_selection": cpt_selection,
            "time": elapsed_time,
        }
