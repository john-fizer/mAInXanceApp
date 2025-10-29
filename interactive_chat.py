"""
Interactive RAG-Based Chat Interface for Maintenance Assistant
==============================================================

This module provides an interactive chat interface that uses:
- Retrieval-Augmented Generation (RAG) approach
- Historical work order similarity search
- ML model predictions for action clusters
- Real-time recommendations based on historical patterns

Author: Enhanced by Claude Code
Date: 2025-10-29
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import sys

# Import SimilaritySearchEngine from the ml_workflow module
try:
    from ml_workflow_enhanced import SimilaritySearchEngine
except ImportError:
    # If import fails, we'll define a placeholder
    SimilaritySearchEngine = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MaintenanceAssistantChat:
    """
    Interactive chat assistant for maintenance technicians.

    Combines ML predictions with RAG-style retrieval to provide
    intelligent recommendations for work orders.
    """

    def __init__(self, model_path: str = 'prediction_model.pkl',
                 search_engine_path: str = 'similarity_search_engine.pkl',
                 data_path: str = 'processed_work_orders.csv'):
        """
        Initialize the chat assistant.

        Args:
            model_path: Path to saved prediction model
            search_engine_path: Path to saved similarity search engine
            data_path: Path to processed work orders data
        """
        logger.info("Initializing Maintenance Assistant Chat...")

        # Load model and search engine
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Loaded prediction model from {model_path}")
        except FileNotFoundError:
            logger.warning(f"Model file not found: {model_path}")
            self.model = None

        try:
            self.search_engine = joblib.load(search_engine_path)
            logger.info(f"Loaded search engine from {search_engine_path}")
        except FileNotFoundError:
            logger.warning(f"Search engine file not found: {search_engine_path}")
            self.search_engine = None

        try:
            self.df = pd.read_csv(data_path)
            logger.info(f"Loaded data with {len(self.df)} work orders")
        except FileNotFoundError:
            logger.warning(f"Data file not found: {data_path}")
            self.df = None

        # Cluster descriptions (based on analysis)
        self.cluster_descriptions = {
            0: "Routine Maintenance & Adjustments",
            1: "Component Replacement (Fuses, Belts, Hoses)",
            2: "Task Completion Logging",
            3: "Repairs (Broken Parts, Wiring)",
            4: "Advanced Repairs (Electrical, Plumbing, Air Systems)"
        }

        # Conversation history
        self.conversation_history = []

    def predict_action_cluster(self, description: str) -> Tuple[int, np.ndarray, str]:
        """
        Predict the most likely action cluster for a description.

        Args:
            description: Work order description

        Returns:
            Tuple of (predicted_cluster, probabilities, cluster_description)
        """
        if self.model is None:
            logger.error("Prediction model not loaded")
            return -1, np.array([]), "Model not available"

        # Predict cluster
        predicted_cluster = self.model.predict([description])[0]

        # Get probabilities
        probabilities = self.model.predict_proba([description])[0]

        # Get cluster description
        cluster_desc = self.cluster_descriptions.get(
            predicted_cluster,
            f"Unknown Cluster {predicted_cluster}"
        )

        return predicted_cluster, probabilities, cluster_desc

    def search_similar_cases(self, description: str, top_n: int = 5) -> pd.DataFrame:
        """
        Search for similar historical work orders.

        Args:
            description: Work order description
            top_n: Number of results to return

        Returns:
            pd.DataFrame: Similar work orders
        """
        if self.search_engine is None:
            logger.error("Search engine not loaded")
            return pd.DataFrame()

        return self.search_engine.search(description, top_n=top_n)

    def get_cluster_statistics(self, cluster_id: int) -> Dict:
        """
        Get statistics about a specific cluster.

        Args:
            cluster_id: Cluster ID

        Returns:
            dict: Cluster statistics
        """
        if self.df is None:
            return {}

        cluster_data = self.df[self.df['Note_Cluster'] == cluster_id]

        return {
            'total_cases': len(cluster_data),
            'percentage': len(cluster_data) / len(self.df) * 100,
            'avg_text_length': cluster_data['Text_cleaned'].str.split().str.len().mean(),
            'sample_solutions': cluster_data['Text'].sample(min(3, len(cluster_data))).tolist()
        }

    def format_recommendation(self, description: str,
                            predicted_cluster: int,
                            probabilities: np.ndarray,
                            cluster_desc: str,
                            similar_cases: pd.DataFrame) -> str:
        """
        Format a comprehensive recommendation response.

        Args:
            description: Original query description
            predicted_cluster: Predicted cluster ID
            probabilities: Prediction probabilities
            cluster_desc: Cluster description
            similar_cases: Similar historical cases

        Returns:
            str: Formatted recommendation
        """
        response = []

        response.append("="*80)
        response.append("MAINTENANCE RECOMMENDATION")
        response.append("="*80)

        # Query
        response.append(f"\nYour Query: {description}")

        # Prediction
        response.append(f"\nPredicted Action Category: Cluster {predicted_cluster}")
        response.append(f"Category Description: {cluster_desc}")

        # Confidence
        confidence = probabilities[predicted_cluster] * 100
        response.append(f"Confidence: {confidence:.2f}%")

        # Alternative clusters
        response.append("\nAlternative Possibilities:")
        sorted_clusters = np.argsort(probabilities)[::-1][1:4]  # Top 3 alternatives
        for rank, cluster in enumerate(sorted_clusters, 1):
            prob = probabilities[cluster] * 100
            desc = self.cluster_descriptions.get(cluster, f"Cluster {cluster}")
            response.append(f"  {rank}. {desc} ({prob:.2f}%)")

        # Cluster statistics
        stats = self.get_cluster_statistics(predicted_cluster)
        if stats:
            response.append(f"\nCluster Statistics:")
            response.append(f"  - Historical cases in this category: {stats['total_cases']}")
            response.append(f"  - Percentage of all cases: {stats['percentage']:.2f}%")

        # Similar cases
        response.append("\n" + "-"*80)
        response.append("SIMILAR HISTORICAL CASES (Top 5)")
        response.append("-"*80)

        if not similar_cases.empty:
            for idx, (_, case) in enumerate(similar_cases.iterrows(), 1):
                response.append(f"\n{idx}. Work Order: {case['WO No.']}")
                response.append(f"   Similarity: {case['similarity_score']*100:.2f}%")
                response.append(f"   Problem: {case['Description']}")
                response.append(f"   Solution: {case['Text']}")
        else:
            response.append("No similar cases found.")

        # Recommendations
        response.append("\n" + "-"*80)
        response.append("RECOMMENDED ACTIONS")
        response.append("-"*80)

        if not similar_cases.empty:
            response.append("\nBased on similar historical cases, consider:")
            # Extract common actions from top 3 similar cases
            top_solutions = similar_cases.head(3)['Text'].tolist()
            response.append("\n1. Review the solutions from the top similar cases above")
            response.append("2. Check if the issue matches one of the common patterns:")

            for i, solution in enumerate(top_solutions, 1):
                response.append(f"   {chr(96+i)}) {solution[:100]}...")

            response.append("\n3. Document your findings and actions taken")
            response.append("4. If issue persists, escalate to senior technician")
        else:
            response.append("\n1. This appears to be a unique case")
            response.append("2. Consult with senior technician before proceeding")
            response.append("3. Document all diagnostic steps thoroughly")

        response.append("\n" + "="*80)

        return "\n".join(response)

    def process_query(self, description: str, verbose: bool = True) -> Dict:
        """
        Process a user query and generate recommendations.

        Args:
            description: Work order description
            verbose: Whether to print formatted response

        Returns:
            dict: Complete response data
        """
        # Predict cluster
        predicted_cluster, probabilities, cluster_desc = self.predict_action_cluster(description)

        # Search similar cases
        similar_cases = self.search_similar_cases(description, top_n=5)

        # Format response
        formatted_response = self.format_recommendation(
            description,
            predicted_cluster,
            probabilities,
            cluster_desc,
            similar_cases
        )

        # Store in conversation history
        conversation_entry = {
            'timestamp': datetime.now(),
            'query': description,
            'predicted_cluster': predicted_cluster,
            'cluster_description': cluster_desc,
            'confidence': probabilities[predicted_cluster],
            'similar_cases_count': len(similar_cases)
        }
        self.conversation_history.append(conversation_entry)

        if verbose:
            print(formatted_response)

        return {
            'query': description,
            'predicted_cluster': predicted_cluster,
            'cluster_description': cluster_desc,
            'probabilities': probabilities,
            'similar_cases': similar_cases,
            'formatted_response': formatted_response
        }

    def show_conversation_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            print("No conversation history yet.")
            return

        print("\n" + "="*80)
        print("CONVERSATION HISTORY")
        print("="*80)

        for i, entry in enumerate(self.conversation_history, 1):
            print(f"\n{i}. [{entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}]")
            print(f"   Query: {entry['query']}")
            print(f"   Prediction: {entry['cluster_description']}")
            print(f"   Confidence: {entry['confidence']*100:.2f}%")
            print(f"   Similar cases found: {entry['similar_cases_count']}")

    def interactive_mode(self):
        """
        Start interactive chat mode.

        Allows users to continuously enter queries and receive recommendations.
        """
        print("\n" + "="*80)
        print("MAINTENANCE ASSISTANT - INTERACTIVE CHAT MODE")
        print("="*80)
        print("\nWelcome! I'm your AI maintenance assistant.")
        print("Describe your work order problem, and I'll provide recommendations")
        print("based on historical patterns and similar cases.")
        print("\nCommands:")
        print("  - Type 'history' to see conversation history")
        print("  - Type 'help' for assistance")
        print("  - Type 'quit' or 'exit' to end session")
        print("="*80)

        while True:
            try:
                # Get user input
                query = input("\n[You] Describe the problem: ").strip()

                # Handle commands
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for using Maintenance Assistant. Goodbye!")
                    break

                elif query.lower() == 'history':
                    self.show_conversation_history()
                    continue

                elif query.lower() == 'help':
                    self._show_help()
                    continue

                elif not query:
                    print("Please enter a problem description.")
                    continue

                # Process query
                print("\n[Assistant] Analyzing your request...")
                self.process_query(query, verbose=True)

            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break

            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                print(f"\nError: {str(e)}")
                print("Please try again with a different query.")

    def _show_help(self):
        """Display help information."""
        print("\n" + "="*80)
        print("HELP - HOW TO USE THE MAINTENANCE ASSISTANT")
        print("="*80)
        print("\n1. Describe your problem clearly and concisely")
        print("   Example: 'conveyor belt not moving'")
        print("   Example: 'motor won't start after power outage'")
        print("\n2. The assistant will:")
        print("   - Predict the most likely action category")
        print("   - Find similar historical cases")
        print("   - Provide recommended actions")
        print("\n3. Review the similar cases carefully - they often contain")
        print("   the exact solution to your problem!")
        print("\n4. Available cluster categories:")
        for cluster_id, desc in self.cluster_descriptions.items():
            print(f"   - Cluster {cluster_id}: {desc}")
        print("\n" + "="*80)

    def batch_process(self, queries: List[str]) -> List[Dict]:
        """
        Process multiple queries in batch mode.

        Args:
            queries: List of problem descriptions

        Returns:
            List of response dictionaries
        """
        logger.info(f"Processing {len(queries)} queries in batch mode...")

        results = []
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}: {query}")
            result = self.process_query(query, verbose=False)
            results.append(result)

        logger.info("Batch processing completed")
        return results


def demo_queries():
    """Return a list of demo queries for testing."""
    return [
        "conveyor not working",
        "machine won't turn on",
        "leak in air line",
        "motor making strange noise",
        "control panel not responding",
        "sensor reading incorrect values",
        "hydraulic system pressure low",
        "emergency stop button stuck",
        "belt slipping on pulley",
        "power supply failure"
    ]


def main():
    """Main function to run the interactive chat."""
    print("\n" + "="*80)
    print("INITIALIZING MAINTENANCE ASSISTANT CHAT SYSTEM")
    print("="*80)

    # Check if model files exist
    import os

    if not os.path.exists('prediction_model.pkl'):
        print("\nERROR: Prediction model not found!")
        print("Please run 'python ml_workflow_enhanced.py' first to train the model.")
        return

    if not os.path.exists('similarity_search_engine.pkl'):
        print("\nERROR: Similarity search engine not found!")
        print("Please run 'python ml_workflow_enhanced.py' first to create the search engine.")
        return

    # Initialize chat assistant
    assistant = MaintenanceAssistantChat()

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'demo':
            # Demo mode
            print("\n" + "="*80)
            print("DEMO MODE - TESTING WITH SAMPLE QUERIES")
            print("="*80)

            queries = demo_queries()
            print(f"\nProcessing {len(queries)} demo queries...\n")

            for i, query in enumerate(queries[:5], 1):  # Show first 5
                print(f"\n{'='*80}")
                print(f"DEMO QUERY {i}/{min(5, len(queries))}")
                print(f"{'='*80}")
                assistant.process_query(query, verbose=True)
                input("\nPress Enter to continue to next demo...")

        elif sys.argv[1] == 'batch':
            # Batch mode with demo queries
            print("\n" + "="*80)
            print("BATCH MODE - PROCESSING ALL DEMO QUERIES")
            print("="*80)

            queries = demo_queries()
            results = assistant.batch_process(queries)

            # Print summary
            print("\n" + "="*80)
            print("BATCH PROCESSING SUMMARY")
            print("="*80)

            for i, result in enumerate(results, 1):
                print(f"\n{i}. Query: {result['query']}")
                print(f"   Prediction: {result['cluster_description']}")
                print(f"   Confidence: {result['probabilities'][result['predicted_cluster']]*100:.2f}%")
                print(f"   Similar cases: {len(result['similar_cases'])}")

        else:
            # Single query mode
            query = ' '.join(sys.argv[1:])
            assistant.process_query(query, verbose=True)
    else:
        # Interactive mode
        assistant.interactive_mode()


if __name__ == "__main__":
    main()
