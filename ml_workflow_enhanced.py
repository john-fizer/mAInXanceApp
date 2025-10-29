"""
Enhanced ML Workflow for Technician Action Prediction
=====================================================

This module implements a comprehensive machine learning pipeline for predicting
technician actions based on work order descriptions.

Key Features:
- Advanced data preprocessing with noise removal
- Unsupervised clustering of technician behaviors
- Supervised classification with cross-validation
- Class imbalance handling
- Comprehensive visualizations and metrics
- Production-ready error handling and logging

Author: Enhanced by Claude Code
Date: 2025-10-29
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from datetime import datetime
from typing import Tuple, List, Dict, Any

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    silhouette_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score
)
from sklearn.metrics.pairwise import cosine_similarity

# Configure warnings and logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class DataPreprocessor:
    """
    Handles data loading, cleaning, and preprocessing.

    This class removes noise, validates data quality, and filters
    technician names and filler words to improve feature quality.
    """

    # Common technician names to filter out (identified from cluster analysis)
    TECHNICIAN_NAMES = {
        'brad', 'fox', 'wfox', 'curt', 'chad', 'jeff', 'js', 'fred',
        'bradjeff', 'cliff', 'joel', 'md', 'payne', 'dale', 'bob', 'cd',
        'randy', 'cliff'
    }

    # Filler words that don't provide meaningful information
    FILLER_WORDS = {
        'completed', 'complete', 'done', 'finish', 'finished', 'ok',
        'okay', 'good', 'checked', 'check'
    }

    def __init__(self, file_path: str):
        """Initialize the preprocessor with data file path."""
        self.file_path = file_path
        self.df = None
        logger.info(f"Initializing DataPreprocessor with file: {file_path}")

    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file with proper encoding.

        Returns:
            pd.DataFrame: Raw loaded data
        """
        try:
            logger.info("Loading data from CSV...")
            self.df = pd.read_csv(
                self.file_path,
                encoding='ISO-8859-1',
                dtype=str,
                low_memory=False
            )
            logger.info(f"Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def clean_data(self) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning.

        Steps:
        1. Remove completely empty rows
        2. Validate work order numbers
        3. Clean and strip text fields
        4. Remove rows with missing critical fields
        5. Remove technician names and filler words

        Returns:
            pd.DataFrame: Cleaned data
        """
        logger.info("Starting data cleaning pipeline...")

        # Remove completely empty rows
        initial_count = len(self.df)
        self.df = self.df.dropna(how='all')
        logger.info(f"Removed {initial_count - len(self.df)} empty rows")

        # Validate work order numbers (must be numeric)
        self.df = self.df[self.df['WO No.'].astype(str).str.match(r'^\d+$')]

        # Clean and strip text fields
        self.df['WO No.'] = self.df['WO No.'].astype(str).str.strip()
        self.df['Description'] = self.df['Description'].astype(str).str.strip()
        self.df['Text'] = self.df['Text'].astype(str).str.strip()

        # Remove rows with missing descriptions and notes
        self.df = self.df[~((self.df['Description'].str.strip() == "") &
                           (self.df['Text'].str.strip() == ""))]

        # Remove rows with invalid technician notes
        self.df = self.df[
            self.df['Text'].notna() &
            (self.df['Text'].str.strip() != "") &
            (self.df['Text'].str.lower().str.strip() != "nan")
        ]

        # Create cleaned text columns
        self.df['Description_cleaned'] = self.df['Description'].apply(self._clean_text)
        self.df['Text_cleaned'] = self.df['Text'].apply(self._clean_text)

        logger.info(f"Data cleaning completed: {len(self.df)} rows remaining")
        logger.info(f"Data quality: {len(self.df) / initial_count * 100:.2f}% retained")

        return self.df

    def _clean_text(self, text: str) -> str:
        """
        Clean individual text field.

        - Convert to lowercase
        - Remove punctuation
        - Remove technician names
        - Remove filler words
        - Strip extra whitespace

        Args:
            text: Raw text string

        Returns:
            str: Cleaned text
        """
        # Convert to lowercase and remove punctuation
        text = str(text).lower()
        text = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in text)

        # Split into words
        words = text.split()

        # Remove technician names and filler words
        words = [
            word for word in words
            if word not in self.TECHNICIAN_NAMES and
               word not in self.FILLER_WORDS and
               len(word) > 2  # Remove very short words
        ]

        # Join back and strip
        return ' '.join(words).strip()

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics about the dataset.

        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_records': len(self.df),
            'unique_work_orders': self.df['WO No.'].nunique(),
            'avg_description_length': self.df['Description_cleaned'].str.split().str.len().mean(),
            'avg_text_length': self.df['Text_cleaned'].str.split().str.len().mean(),
            'missing_descriptions': (self.df['Description_cleaned'] == '').sum(),
            'missing_texts': (self.df['Text_cleaned'] == '').sum()
        }

        logger.info("Data Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")

        return summary


class ClusteringEngine:
    """
    Handles unsupervised clustering of technician notes.

    Uses K-Means clustering with TF-IDF vectorization to identify
    behavior-based patterns in technician actions.
    """

    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        """
        Initialize clustering engine.

        Args:
            n_clusters: Number of clusters for K-Means
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.tfidf = None
        self.kmeans = None
        self.X_notes = None
        logger.info(f"Initializing ClusteringEngine with {n_clusters} clusters")

    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        """
        Fit clustering model and transform texts into cluster labels.

        Args:
            texts: Series of cleaned text documents

        Returns:
            np.ndarray: Cluster labels for each document
        """
        logger.info("Fitting TF-IDF vectorizer...")
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.X_notes = self.tfidf.fit_transform(texts)

        logger.info(f"TF-IDF matrix shape: {self.X_notes.shape}")
        logger.info(f"Fitting K-Means with {self.n_clusters} clusters...")

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )

        labels = self.kmeans.fit_predict(self.X_notes)
        logger.info("Clustering completed successfully")

        return labels

    def get_top_terms_per_cluster(self, n_terms: int = 10) -> Dict[int, List[str]]:
        """
        Extract top terms for each cluster to interpret cluster meanings.

        Args:
            n_terms: Number of top terms to extract per cluster

        Returns:
            dict: Mapping of cluster_id to list of top terms
        """
        feature_names = np.array(self.tfidf.get_feature_names_out())
        cluster_centers = self.kmeans.cluster_centers_

        top_terms = {}
        for cluster_id in range(self.n_clusters):
            center = cluster_centers[cluster_id]
            top_indices = center.argsort()[::-1][:n_terms]
            top_terms[cluster_id] = feature_names[top_indices].tolist()

        return top_terms

    def calculate_silhouette_scores(self, k_range: range = range(2, 10),
                                    sample_size: int = 5000) -> Dict[int, float]:
        """
        Calculate silhouette scores for different numbers of clusters.

        Args:
            k_range: Range of k values to test
            sample_size: Sample size for faster computation

        Returns:
            dict: Mapping of k to silhouette score
        """
        logger.info(f"Calculating silhouette scores for k in {list(k_range)}")

        # Sample data if needed
        if self.X_notes.shape[0] > sample_size:
            sample_indices = np.random.choice(
                self.X_notes.shape[0],
                sample_size,
                replace=False
            )
            X_sample = self.X_notes[sample_indices]
        else:
            X_sample = self.X_notes

        scores = {}
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, n_init=10, max_iter=200, random_state=42)
            labels = kmeans_temp.fit_predict(X_sample)
            score = silhouette_score(X_sample, labels)
            scores[k] = score
            logger.info(f"  k={k}: silhouette_score={score:.4f}")

        return scores


class PredictionModel:
    """
    Supervised classification model for predicting technician action clusters.

    Uses TF-IDF + Logistic Regression pipeline with class balancing
    to handle imbalanced cluster distributions.
    """

    def __init__(self, class_weight: str = 'balanced', max_iter: int = 1000):
        """
        Initialize prediction model.

        Args:
            class_weight: Strategy for handling class imbalance
            max_iter: Maximum iterations for logistic regression
        """
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.pipeline = None
        self.feature_names = None
        logger.info(f"Initializing PredictionModel with class_weight='{class_weight}'")

    def build_pipeline(self) -> Pipeline:
        """
        Build sklearn pipeline with TF-IDF and Logistic Regression.

        Returns:
            Pipeline: Sklearn pipeline
        """
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
            ('clf', LogisticRegression(
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                random_state=42,
                solver='lbfgs'
            ))
        ])

        logger.info("Pipeline created successfully")
        return self.pipeline

    def train(self, X_train: pd.Series, y_train: pd.Series) -> 'PredictionModel':
        """
        Train the prediction model.

        Args:
            X_train: Training descriptions
            y_train: Training labels (cluster IDs)

        Returns:
            self: Trained model
        """
        logger.info(f"Training model on {len(X_train)} samples...")
        logger.info(f"Class distribution: {dict(pd.Series(y_train).value_counts())}")

        self.pipeline.fit(X_train, y_train)

        # Store feature names for later analysis
        self.feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()

        logger.info("Training completed successfully")
        return self

    def evaluate(self, X_test: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance on test set.

        Args:
            X_test: Test descriptions
            y_test: Test labels

        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"Evaluating model on {len(X_test)} test samples...")

        y_pred = self.pipeline.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        results = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_test': y_test
        }

        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 (weighted): {f1_weighted:.4f}")
        logger.info(f"  F1 (macro): {f1_macro:.4f}")

        return results

    def cross_validate(self, X: pd.Series, y: pd.Series,
                       cv: int = 5) -> Dict[str, Any]:
        """
        Perform stratified k-fold cross-validation.

        Args:
            X: Feature data (descriptions)
            y: Labels (cluster IDs)
            cv: Number of folds

        Returns:
            dict: Cross-validation results
        """
        logger.info(f"Performing {cv}-fold stratified cross-validation...")

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        # Calculate cross-validation scores
        cv_accuracy = cross_val_score(
            self.pipeline, X, y,
            cv=skf,
            scoring='accuracy',
            n_jobs=-1
        )

        cv_f1_weighted = cross_val_score(
            self.pipeline, X, y,
            cv=skf,
            scoring='f1_weighted',
            n_jobs=-1
        )

        cv_f1_macro = cross_val_score(
            self.pipeline, X, y,
            cv=skf,
            scoring='f1_macro',
            n_jobs=-1
        )

        results = {
            'cv_accuracy_mean': cv_accuracy.mean(),
            'cv_accuracy_std': cv_accuracy.std(),
            'cv_accuracy_scores': cv_accuracy,
            'cv_f1_weighted_mean': cv_f1_weighted.mean(),
            'cv_f1_weighted_std': cv_f1_weighted.std(),
            'cv_f1_weighted_scores': cv_f1_weighted,
            'cv_f1_macro_mean': cv_f1_macro.mean(),
            'cv_f1_macro_std': cv_f1_macro.std(),
            'cv_f1_macro_scores': cv_f1_macro
        }

        logger.info(f"  CV Accuracy: {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
        logger.info(f"  CV F1 (weighted): {results['cv_f1_weighted_mean']:.4f} ± {results['cv_f1_weighted_std']:.4f}")
        logger.info(f"  CV F1 (macro): {results['cv_f1_macro_mean']:.4f} ± {results['cv_f1_macro_std']:.4f}")

        return results

    def predict(self, description: str) -> int:
        """
        Predict cluster for a single description.

        Args:
            description: Work order description

        Returns:
            int: Predicted cluster ID
        """
        return self.pipeline.predict([description])[0]

    def predict_proba(self, description: str) -> np.ndarray:
        """
        Get prediction probabilities for all clusters.

        Args:
            description: Work order description

        Returns:
            np.ndarray: Probability for each cluster
        """
        return self.pipeline.predict_proba([description])[0]


class Visualizer:
    """
    Handles all visualization and plotting for the ML workflow.

    Creates publication-quality plots for model evaluation,
    cluster analysis, and performance monitoring.
    """

    @staticmethod
    def plot_cluster_distribution(cluster_labels: pd.Series,
                                  save_path: str = 'cluster_distribution.png'):
        """
        Plot distribution of samples across clusters.

        Args:
            cluster_labels: Series of cluster assignments
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))

        counts = cluster_labels.value_counts().sort_index()
        ax = counts.plot(kind='bar', color='steelblue', edgecolor='black')

        plt.title('Cluster Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Cluster ID', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.xticks(rotation=0)

        # Add value labels on bars
        for i, v in enumerate(counts):
            ax.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')

        # Add percentage labels
        total = counts.sum()
        for i, v in enumerate(counts):
            pct = v / total * 100
            ax.text(i, v / 2, f'{pct:.1f}%', ha='center', va='center',
                   color='white', fontweight='bold', fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Cluster distribution plot saved to {save_path}")
        plt.close()

    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray,
                             save_path: str = 'confusion_matrix.png'):
        """
        Plot confusion matrix heatmap.

        Args:
            cm: Confusion matrix array
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar_kws={'label': 'Count'},
            linewidths=0.5,
            linecolor='gray'
        )

        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Cluster', fontsize=12)
        plt.ylabel('True Cluster', fontsize=12)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {save_path}")
        plt.close()

    @staticmethod
    def plot_silhouette_scores(scores: Dict[int, float],
                              save_path: str = 'silhouette_scores.png'):
        """
        Plot silhouette scores for different k values.

        Args:
            scores: Dictionary mapping k to silhouette score
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))

        ks = list(scores.keys())
        score_values = list(scores.values())

        plt.plot(ks, score_values, marker='o', linewidth=2,
                markersize=8, color='steelblue')

        # Highlight best k
        best_k = max(scores, key=scores.get)
        best_score = scores[best_k]
        plt.scatter([best_k], [best_score], color='red', s=200,
                   zorder=5, label=f'Best k={best_k}')

        plt.title('Silhouette Score by Number of Clusters',
                 fontsize=16, fontweight='bold')
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Silhouette scores plot saved to {save_path}")
        plt.close()

    @staticmethod
    def plot_cross_validation_results(cv_results: Dict[str, Any],
                                      save_path: str = 'cross_validation_results.png'):
        """
        Plot cross-validation results across folds.

        Args:
            cv_results: Dictionary containing CV results
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Accuracy scores
        axes[0].bar(range(len(cv_results['cv_accuracy_scores'])),
                   cv_results['cv_accuracy_scores'],
                   color='steelblue', edgecolor='black')
        axes[0].axhline(y=cv_results['cv_accuracy_mean'], color='red',
                       linestyle='--', linewidth=2, label='Mean')
        axes[0].set_title('Cross-Validation Accuracy', fontweight='bold')
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # F1 Weighted scores
        axes[1].bar(range(len(cv_results['cv_f1_weighted_scores'])),
                   cv_results['cv_f1_weighted_scores'],
                   color='forestgreen', edgecolor='black')
        axes[1].axhline(y=cv_results['cv_f1_weighted_mean'], color='red',
                       linestyle='--', linewidth=2, label='Mean')
        axes[1].set_title('Cross-Validation F1 (Weighted)', fontweight='bold')
        axes[1].set_xlabel('Fold')
        axes[1].set_ylabel('F1 Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # F1 Macro scores
        axes[2].bar(range(len(cv_results['cv_f1_macro_scores'])),
                   cv_results['cv_f1_macro_scores'],
                   color='darkorange', edgecolor='black')
        axes[2].axhline(y=cv_results['cv_f1_macro_mean'], color='red',
                       linestyle='--', linewidth=2, label='Mean')
        axes[2].set_title('Cross-Validation F1 (Macro)', fontweight='bold')
        axes[2].set_xlabel('Fold')
        axes[2].set_ylabel('F1 Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Cross-validation results plot saved to {save_path}")
        plt.close()

    @staticmethod
    def plot_class_performance(report_dict: Dict[str, Any],
                              save_path: str = 'class_performance.png'):
        """
        Plot per-class performance metrics.

        Args:
            report_dict: Classification report dictionary
            save_path: Path to save the plot
        """
        # Extract per-class metrics
        classes = [k for k in report_dict.keys() if k.isdigit()]
        classes = sorted(classes, key=int)

        precision = [report_dict[c]['precision'] for c in classes]
        recall = [report_dict[c]['recall'] for c in classes]
        f1 = [report_dict[c]['f1-score'] for c in classes]
        support = [report_dict[c]['support'] for c in classes]

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        x = np.arange(len(classes))
        width = 0.25

        # Precision, Recall, F1 comparison
        axes[0, 0].bar(x - width, precision, width, label='Precision', color='steelblue')
        axes[0, 0].bar(x, recall, width, label='Recall', color='forestgreen')
        axes[0, 0].bar(x + width, f1, width, label='F1-Score', color='darkorange')
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Per-Class Metrics Comparison', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(classes)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Support distribution
        axes[0, 1].bar(x, support, color='purple', edgecolor='black')
        axes[0, 1].set_xlabel('Cluster')
        axes[0, 1].set_ylabel('Number of Samples')
        axes[0, 1].set_title('Class Support Distribution', fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(classes)
        axes[0, 1].grid(True, alpha=0.3)

        # F1-Score by cluster
        colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
        axes[1, 0].barh(classes, f1, color=colors, edgecolor='black')
        axes[1, 0].set_xlabel('F1-Score')
        axes[1, 0].set_ylabel('Cluster')
        axes[1, 0].set_title('F1-Score by Cluster', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')

        # Precision-Recall scatter
        axes[1, 1].scatter(recall, precision, s=[s/5 for s in support],
                          c=range(len(classes)), cmap='viridis',
                          edgecolors='black', linewidth=2, alpha=0.7)
        for i, c in enumerate(classes):
            axes[1, 1].annotate(f'C{c}', (recall[i], precision[i]),
                              fontweight='bold', fontsize=10)
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision-Recall by Cluster (size=support)',
                            fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class performance plot saved to {save_path}")
        plt.close()


class SimilaritySearchEngine:
    """
    Implements RAG-style similarity search for finding relevant historical cases.

    Uses TF-IDF and cosine similarity to retrieve the most similar
    historical work orders for a given query.
    """

    def __init__(self):
        """Initialize similarity search engine."""
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.df_notes = None
        logger.info("Initializing SimilaritySearchEngine")

    def fit(self, df: pd.DataFrame, description_col: str = 'Description_cleaned',
            text_col: str = 'Text'):
        """
        Fit the similarity search engine on the dataset.

        Args:
            df: DataFrame with work orders
            description_col: Column name for descriptions
            text_col: Column name for technician notes
        """
        logger.info("Fitting similarity search engine...")

        # Filter for rows with valid notes
        self.df_notes = df[df[text_col].notna() & (df[text_col].str.strip() != "")].copy()

        # Fit TF-IDF on descriptions
        descriptions = self.df_notes[description_col].fillna("")
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)  # Include bigrams for better matching
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(descriptions)

        logger.info(f"Similarity search engine fitted on {len(self.df_notes)} records")
        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    def search(self, query: str, top_n: int = 10) -> pd.DataFrame:
        """
        Search for similar historical work orders.

        Args:
            query: Query description
            top_n: Number of results to return

        Returns:
            pd.DataFrame: Top matching work orders with similarity scores
        """
        # Transform query
        query_vec = self.tfidf_vectorizer.transform([query.lower().strip()])

        # Calculate cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top N indices
        top_indices = similarities.argsort()[-top_n:][::-1]

        # Create results dataframe
        results = self.df_notes.iloc[top_indices].copy()
        results['similarity_score'] = similarities[top_indices]

        return results[['WO No.', 'Description', 'Text', 'similarity_score']]


def main():
    """
    Main execution function for the ML workflow.

    Orchestrates the entire pipeline from data loading to model evaluation.
    """
    logger.info("="*80)
    logger.info("Starting Enhanced ML Workflow for Technician Action Prediction")
    logger.info("="*80)

    # Step 1: Data Preprocessing
    logger.info("\n" + "="*80)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("="*80)

    preprocessor = DataPreprocessor("Work Orders with description, and notes.csv")
    df = preprocessor.load_data()
    df = preprocessor.clean_data()
    summary = preprocessor.get_data_summary()

    # Step 2: Unsupervised Clustering
    logger.info("\n" + "="*80)
    logger.info("STEP 2: UNSUPERVISED CLUSTERING")
    logger.info("="*80)

    clustering = ClusteringEngine(n_clusters=5)
    df['Note_Cluster'] = clustering.fit_transform(df['Text_cleaned'])

    # Analyze clusters
    top_terms = clustering.get_top_terms_per_cluster(n_terms=10)
    logger.info("\nTop terms per cluster:")
    for cluster_id, terms in top_terms.items():
        logger.info(f"  Cluster {cluster_id}: {', '.join(terms)}")

    # Calculate silhouette scores
    silhouette_scores = clustering.calculate_silhouette_scores()

    # Step 3: Visualizations - Clustering
    logger.info("\n" + "="*80)
    logger.info("STEP 3: CLUSTERING VISUALIZATIONS")
    logger.info("="*80)

    viz = Visualizer()
    viz.plot_cluster_distribution(df['Note_Cluster'])
    viz.plot_silhouette_scores(silhouette_scores)

    # Step 4: Build and Train Prediction Model
    logger.info("\n" + "="*80)
    logger.info("STEP 4: SUPERVISED LEARNING")
    logger.info("="*80)

    # Split data
    X = df['Description_cleaned']
    y = df['Note_Cluster']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Build and train model
    model = PredictionModel(class_weight='balanced')
    model.build_pipeline()
    model.train(X_train, y_train)

    # Step 5: Cross-Validation
    logger.info("\n" + "="*80)
    logger.info("STEP 5: CROSS-VALIDATION")
    logger.info("="*80)

    cv_results = model.cross_validate(X, y, cv=5)

    # Step 6: Model Evaluation
    logger.info("\n" + "="*80)
    logger.info("STEP 6: MODEL EVALUATION")
    logger.info("="*80)

    eval_results = model.evaluate(X_test, y_test)

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(eval_results['y_test'], eval_results['y_pred']))

    # Step 7: Visualizations - Model Performance
    logger.info("\n" + "="*80)
    logger.info("STEP 7: MODEL PERFORMANCE VISUALIZATIONS")
    logger.info("="*80)

    viz.plot_confusion_matrix(eval_results['confusion_matrix'])
    viz.plot_cross_validation_results(cv_results)
    viz.plot_class_performance(eval_results['classification_report'])

    # Step 8: Similarity Search Engine
    logger.info("\n" + "="*80)
    logger.info("STEP 8: SIMILARITY SEARCH ENGINE")
    logger.info("="*80)

    search_engine = SimilaritySearchEngine()
    search_engine.fit(df)

    # Test similarity search
    test_queries = [
        "conveyor not working",
        "machine won't turn on",
        "leak in air line"
    ]

    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        results = search_engine.search(query, top_n=5)
        logger.info(f"Top 5 similar cases found")
        for idx, row in results.iterrows():
            logger.info(f"  WO {row['WO No.']}: {row['similarity_score']:.4f} - {row['Text'][:80]}...")

    # Step 9: Save Results
    logger.info("\n" + "="*80)
    logger.info("STEP 9: SAVING RESULTS")
    logger.info("="*80)

    # Save processed data
    df.to_csv('processed_work_orders.csv', index=False)
    logger.info("Processed data saved to: processed_work_orders.csv")

    # Save model pipeline
    import joblib
    joblib.dump(model.pipeline, 'prediction_model.pkl')
    logger.info("Prediction model saved to: prediction_model.pkl")

    joblib.dump(search_engine, 'similarity_search_engine.pkl')
    logger.info("Similarity search engine saved to: similarity_search_engine.pkl")

    # Generate final report
    logger.info("\n" + "="*80)
    logger.info("FINAL PERFORMANCE SUMMARY")
    logger.info("="*80)
    logger.info(f"Test Accuracy: {eval_results['accuracy']:.4f}")
    logger.info(f"Test F1 (Weighted): {eval_results['f1_weighted']:.4f}")
    logger.info(f"Test F1 (Macro): {eval_results['f1_macro']:.4f}")
    logger.info(f"CV Accuracy: {cv_results['cv_accuracy_mean']:.4f} ± {cv_results['cv_accuracy_std']:.4f}")
    logger.info(f"CV F1 (Weighted): {cv_results['cv_f1_weighted_mean']:.4f} ± {cv_results['cv_f1_weighted_std']:.4f}")
    logger.info(f"CV F1 (Macro): {cv_results['cv_f1_macro_mean']:.4f} ± {cv_results['cv_f1_macro_std']:.4f}")

    logger.info("\n" + "="*80)
    logger.info("WORKFLOW COMPLETED SUCCESSFULLY!")
    logger.info("="*80)

    return {
        'preprocessor': preprocessor,
        'clustering': clustering,
        'model': model,
        'search_engine': search_engine,
        'df': df,
        'eval_results': eval_results,
        'cv_results': cv_results
    }


if __name__ == "__main__":
    results = main()
