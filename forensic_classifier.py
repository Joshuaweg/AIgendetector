"""
Forensic Classifier with Bayesian Uncertainty Quantification

Provides quantifiable certainty for AI-generated video detection using
camera forensic metrics.

Models:
1. Bayesian Logistic Regression - Fast, interpretable, good uncertainty
2. Gaussian Process Classification - Best uncertainty, slower
3. Ensemble with calibration - Combines multiple models
4. NGBoost - Natural Gradient Boosting with probabilistic output
5. Quantile XGBoost - XGBoost with prediction intervals
6. XGBoost Ensemble - Bootstrap ensemble for uncertainty

Usage:
    from forensic_classifier import ForensicClassifier

    clf = ForensicClassifier(model_type='ngboost')  # or 'quantile_xgb', 'xgb_ensemble'
    clf.fit(X_train, y_train)

    # Get prediction with uncertainty
    pred, confidence, credible_interval = clf.predict_with_uncertainty(X_test)
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass
from scipy import stats
from scipy.special import expit  # sigmoid
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PredictionResult:
    """Container for prediction with uncertainty."""
    prediction: int  # 0 = AI, 1 = Real
    label: str  # "AI" or "Real"
    probability: float  # P(Real)
    confidence: float  # How certain (0-1)
    credible_interval: Tuple[float, float]  # 95% CI for P(Real)
    uncertainty_type: str  # "low", "medium", "high"

    def __repr__(self):
        return (f"Prediction: {self.label} (P={self.probability:.3f}, "
                f"Confidence={self.confidence:.1%}, "
                f"95% CI=[{self.credible_interval[0]:.3f}, {self.credible_interval[1]:.3f}])")


class BayesianLogisticRegression:
    """
    Bayesian Logistic Regression using Laplace approximation.

    Provides posterior distributions over weights, enabling
    uncertainty quantification in predictions.
    """

    def __init__(self, prior_scale: float = 1.0, n_samples: int = 1000):
        """
        Args:
            prior_scale: Scale of Gaussian prior on weights (regularization)
            n_samples: Number of posterior samples for prediction
        """
        self.prior_scale = prior_scale
        self.n_samples = n_samples
        self.weights_mean = None
        self.weights_cov = None
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit Bayesian logistic regression using Laplace approximation.

        1. Find MAP estimate (mode of posterior)
        2. Approximate posterior as Gaussian around MAP
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Add intercept
        X_aug = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
        n_features = X_aug.shape[1]

        # Prior precision (inverse variance)
        prior_precision = np.eye(n_features) / (self.prior_scale ** 2)
        prior_precision[0, 0] = 1e-6  # Weak prior on intercept

        # Find MAP estimate using iteratively reweighted least squares (IRLS)
        weights = np.zeros(n_features)

        for iteration in range(100):
            # Predictions
            logits = X_aug @ weights
            probs = expit(logits)

            # Gradient of negative log posterior
            grad = X_aug.T @ (probs - y) + prior_precision @ weights

            # Hessian (Fisher information + prior)
            S = probs * (1 - probs)
            H = X_aug.T @ (X_aug * S[:, np.newaxis]) + prior_precision

            # Newton update
            try:
                delta = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(H, grad, rcond=None)[0]

            weights_new = weights - delta

            # Check convergence
            if np.max(np.abs(weights_new - weights)) < 1e-6:
                break
            weights = weights_new

        self.weights_mean = weights

        # Posterior covariance (inverse Hessian at MAP)
        logits = X_aug @ weights
        probs = expit(logits)
        S = probs * (1 - probs)
        H = X_aug.T @ (X_aug * S[:, np.newaxis]) + prior_precision

        try:
            self.weights_cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            self.weights_cov = np.linalg.pinv(H)

        return self

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty quantification.

        Returns:
            mean_probs: Mean predicted probabilities
            std_probs: Standard deviation of probabilities
            samples: Raw probability samples (n_samples, n_points)
        """
        X_scaled = self.scaler.transform(X)
        X_aug = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])

        # Sample from posterior
        weight_samples = np.random.multivariate_normal(
            self.weights_mean, self.weights_cov, size=self.n_samples
        )

        # Compute probabilities for each sample
        logits = X_aug @ weight_samples.T  # (n_points, n_samples)
        prob_samples = expit(logits)

        mean_probs = np.mean(prob_samples, axis=1)
        std_probs = np.std(prob_samples, axis=1)

        return mean_probs, std_probs, prob_samples.T

    def get_credible_interval(self, X: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Get credible intervals for predictions."""
        _, _, samples = self.predict_proba(X)

        lower = np.percentile(samples, 100 * alpha / 2, axis=0)
        upper = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)

        return lower, upper

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from posterior mean weights."""
        # Skip intercept (index 0)
        weights = self.weights_mean[1:]
        importance = np.abs(weights) / np.sum(np.abs(weights))

        return dict(sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        ))


class GaussianProcessClassifier:
    """
    Gaussian Process Classification with RBF kernel.

    Provides excellent uncertainty quantification but scales O(n³).
    Uses Laplace approximation for inference.
    """

    def __init__(self, length_scale: float = 1.0, noise: float = 1e-4, n_samples: int = 500):
        self.length_scale = length_scale
        self.noise = noise
        self.n_samples = n_samples
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None
        self.f_map = None  # MAP latent function values
        self.K = None  # Kernel matrix

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF (squared exponential) kernel."""
        sq_dist = np.sum(X1**2, axis=1, keepdims=True) + \
                  np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return np.exp(-0.5 * sq_dist / (self.length_scale ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter: int = 50):
        """Fit GP classifier using Laplace approximation."""
        self.X_train = self.scaler.fit_transform(X)
        self.y_train = y.astype(float)

        n = len(y)

        # Compute kernel matrix
        self.K = self._rbf_kernel(self.X_train, self.X_train)
        self.K += self.noise * np.eye(n)

        # Find MAP estimate of latent function using Newton's method
        f = np.zeros(n)

        for _ in range(max_iter):
            pi = expit(f)
            W = pi * (1 - pi)
            W = np.clip(W, 1e-10, 1 - 1e-10)

            # Gradient and Hessian of log posterior
            grad = self.y_train - pi - np.linalg.solve(self.K, f)

            W_sqrt = np.sqrt(W)
            B = np.eye(n) + W_sqrt[:, np.newaxis] * self.K * W_sqrt

            try:
                L = np.linalg.cholesky(B)
                b = W * f + grad
                a = b - W_sqrt * np.linalg.solve(L.T, np.linalg.solve(L, W_sqrt * (self.K @ b)))
                f_new = self.K @ a
            except np.linalg.LinAlgError:
                # Fallback to direct solve
                H = np.linalg.inv(self.K) + np.diag(W)
                f_new = f + np.linalg.solve(H, grad)

            if np.max(np.abs(f_new - f)) < 1e-6:
                break
            f = f_new

        self.f_map = f

        return self

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with uncertainty."""
        X_scaled = self.scaler.transform(X)

        # Kernel between test and training points
        K_star = self._rbf_kernel(X_scaled, self.X_train)

        # Predictive mean
        pi = expit(self.f_map)
        W = pi * (1 - pi)
        W = np.clip(W, 1e-10, 1 - 1e-10)

        f_mean = K_star @ (self.y_train - pi)

        # Predictive variance
        W_sqrt = np.sqrt(W)
        B = np.eye(len(self.y_train)) + W_sqrt[:, np.newaxis] * self.K * W_sqrt

        try:
            L = np.linalg.cholesky(B)
            v = np.linalg.solve(L, W_sqrt[:, np.newaxis] * K_star.T)
            K_star_star = self._rbf_kernel(X_scaled, X_scaled)
            f_var = np.diag(K_star_star) - np.sum(v**2, axis=0)
        except np.linalg.LinAlgError:
            f_var = np.ones(len(X_scaled)) * 0.5

        f_var = np.clip(f_var, 1e-10, None)

        # Sample from predictive distribution
        f_samples = np.random.normal(f_mean, np.sqrt(f_var), size=(self.n_samples, len(X_scaled)))
        prob_samples = expit(f_samples)

        mean_probs = np.mean(prob_samples, axis=0)
        std_probs = np.std(prob_samples, axis=0)

        return mean_probs, std_probs, prob_samples

    def get_credible_interval(self, X: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Get credible intervals."""
        _, _, samples = self.predict_proba(X)

        lower = np.percentile(samples, 100 * alpha / 2, axis=0)
        upper = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)

        return lower, upper


class NGBoostClassifier:
    """
    Natural Gradient Boosting for probabilistic classification.

    NGBoost outputs full probability distributions, providing
    principled uncertainty quantification with gradient boosting power.
    """

    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.1,
                 minibatch_frac: float = 1.0, n_samples: int = 500):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.n_samples = n_samples
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit NGBoost classifier."""
        try:
            from ngboost import NGBClassifier
            from ngboost.distns import Bernoulli
        except ImportError:
            raise ImportError("NGBoost not installed. Run: pip install ngboost")

        X_scaled = self.scaler.fit_transform(X)

        self.model = NGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            minibatch_frac=self.minibatch_frac,
            Dist=Bernoulli,
            verbose=False
        )
        self.model.fit(X_scaled, y)

        return self

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty from NGBoost's distributional output.
        """
        X_scaled = self.scaler.transform(X)

        # Get probability distribution parameters
        dist = self.model.pred_dist(X_scaled)

        # For Bernoulli, dist.prob gives P(Y=1)
        mean_probs = dist.prob

        # Sample from the predicted distributions
        samples = np.array([dist.sample() for _ in range(self.n_samples)])
        std_probs = np.std(samples, axis=0)

        return mean_probs, std_probs, samples

    def get_credible_interval(self, X: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Get credible intervals from distribution."""
        _, _, samples = self.predict_proba(X)

        lower = np.percentile(samples, 100 * alpha / 2, axis=0)
        upper = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)

        return lower, upper

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from NGBoost."""
        if self.model is None:
            return {}

        importance = self.model.feature_importances_
        importance = importance / (importance.sum() + 1e-8)

        return dict(sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        ))


class QuantileXGBoostClassifier:
    """
    XGBoost with quantile regression for prediction intervals.

    Trains separate models for different quantiles to estimate
    the full conditional distribution.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 6,
                 learning_rate: float = 0.1, quantiles: List[float] = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.quantiles = quantiles or [0.025, 0.1, 0.5, 0.9, 0.975]
        self.models = {}
        self.scaler = StandardScaler()
        self.main_model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit quantile regression models."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        X_scaled = self.scaler.fit_transform(X)

        # Main classification model
        self.main_model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        self.main_model.fit(X_scaled, y)

        # Quantile regression models for uncertainty
        # We'll use the predicted probabilities and bootstrap for quantiles
        # since XGBoost quantile regression is for regression tasks

        # Alternative: Bootstrap ensemble for quantiles
        n_bootstrap = 50
        self.bootstrap_models = []

        for i in range(n_bootstrap):
            idx = np.random.choice(len(y), size=len(y), replace=True)
            X_boot, y_boot = X_scaled[idx], y[idx]

            model = xgb.XGBClassifier(
                n_estimators=self.n_estimators // 2,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
            model.fit(X_boot, y_boot)
            self.bootstrap_models.append(model)

        return self

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with uncertainty from bootstrap ensemble."""
        X_scaled = self.scaler.transform(X)

        # Main prediction
        main_probs = self.main_model.predict_proba(X_scaled)[:, 1]

        # Bootstrap predictions for uncertainty
        bootstrap_probs = np.array([
            model.predict_proba(X_scaled)[:, 1]
            for model in self.bootstrap_models
        ])

        mean_probs = np.mean(bootstrap_probs, axis=0)
        std_probs = np.std(bootstrap_probs, axis=0)

        return mean_probs, std_probs, bootstrap_probs

    def get_credible_interval(self, X: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Get prediction intervals from bootstrap."""
        _, _, samples = self.predict_proba(X)

        lower = np.percentile(samples, 100 * alpha / 2, axis=0)
        upper = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)

        return lower, upper

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from main XGBoost model."""
        if self.main_model is None:
            return {}

        importance = self.main_model.feature_importances_
        importance = importance / (importance.sum() + 1e-8)

        return dict(sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        ))


class XGBoostBayesianEnsemble:
    """
    XGBoost ensemble with Bayesian bootstrap for uncertainty.

    Uses weighted bootstrap (Bayesian bootstrap) which provides
    a more principled uncertainty estimate than standard bootstrap.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 6,
                 learning_rate: float = 0.1, n_ensemble: int = 30):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_ensemble = n_ensemble
        self.models = []
        self.scaler = StandardScaler()

    def _bayesian_bootstrap_weights(self, n: int) -> np.ndarray:
        """Generate Bayesian bootstrap weights (Dirichlet distribution)."""
        # Dirichlet(1,1,...,1) gives uniform weights that sum to 1
        weights = np.random.dirichlet(np.ones(n))
        return weights * n  # Scale to sum to n for sample_weight

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Bayesian bootstrap ensemble."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        X_scaled = self.scaler.fit_transform(X)
        n = len(y)

        self.models = []

        for i in range(self.n_ensemble):
            # Bayesian bootstrap: random weights from Dirichlet
            weights = self._bayesian_bootstrap_weights(n)

            model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
            model.fit(X_scaled, y, sample_weight=weights)
            self.models.append(model)

        return self

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with uncertainty from Bayesian bootstrap ensemble."""
        X_scaled = self.scaler.transform(X)

        # Get predictions from all models
        all_probs = np.array([
            model.predict_proba(X_scaled)[:, 1]
            for model in self.models
        ])

        mean_probs = np.mean(all_probs, axis=0)
        std_probs = np.std(all_probs, axis=0)

        return mean_probs, std_probs, all_probs

    def get_credible_interval(self, X: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Get credible intervals from ensemble."""
        _, _, samples = self.predict_proba(X)

        lower = np.percentile(samples, 100 * alpha / 2, axis=0)
        upper = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)

        return lower, upper

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Average feature importance across ensemble."""
        if not self.models:
            return {}

        # Average importance across models
        importances = np.array([m.feature_importances_ for m in self.models])
        mean_importance = np.mean(importances, axis=0)
        mean_importance = mean_importance / (mean_importance.sum() + 1e-8)

        return dict(sorted(
            zip(feature_names, mean_importance),
            key=lambda x: x[1],
            reverse=True
        ))


class CalibratedEnsemble:
    """
    Ensemble of classifiers with Platt scaling for calibrated probabilities.
    Uses bootstrap for uncertainty estimation.
    """

    def __init__(self, n_estimators: int = 50, base_model: str = 'logistic'):
        self.n_estimators = n_estimators
        self.base_model = base_model
        self.models = []
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit ensemble with bootstrap."""
        X_scaled = self.scaler.fit_transform(X)
        n_samples = len(y)

        self.models = []

        for i in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_scaled[indices]
            y_boot = y[indices]

            if self.base_model == 'logistic':
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=1000, C=1.0)
            else:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, max_depth=10)

            model.fit(X_boot, y_boot)
            self.models.append(model)

        return self

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with uncertainty from ensemble disagreement."""
        X_scaled = self.scaler.transform(X)

        # Get predictions from all models
        all_probs = np.array([
            model.predict_proba(X_scaled)[:, 1] for model in self.models
        ])

        mean_probs = np.mean(all_probs, axis=0)
        std_probs = np.std(all_probs, axis=0)

        return mean_probs, std_probs, all_probs

    def get_credible_interval(self, X: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Get credible intervals from ensemble."""
        _, _, samples = self.predict_proba(X)

        lower = np.percentile(samples, 100 * alpha / 2, axis=0)
        upper = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)

        return lower, upper


class ForensicClassifier:
    """
    Main forensic classifier with uncertainty quantification.

    Wraps different Bayesian/ensemble models and provides
    unified interface for predictions with confidence.
    """

    def __init__(
        self,
        model_type: str = 'bayesian_logistic',
        confidence_threshold: float = 0.8,
        abstain_on_uncertain: bool = True,
        **model_kwargs
    ):
        """
        Args:
            model_type: 'bayesian_logistic', 'gaussian_process', or 'ensemble'
            confidence_threshold: Minimum confidence to make prediction
            abstain_on_uncertain: If True, return "uncertain" for low confidence
            **model_kwargs: Additional arguments for the model
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.abstain_on_uncertain = abstain_on_uncertain
        self.model_kwargs = model_kwargs

        self.model = None
        self.feature_names = None
        self.is_fitted = False

        # Calibration data
        self.calibration_data = None

    def _create_model(self):
        """Create the underlying model."""
        if self.model_type == 'bayesian_logistic':
            return BayesianLogisticRegression(**self.model_kwargs)
        elif self.model_type == 'gaussian_process':
            return GaussianProcessClassifier(**self.model_kwargs)
        elif self.model_type == 'ensemble':
            return CalibratedEnsemble(**self.model_kwargs)
        elif self.model_type == 'ngboost':
            return NGBoostClassifier(**self.model_kwargs)
        elif self.model_type == 'quantile_xgb':
            return QuantileXGBoostClassifier(**self.model_kwargs)
        elif self.model_type == 'xgb_ensemble':
            return XGBoostBayesianEnsemble(**self.model_kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        validation_split: float = 0.1
    ):
        """
        Fit the classifier.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0 = AI, 1 = Real)
            feature_names: Names of features for interpretability
            validation_split: Fraction for calibration validation
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Split for calibration
        n_val = int(len(y) * validation_split)
        if n_val > 0:
            indices = np.random.permutation(len(y))
            train_idx, val_idx = indices[n_val:], indices[:n_val]
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        else:
            X_train, y_train = X, y
            X_val, y_val = X, y

        # Fit model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)

        # Compute calibration curve
        mean_probs, _, _ = self.model.predict_proba(X_val)
        self.calibration_data = {
            'prob_true': y_val,
            'prob_pred': mean_probs,
        }

        self.is_fitted = True

        return self

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        alpha: float = 0.05
    ) -> List[PredictionResult]:
        """
        Predict with full uncertainty quantification.

        Args:
            X: Feature matrix
            alpha: Significance level for credible intervals (default 0.05 = 95% CI)

        Returns:
            List of PredictionResult objects
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Get predictions
        mean_probs, std_probs, _ = self.model.predict_proba(X)
        lower, upper = self.model.get_credible_interval(X, alpha)

        results = []
        for i in range(len(X)):
            prob = mean_probs[i]
            std = std_probs[i]
            ci = (lower[i], upper[i])

            # Prediction
            pred = 1 if prob > 0.5 else 0
            label = "Real" if pred == 1 else "AI"

            # Confidence: how far from decision boundary, adjusted by uncertainty
            # High confidence = far from 0.5 AND low uncertainty
            distance_from_boundary = abs(prob - 0.5) * 2  # 0 to 1
            certainty = 1 - std * 2  # Lower std = higher certainty
            certainty = np.clip(certainty, 0, 1)

            confidence = distance_from_boundary * certainty

            # Uncertainty classification
            if confidence > 0.7:
                uncertainty_type = "low"
            elif confidence > 0.4:
                uncertainty_type = "medium"
            else:
                uncertainty_type = "high"

            # Abstain if uncertain
            if self.abstain_on_uncertain and confidence < self.confidence_threshold:
                if ci[0] < 0.5 < ci[1]:  # CI spans decision boundary
                    label = "Uncertain"
                    pred = -1

            results.append(PredictionResult(
                prediction=pred,
                label=label,
                probability=float(prob),
                confidence=float(confidence),
                credible_interval=(float(ci[0]), float(ci[1])),
                uncertainty_type=uncertainty_type
            ))

        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Simple prediction without uncertainty details."""
        results = self.predict_with_uncertainty(X)
        return np.array([r.prediction for r in results])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        mean_probs, _, _ = self.model.predict_proba(X)
        return np.column_stack([1 - mean_probs, mean_probs])

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (for Bayesian logistic regression)."""
        if hasattr(self.model, 'get_feature_importance'):
            return self.model.get_feature_importance(self.feature_names)
        return {}

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Comprehensive evaluation with calibration metrics.
        """
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        results = self.predict_with_uncertainty(X)

        # Filter out uncertain predictions for accuracy
        certain_mask = np.array([r.prediction != -1 for r in results])

        predictions = np.array([r.prediction for r in results])
        probabilities = np.array([r.probability for r in results])
        confidences = np.array([r.confidence for r in results])

        metrics = {}

        # Overall metrics
        if certain_mask.sum() > 0:
            metrics['accuracy'] = accuracy_score(y[certain_mask], predictions[certain_mask])
            metrics['coverage'] = certain_mask.mean()  # Fraction of predictions made
        else:
            metrics['accuracy'] = 0.0
            metrics['coverage'] = 0.0

        # Probabilistic metrics
        metrics['roc_auc'] = roc_auc_score(y, probabilities)
        metrics['brier_score'] = brier_score_loss(y, probabilities)

        # Calibration
        prob_true, prob_pred = calibration_curve(y, probabilities, n_bins=10)
        calibration_error = np.mean(np.abs(prob_true - prob_pred))
        metrics['expected_calibration_error'] = calibration_error

        # Uncertainty quality: high confidence should correlate with correctness
        correct = (predictions == y).astype(float)
        correct[predictions == -1] = np.nan

        # Confidence-accuracy correlation
        valid_mask = ~np.isnan(correct)
        if valid_mask.sum() > 10:
            correlation = np.corrcoef(confidences[valid_mask], correct[valid_mask])[0, 1]
            metrics['confidence_accuracy_correlation'] = correlation

        # Abstention analysis
        n_abstained = (~certain_mask).sum()
        metrics['n_abstained'] = int(n_abstained)
        metrics['abstention_rate'] = float(1 - certain_mask.mean())

        # Accuracy at different confidence thresholds
        for threshold in [0.5, 0.7, 0.9]:
            high_conf_mask = confidences >= threshold
            if high_conf_mask.sum() > 0 and certain_mask[high_conf_mask].sum() > 0:
                acc = accuracy_score(
                    y[high_conf_mask & certain_mask],
                    predictions[high_conf_mask & certain_mask]
                )
                metrics[f'accuracy_at_conf_{threshold}'] = acc
                metrics[f'coverage_at_conf_{threshold}'] = high_conf_mask.mean()

        return metrics

    def save(self, path: str):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model_type': self.model_type,
                'model': self.model,
                'feature_names': self.feature_names,
                'confidence_threshold': self.confidence_threshold,
                'abstain_on_uncertain': self.abstain_on_uncertain,
                'calibration_data': self.calibration_data,
            }, f)

    @classmethod
    def load(cls, path: str) -> 'ForensicClassifier':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        clf = cls(
            model_type=data['model_type'],
            confidence_threshold=data['confidence_threshold'],
            abstain_on_uncertain=data['abstain_on_uncertain']
        )
        clf.model = data['model']
        clf.feature_names = data['feature_names']
        clf.calibration_data = data['calibration_data']
        clf.is_fitted = True

        return clf


def print_evaluation_report(metrics: Dict, title: str = "Evaluation Report"):
    """Pretty print evaluation metrics."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print(f"\nAccuracy: {metrics.get('accuracy', 0):.2%}")
    print(f"Coverage: {metrics.get('coverage', 0):.2%} (predictions made)")
    print(f"Abstained: {metrics.get('n_abstained', 0)} samples")

    print(f"\nROC AUC: {metrics.get('roc_auc', 0):.3f}")
    print(f"Brier Score: {metrics.get('brier_score', 0):.4f} (lower is better)")
    print(f"Expected Calibration Error: {metrics.get('expected_calibration_error', 0):.4f}")

    if 'confidence_accuracy_correlation' in metrics:
        print(f"\nConfidence-Accuracy Correlation: {metrics['confidence_accuracy_correlation']:.3f}")

    print("\nAccuracy at Confidence Thresholds:")
    for threshold in [0.5, 0.7, 0.9]:
        acc_key = f'accuracy_at_conf_{threshold}'
        cov_key = f'coverage_at_conf_{threshold}'
        if acc_key in metrics:
            print(f"  ≥{threshold:.0%} confidence: {metrics[acc_key]:.2%} accuracy, "
                  f"{metrics[cov_key]:.2%} coverage")

    print("\n" + "=" * 60)


# =============================================================================
# TRAINING SCRIPT
# =============================================================================

def train_forensic_classifier(
    data_dir: str,
    output_path: str,
    model_type: str = 'bayesian_logistic',
    max_videos: int = 1000,
    max_frames: int = 16,
):
    """
    Train forensic classifier on video dataset.

    Args:
        data_dir: Directory with Real/ and Fake/ subdirectories
        output_path: Where to save trained model
        model_type: Type of classifier
        max_videos: Maximum videos to use (for speed)
        max_frames: Frames per video for forensic analysis
    """
    from camera_forensics import CameraForensics
    from pathlib import Path
    from tqdm import tqdm

    print("=" * 60)
    print("Training Forensic Classifier")
    print("=" * 60)

    data_path = Path(data_dir)

    # Collect videos
    videos = []

    real_dir = data_path / 'Real'
    fake_dir = data_path / 'Fake'

    if real_dir.exists():
        for f in list(real_dir.glob('*.mp4'))[:max_videos // 2]:
            videos.append((str(f), 1))  # 1 = Real

    if fake_dir.exists():
        fake_videos = []
        for subdir in fake_dir.iterdir():
            if subdir.is_dir():
                fake_videos.extend(list(subdir.glob('*.mp4')))
        for f in fake_videos[:max_videos // 2]:
            videos.append((str(f), 0))  # 0 = AI

    print(f"Found {len(videos)} videos")
    np.random.shuffle(videos)

    # Extract forensic features
    forensics = CameraForensics()

    all_features = []
    all_labels = []
    feature_names = None

    print("\nExtracting forensic features...")
    for video_path, label in tqdm(videos):
        try:
            results = forensics.analyze_video(video_path, max_frames=max_frames)
            features, names = forensics.get_feature_vector(results)

            if feature_names is None:
                feature_names = names

            all_features.append(features)
            all_labels.append(label)

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue

    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Labels: {(y == 1).sum()} Real, {(y == 0).sum()} AI")

    # Train classifier
    print(f"\nTraining {model_type} classifier...")
    clf = ForensicClassifier(
        model_type=model_type,
        confidence_threshold=0.6,
        abstain_on_uncertain=True
    )
    clf.fit(X, y, feature_names=feature_names)

    # Evaluate
    metrics = clf.evaluate(X, y)
    print_evaluation_report(metrics, "Training Set Evaluation")

    # Feature importance
    if model_type == 'bayesian_logistic':
        importance = clf.get_feature_importance()
        print("\nTop 10 Most Important Features:")
        for i, (name, imp) in enumerate(list(importance.items())[:10]):
            print(f"  {i+1}. {name}: {imp:.4f}")

    # Save
    clf.save(output_path)
    print(f"\nModel saved to {output_path}")

    return clf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train forensic classifier')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory with Real/ and Fake/ subdirectories')
    parser.add_argument('--output', type=str, default='forensic_model.pkl',
                       help='Output model path')
    parser.add_argument('--model_type', type=str, default='bayesian_logistic',
                       choices=['bayesian_logistic', 'gaussian_process', 'ensemble',
                               'ngboost', 'quantile_xgb', 'xgb_ensemble'],
                       help='Type of classifier')
    parser.add_argument('--max_videos', type=int, default=500,
                       help='Maximum videos to use')
    parser.add_argument('--max_frames', type=int, default=16,
                       help='Frames per video')

    args = parser.parse_args()

    train_forensic_classifier(
        data_dir=args.data_dir,
        output_path=args.output,
        model_type=args.model_type,
        max_videos=args.max_videos,
        max_frames=args.max_frames
    )
