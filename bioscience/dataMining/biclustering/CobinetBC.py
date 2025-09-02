from bioscience.base import *
import bioscience as bs
import numpy as np 
import pandas as pd
from itertools import combinations
import os
import optuna
import shutil
from copy import deepcopy
from joblib import Parallel, delayed


def processCobinetBC(dataset, deviceCount=1, mode=1, debug=False):
    """
    Main processing function for the CoBiNet biclustering algorithm.
    Follows the execution pattern established in BCCA:
      - mode=1: Sequential execution (implemented)
      - mode=2: CPU-parallel execution (to be developed)
      - mode=3: GPU-parallel execution (to be developed)

    :param dataset: The dataset object storing the data of the input file.
    :type dataset: :class:`bioscience.base.models.Dataset`
    :param deviceCount: Number of GPU devices to execute, defaults to 1.
    :type deviceCount: int, optional
    :param mode: Type of execution of the algorithm.
                 1 = sequential, 2 = CPU parallel, 3 = GPU parallel.
    :type mode: int, optional
    :param debug: Run the algorithm in debug mode, defaults to False.
    :type debug: bool, optional

    :return: A BiclusteringModel object containing the biclusters.
    :rtype: :class:`bioscience.base.models.BiclusteringModel`
    """
    oModel = None

    sMode = ""
    if mode == 2:  # NUMBA: CPU Parallel mode
        # To be developed
        sMode = "NUMBA - CPU Parallel mode (to be developed)"
    elif mode == 3:  # NUMBA: GPU Parallel mode
        # To be developed
        sMode = "NUMBA - GPU Parallel mode (to be developed)"
    else:  # Sequential mode
        oModel = __cobinetSequential(dataset, debug)
        deviceCount = 0
        sMode = "CPU Sequential"

    return oModel


def __cobinetSequential(dataset,debug=False, n_trials=20, tolerancyThreshold=10, geneMin=2,  save=False):

    """
    Main entry point for executing the CoBiNet algorithm pipeline.
    This function orchestrates the full workflow:
    1. Data loading and preprocessing.
    2. Automatic hyperparameter optimization using multiple correlation measures (Pearson, Spearman, and Normalized Mutual Information - NMI) along with column percentage.
    3. Execution of the CoBiNet algorithm with the best hyperparameters.
    4. Stability evaluation of the resulting biclusters via perturbation analysis.

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    Gene expression datasets are often noisy, high-dimensional, and heterogeneous.
    The CoBiNet method constructs biclusters using an ensemble of correlation measures, optimizing thresholds and the proportion of columns retained to maximize biological coherence and statistical robustness.
    
    This function coordinates:
    - Automatic hyperparameter optimization: Uses Optuna to search for correlation thresholds and percentage of retained columns that yield the most cohesive and least redundant biclusters.
    - Final biclustering run: Executes the biclustering with the best hyperparameters identified in the optimization phase.
    - Stability evaluation: Applies subsampling and noise perturbations to measure how consistently biclusters appear, using Jaccard similarity as a stability metric.

    Parameters
    ----------
    :param path: Path to the input CSV file containing gene expression data.
                 The file must have genes as rows and conditions/samples as columns.
                 The first column should contain gene identifiers.
    :type path: str or None
    :param n_trials: Number of trials for automatic hyperparameter optimization.
    :type n_trials: int
    :param tolerancyThreshold: Allowed tolerance (in percentage) when adding genes
                                to a bicluster during the growth phase. Converted
                                internally to a decimal fraction.
    :type tolerancyThreshold: int
    :param geneMin: Minimum number of genes (rows) required for a valid bicluster.
    :type geneMin: int
    :param debug: If True, prints detailed execution logs for debugging purposes.
    :type debug: bool
    :param save: If True, saves the resulting biclusters and associated metadata
                 to disk.
    :type save: bool

    Returns
    -------
    :return: None
    :rtype: None
    """
        
        
    data = np.asarray(dataset.data, dtype=float)
    genes = np.asarray(dataset.geneNames)
    conditions = np.asarray(dataset.columnsNames)
    rows, cols = data.shape
    tolerancyThreshold = tolerancyThreshold / 100.0
    resultsPairs = set()
    
    
    
    # 1) First phase: Automatic hyperparameter optimization phase (spearman, pearson, NMI and colMin)
    bestParams = _autoHPOPhase(data, genes, conditions, rows, cols, tolerancyThreshold, geneMin, n_trials)
    
    # 2) Second phase: Run algorithm with the best parameters
    if bestParams is not None:
        resultsPairs, biclustersRows, biclustersCols, biclustersCorrelation = _executeAlgorithm(
            data, genes, conditions, rows, cols, tolerancyThreshold,
            bestParams["pearson"], bestParams["spearman"], bestParams["nmi"],
            int(round(bestParams["cols_percent"] * cols)), geneMin, debug, save, resultsPairs
        )
        
        # 3) Evaluation phase.
        stabilityAverage = _evaluateStability(
            data, genes, conditions, rows, cols,
            bestParams["pearson"], bestParams["spearman"], bestParams["nmi"],
            int(round(bestParams["cols_percent"] * cols)), geneMin,
            tolerancyThreshold=tolerancyThreshold, num_runs=n_trials
        )
        print(f"[EVALUATION] Jaccard average over {n_trials} executions: {stabilityAverage:.4f}")
    else:
        print("No biclusters were found with the parameters entered.")
    
    
    
    

    oModel = BiclusteringModel()
    return oModel











def _executeAlgorithm(data, genes, conditions, rows, cols, tolerancyThreshold, pearsonThreshold, spearmanThreshold, nmiThreshold, colMin, geneMin, debug, save, resultsPairs):
    """
    Executes the CoBiNet biclustering algorithm for a given dataset and set of hyperparameters. This function runs the three core steps:
    1. Pattern generation: Identify initial column patterns (co-expression patterns) for all possible gene pairs.
    2. Bicluster construction: Expand these patterns into full biclusters by adding compatible genes.
    3. Duplicate elimination: Remove redundant or overlapping biclusters.

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    The CoBiNet algorithm is based on the concept that genes participating in the same biological process will exhibit similar expression patterns under certain subsets of experimental conditions. This function:
    - Generates candidate bicluster seeds by evaluating correlation-based similarity measures for every possible gene pair.
    - Expands seeds into biclusters according to correlation thresholds and tolerance values, ensuring biological coherence.
    - Filters out duplicates to maintain a non-redundant set of high-quality biclusters.
    - Optionally saves results for further downstream analysis.

    Parameters
    ----------
    :param data: Gene expression matrix (genes x conditions), without gene identifiers.
    :type data: numpy.ndarray
    :param genes: Array of gene identifiers corresponding to the rows of `data`.
    :type genes: numpy.ndarray
    :param conditions: Array of condition/sample identifiers corresponding to the columns of `data`.
    :type conditions: numpy.ndarray
    :param rows: Total number of genes in the dataset.
    :type rows: int
    :param cols: Total number of conditions/samples in the dataset.
    :type cols: int
    :param tolerancyThreshold: Allowed tolerance (as a decimal fraction) for adding genes
                                to an existing bicluster during the growth phase.
    :type tolerancyThreshold: float
    :param pearsonThreshold: Minimum Pearson correlation coefficient required for
                              inclusion of genes in a bicluster.
    :type pearsonThreshold: float
    :param spearmanThreshold: Minimum Spearman correlation coefficient required for
                               inclusion of genes in a bicluster.
    :type spearmanThreshold: float
    :param nmiThreshold: Minimum Normalized Mutual Information (NMI) score required for
                          inclusion of genes in a bicluster.
    :type nmiThreshold: float
    :param colMin: Minimum number of columns (conditions) required for a valid bicluster.
    :type colMin: int
    :param geneMin: Minimum number of genes required for a valid bicluster.
    :type geneMin: int
    :param debug: If True, prints detailed debugging information during execution.
    :type debug: bool
    :param save: If True, saves the resulting biclusters and metadata to disk.
    :type save: bool
    :param resultsPairs: Set used to store the number of biclusters generated in
                         previous runs to avoid repeated execution for the same configuration.
    :type resultsPairs: set

    Returns
    -------
    :return: 
        - resultsPairs (set): Updated set of previously executed configurations.
        - biclustersRows (list): List of arrays containing gene indices for each bicluster.
        - biclustersCols (list): List of arrays containing condition indices for each bicluster.
        - biclustersCorrelation (list): List of average correlation scores for each bicluster.
    :rtype: tuple
    """
    # 1) Get max patterns
    maxPatterns = _getNumPatterns(rows)

    # 2) Calculate patterns 
    
    patternColumns, correlations = _calculatePatterns(data, rows, cols, pearsonThreshold, spearmanThreshold, nmiThreshold, colMin, maxPatterns)
    
    # 3) Get biclusters
    biclustersRows, biclustersCols, biclustersCorrelation, numBiclusters = _getBiclusters(
        data, patternColumns, correlations, rows, cols, tolerancyThreshold, 
        pearsonThreshold, spearmanThreshold, nmiThreshold, maxPatterns, geneMin
    )
    
    # 4) Optional: Save biclusters
    pairResult = numBiclusters
    if pairResult not in resultsPairs:
        
        if debug:
            print("\n########", flush=True)
            print("RESULTS:", flush=True)
            print("########", flush=True)
            print(f"- Dataset size: {rows}x{cols}", flush=True)
            print(f"- Spearman threshold: {spearmanThreshold}", flush=True)
            print(f"- NMI threshold: {nmiThreshold}", flush=True)
            print(f"- Pearson threshold: {pearsonThreshold}", flush=True)    
            print(f"- Tolerancy threshold (%): {tolerancyThreshold*100} ", flush=True)
            print(f"- Minimum number of columns (biclusters): {colMin}", flush=True)
            print(f"- Minimum number of rows (genes): {geneMin}", flush=True)
            print("-------------------------", flush=True)    
            print(f"Biclusters: {numBiclusters}", flush=True)
        
        if save:
            if not os.path.exists(f"results/biclusters/genes"):
                os.makedirs(f"results/biclusters/genes")
                
            with open(f"results/info.txt", "w") as f:
                f.write(f"\n- Dataset size: {rows}x{cols}")
                f.write(f"\n- Spearman threshold: {spearmanThreshold}")
                f.write(f"\n- NMI threshold: {nmiThreshold}")
                f.write(f"\n- Pearson threshold: {pearsonThreshold}")    
                f.write(f"\n- Tolerancy threshold (%): {tolerancyThreshold*100} ")
                f.write(f"\n- Minimum number of columns (biclusters): {colMin}")
                f.write(f"\n- Minimum number of rows (genes): {geneMin}")
                f.write("\n-------------------------")    
                f.write(f"\nBiclusters: {numBiclusters}")
            
            # Save biclusters
            _saveBiclusters(genes, conditions, biclustersRows, biclustersCols, biclustersCorrelation, debug)
            
        resultsPairs.add(pairResult)
        
    return resultsPairs, biclustersRows, biclustersCols, biclustersCorrelation
  
##################################################
# PHASE 1: AUTOMATIC HYPERPARAMETER OPTIMIZATION #
##################################################
def _autoHPOPhase(data, genes, conditions, rows, cols, tolerancyThreshold, geneMin, n_trials):
    """
    Automatically tunes the main hyperparameters of the CoBiNet biclustering algorithm using Optuna's Bayesian optimization framework. 
    
    The goal is to find the correlation thresholds (Pearson, Spearman, NMI) and the percentage of columns retained (`cols_percent`) that maximize the overall quality, coherence, and size of biclusters while minimizing redundancy.

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    Hyperparameter selection is critical in biclustering because correlation thresholds and the fraction of retained conditions directly impact the ability to detect biologically meaningful co-expression patterns. 
    Manual tuning is inefficient and biased, especially with large search spaces.
    
    This function uses Optuna to:
    - Systematically explore the parameter space.
    - Evaluate bicluster quality via multiple internal validation metrics
      (average intra-bicluster correlation, coherence with the mean profile,
       variance, density, and redundancy).
    - Balance trade-offs between bicluster size, quality, and coverage.

    Parameters
    ----------
    :param data: Gene expression matrix (genes x conditions), without gene identifiers.
    :type data: numpy.ndarray
    :param genes: Array of gene identifiers corresponding to the rows of `data`.
    :type genes: numpy.ndarray
    :param conditions: Array of condition/sample identifiers corresponding to the columns of `data`.
    :type conditions: numpy.ndarray
    :param rows: Total number of genes in the dataset.
    :type rows: int
    :param cols: Total number of conditions/samples in the dataset.
    :type cols: int
    :param tolerancyThreshold: Allowed tolerance (decimal fraction) for adding genes
                                to a bicluster during growth phase.
    :type tolerancyThreshold: float
    :param geneMin: Minimum number of genes required for a valid bicluster.
    :type geneMin: int
    :param n_trials: Number of optimization trials to perform in Optuna.
    :type n_trials: int

    Returns
    -------
    :return: Dictionary containing the best parameter values found:
             - "spearman": optimal Spearman correlation threshold
             - "pearson": optimal Pearson correlation threshold
             - "nmi": optimal NMI threshold
             - "cols_percent": optimal percentage of columns to retain
             Returns None if no valid biclusters are found in any trial.
    :rtype: dict or None
    """
    def objective(trial):
        # 1) Sample hyperparameters within defined search space
        spearman_thr = trial.suggest_float("spearman", 0.7, 0.95, step=0.05)
        pearson_thr = trial.suggest_float("pearson", 0.7, spearman_thr, step=0.05)
        nmi_thr = trial.suggest_float("nmi", pearson_thr, 0.95, step=0.05)
        col_percentage = trial.suggest_float("cols_percent", 0.7, 1.0, step=0.05)
        num_cols_kept = int(round(col_percentage * cols))

        # 2) Run the algorithm with the sampled parameters
        _, biclustersRows, biclustersCols, biclustersCorrelation = _executeAlgorithm(
            data, genes, conditions, rows, cols, tolerancyThreshold,
            round(pearson_thr, 2), round(spearman_thr, 2), round(nmi_thr, 2),
            num_cols_kept, geneMin, False, False, set()
        )
        
        # If no biclusters are found, this trial is pruned (early stop)
        if len(biclustersCorrelation) == 0:
            raise optuna.exceptions.TrialPruned()
        
        # 3) Evaluate internal validation metrics for each bicluster
        validations = Parallel(n_jobs=-1)(
            delayed(_evaluate_bicluster)(
                data, rows_bic, cols_bic,
                pearson_thr, spearman_thr, nmi_thr,
                bs.ensembleCorrelation
            )
            for rows_bic, cols_bic in zip(biclustersRows, biclustersCols)
        )

        # If no valid metrics could be computed, prune trial
        if not validations:
            raise optuna.exceptions.TrialPruned()
        
        # Compute aggregate metrics across biclusters
        avg_row_corr = np.nanmean([v['avg_row_corr'] for v in validations])
        avg_col_corr = np.nanmean([v['avg_col_corr'] for v in validations])
        avg_coherence = np.nanmean([v['avg_profile_coherence'] for v in validations])
        avg_variance = np.nanmean([v['variance'] for v in validations])

        # Penalization for excessive column removal
        total_cols_dataset = data.shape[1]
        used_cols = len(set(col for cols_set in biclustersCols for col in cols_set))
        pct_cols_removed = 1 - (used_cols / total_cols_dataset)
        penalty_cols = (pct_cols_removed ** 2) * 0.50

        # Reward for larger average bicluster size
        bicluster_sizes = [len(rows_bic) for rows_bic in biclustersRows]
        avg_size = np.mean(bicluster_sizes) if bicluster_sizes else 0
        size_factor = np.log1p(avg_size) / np.log1p(data.shape[0])  # Normalized to [0,1]

        # Penalization for redundancy between biclusters
        def jaccard(a, b):
            inter = len(set(a) & set(b))
            union = len(set(a) | set(b))
            return inter / union if union > 0 else 0

        redundancies = []
        for i in range(len(biclustersRows)):
            for j in range(i + 1, len(biclustersRows)):
                redundancies.append(jaccard(biclustersRows[i], biclustersRows[j]))
        avg_redundancy = np.mean(redundancies) if redundancies else 0
        penalty_redundancy = avg_redundancy * 0.20  # Adjustable weight

        # 5) Final score. Balances correlation quality, coherence, size, and redundancy.
        score = (
            0.30 * avg_row_corr +
            0.25 * avg_coherence +
            0.15 * avg_col_corr -
            0.10 * avg_variance -
            penalty_cols +
            0.20 * size_factor -           
            penalty_redundancy
        )
        
        return score
    
    # Run Bayesian optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    completed_trials = [t for t in study.trials if t.value is not None]  # Collect trials that yielded a score

    # Return best hyperparameters or None if no trial succeeded
    if len(completed_trials) > 0:
        best_params = study.best_params
    else:
        best_params = None
    
    return best_params

# Define the bicluster evaluation function again after reset
def _evaluate_bicluster(data, rows_bic, cols_bic, pearson_thr, spearman_thr, nmi_thr, ensembleCorrelation):
    submatrix = data[np.ix_(rows_bic, cols_bic)]

    # Average correlation between rows
    row_corrs = []
    for i in range(len(rows_bic)):
        for j in range(i + 1, len(rows_bic)):
            r1 = data[rows_bic[i]][cols_bic]
            r2 = data[rows_bic[j]][cols_bic]
            validCorr, corr = bs.ensembleCorrelation(r1, r2, pearson_thr, spearman_thr, nmi_thr)
            if validCorr:
                row_corrs.append(corr)
    avg_row_corr = np.mean(row_corrs) if row_corrs else np.nan

    # Average correlation between columns
    col_corrs = []
    for i in range(len(cols_bic)):
        for j in range(i + 1, len(cols_bic)):
            c1 = data[rows_bic, cols_bic[i]]
            c2 = data[rows_bic, cols_bic[j]]
            validCorr, corr = bs.ensembleCorrelation(c1, c2, pearson_thr, spearman_thr, nmi_thr)
            if validCorr:
                col_corrs.append(corr)
    avg_col_corr = np.mean(col_corrs) if col_corrs else np.nan

    # Coherence with the average profile
    profile = np.mean(submatrix, axis=0)
    profile_corrs = []
    for i in range(submatrix.shape[0]):
        row_data = submatrix[i]
        validCorr, corr = bs.ensembleCorrelation(row_data, profile, pearson_thr, spearman_thr, nmi_thr)
        if validCorr:
            profile_corrs.append(corr)
    avg_profile_coherence = np.mean(profile_corrs) if profile_corrs else np.nan

    # Variance
    variance_val = np.var(submatrix)

    # Density
    total_cells = submatrix.shape[0] * submatrix.shape[1]
    valid_cells = np.count_nonzero(~np.isnan(submatrix))
    density = valid_cells / total_cells if total_cells > 0 else np.nan

    return {
        "rows": len(rows_bic),
        "cols": len(cols_bic),
        "avg_row_corr": avg_row_corr,
        "avg_col_corr": avg_col_corr,
        "avg_profile_coherence": avg_profile_coherence,
        "variance": variance_val,
        "density": density
    }


############################################################
# PHASE 2: COBINET ALGORITHM WITH THE BEST HYPERPARAMETERS #
############################################################

#################
# 2.1. PATTERNS #
#################

def _calculatePatterns(data, rows, cols, pearsonThreshold, spearmanThreshold, nmiThreshold, colMin, maxPatterns):
    """
    Generates the initial set of column patterns (candidate bicluster seeds) 
    for all possible gene pairs in the dataset.  
    This is achieved in three sequential stages:
    1. **Ensemble correlation computation**: For each gene pair, calculate 
       combined correlation scores (Pearson, Spearman, and NMI) and record 
       patterns that meet the required thresholds.
    2. **Optimal correlation refinement**: For patterns that nearly meet the 
       thresholds, apply a column removal strategy to improve their correlation 
       scores while retaining enough columns.
    3. **Duplicate pattern removal**: Eliminate exact duplicates to avoid 
       redundant computation in later stages.

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    The pattern generation step is critical in biclustering because it defines 
    the seeds from which biclusters will grow.  
    - Using multiple correlation measures in an ensemble ensures robustness 
      against noise and different data distributions.
    - Refining patterns by removing low-contribution columns prevents high 
      correlation patterns from being discarded prematurely.
    - Removing duplicates at this stage reduces computational cost and prevents 
      artificial inflation of bicluster counts.

    Parameters
    ----------
    :param data: Gene expression matrix (genes x conditions), without gene identifiers.
    :type data: numpy.ndarray
    :param rows: Total number of genes in the dataset.
    :type rows: int
    :param cols: Total number of conditions/samples in the dataset.
    :type cols: int
    :param pearsonThreshold: Minimum Pearson correlation coefficient for a pattern 
                             to be considered valid.
    :type pearsonThreshold: float
    :param spearmanThreshold: Minimum Spearman correlation coefficient for a pattern 
                              to be considered valid.
    :type spearmanThreshold: float
    :param nmiThreshold: Minimum Normalized Mutual Information (NMI) score for a 
                         pattern to be considered valid.
    :type nmiThreshold: float
    :param colMin: Minimum number of columns required for a valid pattern.
    :type colMin: int
    :param maxPatterns: Maximum number of patterns (equal to number of possible 
                        unique gene pairs).
    :type maxPatterns: int

    Returns
    -------
    :return: 
        - patternColumns (numpy.ndarray): Encoded column patterns for each gene pair 
          (stored as decimal representation of binary masks).
        - correlations (numpy.ndarray): Average correlation score for each pattern.
    :rtype: tuple (numpy.ndarray, numpy.ndarray)
    """
    patternColumns, correlations = _calculateEnsembleCorrelations(data, rows, cols, pearsonThreshold, spearmanThreshold, nmiThreshold, colMin, maxPatterns)   
    patternColumns, correlations = _calculateOptimalCorrelations(patternColumns, correlations, data, rows, cols, pearsonThreshold, spearmanThreshold, nmiThreshold, colMin, maxPatterns)    
    patternColumns, correlations = _removePatternsDuplicated(patternColumns, correlations, rows, cols, maxPatterns) 
    return patternColumns, correlations

def _calculateEnsembleCorrelations(data, rows, cols, pearsonThreshold, spearmanThreshold, nmiThreshold, colMin, maxPatterns):
    """
    Computes ensemble correlation scores for every possible gene pair in the dataset
    and generates the initial set of column patterns.  
    For each pair of genes, the function:
    1. Calculates combined correlation scores using Pearson, Spearman, and NMI.
    2. Marks patterns that meet or exceed the correlation thresholds as valid seeds.
    3. Flags patterns that are near the threshold as candidates for later optimization.
    4. Rejects patterns that do not meet minimum criteria.

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    Gene co-expression patterns often exhibit different correlation structures depending
    on data distribution, outliers, and measurement noise.  
    - Pearson captures linear relationships.
    - Spearman captures monotonic (rank-based) relationships, reducing sensitivity to outliers.
    - NMI measures mutual dependence, capturing nonlinear relationships.
    
    Combining these in an ensemble increases robustness and reduces false negatives.
    Patterns that are close to the thresholds are not discarded outright; instead, they are
    flagged for refinement in `_calculateOptimalCorrelations`.

    Parameters
    ----------
    :param data: Gene expression matrix (genes x conditions), without gene identifiers.
    :type data: numpy.ndarray
    :param rows: Total number of genes in the dataset.
    :type rows: int
    :param cols: Total number of conditions/samples in the dataset.
    :type cols: int
    :param pearsonThreshold: Minimum Pearson correlation coefficient for a pattern 
                             to be considered valid.
    :type pearsonThreshold: float
    :param spearmanThreshold: Minimum Spearman correlation coefficient for a pattern 
                              to be considered valid.
    :type spearmanThreshold: float
    :param nmiThreshold: Minimum Normalized Mutual Information (NMI) score for a pattern 
                         to be considered valid.
    :type nmiThreshold: float
    :param colMin: Minimum number of columns required for a valid pattern.
    :type colMin: int
    :param maxPatterns: Maximum number of patterns (equal to the number of unique gene pairs).
    :type maxPatterns: int

    Returns
    -------
    :return: 
        - patternColumns (numpy.ndarray): Encoded column patterns for each gene pair 
          (stored as decimal representation of binary masks).
        - correlations (numpy.ndarray): Average ensemble correlation score for each pattern.
          Values:
            - `>=0`: valid pattern
            - `-2`: optimization candidate (near threshold)
            - `-1`: rejected pattern
    :rtype: tuple (numpy.ndarray, numpy.ndarray)
    """
    patternColumns = np.full(maxPatterns, -1)
    correlations = np.full(maxPatterns, -1, dtype=np.float64)

    # Estimate a tolerance margin to flag patterns "near" the correlation thresholds
    margin_tolerance = _estimate_improvement_margin(data, rows, cols)
    
    # Use the strictest threshold among Pearson, Spearman, and NMI as the reference for "near-threshold" detection
    max_threshold = max(pearsonThreshold, spearmanThreshold, nmiThreshold)

    def process_pattern(iPattern):
        # Decode the gene pair corresponding to this pattern index
        r1, r2 = _getIdsFromPattern(iPattern, rows)
        vector1 = data[r1, :]
        vector2 = data[r2, :]
        
        # Compute the ensemble correlation for the two gene expression profiles
        validCorr, corr_val = bs.ensembleCorrelation(
            vector1, vector2,
            pearsonThreshold, spearmanThreshold, nmiThreshold
        )

        if validCorr and cols >= colMin:
            binaryVector = _indexBinary(np.arange(cols), cols)
            decimalColumns = _binaryToDecimal(binaryVector)
            return (iPattern, corr_val, decimalColumns)
        else:
            # Near-threshold case: pattern did not fully meet requirements, but might be improved
            if not np.isnan(corr_val) and corr_val >= (max_threshold - margin_tolerance):
                return (iPattern, -2, -2)  # near-threshold candidate
            else:
                return (iPattern, -1, -1)  # rejected
    
    # Parallel job
    results = Parallel(n_jobs=-1, backend="loky")(delayed(process_pattern)(iPattern) for iPattern in range(maxPatterns))

    for iPattern, corr_val, col_val in results:
        correlations[iPattern] = corr_val
        patternColumns[iPattern] = col_val

    return patternColumns, correlations

def _calculateOptimalCorrelations(patternColumns, correlations, data, rows, cols, pearsonThreshold, spearmanThreshold, nmiThreshold, colMin, maxPatterns):
    """
    Refines near-threshold column patterns by iteratively removing columns
    that negatively impact correlation scores.  
    This function is applied to patterns flagged as optimization candidates
    (with correlation value == -2) in `_calculateEnsembleCorrelations`.

    The algorithm:
    1. For each candidate pattern, evaluate the correlation for all columns included.
    2. Calculate the "impact" of removing each column on the correlation score.
    3. Iteratively remove the column(s) with the highest positive impact while
       ensuring the pattern still meets the `colMin` requirement.
    4. Accept the refined pattern if it meets the ensemble correlation thresholds.

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    In biclustering, some columns (conditions) can introduce noise and weaken
    correlation between genes that otherwise co-express strongly.  
    Removing such columns can:
    - Increase the coherence of the bicluster seed.
    - Improve the likelihood of growing it into a valid bicluster.
    
    This step allows patterns that were near the thresholds to become valid,
    instead of being discarded prematurely.

    Parameters
    ----------
    :param patternColumns: Encoded column patterns for each gene pair 
                           (stored as decimal representation of binary masks).
    :type patternColumns: numpy.ndarray
    :param correlations: Correlation values for each pattern. 
                         Values of `-2` indicate optimization candidates.
    :type correlations: numpy.ndarray
    :param data: Gene expression matrix (genes x conditions), without gene identifiers.
    :type data: numpy.ndarray
    :param rows: Total number of genes in the dataset.
    :type rows: int
    :param cols: Total number of conditions/samples in the dataset.
    :type cols: int
    :param pearsonThreshold: Minimum Pearson correlation coefficient for a pattern 
                             to be considered valid.
    :type pearsonThreshold: float
    :param spearmanThreshold: Minimum Spearman correlation coefficient for a pattern 
                              to be considered valid.
    :type spearmanThreshold: float
    :param nmiThreshold: Minimum Normalized Mutual Information (NMI) score for a 
                         pattern to be considered valid.
    :type nmiThreshold: float
    :param colMin: Minimum number of columns required for a valid pattern.
    :type colMin: int
    :param maxPatterns: Maximum number of patterns (equal to number of unique gene pairs).
    :type maxPatterns: int

    Returns
    -------
    :return:
        - patternColumns (numpy.ndarray): Updated column patterns, replacing optimized
          patterns with their refined versions.
        - correlations (numpy.ndarray): Updated correlation scores for all patterns.
          Invalid patterns are set to `-1`.
    :rtype: tuple (numpy.ndarray, numpy.ndarray)
    """

    max_removals = cols - colMin # Max columns that can be removed while keeping colMin
    def process_pattern(iPattern):
        corr = correlations[iPattern]
        if corr != -2: # Only optimize patterns flagged as "near-threshold" candidates
            return (iPattern, patternColumns[iPattern], correlations[iPattern])

        # Decode the gene pair for this pattern
        r1, r2 = _getIdsFromPattern(iPattern, rows)
        vector1 = data[r1, :]
        vector2 = data[r2, :]
        selectedColumns = np.arange(cols)
        
        # Compute correlation using all columns before any removal
        validCorr, corrFull = bs.ensembleCorrelation(vector1, vector2, pearsonThreshold, spearmanThreshold, nmiThreshold)

        for k in range(1, max_removals + 1): # k = number of columns to remove in each iteration
            impacts = []
            for col in selectedColumns:
                
                # Simulate removal of a single column
                temp_cols = selectedColumns[selectedColumns != col]
                if len(temp_cols) < colMin:
                    continue # Skip if removal violates colMin constraint

                # Correlation after removing this column
                _, corr_temp = bs.ensembleCorrelation(
                    vector1[temp_cols], vector2[temp_cols],
                    pearsonThreshold, spearmanThreshold, nmiThreshold
                )
                impact = corr_temp - corrFull # Impact = improvement (or deterioration) vs. full correlation
                impacts.append((col, impact))
                
            # Sort columns by impact (highest improvement first)
            impacts.sort(key=lambda x: (-x[1], x[0]))
            
            # Candidate columns to consider for removal
            candidates = [col for col, _ in impacts[:k+1]]
            
            # Try all combinations of k columns from the candidate list
            for subset_remove in combinations(candidates, k):
                # Columns to keep after removal
                subset_keep = [c for c in selectedColumns if c not in subset_remove]
                valid, corr_actual = bs.ensembleCorrelation(
                    vector1[subset_keep], vector2[subset_keep],
                    pearsonThreshold, spearmanThreshold, nmiThreshold
                )
                if valid:
                    # Store optimized pattern in decimal representation
                    binaryVector = _indexBinary(subset_keep, cols)
                    decimalColumns = _binaryToDecimal(binaryVector)
                    return (iPattern, decimalColumns, corr_actual) # Stop if a valid improved pattern is found
        
        # No improvement found -> reject the pattern
        return (iPattern, -1, -1)

    # Paralelización de los patrones
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_pattern)(iPattern) for iPattern in range(maxPatterns)
    )

    # Actualización de arrays originales
    for iPattern, newCol, newCorr in results:
        patternColumns[iPattern] = newCol
        correlations[iPattern] = newCorr

    return patternColumns, correlations
  
def _removePatternsDuplicated(patternColumns, correlations, rows, cols, maxPatterns):
    """
    Removes exact duplicate patterns (gene pair seeds) from the set of generated 
    bicluster candidates.  
    A pattern is considered a duplicate if it involves the same pair of genes 
    (regardless of order) and differs only in the number of selected columns.

    The algorithm:
    1. For each pattern, generate a key based on the two genes in the pair.
    2. If the same gene pair appears multiple times, keep only the one with 
       the largest number of selected columns.
    3. Mark all other duplicates as invalid (`-1` correlation and column pattern).

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    Duplicate patterns can occur due to multiple equivalent binary representations 
    of the same gene pair or when patterns differ only in non-significant columns.
    Keeping multiple versions of the same gene pair:
    - Inflates the total number of patterns without adding new information.
    - Wastes computational resources in later bicluster construction stages.
    
    By keeping only the most "informative" version (largest number of columns), 
    we ensure:
    - Higher coverage of the condition space.
    - Reduced redundancy in subsequent steps.

    Parameters
    ----------
    :param patternColumns: Encoded column patterns for each gene pair 
                           (stored as decimal representation of binary masks).
    :type patternColumns: numpy.ndarray
    :param correlations: Correlation values for each pattern.
    :type correlations: numpy.ndarray
    :param rows: Total number of genes in the dataset.
    :type rows: int
    :param cols: Total number of conditions/samples in the dataset.
    :type cols: int
    :param maxPatterns: Maximum number of patterns (equal to number of unique gene pairs).
    :type maxPatterns: int

    Returns
    -------
    :return:
        - patternColumns (numpy.ndarray): Updated column patterns with duplicates removed.
        - correlations (numpy.ndarray): Updated correlations with duplicates set to `-1`.
    :rtype: tuple (numpy.ndarray, numpy.ndarray)
    """
    seen = {}

    for iPattern in range(maxPatterns):
        correlation_val = correlations[iPattern]
        if correlation_val == -1:
            continue # Skip already invalid patterns
        
        # Decode the gene pair for this pattern index
        r1, r2 = _getIdsFromPattern(iPattern, rows)
        # Use frozenset so that order of genes does not matter (A,B == B,A)
        genes_set = frozenset([r1, r2])
        
        # Count the number of active columns for this pattern
        num_cols = len(_getColumns(_decimalToBinary(patternColumns[iPattern], cols))) \
                   if patternColumns[iPattern] != -1 else 0

        if genes_set in seen:
            # We already have a pattern for this gene pair
            existing_idx, existing_cols = seen[genes_set]
            if num_cols > existing_cols:
                # If current pattern has more active columns -> keep it, discard previous one
                correlations[existing_idx] = -1
                patternColumns[existing_idx] = -1
                seen[genes_set] = (iPattern, num_cols)
            else:
                # If previous pattern had more or equal columns -> discard current one
                correlations[iPattern] = -1
                patternColumns[iPattern] = -1
        else:
            # First time seeing this gene pair -> store it
            seen[genes_set] = (iPattern, num_cols)
            
    return patternColumns, correlations


###################
# 2.2. BICLUSTERS #
###################

def _getBiclusters(data, patternColumns, correlations, rows, cols, tolerancyThreshold,
                   pearsonThreshold, spearmanThreshold, nmiThreshold, maxPatterns, geneMin): 
    """
    Constructs full biclusters from the initial set of column patterns (gene pair seeds).
    This function:
    1. Expands each valid pattern by adding compatible genes that meet the correlation criteria.
    2. Performs a tolerance-based second pass to include genes that are slightly below
       the main threshold but above the tolerance level.
    3. Recalculates bicluster correlation after adding new genes.
    4. Removes exact duplicate biclusters and optionally merges highly overlapping ones
       while preserving correlation quality.

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    Biclustering aims to find subsets of genes that co-express under specific subsets 
    of conditions. Starting from highly correlated gene pairs:
    - Step 1 (strict growth) ensures that added genes match or exceed the seed correlation,
      maintaining strong coherence.
    - Step 2 (tolerance growth) allows inclusion of borderline genes that may carry 
      biological relevance but were excluded due to noise or small deviations.
    - Merging overlapping biclusters prevents fragmentation and improves interpretability,
      while the correlation loss threshold ensures merged biclusters remain meaningful.

    Parameters
    ----------
    :param data: Gene expression matrix (genes x conditions), without gene identifiers.
    :type data: numpy.ndarray
    :param patternColumns: Encoded column patterns for each gene pair 
                           (stored as decimal representation of binary masks).
    :type patternColumns: numpy.ndarray
    :param correlations: Correlation values for each pattern.
    :type correlations: numpy.ndarray
    :param rows: Total number of genes in the dataset.
    :type rows: int
    :param cols: Total number of conditions/samples in the dataset.
    :type cols: int
    :param tolerancyThreshold: Allowed tolerance (decimal fraction) for adding genes 
                               to a bicluster during the growth phase.
    :type tolerancyThreshold: float
    :param pearsonThreshold: Minimum Pearson correlation coefficient required for gene inclusion.
    :type pearsonThreshold: float
    :param spearmanThreshold: Minimum Spearman correlation coefficient required for gene inclusion.
    :type spearmanThreshold: float
    :param nmiThreshold: Minimum Normalized Mutual Information (NMI) score required for gene inclusion.
    :type nmiThreshold: float
    :param maxPatterns: Maximum number of patterns (equal to number of unique gene pairs).
    :type maxPatterns: int
    :param geneMin: Minimum number of genes required for a valid bicluster.
    :type geneMin: int

    Returns
    -------
    :return:
        - biclustersRows (list): List of numpy arrays containing indices of genes in each bicluster.
        - biclustersCols (list): List of numpy arrays containing indices of columns in each bicluster.
        - biclustersCorrelation (list): List of average correlation scores for each bicluster.
        - numBiclusters (int): Total number of valid biclusters constructed.
    :rtype: tuple (list, list, list, int)
    """
    biclustersRows = np.full(maxPatterns, -1, dtype=object)
    biclustersCols = np.full(maxPatterns, -1, dtype=object)
    biclustersCorrelation = np.full(maxPatterns, -1.0, dtype=np.float64) 
    def process_pattern(iPattern):
        if correlations[iPattern] == -1:
            return (iPattern, -1, -1, -1.0)  # patrón inválido

        r1, r2 = _getIdsFromPattern(iPattern, rows)
        candidateRows, toleranceRows = [], []
        
        # Selected columns
        biclusterCols = _getColumns(_decimalToBinary(patternColumns[iPattern], cols))
        
        # Tolerance
        tolerance_val = correlations[iPattern] - (correlations[iPattern] * tolerancyThreshold)

        vec_r1 = data[r1, biclusterCols]
        vec_r2 = data[r2, biclusterCols]

        for iRow in range(rows):
            if iRow == r1 or iRow == r2:
                continue

            vec_i = data[iRow, biclusterCols]

            validCorr1, corr1 = bs.ensembleCorrelation(
                vec_r1, vec_i,
                pearsonThreshold, spearmanThreshold, nmiThreshold
            )                    
            validCorr2, corr2 = bs.ensembleCorrelation(
                vec_r2, vec_i,
                pearsonThreshold, spearmanThreshold, nmiThreshold
            )                    

            if validCorr1 and validCorr2:
                avg_corr = (corr1 + corr2) / 2.0
                if avg_corr >= correlations[iPattern]:
                    candidateRows.append(iRow)
                elif avg_corr >= tolerance_val:
                    toleranceRows.append(iRow)

        # Phase 1: add rows via competition
        biclusterRows, biclusterCorr = _getBiclustersPhase1(
            iPattern, r1, r2, data,
            pearsonThreshold, spearmanThreshold, nmiThreshold,
            candidateRows, biclusterCols, correlations
        )

        # Phase 2: evaluate tolerance rows
        biclusterRows, biclusterCorr = _getBiclustersPhase2(
            data,
            pearsonThreshold, spearmanThreshold, nmiThreshold,
            toleranceRows, biclusterRows, biclusterCols, biclusterCorr
        )
        
        # Save if valid
        if len(biclusterRows) >= geneMin:
            return (iPattern, biclusterRows, biclusterCols, biclusterCorr)
        else:
            return (iPattern, -1, -1, -1)
    
    # Parallel job
    results = Parallel(n_jobs=-1, backend="loky")(delayed(process_pattern)(iPattern) for iPattern in range(maxPatterns)
    )

    # Parallel results
    numBiclusters = 0
    for iPattern, rowsOut, colsOut, corrOut in results:
        if isinstance(rowsOut, np.ndarray):
            biclustersRows[iPattern] = rowsOut
            biclustersCols[iPattern] = colsOut
            biclustersCorrelation[iPattern] = corrOut
            numBiclusters += 1
        else:
            biclustersRows[iPattern] = -1
            biclustersCols[iPattern] = -1
            biclustersCorrelation[iPattern] = -1
            
    biclustersRows, biclustersCols, biclustersCorrelation = _removeEmptyandDuplicateBiclusters(biclustersRows, biclustersCols, biclustersCorrelation)
    biclustersRows, biclustersCols, biclustersCorrelation = _detectOverlapping(data, pearsonThreshold, spearmanThreshold, nmiThreshold, biclustersRows, biclustersCols, biclustersCorrelation)
    return biclustersRows, biclustersCols, biclustersCorrelation, numBiclusters

def _getBiclustersPhase1(iPattern, r1, r2, data, pearsonThreshold, spearmanThreshold, nmiThreshold, candidateRows, biclusterCols, correlations):
    """
    Executes the first growth phase of bicluster construction by expanding the 
    initial gene pair (pattern seed) with strictly compatible genes.

    This phase:
    1. Applies a **competition-based selection** (`_rowCompetition`) to the set of 
       candidate genes that meet or exceed the correlation threshold with both 
       seed genes.
    2. Ensures that all added genes are mutually consistent with each other and 
       with the original seed pair under the current bicluster’s column pattern.
    3. Appends the original seed genes (`r1`, `r2`) to the final bicluster row set.
    4. Recalculates the bicluster’s average correlation if additional genes were added.

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    Starting from a highly correlated gene pair, this phase focuses on **strict 
    inclusion**:
    - Only genes that fully satisfy the correlation thresholds with *all* current 
      bicluster members are accepted.
    - This conservative approach guarantees high internal coherence, avoiding 
      premature inclusion of borderline genes (handled later in Phase 2).
    
    By recalculating the average correlation after expansion, the algorithm ensures 
    that the bicluster’s quality metric reflects the actual set of included genes.

    Parameters
    ----------
    :param iPattern: Index of the current gene pair pattern being expanded.
    :type iPattern: int
    :param r1: Index of the first gene in the original pattern seed.
    :type r1: int
    :param r2: Index of the second gene in the original pattern seed.
    :type r2: int
    :param data: Gene expression matrix (genes x conditions), without gene identifiers.
    :type data: numpy.ndarray
    :param pearsonThreshold: Minimum Pearson correlation coefficient required for inclusion.
    :type pearsonThreshold: float
    :param spearmanThreshold: Minimum Spearman correlation coefficient required for inclusion.
    :type spearmanThreshold: float
    :param nmiThreshold: Minimum Normalized Mutual Information (NMI) score required for inclusion.
    :type nmiThreshold: float
    :param candidateRows: List of row indices representing genes that passed the 
                          initial correlation check with both seed genes.
    :type candidateRows: list[int]
    :param biclusterCols: List or array of column indices defining the bicluster’s 
                          condition subset.
    :type biclusterCols: list[int] or numpy.ndarray
    :param correlations: Array containing correlation values for all patterns.
    :type correlations: numpy.ndarray

    Returns
    -------
    :return:
        - biclusterRows (numpy.ndarray): Indices of genes included in the bicluster 
          after Phase 1 expansion.
        - biclusterCorr (float): Average ensemble correlation of the bicluster 
          after Phase 1.
    :rtype: tuple (numpy.ndarray, float)
    """
    biclusterRows = _rowCompetition(candidateRows, biclusterCols, data, correlations[iPattern], pearsonThreshold, spearmanThreshold, nmiThreshold) 
            
    # Add the two original rows
    biclusterRows = np.append(biclusterRows, [r1, r2])
    
    # Recalculate correlation if new rows were added
    biclusterCorr = correlations[iPattern]       
    if len(biclusterRows) > 2:
        biclusterCorr = _autoCalculateCorrelation(biclusterRows, biclusterCols, data, pearsonThreshold, spearmanThreshold, nmiThreshold)
    
    return biclusterRows, biclusterCorr
    
def _getBiclustersPhase2(data, pearsonThreshold, spearmanThreshold, nmiThreshold, toleranceRows, biclusterRows, biclusterCols, biclusterCorr):
    """
    Executes the second growth phase of bicluster construction by incorporating 
    **tolerance-phase genes** — those whose correlation with the current bicluster 
    members is slightly below the main threshold but above the tolerance level.

    This phase:
    1. Applies a **competition-based selection** (`_rowCompetition`) to 
       `toleranceRows` to ensure that all newly added genes are mutually consistent 
       and do not significantly reduce bicluster coherence.
    2. Appends all tolerance-phase genes that survive competition to the bicluster.
    3. Recalculates the bicluster’s average correlation score to reflect the updated 
       set of genes.

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    While Phase 1 ensures strict inclusion of only highly correlated genes, this 
    phase provides **controlled flexibility**:
    - Biological data often contain genes that are functionally relevant but fall 
      just short of strict correlation thresholds due to noise or experimental 
      variability.
    - By allowing such genes to be added if they meet a tolerance-adjusted correlation 
      level and pass mutual consistency checks, the algorithm can recover biologically 
      meaningful biclusters that would otherwise be too small or fragmented.

    The competition mechanism ensures that the inclusion of these borderline genes 
    does not introduce internal inconsistencies that could compromise interpretability.

    Parameters
    ----------
    :param data: Gene expression matrix (genes x conditions), without gene identifiers.
    :type data: numpy.ndarray
    :param pearsonThreshold: Minimum Pearson correlation coefficient required for validation.
    :type pearsonThreshold: float
    :param spearmanThreshold: Minimum Spearman correlation coefficient required for validation.
    :type spearmanThreshold: float
    :param nmiThreshold: Minimum Normalized Mutual Information (NMI) score required for validation.
    :type nmiThreshold: float
    :param toleranceRows: List of gene indices that passed the tolerance-level correlation 
                          check in Phase 1 but not the strict threshold.
    :type toleranceRows: list[int]
    :param biclusterRows: Current array of gene indices in the bicluster after Phase 1.
    :type biclusterRows: numpy.ndarray
    :param biclusterCols: Array of column indices defining the bicluster’s condition subset.
    :type biclusterCols: numpy.ndarray
    :param biclusterCorr: Average ensemble correlation of the bicluster after Phase 1.
    :type biclusterCorr: float

    Returns
    -------
    :return:
        - biclusterRows (numpy.ndarray): Updated array of gene indices after adding tolerance-phase genes.
        - biclusterCorr (float): Updated average correlation score after Phase 2 expansion.
    :rtype: tuple (numpy.ndarray, float)
    """
    if len(toleranceRows) > 0:
        resultingToleranceRows = _rowCompetition(
            toleranceRows, biclusterCols, data,
            biclusterCorr, pearsonThreshold, spearmanThreshold, nmiThreshold
        )   
                        
        if len(resultingToleranceRows) > 0:
            for f in resultingToleranceRows:                        
                biclusterRows = np.append(biclusterRows, f)
                            
            biclusterCorr = _autoCalculateCorrelation(
                biclusterRows, biclusterCols, data,
                pearsonThreshold, spearmanThreshold, nmiThreshold
            )
    return biclusterRows, biclusterCorr
    
def _rowCompetition(candidateRows, selectedCols, data, patternCorrelation, pearsonThreshold, spearmanThreshold, nmiThreshold): 
    """
    Selects a subset of candidate genes (rows) for inclusion in a bicluster 
    using a pairwise competition mechanism.  
    The idea is to compare each candidate gene against all others and discard
    those whose correlation with at least one other candidate falls below the 
    target bicluster correlation threshold.

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    In biclustering, simply adding all genes that meet the correlation threshold 
    with the seed genes can result in internal inconsistency:  
    - A gene might correlate strongly with the seed genes but poorly with other 
      members of the bicluster.
    - Including such a gene can reduce overall coherence.

    The competition phase ensures:
    - Every gene in the bicluster is mutually consistent with others.
    - Internal average correlation remains high, preserving biological interpretability.

    Parameters
    ----------
    :param candidateRows: List of row indices representing candidate genes for inclusion.
    :type candidateRows: list[int]
    :param selectedCols: List or array of column indices representing the subset of conditions 
                         defining the bicluster pattern.
    :type selectedCols: list[int] or numpy.ndarray
    :param data: Gene expression matrix (genes x conditions), without gene identifiers.
    :type data: numpy.ndarray
    :param patternCorrelation: Minimum average correlation score required for inclusion 
                                in the bicluster.
    :type patternCorrelation: float
    :param pearsonThreshold: Minimum Pearson correlation coefficient required for validation.
    :type pearsonThreshold: float
    :param spearmanThreshold: Minimum Spearman correlation coefficient required for validation.
    :type spearmanThreshold: float
    :param nmiThreshold: Minimum Normalized Mutual Information (NMI) score required for validation.
    :type nmiThreshold: float

    Returns
    -------
    :return: Array of selected row indices (genes) that passed the competition phase.
    :rtype: numpy.ndarray
    """
    resultingRows = []
    forbiddenRows = set()
    
    for i in range(len(candidateRows)):        
        currentRow = candidateRows[i]
        discarded = False
        j = 0
        while j < len(candidateRows) and discarded is False:            
            if i != j and candidateRows[j] not in forbiddenRows:                                
                validCorr, correlation_val = bs.ensembleCorrelation(
                    data[currentRow][selectedCols],
                    data[candidateRows[j]][selectedCols],
                    pearsonThreshold, spearmanThreshold, nmiThreshold
                )                                    
                if correlation_val < patternCorrelation:  # Discard the row in competition
                    discarded = True
                    forbiddenRows.add(currentRow)
            j += 1
                    
        if discarded is False:
            resultingRows.append(currentRow)  # Add the row after the competition        
    
    return np.array(resultingRows, dtype=np.int64)

def _autoCalculateCorrelation(biclusterRows, biclusterCols, data, pearsonThreshold, spearmanThreshold, nmiThreshold):
    """
    Calculates the average pairwise ensemble correlation score for all 
    gene pairs within a bicluster, considering only the selected columns 
    (conditions) that define the bicluster.

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    The coherence of a bicluster is determined by the degree to which 
    its member genes co-express under the subset of conditions defining it.
    This function:
    - Iterates over all unique gene pairs in the bicluster.
    - Calculates an ensemble correlation (Pearson, Spearman, NMI) for each pair.
    - Returns the average correlation, which serves as a quality metric.

    The average correlation is:
    - Used for deciding whether to include tolerance-phase genes.
    - Useful for merging overlapping biclusters without losing quality.
    - A core measure in evaluating stability and robustness.

    Parameters
    ----------
    :param biclusterRows: List or array of row indices (genes) in the bicluster.
    :type biclusterRows: list[int] or numpy.ndarray
    :param biclusterCols: List or array of column indices (conditions) defining the bicluster.
    :type biclusterCols: list[int] or numpy.ndarray
    :param data: Gene expression matrix (genes x conditions), without gene identifiers.
    :type data: numpy.ndarray
    :param pearsonThreshold: Minimum Pearson correlation coefficient for validation.
    :type pearsonThreshold: float
    :param spearmanThreshold: Minimum Spearman correlation coefficient for validation.
    :type spearmanThreshold: float
    :param nmiThreshold: Minimum Normalized Mutual Information (NMI) score for validation.
    :type nmiThreshold: float

    Returns
    -------
    :return: Average ensemble correlation score for all valid gene pairs in the bicluster.
             Returns `-1` if no valid pairs were found.
    :rtype: float
    """ 
    correlations = []
    totalGenePairs = 0

    # Generate all combinations of pairs of rows (genes)
    for row1, row2 in combinations(biclusterRows, 2):
        totalGenePairs += 1
        
        # Extract the values of the selected columns for both rows
        values1 = np.array(data[row1])[biclusterCols]
        values2 = np.array(data[row2])[biclusterCols]

        # Calculate Pearson/Spearman/NMI correlation between both rows
        validCorr, correlation_val = bs.ensembleCorrelation(values1, values2, pearsonThreshold, spearmanThreshold, nmiThreshold)
        
        if validCorr:
            correlations.append(correlation_val)

    # Return the average of the correlations
    if not correlations:
        return -1
    
    return float(np.mean(correlations))

def _removeEmptyandDuplicateBiclusters(biclustersRows, biclustersCols, biclustersCorrelation):
    """
    Cleans the list of constructed biclusters by:
    1. Removing **empty** entries (placeholders with value `-1`).
    2. Eliminating **duplicate biclusters** that contain the same set of genes, 
       keeping only the one with the highest correlation score.

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    During bicluster generation, intermediate placeholders and duplicates may be 
    introduced:
    - Empty placeholders (`-1`) represent patterns that failed the validation 
      criteria or were discarded during processing.
    - Duplicate biclusters can arise when different seed patterns lead to the same 
      final gene set, often with minor variations in correlation score.

    Retaining duplicates:
    - Artificially inflates the bicluster count.
    - Wastes computation time in downstream analysis.
    - Increases redundancy in biological interpretation.

    By removing empty entries and keeping only the **highest-quality** version 
    (maximum correlation) of each unique gene set, we ensure:
    - Non-redundant, clean results.
    - Higher reliability in biological interpretation.
    - Reduced memory footprint for storing final results.

    Parameters
    ----------
    :param biclustersRows: List of arrays representing the gene indices for each bicluster.
                           Empty/invalid entries are stored as `-1`.
    :type biclustersRows: list[numpy.ndarray or int]
    :param biclustersCols: List of arrays representing the condition indices for each bicluster.
                           Empty/invalid entries are stored as `-1`.
    :type biclustersCols: list[numpy.ndarray or int]
    :param biclustersCorrelation: List of average correlation scores for each bicluster.
                                  Invalid entries are stored as `-1`.
    :type biclustersCorrelation: list[float or numpy.float64]

    Returns
    -------
    :return:
        - biclustersRows (list): Cleaned list of bicluster gene sets (no empties, no duplicates).
        - biclustersCols (list): Corresponding cleaned list of bicluster condition sets.
        - biclustersCorrelation (list): Corresponding cleaned list of correlation scores.
    :rtype: tuple (list, list, list)
    """
    biclustersRows = [elem for elem in biclustersRows if not (isinstance(elem, int) and elem == -1)]
    biclustersCols = [elem for elem in biclustersCols if not (isinstance(elem, int) and elem == -1)]
    biclustersCorrelation = [elem for elem in biclustersCorrelation if not (isinstance(elem, np.float64) and elem == -1)] 
    
    seen = {}
    for i, (genes, corr) in enumerate(zip(biclustersRows, biclustersCorrelation)):
        gene_key = frozenset(genes)
        if gene_key in seen:
            existing_idx, existing_corr = seen[gene_key]
            if corr > existing_corr:
                seen[gene_key] = (i, corr)
        else:
            seen[gene_key] = (i, corr)
    final_indices = sorted(idx for idx, _ in seen.values())
    biclustersRows = [biclustersRows[i] for i in final_indices]
    biclustersCols = [biclustersCols[i] for i in final_indices]
    biclustersCorrelation = [biclustersCorrelation[i] for i in final_indices]
    return biclustersRows, biclustersCols, biclustersCorrelation

def _detectOverlapping(data, pearsonThreshold, spearmanThreshold, nmiThreshold, biclustersRows, biclustersCols, biclustersCorrelation):
    """
    Detects and resolves high-overlap cases between biclusters, either by:
    1. **Merging** highly overlapping biclusters into a single one (if the 
       merged correlation quality is acceptable).
    2. **Removing** the lower-quality bicluster when merging would cause 
       excessive loss in correlation.

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    Multiple biclusters may share a large proportion of genes, especially 
    when derived from similar seed patterns.  
    - Excessive overlap can reduce interpretability, as nearly identical 
      biclusters do not contribute new biological information.
    - Merging such biclusters can yield more complete gene modules, provided 
      that correlation quality is preserved.

    The algorithm:
    - Uses **Jaccard similarity** to measure the proportion of shared genes 
      between two biclusters.
    - If the overlap exceeds a predefined fusion threshold (`fusion_threshold`), 
      attempts to merge them into a single bicluster.
    - Calculates the average correlation of the merged bicluster using 
      `_autoCalculateCorrelation`.
    - Accepts the merge only if the correlation loss compared to the 
      best original bicluster is below a maximum allowed ratio (`max_loss_ratio`).
    - If merging is not acceptable, removes the bicluster with the lower 
      correlation score.

    Parameters
    ----------
    :param data: Gene expression matrix (genes x conditions), without gene identifiers.
    :type data: numpy.ndarray
    :param pearsonThreshold: Minimum Pearson correlation coefficient for validation.
    :type pearsonThreshold: float
    :param spearmanThreshold: Minimum Spearman correlation coefficient for validation.
    :type spearmanThreshold: float
    :param nmiThreshold: Minimum Normalized Mutual Information (NMI) score for validation.
    :type nmiThreshold: float
    :param biclustersRows: List of arrays containing gene indices for each bicluster.
    :type biclustersRows: list[numpy.ndarray]
    :param biclustersCols: List of arrays containing condition indices for each bicluster.
    :type biclustersCols: list[numpy.ndarray]
    :param biclustersCorrelation: List of average correlation scores for each bicluster.
    :type biclustersCorrelation: list[float]

    Returns
    -------
    :return:
        - biclustersRows (list): Updated list of bicluster gene sets after overlap resolution.
        - biclustersCols (list): Updated list of bicluster condition sets.
        - biclustersCorrelation (list): Updated list of bicluster correlation scores.
    :rtype: tuple (list, list, list)
    """
    def jaccard(a, b):
        inter = len(set(a) & set(b))
        union = len(set(a) | set(b))
        return inter / union if union > 0 else 0

    fusion_threshold = 0.9   # Overlap threshold for considering fusion
    max_loss_ratio = 0.05    # Maximum acceptable correlation loss (5%)

    i = 0
    while i < len(biclustersRows):
        j = i + 1
        while j < len(biclustersRows):
            overlap = jaccard(biclustersRows[i], biclustersRows[j])
            if overlap >= fusion_threshold:
                # Merge genes (columns remain the same)
                merged_genes = np.unique(np.concatenate([biclustersRows[i], biclustersRows[j]]))
                merged_cols = biclustersCols[i]

                # Merged correlation
                merged_corr = _autoCalculateCorrelation(
                    merged_genes, merged_cols, data,
                    pearsonThreshold, spearmanThreshold, nmiThreshold
                )
                max_original_corr = max(biclustersCorrelation[i], biclustersCorrelation[j])

                if merged_corr >= (1 - max_loss_ratio) * max_original_corr:
                    biclustersRows[i] = merged_genes
                    biclustersCorrelation[i] = merged_corr
                    del biclustersRows[j]
                    del biclustersCols[j]
                    del biclustersCorrelation[j]
                    continue
                else:
                    if biclustersCorrelation[i] >= biclustersCorrelation[j]:
                        del biclustersRows[j]
                        del biclustersCols[j]
                        del biclustersCorrelation[j]
                        continue
                    else:
                        del biclustersRows[i]
                        del biclustersCols[i]
                        del biclustersCorrelation[i]
                        i -= 1
                        break
            j += 1
        i += 1
    
    return biclustersRows, biclustersCols, biclustersCorrelation


#######################
# PHASE 3: EVALUATION #
#######################
def _evaluateStability(data, genes, conditions, rows, cols, pearson_thr, spearman_thr, nmi_thr, colMin, geneMin, tolerancyThreshold=10, num_runs=20):
    """
    Evaluates the stability of the biclustering results by introducing controlled
    perturbations to the dataset and measuring the similarity of biclusters across runs.  
    This is done via:
    1. Running the biclustering algorithm on the original dataset to obtain reference biclusters.
    2. Creating perturbed datasets by adding Gaussian noise and randomly subsampling columns.
    3. Matching perturbed-run biclusters to reference biclusters based on Jaccard similarity 
       and core gene overlap.
    4. Reporting the average stability score across multiple perturbation runs.

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    Biclustering stability is a critical quality indicator:
    - A high stability score suggests that the biclusters are robust to data noise 
      and sampling variability, which is desirable in biological applications where 
      datasets can be noisy or incomplete.
    - A low stability score may indicate overfitting or high sensitivity to specific conditions.

    Stability is quantified using:
    - **Jaccard similarity**: Measures the proportion of shared genes between 
      a reference bicluster and its closest match in a perturbed run.
    - **Core gene ratio**: Ensures that at least a certain proportion of core genes 
      are preserved in the perturbed version.

    Parameters
    ----------
    :param data: Gene expression matrix (genes x conditions), without gene identifiers.
    :type data: numpy.ndarray
    :param genes: Array of gene identifiers corresponding to the rows of `data`.
    :type genes: numpy.ndarray
    :param conditions: Array of condition/sample identifiers corresponding to the columns of `data`.
    :type conditions: numpy.ndarray
    :param rows: Total number of genes in the dataset.
    :type rows: int
    :param cols: Total number of conditions/samples in the dataset.
    :type cols: int
    :param pearson_thr: Pearson correlation threshold used in the reference run.
    :type pearson_thr: float
    :param spearman_thr: Spearman correlation threshold used in the reference run.
    :type spearman_thr: float
    :param nmi_thr: NMI threshold used in the reference run.
    :type nmi_thr: float
    :param colMin: Minimum number of columns required for a valid bicluster.
    :type colMin: int
    :param geneMin: Minimum number of genes required for a valid bicluster.
    :type geneMin: int
    :param tolerancyThreshold: Tolerance (decimal fraction) for adding genes during bicluster growth.
    :type tolerancyThreshold: float
    :param num_runs: Number of perturbation runs to perform.
    :type num_runs: int
    :param noise_std: Standard deviation of Gaussian noise added to perturb the data.
    :type noise_std: float
    :param subsample_ratio: Fraction of columns randomly retained in each perturbed run.
    :type subsample_ratio: float
    :param min_genes_stability: Minimum number of genes required for a bicluster to be considered in stability evaluation.
    :type min_genes_stability: int
    :param core_threshold: Minimum proportion of reference genes that must be preserved in the matched bicluster.
    :type core_threshold: float
    :param size_tolerance: Maximum allowed relative size difference between reference and matched biclusters.
    :type size_tolerance: float

    Returns
    -------
    :return: Average Jaccard stability score across all perturbation runs.
             Values closer to 1 indicate higher stability; values closer to 0 indicate low stability.
    :rtype: float
    """
    
    noise_std=0.001
    subsample_ratio=0.95
    min_genes_stability=2
    core_threshold=0.65
    size_tolerance=0.25
    
    # 1) Reference biclusters
    _, biclustersRows_ref, _, _ = _executeAlgorithm(
        data, genes, conditions, rows, cols, tolerancyThreshold,
        pearson_thr, spearman_thr, nmi_thr,
        colMin, geneMin, False, False, set()
    )

    # Filter out small biclusters
    biclustersRows_ref = [
        b for b in biclustersRows_ref
        if not isinstance(b, (int, np.integer)) and len(b) >= min_genes_stability
    ]

    if not biclustersRows_ref:
        print("[EVALUATION] No large biclusters in reference.")
        return 0

    gene_occurrences = {tuple(b): [] for b in biclustersRows_ref}

    jaccard_scores = []

    for run in range(num_runs):
        data_perturbed = deepcopy(data)

        # Add Gaussian noise
        data_perturbed += np.random.normal(0, noise_std, data_perturbed.shape)

        # Column subsampling
        selected_cols = np.random.choice(cols, size=int(cols * subsample_ratio), replace=False)
        data_perturbed = data_perturbed[:, selected_cols]
        conditions_sub = conditions[selected_cols]
        colMin_sub = max(1, int(colMin * subsample_ratio))

        # New biclusters from perturbed data
        _, biclustersRows_new, _, _ = _executeAlgorithm(
            data_perturbed, genes, conditions_sub, rows, len(selected_cols), tolerancyThreshold,
            pearson_thr, spearman_thr, nmi_thr,
            colMin_sub, geneMin, False, False, set()
        )

        # Filter out small biclusters
        biclustersRows_new = [
            b for b in biclustersRows_new
            if not isinstance(b, (int, np.integer)) and len(b) >= min_genes_stability
        ]

        # Match each reference bicluster to the most similar one in the perturbed run
        for b_ref in biclustersRows_ref:
            max_score = 0
            size_ref = len(b_ref)
            for b_new in biclustersRows_new:
                size_new = len(b_new)
                size_diff_ratio = abs(size_ref - size_new) / size_ref
                if size_diff_ratio > size_tolerance:
                    continue

                common_genes = set(b_ref) & set(b_new)
                jaccard_val = len(common_genes) / len(set(b_ref) | set(b_new))
                core_ratio = len(common_genes) / size_ref

                if core_ratio >= core_threshold:
                    jaccard_val = max(jaccard_val, core_ratio)

                if jaccard_val > max_score:
                    max_score = jaccard_val

            jaccard_scores.append(max_score)

    stability_average = np.mean(jaccard_scores) if jaccard_scores else 0

    return stability_average


######################
# AUXILIAR FUNCTIONS #
######################

def _estimate_improvement_margin(data, n_rows, n_cols, confidence_factor=2):
    """
    Estimate the typical correlation improvement margin when removing a single column
    from a large gene-expression dataset.

    This function identifies the subset of rows (genes) and columns (conditions/patients)
    with the highest variance, as these are statistically the most likely to cause significant
    changes in correlation when removed. The margin is computed as the standard deviation of
    these changes, multiplied by a confidence factor (default 2 = 95% confidence interval).

    Scientific rationale:
    ---------------------
    - **Column selection**: The top 10% most variable columns are considered, as these tend to
      concentrate the majority of the impact on correlation metrics. A minimum of 5 columns is enforced
      to ensure representativeness in smaller datasets.
    - **Row selection**: The top 5% most variable genes are selected, as gene expression datasets typically
      have a skewed variance distribution where only a minority of genes show high variability.
      A minimum of 10 genes is enforced to ensure diversity of gene pairs.
    - **Confidence factor**: Multiplying the standard deviation by 2 corresponds to an approximate 95%
      confidence interval for normally distributed variations, ensuring that the margin covers nearly all
      realistic fluctuation scenarios in the dataset.

    Parameters
    ----------
    data : np.ndarray
        Gene-expression matrix (rows = genes, columns = conditions/patients).
    n_rows : int
        Number of genes in the dataset.
    n_cols : int
        Number of columns (conditions/patients) in the dataset.
    confidence_factor : float, optional
        Multiplier for the standard deviation (default = 2, ~95% confidence).

    Returns
    -------
    float
        Estimated tolerance margin for correlation improvement.
    """
    deltas = []

    # 1. Select critical columns by variance
    if n_cols <= 50:
        cols_criticas = np.arange(n_cols)
    else:
        num_top_cols = max(5, n_cols // 10)  # 10% or minimum 5
        col_vars = np.var(data, axis=0)
        cols_criticas = np.argsort(col_vars)[-num_top_cols:]

    # 2. Select critical genes by variance
    if n_rows <= 200:
        genes_criticos = np.arange(n_rows)
    else:
        num_top_genes = max(10, n_rows // 20)  # 5% or minimum 10
        row_vars = np.var(data, axis=1)
        genes_criticos = np.argsort(row_vars)[-num_top_genes:]

    # 3. Measure correlation changes upon column removal in the critical subset
    for i_idx in range(len(genes_criticos)):
        for j_idx in range(i_idx + 1, len(genes_criticos)):
            r1, r2 = genes_criticos[i_idx], genes_criticos[j_idx]
            v1, v2 = data[r1, :], data[r2, :]

            full_corr = np.corrcoef(v1, v2)[0, 1]
            for col in cols_criticas:
                reduced_corr = np.corrcoef(
                    np.delete(v1, col),
                    np.delete(v2, col)
                )[0, 1]
                if not np.isnan(full_corr) and not np.isnan(reduced_corr):
                    deltas.append(abs(reduced_corr - full_corr))

    if len(deltas) == 0:
        return 0.0

    # 4. Return margin (standard deviation * confidence factor)
    return confidence_factor * np.std(deltas)

def _indexBinary(indice, columnas):
    """
    Converts a list of selected column indices into a binary mask representation.  
    Each position in the binary mask corresponds to a column:
    - `1` indicates that the column is included.
    - `0` indicates that the column is excluded.

    This binary mask is used to represent column patterns compactly and is later 
    converted into a decimal number for efficient storage and comparison.

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    In biclustering, it is common to work with subsets of columns (conditions) 
    that define each bicluster.  
    Representing these subsets as binary masks allows:
    - Fast storage (by converting to decimal form).
    - Easy comparison between patterns.
    - Reduced memory usage for large datasets.
    
    This function is a low-level utility used in:
    - `_calculateEnsembleCorrelations`
    - `_calculateOptimalCorrelations`
    - Any step where column subsets must be encoded or compared.

    Parameters
    ----------
    :param indices: Indices of the columns that should be marked as `1` in the binary mask.
    :type indices: list[int] or numpy.ndarray
    :param totalLength: Total number of columns in the dataset (length of the binary mask).
    :type totalLength: int

    Returns
    -------
    :return: Binary mask as a numpy array of integers (0s and 1s), length = `totalLength`.
    :rtype: numpy.ndarray
    """
    vector_binario = None
    if indice is not None:
        vector_binario = [0] * columnas

        # Mark with ‘1’ the indicated positions
        for idx in indice:
            vector_binario[idx] = 1 
        
    return vector_binario    
        
def _binaryToDecimal(vector): 
    """
    Converts a binary mask (array of 0s and 1s) into its decimal representation.  
    This is used to store column patterns compactly and compare them efficiently.

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    In the CoBiNet biclustering algorithm, each bicluster pattern is defined by 
    a specific subset of columns (conditions).  
    - Representing this subset as a binary vector allows easy visualization of 
      included/excluded columns.
    - Converting the binary vector into a decimal integer enables:
        - Fast equality comparisons.
        - Compact storage in arrays.
        - Avoidance of expensive array-based comparisons during large-scale processing.

    This conversion is part of the column pattern encoding-decoding system used in:
    - `_calculateEnsembleCorrelations`
    - `_calculateOptimalCorrelations`
    - `_removePatternsDuplicated`

    Parameters
    ----------
    :param binaryVector: Binary mask indicating which columns are included (1) or excluded (0).
    :type binaryVector: list[int] or numpy.ndarray

    Returns
    -------
    :return: Decimal integer representation of the binary vector.
    :rtype: int
    """   
    numero_decimal = 0
    for i, bit in enumerate(reversed(vector)):
        numero_decimal += bit * (2 ** i)
    return numero_decimal

def _decimalToBinary(numero_decimal, dimension):
    """
    Converts a decimal integer back into its binary mask representation 
    (array of 0s and 1s) of fixed length.  
    This is the inverse operation of `_binaryToDecimal`.

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    In the CoBiNet biclustering algorithm, column patterns are stored 
    as decimal integers for compactness and fast comparisons.  
    When these patterns need to be interpreted or processed further, 
    they must be decoded back into binary masks indicating which 
    columns (conditions) are part of the bicluster.

    This decoding step is necessary for:
    - Extracting the actual subset of conditions from stored patterns.
    - Performing correlation calculations on the correct columns.
    - Expanding biclusters during the growth phase.

    Parameters
    ----------
    :param decimalValue: Integer representation of the binary column pattern.
    :type decimalValue: int
    :param totalLength: Total number of columns in the dataset (length of the binary mask).
    :type totalLength: int

    Returns
    -------
    :return: Binary mask as a numpy array of integers (0s and 1s), length = `totalLength`.
    :rtype: numpy.ndarray
    """
    
    # Convert the number to binary (without the prefix ‘0b’) and fill with zeros.
    binario = bin(numero_decimal)[2:].zfill(dimension)
    
    # Converts the binary string to a list of integers (binary vector).
    vector_binario = [int(bit) for bit in binario]
    return vector_binario

def _getColumns(vector_binario):
    """
    Retrieves the indices of columns that are marked as active (`1`) in a binary mask.  
    This function is used to translate a binary column pattern into the actual 
    set of condition indices that define a bicluster.

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    In the CoBiNet biclustering algorithm, subsets of conditions are represented 
    as binary masks (arrays of 0s and 1s) for compactness.  
    When a bicluster is being expanded, validated, or saved, it is necessary 
    to extract the actual column indices from this mask to:
    - Perform correlation calculations on the correct subset of conditions.
    - Reconstruct the bicluster's data matrix from the global dataset.
    - Save results in a human-readable format (e.g., gene list × condition list).

    Parameters
    ----------
    :param binaryVector: Binary mask indicating active columns (1) and inactive columns (0).
    :type binaryVector: list[int] or numpy.ndarray

    Returns
    -------
    :return: Array of integer indices corresponding to columns with value `1` in the binary mask.
    :rtype: numpy.ndarray
    """
    # Uses enumerate to traverse the vector and stores the positions where the bit is 1
    posiciones = [indice for indice, bit in enumerate(vector_binario) if bit == 1]
    return np.array(posiciones)

def _getIdsFromPattern(pattern, rows):
    """
    Decodes a pattern index (an integer representing a unique gene pair) into 
    the corresponding two gene indices (row IDs) from the dataset.  
    This mapping is used to iterate over all unique gene pairs in a systematic 
    and reproducible way.

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    In the CoBiNet biclustering algorithm, the total number of possible patterns 
    corresponds to all unique unordered pairs of genes from the dataset:
        Total patterns = (totalRows × (totalRows - 1)) / 2

    Instead of storing all pairs explicitly, the algorithm:
    - Iterates over pattern indices from `0` to `maxPatterns - 1`.
    - Uses this function to map each index to the corresponding `(row1, row2)` pair.

    This approach:
    - Reduces memory usage.
    - Ensures a consistent ordering of gene pair evaluations.
    - Facilitates reproducibility in correlation calculations.

    Parameters
    ----------
    :param patternIndex: Index of the pattern (0-based) representing a unique gene pair.
    :type patternIndex: int
    :param totalRows: Total number of rows (genes) in the dataset.
    :type totalRows: int

    Returns
    -------
    :return: Tuple `(row1, row2)` with the indices of the two genes forming the pattern.
    :rtype: tuple[int, int]
    """
    r1 = 0
    r2 = -1
    auxPat = pattern - rows + 1
    if (auxPat < 0):
        r2 = auxPat + rows
    
    j = rows - 2
    while(r2 == -1):
        auxPat = auxPat - j
        r1 += 1
        if (auxPat < 0):
            r2 = (j + auxPat) + (r1 + 1)
        j -= 1
    
    return r1, r2
    
def _getNumPatterns(rows):
    """
    Calculates the total number of unique unordered gene pairs 
    (patterns) that can be formed from the dataset.  
    This value is used to determine the maximum number of patterns 
    (`maxPatterns`) to be processed in the biclustering algorithm.

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    In the CoBiNet biclustering algorithm, each initial pattern 
    corresponds to a unique pair of genes.  
    Since gene pairs are unordered (pair `(A, B)` is the same as `(B, A)`), 
    the total number of unique patterns is given by the combinatorial formula:

        Total patterns = C(totalRows, 2) = totalRows × (totalRows - 1) / 2

    This ensures:
    - No duplicate evaluations of the same gene pair.
    - Efficient memory allocation for storing pattern-related information.
    - Consistent iteration over all possible gene combinations.

    Parameters
    ----------
    :param totalRows: Total number of genes (rows) in the dataset.
    :type totalRows: int

    Returns
    -------
    :return: Number of unique unordered gene pairs in the dataset.
    :rtype: int
    """
    maxPatterns = 0
    for i in range(rows):
        for j in range(i + 1, rows):
            maxPatterns += 1
    return maxPatterns
    
import shutil

def _saveBiclusters(genes, condiciones, biclustersFilas, biclustersColumnas, biclustersCorrelacion, debug):
    """
    Saves the resulting biclusters to disk, storing for each bicluster:
    - The list of genes included.
    - The list of conditions (columns) under which the bicluster is defined.
    - The bicluster’s average correlation score.

    Scientific Rationale / Methodological Context
    ----------------------------------------------
    Saving bicluster results is essential for downstream analysis, such as:
    - Biological enrichment analysis (e.g., Gene Ontology, KEGG).
    - Biomarker discovery.
    - Cross-comparison with other biclustering algorithms.

    The saved output provides:
    - **Gene list**: For functional and pathway analysis.
    - **Condition list**: To identify experimental contexts where the co-expression is observed.
    - **Correlation score**: A quality indicator of the bicluster.

    This function is typically executed at the end of `_executeAlgorithm` when 
    the `save` flag is enabled.

    Parameters
    ----------
    :param genes: Array of gene identifiers corresponding to the dataset rows.
    :type genes: numpy.ndarray
    :param conditions: Array of condition/sample identifiers corresponding to the dataset columns.
    :type conditions: numpy.ndarray
    :param biclustersRows: List of numpy arrays containing row indices (genes) for each bicluster.
    :type biclustersRows: list[numpy.ndarray]
    :param biclustersCols: List of numpy arrays containing column indices (conditions) for each bicluster.
    :type biclustersCols: list[numpy.ndarray]
    :param biclustersCorrelation: List of average correlation scores for each bicluster.
    :type biclustersCorrelation: list[float]
    :param debug: If True, prints bicluster details to the console as they are saved.
    :type debug: bool

    Returns
    -------
    :return: None. Writes bicluster files to the `results/biclusters/genes` directory.
    :rtype: None
    """
    
    if (biclustersFilas is None or len(biclustersFilas) == 0 or
        all(isinstance(x, int) and x == -1 for x in biclustersFilas)):
        if debug:
            print(f"Skipping saving biclusters: no valid biclusters found.")
        return
    
    path = "results/biclusters/results_biclusters.csv"
    
    genes_dir = "results/biclusters/genes"
    if os.path.exists(genes_dir):
        shutil.rmtree(genes_dir)
    os.makedirs(genes_dir)
    
    # Remove all elements at -1 to be left with only the biclusters
    biclustersFilas = [np.array(x, dtype=np.int64) for x in biclustersFilas if isinstance(x, (list, np.ndarray))]
    biclustersColumnas = [np.array(x, dtype=np.int64) for x in biclustersColumnas if isinstance(x, (list, np.ndarray))]
            
    bicluster_geneNames = [",".join(genes[indices]) for indices in biclustersFilas]
    bicluster_columnNames = [",".join(condiciones[indices]) for indices in biclustersColumnas]
    
    df = pd.DataFrame({
        "BiclusterID": np.arange(1, len(biclustersFilas) + 1),
        "Correlation": biclustersCorrelacion,
        "Genes": bicluster_geneNames,
        "Conditions": bicluster_columnNames            
    })
    
    df = df.sort_values(by="Correlation", ascending=False)    
    df.to_csv(path, index=False)
    print(f"Biclusters and list of genes for each bicluster saved in results/biclusters", flush=True)
    
    for i, indices_genes in enumerate(biclustersFilas):
        gene_names = genes[indices_genes]
        output_path = os.path.join(genes_dir, f"gene_bicluster_{i+1}.csv")
        pd.DataFrame(gene_names).to_csv(output_path, index=False, header=False)

