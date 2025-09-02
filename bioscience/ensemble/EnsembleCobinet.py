import numpy as np 


def ensembleCorrelation(vector1, vector2, pearsonThreshold, spearmanThreshold, nmiThreshold): 
    
    pearson_val = pearson(vector1, vector2)   # siempre devuelve valor real
    spearman_val = spearman(vector1, vector2)
    nmi_val = nmi(vector1, vector2)
    
    metrics = [pearson_val, spearman_val, nmi_val]
    metrics = [np.nan if m is None else m for m in metrics]
    
    passed = 0
    if not np.isnan(metrics[0]) and metrics[0] >= pearsonThreshold:
        passed += 1
    if not np.isnan(metrics[1]) and metrics[1] >= spearmanThreshold:
        passed += 1
    if not np.isnan(metrics[2]) and metrics[2] >= nmiThreshold:
        passed += 1
    
    valid = passed >= int(np.ceil(len([m for m in metrics if not np.isnan(m)]) / 2))

    return valid, np.nanmean(metrics)



def pearson(x, y, debug = False):
    
    n = len(x)    
    if n != len(y):
        return None

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    diff_x = [xi - mean_x for xi in x]
    diff_y = [yi - mean_y for yi in y]

    numerator = sum(diff_x[i] * diff_y[i] for i in range(n))

    std_x = (sum(dx ** 2 for dx in diff_x) ** 0.5)
    std_y = (sum(dy ** 2 for dy in diff_y) ** 0.5)

    pearson = numerator / (std_x * std_y) if std_x * std_y != 0 else 0
    
    if pearson != None:
        pearson = abs(pearson)

    return pearson


def spearman(x, y, debug=False):
    
    def get_ranks(data):
            
        sorted_data = sorted(enumerate(data), key=lambda item: item[1])        
        ranks = [0] * len(data)
        
        for rank, (index, _) in enumerate(sorted_data, start=1):
            ranks[index] = rank
        
        return ranks

    rank_x = get_ranks(x)
    rank_y = get_ranks(y)
    
    d_squared = [(rank_x[i] - rank_y[i]) ** 2 for i in range(len(x))]

    n = len(x)
    if n * (n**2 - 1) == 0:
        spearman_coefficient = 0.0
    else:
        spearman_coefficient = 1 - (6 * sum(d_squared)) / (n * (n**2 - 1))
    
    if spearman != None:
        spearman_coefficient = abs(spearman_coefficient)
    
    return spearman_coefficient


def discretize_vector(vec, precision=5):
    return np.round(vec, decimals=precision)

def nmi(x, y, debug=False):
    x = discretize_vector(x)
    y = discretize_vector(y)

    n = len(x)
    assert n == len(y)

    x_labels = np.unique(x)
    y_labels = np.unique(y)

    # Build joint distribution table
    px = np.array([np.sum(x == label) for label in x_labels]) / n
    py = np.array([np.sum(y == label) for label in y_labels]) / n
    pxy = np.zeros((len(x_labels), len(y_labels)))

    for i, lx in enumerate(x_labels):
        for j, ly in enumerate(y_labels):
            pxy[i, j] = np.sum((x == lx) & (y == ly)) / n

    # Compute entropies
    Hx = -np.sum(px * np.log2(px + 1e-10))
    Hy = -np.sum(py * np.log2(py + 1e-10))
    MI = np.sum(pxy * np.log2((pxy + 1e-10) / ((px[:, None] * py[None, :]) + 1e-10)))

    # Normalize
    NMI = 2 * MI / (Hx + Hy) if (Hx + Hy) > 0 else 0
    
    if NMI != None:
        NMI = abs(NMI)
        
    return NMI