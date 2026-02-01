"""
ga_search.py - Genetic Algorithm for Threshold Search

This module implements the GA (Genetic Algorithm) component of the
Pareto-LP-GA framework. The GA searches for optimal class-wise thresholds
(α_k) that are used as constraints in the LP optimization.

Why GA is needed:
    - The LP needs thresholds α_k for non-target classes
    - These thresholds control the Pareto tradeoff
    - Too strict thresholds → No improvement possible
    - Too loose thresholds → Non-target classes may degrade too much
    - GA explores the threshold space to find the best balance

Fitness Function:
    - High fitness if target classes improve
    - Penalty if any target class degrades
    - Penalty proportional to non-target class degradation

Reference:
    Nahin et al. (2025). Algorithm 1 "Pareto-LP-GA"
"""

import numpy as np
from typing import Callable, Tuple, List, Optional


class GeneticAlgorithm:
    """
    Genetic Algorithm for optimizing class-wise thresholds.
    
    The GA maintains a population of candidate threshold vectors and
    evolves them over generations using selection, crossover, and mutation.
    
    Each individual in the population is a vector α = [α_0, α_1, ..., α_K-1]
    where α_k is the minimum performance threshold for class k.
    """
    
    def __init__(
        self,
        n_classes: int,
        population_size: int = 50,
        n_generations: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_fraction: float = 0.1,
        threshold_range: Tuple[float, float] = (0.0, 1.0),
        random_seed: Optional[int] = None
    ):
        """
        Initialize the Genetic Algorithm.

        Args:
            n_classes: Number of classes (dimension of threshold vector)
            population_size: Number of individuals in each generation
            n_generations: Number of evolutionary generations
            mutation_rate: Probability of mutating each gene
            crossover_rate: Probability of crossover between parents
            elite_fraction: Fraction of top individuals to preserve
            threshold_range: (min, max) range for RELATIVE threshold values in [0, 1]
                             α_k = 0.8 means "preserve 80% of total class k influence"
            random_seed: Optional seed for reproducibility
        """
        self.n_classes = n_classes
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = max(1, int(elite_fraction * population_size))
        self.threshold_range = threshold_range
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize population randomly
        self.population = self._initialize_population()
        
        # Track best solution
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
    
    def _initialize_population(self) -> np.ndarray:
        """
        Initialize a random population of threshold vectors.
        
        Each individual is a vector of thresholds for each class.
        We sample uniformly in the given range.
        
        Returns:
            Population array of shape (population_size, n_classes)
        """
        low, high = self.threshold_range
        population = np.random.uniform(
            low, high, 
            size=(self.population_size, self.n_classes)
        )
        return population
    
    def evaluate_population(
        self, 
        fitness_func: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        """
        Evaluate fitness for all individuals in the population.
        
        The fitness function is provided externally and encapsulates:
        1. Solving the LP with the given thresholds
        2. Computing performance change after reweighted training
        3. Returning a scalar fitness score
        
        Args:
            fitness_func: Function that takes threshold vector, returns fitness
            
        Returns:
            Fitness array of shape (population_size,)
        """
        fitness_scores = np.array([
            fitness_func(individual) for individual in self.population
        ])
        return fitness_scores
    
    def select_parents(self, fitness_scores: np.ndarray) -> np.ndarray:
        """
        Select parents for the next generation using tournament selection.
        
        Tournament selection:
        1. Randomly pick k individuals
        2. Select the one with highest fitness
        3. Repeat to get required number of parents
        
        This is a common selection method that balances exploration
        (random component) and exploitation (fitness-based selection).
        
        Args:
            fitness_scores: Fitness array of shape (population_size,)
            
        Returns:
            Selected parent indices of shape (n_parents,)
        """
        n_parents = self.population_size - self.elite_size
        tournament_size = 3
        parent_indices = []
        
        for _ in range(n_parents):
            # Random tournament
            contestants = np.random.choice(
                self.population_size, 
                size=tournament_size, 
                replace=False
            )
            # Winner is the one with highest fitness
            winner = contestants[np.argmax(fitness_scores[contestants])]
            parent_indices.append(winner)
        
        return np.array(parent_indices)
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform crossover between two parents to create offspring.
        
        We use uniform crossover:
        - For each gene, randomly choose from parent1 or parent2
        - This allows mixing of threshold values across all classes
        
        Args:
            parent1: First parent threshold vector
            parent2: Second parent threshold vector
            
        Returns:
            Tuple of two offspring threshold vectors
        """
        if np.random.random() > self.crossover_rate:
            # No crossover - return copies of parents
            return parent1.copy(), parent2.copy()
        
        # Uniform crossover
        mask = np.random.random(self.n_classes) < 0.5
        
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        
        return child1, child2
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Apply mutation to an individual.
        
        For each gene (threshold), with probability mutation_rate:
        - Add Gaussian noise to the threshold value
        - Clip to valid range
        
        Mutation helps explore new areas of the search space
        and prevents premature convergence.
        
        Args:
            individual: Threshold vector to mutate
            
        Returns:
            Mutated threshold vector
        """
        mutant = individual.copy()
        
        for i in range(self.n_classes):
            if np.random.random() < self.mutation_rate:
                # Gaussian mutation
                mutation = np.random.normal(0, 0.2)
                mutant[i] += mutation
                
                # Clip to valid range
                mutant[i] = np.clip(
                    mutant[i], 
                    self.threshold_range[0], 
                    self.threshold_range[1]
                )
        
        return mutant
    
    def evolve_generation(self, fitness_scores: np.ndarray) -> None:
        """
        Evolve the population for one generation.
        
        Steps:
        1. Elitism: Keep top individuals unchanged
        2. Selection: Choose parents for breeding
        3. Crossover: Create offspring from parent pairs
        4. Mutation: Add random variations
        
        Args:
            fitness_scores: Fitness array from current generation
        """
        new_population = []
        
        # Elitism: Preserve top individuals
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())
        
        # Selection and breeding
        parent_indices = self.select_parents(fitness_scores)
        
        # Create offspring through crossover and mutation
        for i in range(0, len(parent_indices) - 1, 2):
            parent1 = self.population[parent_indices[i]]
            parent2 = self.population[parent_indices[i + 1]]
            
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        # Handle odd case
        while len(new_population) < self.population_size:
            idx = np.random.choice(len(parent_indices))
            parent = self.population[parent_indices[idx]]
            new_population.append(self.mutate(parent))
        
        self.population = np.array(new_population[:self.population_size])
    
    def run(
        self, 
        fitness_func: Callable[[np.ndarray], float],
        verbose: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Run the full GA optimization.
        
        This is the main entry point for the GA search.
        It runs for n_generations, tracking the best solution found.
        
        Args:
            fitness_func: Function that evaluates a threshold vector
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best_thresholds, best_fitness)
        """
        for gen in range(self.n_generations):
            # Evaluate fitness
            fitness_scores = self.evaluate_population(fitness_func)
            
            # Track best
            gen_best_idx = np.argmax(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]
            
            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_individual = self.population[gen_best_idx].copy()
            
            self.fitness_history.append(gen_best_fitness)
            
            if verbose:
                print(f"Generation {gen + 1}/{self.n_generations}: "
                      f"Best fitness = {gen_best_fitness:.4f}")
            
            # Evolve (except on last generation)
            if gen < self.n_generations - 1:
                self.evolve_generation(fitness_scores)
        
        return self.best_individual, self.best_fitness


def create_fitness_function(
    influence_vectors: np.ndarray,
    target_classes: List[int],
    solve_lp_func: Callable,
    train_and_evaluate_func: Callable,
    baseline_accuracies: np.ndarray
) -> Callable[[np.ndarray], float]:
    """
    Create a fitness function for the GA.
    
    The fitness function:
    1. Takes threshold vector α
    2. Solves LP to get optimal weights
    3. Trains model with weighted loss
    4. Evaluates performance change
    5. Returns fitness score
    
    Fitness is high when:
    - Target classes improve (main goal)
    - No severe degradation on non-target classes
    
    Args:
        influence_vectors: Influence matrix (n_samples, n_classes)
        target_classes: Classes to improve
        solve_lp_func: Function to solve LP given thresholds
        train_and_evaluate_func: Function to train model and get accuracies
        baseline_accuracies: Original per-class accuracies
        
    Returns:
        Fitness function that takes threshold vector, returns scalar
    """
    n_classes = influence_vectors.shape[1]
    non_target = [k for k in range(n_classes) if k not in target_classes]
    
    def fitness(alpha: np.ndarray) -> float:
        # Solve LP with these thresholds
        weights, success = solve_lp_func(
            influence_vectors, target_classes, alpha
        )
        
        if not success:
            return -1000.0  # Penalty for infeasible LP
        
        # Train with weighted loss and evaluate
        new_accuracies = train_and_evaluate_func(weights)
        
        # Compute performance change
        delta = new_accuracies - baseline_accuracies
        
        # Fitness components
        target_improvement = np.sum(delta[target_classes])
        
        # Penalty if any target class degrades
        target_degradation = np.sum(np.minimum(delta[target_classes], 0))
        
        # Penalty for non-target degradation
        nontarget_degradation = np.sum(np.minimum(delta[non_target], 0)) if non_target else 0
        
        # Combined fitness
        # Large negative penalty for target degradation
        # Smaller penalty for non-target degradation
        fitness_score = (
            target_improvement 
            + 10 * target_degradation  # Heavy penalty
            + 1 * nontarget_degradation  # Light penalty
        )
        
        return fitness_score
    
    return fitness


def simple_threshold_search(
    influence_vectors: np.ndarray,
    target_classes: List[int],
    solve_lp_func: Callable,
    n_trials: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplified threshold search without full training loop.

    This is a faster alternative for initial experiments.
    Instead of training the model, we use the influence vectors
    directly to estimate fitness.

    Args:
        influence_vectors: Influence matrix (n_samples, n_classes)
        target_classes: Classes to improve
        solve_lp_func: Function to solve LP
        n_trials: Number of random threshold trials

    Returns:
        Tuple of (best_thresholds, best_weights)
    """
    n_classes = influence_vectors.shape[1]
    best_score = float('-inf')
    best_thresholds = None
    best_weights = None

    for _ in range(n_trials):
        # Random RELATIVE thresholds in [0, 1]
        # α_k = 0.8 means "preserve at least 80% of total class k influence"
        # Lower values allow more aggressive optimization on target classes
        alpha = np.random.uniform(0.0, 0.9, size=n_classes)
        
        # Solve LP
        weights, success = solve_lp_func(
            influence_vectors, target_classes, alpha
        )
        
        if not success:
            continue
        
        # Estimate improvement using influence vectors
        predicted_gain = influence_vectors.T @ weights
        
        # Score: improvement on targets, penalize non-target drops
        score = np.sum(predicted_gain[target_classes])
        for k in range(n_classes):
            if k not in target_classes:
                score += min(0, predicted_gain[k])  # Penalty for drops
        
        if score > best_score:
            best_score = score
            best_thresholds = alpha.copy()
            best_weights = weights.copy()
    
    if best_thresholds is None:
        # Fallback to zeros
        best_thresholds = np.zeros(n_classes)
        best_weights = np.ones(influence_vectors.shape[0]) / influence_vectors.shape[0]
    
    return best_thresholds, best_weights
