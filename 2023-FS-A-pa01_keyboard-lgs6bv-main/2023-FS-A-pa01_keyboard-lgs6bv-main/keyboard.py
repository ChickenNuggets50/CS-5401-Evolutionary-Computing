#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Evolve a better keyboard.
This assignment is mostly open-ended,
with a couple restrictions:

# DO NOT MODIFY >>>>
Do not edit the sections between these marks below.
# <<<< DO NOT MODIFY
"""

# %%
import random
from typing import TypedDict
import math
import json
import datetime
from typing import List

# DO NOT MODIFY >>>>
# First, what should our representation look like?
# Is there any modularity in adjacency?
# What mechanisms capitalize on such modular patterns?
# ./corpus/2_count.py specificies this same structure
# Positions    01234   56789   01234
LEFT_DVORAK = "',.PY" "AOEUI" ";QJKX"
LEFT_QWERTY = "QWERT" "ASDFG" "ZXCVB"
LEFT_COLEMK = "QWFPG" "ARSTD" "ZXCVB"
LEFT_WORKMN = "QDRWB" "ASHTG" "ZXMCV"

LEFT_DISTAN = "22222" "11112" "22222"
LEFT_ERGONO = "11112" "11112" "22323"
LEFT_EDGE_B = "12345" "12345" "12345"

# Positions     56   7890123   456789   01234
RIGHT_DVORAK = "[]" "FGCRL/=" "DHTNS-" "BMWVZ"
RIGHT_QWERTY = "-=" "YUIOP[]" "HJKL;'" "NM,./"
RIGHT_COLEMK = "-=" "JLUY;[]" "HNEIO'" "KM,./"
RIGHT_WOKRMN = "-=" "JFUP;[]" "YNEOI'" "KL,./"

RIGHT_DISTAN = "34" "2222223" "211112" "22222"
RIGHT_ERGONO = "33" "3111134" "211112" "21222"
RIGHT_EDGE_B = "21" "7654321" "654321" "54321"

DVORAK = LEFT_DVORAK + RIGHT_DVORAK
QWERTY = LEFT_QWERTY + RIGHT_QWERTY
COLEMAK = LEFT_COLEMK + RIGHT_COLEMK
WORKMAN = LEFT_WORKMN + RIGHT_WOKRMN

DISTANCE = LEFT_DISTAN + RIGHT_DISTAN
ERGONOMICS = LEFT_ERGONO + RIGHT_ERGONO
PREFER_EDGES = LEFT_EDGE_B + RIGHT_EDGE_B

# Real data on w.p.m. for each letter, normalized.
# Higher values is better (higher w.p.m.)
with open(file="typing_data/manual-typing-data_qwerty.json", mode="r") as f:
    data_qwerty = json.load(fp=f)
with open(file="typing_data/manual-typing-data_dvorak.json", mode="r") as f:
    data_dvorak = json.load(fp=f)
data_values = list(data_qwerty.values()) + list(data_dvorak.values())
mean_value = sum(data_values) / len(data_values)
data_combine = []
for dv, qw in zip(DVORAK, QWERTY):
    if dv in data_dvorak.keys() and qw in data_qwerty.keys():
        data_combine.append((data_dvorak[dv] + data_qwerty[qw]) / 2)
    elif dv in data_dvorak.keys() and qw not in data_qwerty.keys():
        data_combine.append(data_dvorak[dv])
    elif dv not in data_dvorak.keys() and qw in data_qwerty.keys():
        data_combine.append(data_qwerty[qw])
    else:
        # Fill missing data with the mean
        data_combine.append(mean_value)


class Individual(TypedDict):
    genome: str
    fitness: int


Population = list[Individual]


def render_keyboard(individual: Individual) -> str:
    layout = individual["genome"]
    fitness = individual["fitness"]
    """Prints the keyboard in a nice way"""
    return (
        f"______________  ________________\n"
        f" ` 1 2 3 4 5 6  7 8 9 0 " + " ".join(layout[15:17]) + " Back\n"
        f"Tab " + " ".join(layout[0:5]) + "  " + " ".join(layout[17:24]) + " \\\n"
        f"Caps " + " ".join(layout[5:10]) + "  " + " ".join(layout[24:30]) + " Enter\n"
        f"Shift "
        + " ".join(layout[10:15])
        + "  "
        + " ".join(layout[30:35])
        + " Shift\n"
        f"\nAbove keyboard has fitness of: {fitness}"
    )


# <<<< DO NOT MODIFY


def ensure_unique_chars(genome: str) -> str:
    unseen_chars = list(DVORAK)
    unique_layout = ""

    for char in genome:
        if char in unseen_chars:
            unseen_chars.remove(char)
            unique_layout += char
        else:
            unique_layout += "0"
    for char in unique_layout:
        if char == "0":
            unique_char = random.choice(unseen_chars)
            unique_layout += unique_char
            unseen_chars.remove(unique_char)

    unique_layout = "".join([char for char in unique_layout if char != "0"])
    return unique_layout


def initialize_individual(genome: str, fitness: int) -> Individual:
    """
    Purpose:        Create one individual
    Parameters:     genome as string, fitness as integer (higher better)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a dict[str, int]
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    return {"genome": genome, "fitness": fitness}


def initialize_pop(example_genome: str, pop_size: int) -> Population:
    """
    Purpose:        Create population to evolve
    Parameters:     Goal string, population size as int
    User Input:     no
    Prints:         no
    Returns:        a population, as a list of Individuals
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    population = []
    for _ in range(pop_size):
        genome = list(example_genome)
        random.shuffle(genome)
        individual = initialize_individual("".join(genome), fitness=0)
        population.append(individual)
    return population


def recombine_pair(parent1: Individual, parent2: Individual) -> Population:
    """
    Purpose:        Recombine two parents to produce two children
    Parameters:     Two parents as Individuals
    User Input:     no
    Prints:         no
    Returns:        A population of size 2, the children
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    place = random.choice(range(len(parent1["genome"])))

    child1_genome = parent1["genome"][:place] + parent2["genome"][place:]
    child2_genome = parent2["genome"][:place] + parent1["genome"][place:]

    child1 = initialize_individual(genome=child1_genome, fitness=0)
    child2 = initialize_individual(genome=child2_genome, fitness=0)
    return [child1, child2]


def recombine_group(parents: Population, recombine_rate: float) -> Population:
    """
    Purpose:        Recombines a whole group, returns the new population
                    Pair parents 1-2, 2-3, 3-4, etc..
                    Recombine at rate, else clone the parents.
    Parameters:     parents and recombine rate
    User Input:     no
    Prints:         no
    Returns:        New population of children
    Modifies:       Nothing
    Calls:          ?
    """
    children: Population = []
    for ipair in range(0, len(parents) - 1, 2):
        if random.random() < recombine_rate:
            child1, child2 = recombine_pair(
                parent1=parents[ipair], parent2=parents[ipair + 1]
            )
        else:
            child1, child2 = parents[ipair], parents[ipair + 1]
        children.extend([child1, child2])
    return children


def mutate_individual(parent: Individual, mutate_rate: float) -> Individual:
    """
    Purpose:        Mutate one individual
    Parameters:     One parents as Individual, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    new_genome = list(parent["genome"])
    for i in range(len(new_genome)):
        if random.random() < mutate_rate:
            j = random.randint(0, len(new_genome) - 1)
            new_genome[i], new_genome[j] = new_genome[j], new_genome[i]
    mutant = initialize_individual(
        genome=ensure_unique_chars("".join(new_genome)), fitness=0
    )
    return mutant


def mutate_group(children: Population, mutate_rate: float) -> Population:
    """
    Purpose:        Mutates a whole Population, returns the mutated group
    Parameters:     Population, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    mutants: Population = []
    for child in children:
        mutants.append(mutate_individual(parent=child, mutate_rate=mutate_rate))
    return mutants


# DO NOT MODIFY >>>>
def evaluate_individual(individual: Individual) -> None:
    """
    Purpose:        Computes and modifies the fitness for one individual
                    Assumes and relies on the logc of ./corpus/2_counts.py
    Parameters:     One Individual
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The individual (mutable object)
    Calls:          Basic python only
    Example doctest:
    """
    layout = individual["genome"]

    # Basic return to home row, with no differential cost for repeats.
    fitness = 0
    for pos, key in enumerate(layout):
        fitness += count_dict[key] * int(DISTANCE[pos])

    # Top-down guess at ideal ergonomics
    for pos, key in enumerate(layout):
        fitness += count_dict[key] * int(ERGONOMICS[pos])

    # Keybr.com querty-dvorak average data as estimate of real hand
    for pos, key in enumerate(layout):
        fitness += count_dict[key] / data_combine[pos]

    # Symbols should be toward edges.
    for pos, key in enumerate(layout):
        if key in "-[],.';/=":
            fitness += int(PREFER_EDGES[pos])

    # Vowels on the left, Consosants on the right
    for pos, key in enumerate(layout):
        if key in "AEIOUY" and pos > 14:
            fitness += 3

    # [] {} () <> should be adjacent.
    # () are fixed by design choice (number line).
    # [] and {} are on same keys.
    # Perhaps ideally, <> and () should be on same keys too...
    right_edges = [4, 9, 14, 16, 23, 29, 34]
    for pos, key in enumerate(layout):
        # order of (x or y) protects index on far right:
        if key == "[" and (pos in right_edges or "]" != layout[pos + 1]):
            fitness += 1
        if key == "," and (pos in right_edges or "." != layout[pos + 1]):
            fitness += 1

    # high transitional probabilities should be rolls or alternates?
    # ing, ch, th, the, etc?
    # Would need to build a new dataset of 2 and 3 char transitions?
    for pos in range(len(layout) - 1):
        if pos in right_edges:
            continue
        char1 = layout[pos]
        char2 = layout[pos + 1]
        dict_key = char1 + char2
        fitness -= count_run2_dict[dict_key]

    individual["fitness"] = fitness


# <<<< DO NOT MODIFY


def evaluate_group(individuals: Population) -> None:
    """
    Purpose:        Computes and modifies the fitness for population
    Parameters:     Objective string, Population
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The Individuals, all mutable objects
    Calls:          ?
    Example doctest:
    """
    for individual in individuals:
        evaluate_individual(individual)


def rank_group(individuals: Population) -> None:
    """
    Purpose:        Create one individual
    Parameters:     Population of Individuals
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The population's order (a mutable object)
    Calls:          ?
    Example doctest:
    """
    individuals.sort(key=lambda x: x["fitness"])


def parent_select(individuals: Population, number: int) -> Population:
    """
    Purpose:        Choose parents in direct probability to their fitness
    Parameters:     Population, the number of individuals to pick.
    User Input:     no
    Prints:         no
    Returns:        Sub-population
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    total_fitness = sum(individual["fitness"] for individual in individuals)
    probabilities = [
        individual["fitness"] / total_fitness for individual in individuals
    ]
    selected_parents = random.choices(individuals, probabilities, k=number)
    return selected_parents


def survivor_select(individuals: Population, pop_size: int) -> Population:
    """
    Purpose:        Picks who gets to live!
    Parameters:     Population, and population size to return.
    User Input:     no
    Prints:         no
    Returns:        Population, of pop_size
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    individuals.sort(key=lambda x: x["fitness"])
    return individuals[:pop_size]


def evolve(example_genome: str, pop_size: int = 100) -> Population:
    """
    Purpose:        A whole EC run, main driver
    Parameters:     The evolved population of solutions
    User Input:     No
    Prints:         Updates every time fitness switches.
    Returns:        Population
    Modifies:       Various data structures
    Calls:          Basic python, all your functions
    """
    # To debug doctest test in pudb
    # Highlight the line of code below below
    # Type 't' to jump 'to' it
    # Type 's' to 'step' deeper
    # Type 'n' to 'next' over
    # Type 'f' or 'r' to finish/return a function call and go back to caller
    population = initialize_pop(example_genome=example_genome, pop_size=pop_size)
    evaluate_group(individuals=population)
    rank_group(individuals=population)
    best_fitness = population[0]["fitness"]
    perfect_fitness = 0.0
    counter = 0
    while counter < 10000:
        counter += 1
        parents = parent_select(individuals=population, number=80)
        children = recombine_group(parents=parents, recombine_rate=0.7)
        mutate_rate = 0.1 - (0.1 - 0.01) * (best_fitness / 100)
        mutants = mutate_group(children=children, mutate_rate=mutate_rate)
        evaluate_group(individuals=mutants)
        everyone = population + mutants
        rank_group(individuals=everyone)
        population = survivor_select(individuals=everyone, pop_size=pop_size)
        if best_fitness != population[0]["fitness"]:
            best_fitness = population[0]["fitness"]
            print("Iteration number", counter, "with best individual", population[0])
    return population


seed = False

# DO NOT MODIFY >>>>
if __name__ == "__main__":
    divider = "===================================================="
    # Execute doctests to protect main:
    # import doctest

    # doctest.testmod()
    # doctest.testmod(verbose=True)

    if seed:
        random.seed(42)

    with open("corpus/counts.json") as fhand:
        count_dict = json.load(fhand)
    # print({k: v for k, v in sorted(count_dict.items(), key=lambda item: item[1], reverse=True)})
    # print("Above is the order of frequency of letters in English.")

    # print("Counts of characters in big corpus, ordered by freqency:")
    # ordered = sorted(count_dict, key=count_dict.__getitem__, reverse=True)
    # for key in ordered:
    #     print(key, count_dict[key])

    with open("corpus/counts_run2.json") as fhand:
        count_run2_dict = json.load(fhand)
    # print({k: v for k, v in sorted(count_run2_dict.items(), key=lambda item: item[1], reverse=True)})
    # print("Above is the order of frequency of letter-pairs in English.")

    print(divider)
    print(
        f"Number of possible permutations of standard keyboard: {math.factorial(len(DVORAK)):,e}"
    )
    print("That's a huge space to search through")
    print("The messy landscape is a difficult to optimize multi-modal space")
    print("Lower fitness is better.")

    print(divider)
    print("\nThis is the Dvorak keyboard:")
    dvorak = Individual(genome=DVORAK, fitness=0)
    evaluate_individual(dvorak)
    print(render_keyboard(dvorak))

    print(divider)
    print("\nThis is the Workman keyboard:")
    workman = Individual(genome=WORKMAN, fitness=0)
    evaluate_individual(workman)
    print(render_keyboard(workman))

    print(divider)
    print("\nThis is the Colemak keyboard:")
    colemak = Individual(genome=COLEMAK, fitness=0)
    evaluate_individual(colemak)
    print(render_keyboard(colemak))

    print(divider)
    print("\nThis is the QWERTY keyboard:")
    qwerty = Individual(genome=QWERTY, fitness=0)
    evaluate_individual(qwerty)
    print(render_keyboard(qwerty))

    print(divider)
    print("\nThis is a random layout:")
    badarr = list(DVORAK)
    random.shuffle(badarr)
    badstr = "".join(badarr)
    badkey = Individual(genome=badstr, fitness=0)
    evaluate_individual(badkey)
    print(render_keyboard(badkey))

    print(divider)
    input("Press any key to start")
    population = evolve(example_genome=DVORAK)

    print("Here is the best layout:")
    print(render_keyboard(population[0]))

    grade = 0
    if qwerty["fitness"] < population[0]["fitness"]:
        grade = 0
    if colemak["fitness"] < population[0]["fitness"]:
        grade = 50
    if workman["fitness"] < population[0]["fitness"]:
        grade = 60
    elif dvorak["fitness"] < population[0]["fitness"]:
        grade = 70
    else:
        grade = 80

    with open(file="results.txt", mode="w") as f:
        f.write(str(grade))

    with open(file="best.json", mode="w") as f:
        f.write(json.dumps(population[0]))

    with open(file="best.txt", mode="w") as f:
        f.write(render_keyboard(population[0]))
# <<<< DO NOT MODIFY
