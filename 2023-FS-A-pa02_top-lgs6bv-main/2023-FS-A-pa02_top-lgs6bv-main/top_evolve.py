#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
This one is more open-ended.
You will write most of the functions,
and can even change their definitions.
The only things you can't change inclued:
* Individual and Population type
* xml_string_builder
* evaluate_individual
* if __name__ == "__main__":
"""

import numpy as np
import mujoco  # type: ignore
import random
from typing import TypedDict
import multiprocessing as mp
from multiprocessing.pool import Pool
import json


class Individual(TypedDict):
    """
    Don't change this.
    """

    genome: list[float]
    fitness: float


# Don't change this:
Population = list[Individual]


def xml_string_builder(genome: list[float]) -> str:
    """
    Don't change this.
    """
    top_str = """
    <mujoco model="single top">
      <!-- <option integrator="RK4"/> -->
      <option density="1.204" viscosity="1.8e-5" integrator="implicit"/>
      <!-- <option density="1.204" viscosity="1.8e-5" integrator="RK4"/> -->

      <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
         rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
      </asset>

      <default>
        <!-- <geom condim="6" friction="1 0.01 0.01"/> -->
        <geom condim="6" friction="1 0.001 0.001"/>
      </default>

      <worldbody>
        <light pos="0 0 .6"/>
        <camera name="closeup" pos="0 -.3 .3" xyaxes="1 0 0 0 1 1"/>
        <geom size=".2 .2 .01" type="plane" material="grid"/>
        <geom name="r_wall" type="box" pos=".2 0 .04" size=".01 .22 .05"/>
        <geom name="l_wall" type="box" pos="-.2 0 .04" size=".01 .22 .05"/>
        <geom name="b_wall" type="box" pos="0 .2 .04" size=".22 .01 .05"/>
        <geom name="f_wall" type="box" pos="0 -.2 .04" size=".22 .01 .05"/>
        <body name="top" pos="0 0 .02">
          <freejoint/>
          <!--<geom name="tip" type="ellipsoid" size=".005 .005 .02" />-->
    """

    z = 0.0
    for layer in range(0, 24):
        disc_d = genome[layer]
        top_str += f'      <geom name="l{layer}" type="cylinder" pos="0 0 {z}" size="{disc_d} .0005"/>\n'
        z = round(number=z + 0.001, ndigits=4)

    rand_x_vel = random.uniform(a=0, b=1)
    rand_y_vel = random.uniform(a=0, b=1)
    rand_x_rot = random.randint(a=1, b=10)
    rand_y_rot = random.randint(a=1, b=10)
    rand_z_rot = random.randint(a=100, b=350)
    top_str += f"""
          <!--<geom name="spinner" type="capsule" pos="0 0 0.025" size="0.0015 .005"/>-->
          <geom name="tip" type="ellipsoid" pos="0 0 0.021" size=".0025 .0025 .02" />
        </body>
      </worldbody>

      <keyframe>
        <!-- qpos =  -->
        <!-- qvel = x(left-right) y(in-out) z(up-down)
                    x(rot in x plane lr) y (rot in y plane in-out) z (rot in z plane top-spin) -->
        <key name="spinning" qpos="0 0 0.02 1 0 0 0" qvel="{rand_x_vel} {rand_y_vel} 0 {rand_x_rot} {rand_y_rot} {rand_z_rot}"/>
      </keyframe>
    </mujoco>
    """
    return top_str


def initialize_individual(genome: list[float], fitness: float) -> Individual:
    """
    Purpose:        Create one individual
    Parameters:     genome, fitness as float (higher better)
    User Input:     no
    Prints:         no
    Returns:        One Individual
    Modifies:       Nothing
    Calls:          Basic python only
    Tests:          ./unit_tests/*
    Status:         Do this one!
    """
    # print("Do this one.")
    return {"genome": genome, "fitness": fitness}


def initialize_pop(pop_size: int) -> Population:
    """
    Purpose:        Create population to evolve
    Parameters:     population size as int
    User Input:     no
    Prints:         no
    Returns:        a population, as a list of Individuals
    Modifies:       Nothing
    Calls:          random.something
    Tests:          ./unit_tests/*
    Status:         Do this one!
    """
    # print("Do this one.")
    return [
        initialize_individual(
            genome=[random.uniform(0.001, 0.1) for _ in range(24)], fitness=0
        )
        for _ in range(pop_size)
    ]


def recombine_pair(parent1: Individual, parent2: Individual) -> Population:
    """
    Purpose:        Recombine two parents to produce two children
    Parameters:     Two parents as Individuals
    User Input:     no
    Prints:         no
    Returns:        A population of size 2, the children
    Modifies:       Nothing
    Calls:          Basic python, random.choice-1, initialize_individual-2
    Tests:          ./unit_tests/*
    Status:         Do this one!
    """
    crossover_point = random.randint(1, len(parent1["genome"]) - 1)
    child1_genome = (
        parent1["genome"][:crossover_point] + parent2["genome"][crossover_point:]
    )
    child2_genome = (
        parent2["genome"][:crossover_point] + parent1["genome"][crossover_point:]
    )
    return [
        initialize_individual(genome=child1_genome, fitness=0),
        initialize_individual(genome=child2_genome, fitness=0),
    ]


def recombine_group(parents: Population, recombine_rate: float) -> Population:
    """
    Purpose:        Recombines a whole group, returns the new population
                    Pair parents 1-2, 3-4, 5-6, etc..
                    Recombine at rate, else clone the parents.
    Parameters:     genome as string, fitness as integer (higher better)
    User Input:     no
    Prints:         no
    Returns:        New population of children
    Modifies:       Nothing
    Calls:          Basic python, random.random~n/2, recombine pair-n
    Tests:          ./unit_tests/*
    Status:         Do this one!
    """
    # print("Do this one.")
    children = []
    for i in range(0, len(parents), 2):
        if random.random() < recombine_rate:
            children.extend(recombine_pair(parents[i], parents[i + 1]))
        else:
            children.extend([parents[i].copy(), parents[i + 1].copy()])
    return children


def mutate_individual(parent: Individual, mutate_sigma: float) -> Individual:
    """
    Purpose:        Mutate one individual, re-init it's fitness to 0
    Parameters:     One parents as Individual, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual
    Modifies:       Nothing
    Calls:          Basic python, random,choice-1,
                    random.random-1, initialize_individual-1
    Tests:          ./unit_tests/*
    Status:         Do this one!
    """
    # print("Do this one.")
    mutated_genome = [
        max(0.001, gene + random.gauss(0, mutate_sigma)) for gene in parent["genome"]
    ]
    return initialize_individual(genome=mutated_genome, fitness=0)


def mutate_group(children: Population, mutate_sigma: float) -> Population:
    """
    Purpose:        Mutates a whole Population, returns the mutated group
    Parameters:     Population, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    Calls:          Basic python, mutate_individual-n
    Tests:          ./unit_tests/*
    Status:         Do this one!
    """
    # print("Do this one.")
    return [mutate_individual(child, mutate_sigma) for child in children]


def evaluate_individual(individual: Individual) -> Individual:
    """
    Purpose:        Computes and modifies the fitness for one individual
    Parameters:     One Individual
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The individual (mutable object)
    Calls:          Basic python only
    Tests:          ./unit_tests/*
    Status:         Don't change this one.
    Note:           pool.map required returning rather than mutating
    """
    top_str = xml_string_builder(genome=individual["genome"])
    model = mujoco.MjModel.from_xml_string(xml=top_str)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    quit_rot_vel = 50
    while quit_rot_vel < data.qvel[5]:
        mujoco.mj_step(model, data)
    individual["fitness"] = data.time
    return individual


def evaluate_group(individuals: Population) -> Population:
    """
    Purpose:        Computes and modifies the fitness for population
    Parameters:     Objective string, Population
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The Individuals, all mutable objects
    Calls:          Basic python, evaluate_individual-n
    Tests:          ./unit_tests/*
    Status:         I give you this one, but you can change it.
    Note:           pool.map required returning rather than mutating
    """
    with Pool(processes=mp.cpu_count()) as pool:
        individuals = pool.map(func=evaluate_individual, iterable=individuals)
    return individuals


def rank_group(individuals: Population) -> None:
    """
    Purpose:        Create one individual
    Parameters:     Population of Individuals
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The population's order (a mutable object)
    Calls:          Basic python only
    Tests:          ./unit_tests/*
    Status:         Do this one!
    """
    # print("Do this one.")
    individuals.sort(key=lambda x: x["fitness"], reverse=True)


def parent_select(individuals: Population, number: int) -> Population:
    """
    Purpose:        Choose parents in direct probability to their fitness
    Parameters:     Population, the number of individuals to pick.
    User Input:     no
    Prints:         no
    Returns:        Sub-population
    Modifies:       Nothing
    Calls:          Basic python, random.choices-1
    Tests:          ./unit_tests/*
    Status:         Do this one!
    """
    # print("Do this one.")
    probabilities = np.array([1 / (i + 1) for i in range(len(individuals))])
    probabilities /= probabilities.sum()
    selected_indices = np.random.choice(len(individuals), size=number, p=probabilities)
    return [individuals[i] for i in selected_indices]


def survivor_select(individuals: Population, pop_size: int) -> Population:
    """
    Purpose:        Picks who gets to live!
    Parameters:     Population, and population size to return.
    User Input:     no
    Prints:         no
    Returns:        Population, of pop_size
    Modifies:       Nothing
    Calls:          Basic python only
    Tests:          ./unit_tests/*
    Status:         Do this one!
    """
    # print("Do this one.")
    rank_group(individuals)
    return individuals[:pop_size]


def evolve(pop_size: int = 1000) -> Population:
    """
    Purpose:        A whole EC run, main driver
    Parameters:     The evolved population of solutions
    User Input:     No
    Prints:         Updates every time fitness switches.
    Returns:        Population
    Modifies:       Various data structures
    Calls:          Basic python only, all your functions
    Tests:          ./stdio_tests/* and ./arg_tests/
    Status:         Giving you this one, but you can change it.
    """
    # To debug doctest test in pudb
    # Highlight the line of code below below
    # Type 't' to jump 'to' it
    # Type 's' to 'step' deeper
    # Type 'n' to 'next' over
    # Type 'f' or 'r' to finish/return a function call and go back to caller
    population = initialize_pop(pop_size=pop_size)
    with open(file="evolved_pop.json", mode="r") as fh:
        population = json.load(fp=fh)
    population = evaluate_group(individuals=population)
    rank_group(individuals=population)
    best_fitness = population[0]["fitness"]
    counter = 0
    while counter < 100:
        counter += 1
        parents = parent_select(individuals=population, number=70)
        children = recombine_group(parents=parents, recombine_rate=0.7)
        mutate_sigma = 0.01
        mutants = mutate_group(children=children, mutate_sigma=mutate_sigma)
        mutants = evaluate_group(individuals=mutants)
        everyone = population + mutants
        rank_group(individuals=everyone)
        population = survivor_select(individuals=everyone, pop_size=pop_size)
        if best_fitness != population[0]["fitness"]:
            best_fitness = population[0]["fitness"]
            print("Iteration number", counter, "with best individual", population[0])
    return population


if __name__ == "__main__":
    """
    Don't change this section:
    """
    population = evolve()
    with open(file="evolved_pop.json", mode="w") as fh:
        json.dump(obj=population, fp=fh)
    grade = int(round((population[0]["fitness"] / 23 * 0.9) * 100))
    with open(file="results.txt", mode="w") as f:
        f.write(str(grade))
