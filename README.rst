=======
Rxitect
=======
------------------------------------------------------------------------------------
A de-novo drug design library for creating retrosynthesis-aware reward-driven models
------------------------------------------------------------------------------------

Introduction
============

This library was made for my M.Sc. thesis research with the aim of understanding
how computational chemists can incorporate synthesis planning into de-novo drug design
systems. Many molecule generators propose interesting but impractical molecules, which is why we need
to design them with synthesizability in mind. Modern Computer-Assisted Synthesis Planning (CASP) tools are quite powerful
but are of limited use in algorithms that need to call said tools many times (e.g., > 100.000 calls)
due to the time it takes to solve a single molecule on average. This research aims to
create a useful proxy that is cheap to call yet robust, and then using a myriad of techniques
that are known to be effective in searching the vast molecular search space such as Reinforcement Learning (RL),
and Generative Flow Networks (GFlowNets), we can experimentally test if these proxies are useful to propose more
practical and synthesizable molecules.

Quickstart
----------
Run the following code to get up and running
```
conda env create -f environment.yml
# alternatively you can use mamba, which I recommend
conda activate rx
poetry install
```

Examples
--------
Coming Soon!
