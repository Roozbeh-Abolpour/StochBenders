# StochBenders

A structure-exploiting Benders decomposition framework for large-scale MILPs with scenario-wise and component-wise separability.

---

## 🔍 Motivation

Modern optimization problems in energy systems, microgrids, and stochastic programming exhibit:

- Scenario-wise decomposition  
- Weak coupling through binary variables  
- Natural partition into subsystems (DERs, batteries, grid, etc.)

Standard Benders decomposition does not fully exploit this structure. In many real-world problems, the optimization model is inherently modular, but classical BD treats it as monolithic, leading to inefficiencies in both convergence and computational time.

---

## 🚀 Contribution

This repository implements a **two-phase, structure-exploiting Benders framework** designed for real large-scale MILPs.

### Implemented methods

- Classical Benders Decomposition (**BD**)  
- Strengthened Benders Decomposition (**SBD**)  
- Two-phase decomposition with structural exploitation  

### Key features

- Multi-level decomposition (system → master → subproblems)  
- Parallelizable subproblem structure  
- Cut aggregation across subsystems  
- Efficient feasibility cut generation (Farkas-based)  
- Exploitation of sparsity and block structure  

---

## 🧠 Methodological Insight

The framework is based on the observation that large MILPs often admit **three levels of separability**:

1. **Problem-level decomposition**  
   Independent or weakly coupled subsystems (e.g., DERs, loads, markets)

2. **Master-level decomposition**  
   Binary or here-and-now variables coordinating subsystems

3. **Subproblem-level decomposition**  
   Scenario-wise or component-wise LPs

Instead of applying BD once, this framework:

- Identifies structure  
- Separates components  
- Solves subproblems in parallel  
- Aggregates cuts efficiently  

This leads to significantly reduced wall-clock time in large-scale problems.

---

## 📦 Installation

```bash
pip install -e .