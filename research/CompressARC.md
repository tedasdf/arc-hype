# ARC-AGI Without Pretraining — CompressARC

## Overview

CompressARC is a **76k-parameter neural network with no pretraining** designed to solve ARC-AGI puzzles through compression by finding a **short program or hypothesis** that explains the observed input-output examples and perform this search inside a constrained neural network. As training happens **independently for each puzzle**. The model is not pretrained on a large ARC dataset. For every test puzzle, its parameters and latent variables are optimized using only the information available in that puzzle. This gives CompressARC an unusual form of generalisation: rather than learning how to solve ARC from previous examples, it tries to discover a compact explanation for each new puzzle from scratch.

---

## Core Idea: Compression as Program Search

CompressARC follows a Minimum Description Length style with the single goal of explaining the puzzle with little puzzle-specific information. While A hypothesis that simply memorises every output pixel could obtain very low reconstruction error, but it would require a large amount of information. CompressARC therefore penalises the amount of information stored in its puzzle-specific latent variable (z).

Conceptually:

$$
\mathcal{L}
=
\text{reconstruction error}
+
\text{description length}
$$

where reconstruction is measured using cross-entropy and the information stored in (z) is controlled using KL divergence.

This makes CompressARC similar to **program synthesis implemented through a neural network**. The architecture defines a restricted neural "program language", while optimisation searches for a short computation within that language that explains the puzzle.

---


## Multitensor Representation

The internal representation used by CompressARC is called a **multitensor**.

A multitensor is a collection of tensors with different legal shapes. These shapes are formed using subsets of several predefined axes:

* (E): example
* (C): colour
* (D): direction
* (H): height
* (W): width

The channel dimension is always included.

Rather than having a single representation such as:

`[batch, channel, H, W]`

CompressARC maintains multiple representations such as:

* `[E, H, channel]`
* `[E, W, channel]`
* `[E, C, H, W, channel]`
* other legal combinations

There are **18 legal tensor types**.

These representations do not correspond directly to semantic concepts such as "object", "shape", or "symmetry". Instead, each tensor provides a representational space for relationships between its included axes.

For example:

`[E, C, H, W]`

can represent information involving a particular colour at a particular spatial position within a particular example.

This means CompressARC's representation is powerful but heavily structured around a fixed set of relatively low-level relational axes.

---

## Decoding Layer and Latent (z)

CompressARC begins with a puzzle-specific latent representation (z).

The decoding layer determines how much information (z) is allowed to contain.

A learned **target capacity** specifies approximately how many bits of puzzle-specific information each latent tensor should carry.

Gaussian noise is added to the latent variables. The amount of noise is chosen according to the desired channel capacity:

$$
\text{more noise}
\Rightarrow
\text{less information}
$$

$$
\text{less noise}
\Rightarrow
\text{more information}
$$

KL divergence measures the information cost of deviating from the prior distribution.

Therefore:

$$
KL \uparrow
\Rightarrow
z \text{ contains more puzzle-specific information}
$$

while:

$$
KL \downarrow
\Rightarrow
z \text{ contains less information}
$$

This prevents the model from simply storing every output pixel inside (z).

The latent variable therefore behaves similarly to a **compressed neural program seed**: it contains the minimum puzzle-specific information required for the network computation to reconstruct the solution.

---

## Core Network

After decoding (z), the multitensor is processed by an equivariant core network.

The architecture deliberately contains computational primitives useful for ARC-style reasoning.

### Multitensor Communication

Allows information to move between different multitensor types.

A representation containing additional dimensions can be reduced, while information from a lower-rank tensor can be broadcast into a higher-rank representation.

For example:

`[E, C, H, W] -> [E, H, W]`

can aggregate information across colours.

This allows information discovered in one relational representation to influence another.

### Softmax

Softmax provides a **selection and sharpening mechanism**.

Rather than only converting output logits into probabilities, it allows internal representations to become approximately one-hot along selected dimensions.

This can allow the network to strongly select a particular colour, position, direction, or other candidate.

### Directional Shift

Shift operations allow information to move spatially by one step in a particular direction.

This gives the network a primitive similar to:

`move / translate information`

while preserving the architecture's spatial equivariances.

### Directional Cummax

Cummax allows information to propagate over larger spatial distances along a direction.

For example:

`[0, 0, 5, 0, 0, 0] -> [0, 0, 5, 5, 5, 5]`

Once a strong feature is encountered, it remains available to later positions along that direction.

This enables **long-range spatial propagation in a single operation**.

### Directional Communication

Allows information associated with different directions to interact.

Because the direction axis transforms consistently under rotations and reflections, the network can reason across orientations without arbitrarily privileging a particular direction.

### Nonlinear Layers

Nonlinear transformations allow the network to compose simple features into more complex computations.

Without nonlinearities, repeatedly stacking linear operations would still reduce to a linear transformation.

### Normalisation

Normalisation stabilises internal activations while preserving the required equivariances.

### Linear Head

The final linear head converts the final multitensor representation into predictions for the ARC output grid, including colour logits and output-grid shape information.

---

## Equivariance as an Inductive Bias

A major part of CompressARC's efficiency comes from hard-coding symmetries into the architecture.

The network is designed to respect transformations such as:

* permutation of examples
* permutation of colours
* rotations
* reflections

This means the model does not need to independently learn every equivalent version of the same rule.

For example, if a solution works after rotating a puzzle by (90^\circ), the architecture is structured so that its internal representation rotates consistently as well.

If the puzzle requires two otherwise equivalent entities to be treated differently, the latent variable (z) must encode that distinction.

This is referred to as **breaking a symmetry**, and doing so requires additional information, increasing the KL cost.

Therefore the model prefers symmetric, simple explanations unless breaking the symmetry significantly improves reconstruction.

---

## Interpretation

CompressARC can be viewed as a form of **compressed neural program synthesis**.

Traditional symbolic program synthesis might search over operations such as:

* rotate
* translate
* crop
* recolour

CompressARC instead provides neural computational primitives such as:

* shift
* cummax
* multitensor communication
* softmax
* directional communication

Optimisation then searches for a compact puzzle-specific latent and network configuration that composes these primitives into a solution.

The restricted architecture keeps this search feasible, but this restriction also creates an important limitation.

The explicit representational axes are limited to:

$$
E, C, D, H, W
$$

Higher-level concepts such as:

* objects
* object-object relations
* counting
* connectivity
* arbitrary shape transformations

do not have explicit representational axes or dedicated computational primitives.

Therefore CompressARC exposes an important trade-off:

$$
\text{restricted hypothesis space}
\Rightarrow
\text{efficient search}
$$

but also:

$$
\text{restricted hypothesis space}
\Rightarrow
\text{limited expressiveness}
$$



Research question including :

> **How can CompressARC’s representation or computational space be relaxed while preserving the compression bias that keeps search tractable?**

> **Can several hand-designed axis-specific primitives be replaced by a more general operator over arbitrary multitensor axes without hurting search efficiency or generalisation?**



## plan
- Debug stage: 3–5 tasks, just to make sure the relaxed tensor system runs and the new tensors actually appear.
- Exploration stage: 20 tasks, ideally mixed difficulty / transformation types. Run multiple seeds or repeated restarts per task because CompressARC is stochastic.
- Evidence stage: 50–100 tasks if the 20-task result looks interesting.
- Full 400: only after the experiment, metrics, and run procedure are stable.