# Introduction

Vision is the process of discovering from images what is present in the world and where it is.

## Perception/Understanding
High-Level Representaiton that captures semantic structure and objects in scene

Vision as Measurement
- Stereo and Structure from Motion

For Robotics
- Object Segmentation and Categorization
- 3D Understanding
- Functional Understanding
    - Semantic relations
- Activity Recognition

Why are these problems hard?
- Hard to tokenize
- view-point variation
- lighting/illumination
- occlusion (partially blocked by something in front of it)
    - requires lots of priors
- scale
- deformation
    - more difficult than rigid objects(cup vs a horse with legs)
- background clutter
- Object intra-class variation
    - Ex: various types of "chairs" from normal to modern art looking
    - However, fortunately, Army stuff and cars and large outdoors objects are all only subtly variable
- local ambiguity