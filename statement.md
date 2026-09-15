# Problem Statement

## The Problem

Managing crowd density in public venues such as malls, metro stations,
exhibitions, and event halls is a persistent safety challenge.  Manual
headcounts are slow and inaccurate; traditional CCTV gives only a raw
video feed with no automated analysis.  When a zone becomes dangerously
overcrowded, security staff may not notice until it is too late.

## Scope

This project delivers a real-time, automated crowd monitoring system
that processes live video (webcam or recorded footage) and provides:

- Continuous person detection and tracking
- Entry/exit counting through a configurable virtual line
- Zone-level occupancy monitoring with capacity enforcement
- Instant overcrowding alerts on screen and console
- Crowd density heatmap for spatial analysis
- Higher-level analytics: cluster detection, density classification,
  spread index, and scene-state classification

The system runs on a standard laptop CPU without specialised hardware.
A GPU will increase frame rate but is not required.

## Target Users

- **Venue security teams** — receive real-time alerts before situations
  become dangerous
- **Event organisers** — monitor occupancy across different sections of
  a venue
- **Urban planners / researchers** — analyse crowd flow patterns over time
- **Students** — learn applied computer vision across the full pipeline
  from raw pixels to high-level analytics

## High-Level Features

1. Plug-and-play support for any USB webcam or video file
2. Configurable zones, capacities, and counting line in a single file
3. Modular architecture — each processing stage is an independent module
4. Keyboard shortcuts for toggling all visual overlays at runtime
5. Snapshot saving for incident documentation
6. Full coverage of the Computer Vision curriculum:
   - Image formation and representation (Module 1)
   - Enhancement, morphological ops, histogram equalisation (Module 2)
   - Edge detection, corner detection, background subtraction (Module 3)
   - Segmentation, K-Means spatial clustering, density heatmap (Module 4)
   - KNN classification, Naive Bayes scene state, PCA spread, YOLOv8 detection,
     centroid tracking (Module 5)
