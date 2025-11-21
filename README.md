# Multi-Peaks Autopointer for SAM2
A lightweight extension of PerSAM for multi-part object segmentation on SAM2.
Given one reference image + mask, the method extracts dense foreground features, clusters them into K semantic centers, and generates multi-peak similarity maps on test images.
Peak locations become positive points, plus one negative point, forming an automatic prompt set for SAM2.
### Features
- Supports multi-component / separated objects
- K-Means on dense foreground features
- Multi-peak similarity → automatic multi-point prompts
- Training free
