**Fixed-Point Convergence of Multi-block PnP ADMM and Its Application to Hyperspectral Image Restoration**

Abstract: Coupling methods of integrating multiple priors  have emerged as a pivotal research focus in  hyperspectral image (HSI)  restoration. Among these methods, the Plug-and-Play (PnP) framework stands out and pioneers a novel coupling approach, enabling flexible integration of diverse methods into model-based approaches. However, the current convergence analyses of the PnP framework are highly unexplored, as they are limited to 2-block composite optimization problems, failing to meet the need of coupling modeling for incorporating multiple priors. This paper focuses on the  convergence analysis of PnP-based algorithms for multi-block composite optimization problems. In this work, under the PnP framework and utilizing the alternating direction method of multipliers (ADMM) of the continuation scheme, we propose a unified multi-block PnP ADMM algorithm framework for HSI restoration. Inspired by the fixed-point convergence theory of the 2-block PnP ADMM, we establish a similar fixed-point convergence guarantee for the multi-block PnP ADMM with extended condition and provide a feasible parameter tuning methodology. Based on this framework, we design an effective mixed noise removal algorithm incorporating global, nonlocal and deep priors. Extensive experiments validate the algorithm's superiority and competitiveness. The open-source code is available online at https://github.com/Weijie-Liang/FBGND.

**Citation**

@ARTICLE{FBGND,  
author={Liang, Weijie and Tu, Zhihui and Lu, Jian and Tu, Kai and Ng, Michael K. and Xu, Chen},  
journal={IEEE Transactions on Computational Imaging},   
title={Fixed-Point Convergence of Multi-block PnP ADMM and Its Application to Hyperspectral Image Restoration},   
year={2024},  
doi={10.1109/TCI.2024.3485467}}