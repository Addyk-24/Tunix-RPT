# TUNIX: Post-Training for Reasoning

A high-performance pipeline with Scaling Logical Inference for Post Training **Gemma 3 1B** into a specialized reasoning engine using **Tunix, JAX, and Cloud TPUs v3–8/v5e-1**

### Check out My blog related to This 👉 **[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/@addykatkar24/tunix-reasoning-post-training-9bfa54c3e974)**

## 📌 Project Overview
This project focuses on the post-training phase of Large Language Models to enhance logical deduction and multi-step reasoning. By leveraging the **JAX/Flax** ecosystem, we achieve massive throughput on TPU v3-8 hardware, overcoming traditional bottlenecks associated with dynamic shapes in transformer training.

### Key Objectives:
* **Reasoning Alignment:** Transforming general-purpose knowledge into structured "Chain of Thought" (CoT) logic.
* **JAX Optimization:** Solving the XLA "compilation hang" and memory fragmentation on TPUs.
* **Hardware Efficiency:** Utilizing a TPU v3-8 mesh with LoRA (Low-Rank Adaptation) via Tunix.

---

## 🛠️ Technical Stack
* **Model:** Google Gemma 3 (1B Variant)
* **Framework:** Tunix (Post-training framework for JAX/Flax)
* **Compute:** Google Cloud TPU v3-8
* **Optimization:** Optax (AdamW with Cosine Schedule)
* **Data:** Hugging Face Datasets (Multi-domain rebalanced)

---

## 🚀 Key Technical Hurdles & Solutions

### 1. The "Inhomogeneous Shape" Fix (Static Padding)
TPUs require fixed-size buffers. Standard tokenization creates "jagged" arrays that crash the XLA compiler.
* **Solution:** Implemented a custom data collator that enforces a strict `MAX_TARGET_LENGTH`. This ensures every batch is a perfect rectangle, allowing JAX to compile the computation graph exactly once.



### 2. Solving the "BTNS" Einsum Error
Gemma 3's Multi-Head Attention expects specific 4D tensor alignments.
* **Solution:** Manually expanded 2D attention masks into a **4D broadcastable format** `[Batch, 1, 1, Sequence]`. This aligned the dimensions for the Einstein Summation (`einsum`) kernels in the attention layers.



### 3. JAX Data Tracing
JAX cannot trace Python strings. Residual metadata in datasets often causes `_str_abstractify` errors.
* **Solution:** Developed a pre-processing pipeline that strips all non-numeric columns, leaving only the raw integer `input_ids` and `attention_mask` for the TPU.


## 📈 Performance & Results
* **Compilation Speed:** After the initial Step 0 JAX trace, training stabilized at millisecond execution speeds per step.
* **Reasoning Delta:** Post-trained models showed a marked increase in using "Chain of Thought" markers compared to the base model.

## Tunix Reasoning Architecture & Hyperparameters Configs:


<p align="center">
<img width="529" height="366" alt="balance_ds_image" src="https://github.com/user-attachments/assets/bdcfb24d-4da6-49ce-bc91-23b46a8aa907" />
<br/>
<i>Final Rebalancing Dataset</i>
</p>
<br/>

<p align="center">
<img width="940" height="232" alt="enhanced_hardware_check" src="https://github.com/user-attachments/assets/d9285938-74c9-4a86-9684-614f4614ea8a" />
<i>Hardware Check</i>
</p>
<br/>
<p align="center">
<img width="529" height="366" alt="model_config" src="https://github.com/user-attachments/assets/6a5e204f-4bf3-4ff0-896f-20b32b0eb24c" />
<br/>
<i>Model Configs</i>
</p>
<br/>

---

## 📊 Data Strategy
We utilized **Stratified Post-Training**, rebalancing the model across six critical reasoning domains:
1.  **Mathematics:** Step-by-step problem solving.
2.  **Coding:** Logic-heavy algorithm generation.
3.  **Science:** Deductive reasoning.
4.  **Creative:** Instruction following.
5.  **Summarization:** Contextual logic.
6.  **General:** General-knowledge reasoning.

---
## 👩‍💻 Author
Aditya Katkar </br>
[GitHub](https://github.com/Addyk-24) </br>
[LinkedIn](https://www.linkedin.com/in/aditya-katkar-673930340)