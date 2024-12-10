---
title: Lab2
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.0.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: lab2
---

# Improving Model Performance: Model-Centric Approach

## Model-Centric Approach
To enhance the performance of our model, we adopted a **model-centric strategy**, specifically tuning the rank of the LoRA (Low-Rank Adaptation) model. The evaluation results below show the model's performance across three datasets: **HellaSwag**, **WinoGrande**, and **Abstract Algebra**. By adjusting the rank (denoted as `r`), we observed meaningful variations in accuracy across tasks, demonstrating that the choice of rank is critical for model optimization.

### Datasets
1. **HellaSwag**:  
   A benchmark designed to test a model's ability to complete sentences based on commonsense reasoning. It features adversarial examples to ensure that high performance requires deep understanding rather than superficial pattern matching.
   
2. **WinoGrande**:  
   A dataset focused on commonsense pronoun resolution. Each question presents an ambiguous pronoun and requires selecting the correct referent, which is challenging for models due to the subtle reasoning involved.
   
3. **Abstract Algebra**:  
   A dataset that evaluates a model's understanding of advanced mathematical concepts, specifically algebraic structures like groups, rings, and fields.

### Evaluation Results

| Rank | HellaSwag       | WinoGrande      | Abstract Algebra |
|------|-----------------|-----------------|------------------|
| r8   | 0.591           | <span style="color:red">0.200</span> | 0.595            |
| r16  | **<span style="color:green">0.592</span>** | 0.210           | 0.594            |
| r32  | 0.590           | **<span style="color:green">0.230</span>** | **<span style="color:green">0.597</span>** |
| r64  | <span style="color:red">0.588</span> | 0.210           | <span style="color:red">0.585</span> |

### Findings
- On **HellaSwag**, the performance peaked at rank **r16** (0.592) but showed minimal sensitivity to rank changes overall.
- For **WinoGrande**, performance improved consistently up to rank **r32**, reaching 0.230. This demonstrates that more capacity (via higher rank) benefits tasks requiring subtle reasoning.
- In **Abstract Algebra**, rank **r32** yielded the best result (0.597), while larger ranks (e.g., r64) slightly degraded performance, likely due to overfitting.

**Conclusion**: These findings suggest that while increasing the rank provides more capacity for adaptation and can improve performance, overly large ranks can lead to diminishing returns and overfitting. Careful tuning is essential to balance these effects and achieve optimal results.



