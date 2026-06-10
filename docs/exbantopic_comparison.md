# Ban-topic exclusion check (Check 1)

Compare baseline subreddit-day DiD to samples excluding comments matching the ban-topic regex (`is_ban_topic`).

**Interpretation:**
- If `sem_axis_emotion` week-1 dip and `ai_style_rate` / `style_index_llm` bumps vanish on the exclusion sample → attention-shock (ban-discussion vocabulary) confirmed.
- If the emotion dip survives exclusion → more consistent with a genuine discourse shift.

## Key coefficients (`cross_country_all`)

- **ai_style_rate** (early_ban_7d): baseline β=0.0208 (p=0.191) → exbantopic β=0.0173 (p=0.266); Δβ=-0.0035
- **ai_style_rate** (full_ban): baseline β=0.0013 (p=0.891) → exbantopic β=0.0034 (p=0.717); Δβ=0.0021
- **ai_style_rate** (post_first_2bd): baseline β=0.0287 (p=0.168) → exbantopic β=0.0301 (p=0.143); Δβ=0.0014
- **cognition_rate** (early_ban_7d): baseline β=0.1218 (p=0.0482) → exbantopic β=0.1128 (p=0.0686); Δβ=-0.0090
- **cognition_rate** (full_ban): baseline β=0.0041 (p=0.902) → exbantopic β=0.0016 (p=0.963); Δβ=-0.0025
- **cognition_rate** (post_first_2bd): baseline β=0.0760 (p=0.306) → exbantopic β=0.0642 (p=0.394); Δβ=-0.0118
- **emotion_rate** (early_ban_7d): baseline β=-0.0429 (p=0.168) → exbantopic β=-0.0421 (p=0.175); Δβ=0.0008
- **emotion_rate** (full_ban): baseline β=-0.0136 (p=0.558) → exbantopic β=-0.0147 (p=0.529); Δβ=-0.0010
- **emotion_rate** (post_first_2bd): baseline β=-0.0346 (p=0.0731) → exbantopic β=-0.0348 (p=0.0726); Δβ=-0.0001
- **pole_share** (early_ban_7d): baseline β=-0.0087 (p=0.771) → exbantopic β=-0.0096 (p=0.748); Δβ=-0.0009
- **pole_share** (full_ban): baseline β=0.0623 (p=0.00324) → exbantopic β=0.0649 (p=0.0027); Δβ=0.0026
- **pole_share** (post_first_2bd): baseline β=0.0081 (p=0.883) → exbantopic β=0.0090 (p=0.871); Δβ=0.0009
- **sem_axis_emotion** (early_ban_7d): baseline β=-0.0087 (p=0.00511) → exbantopic β=-0.0088 (p=0.0073); Δβ=-0.0000
- **sem_axis_emotion** (full_ban): baseline β=-0.0033 (p=0.1) → exbantopic β=-0.0033 (p=0.108); Δβ=-0.0000
- **sem_axis_emotion** (post_first_2bd): baseline β=-0.0123 (p=0.0146) → exbantopic β=-0.0123 (p=0.0136); Δβ=-0.0001
- **sem_axis_ideology** (early_ban_7d): baseline β=-0.0072 (p=0.00333) → exbantopic β=-0.0073 (p=0.00346); Δβ=-0.0001
- **sem_axis_ideology** (full_ban): baseline β=-0.0038 (p=0.0675) → exbantopic β=-0.0039 (p=0.0557); Δβ=-0.0002
- **sem_axis_ideology** (post_first_2bd): baseline β=-0.0065 (p=0.108) → exbantopic β=-0.0060 (p=0.149); Δβ=0.0006
- **style_index_llm** (early_ban_7d): baseline β=0.0328 (p=0.253) → exbantopic β=0.0272 (p=0.337); Δβ=-0.0056
- **style_index_llm** (full_ban): baseline β=0.0037 (p=0.772) → exbantopic β=0.0083 (p=0.445); Δβ=0.0046
- **style_index_llm** (post_first_2bd): baseline β=0.0506 (p=0.144) → exbantopic β=0.0523 (p=0.121); Δβ=0.0018

Full table: `results/tables/.../did/exbantopic_comparison.csv` (path resolved from study config).
