# Ban-topic exclusion check (Check 1)

Compare baseline subreddit-day DiD to samples excluding comments matching the ban-topic regex (`is_ban_topic`).

**Interpretation:**
- If `sem_axis_emotion` week-1 dip and `ai_style_rate` / `style_index_llm` bumps vanish on the exclusion sample → attention-shock (ban-discussion vocabulary) confirmed.
- If the emotion dip survives exclusion → more consistent with a genuine discourse shift.

## Key coefficients (`cross_country_all`)

- **ai_style_rate** (early_ban_7d): baseline β=0.0190 (p=0.277) → exbantopic β=0.0170 (p=0.326); Δβ=-0.0019
- **ai_style_rate** (full_ban): baseline β=0.0013 (p=0.891) → exbantopic β=0.0034 (p=0.717); Δβ=0.0021
- **ai_style_rate** (post_first_2bd): baseline β=0.0287 (p=0.168) → exbantopic β=0.0301 (p=0.143); Δβ=0.0014
- **cognition_rate** (early_ban_7d): baseline β=0.1109 (p=0.083) → exbantopic β=0.0997 (p=0.126); Δβ=-0.0112
- **cognition_rate** (full_ban): baseline β=0.0041 (p=0.902) → exbantopic β=0.0016 (p=0.963); Δβ=-0.0025
- **cognition_rate** (post_first_2bd): baseline β=nan (p=nan) → exbantopic β=0.0642 (p=0.394); Δβ=nan
- **emotion_rate** (early_ban_7d): baseline β=-0.0428 (p=0.223) → exbantopic β=-0.0427 (p=0.222); Δβ=0.0001
- **emotion_rate** (full_ban): baseline β=-0.0136 (p=0.558) → exbantopic β=-0.0147 (p=0.529); Δβ=-0.0010
- **emotion_rate** (post_first_2bd): baseline β=nan (p=nan) → exbantopic β=-0.0348 (p=0.0726); Δβ=nan
- **pole_share** (early_ban_7d): baseline β=0.0171 (p=0.559) → exbantopic β=0.0170 (p=0.557); Δβ=-0.0001
- **pole_share** (full_ban): baseline β=0.0623 (p=0.00324) → exbantopic β=0.0649 (p=0.0027); Δβ=0.0026
- **pole_share** (post_first_2bd): baseline β=0.0081 (p=0.883) → exbantopic β=0.0090 (p=0.871); Δβ=0.0009
- **sem_axis_emotion** (early_ban_7d): baseline β=-0.0093 (p=0.00492) → exbantopic β=-0.0093 (p=0.00738); Δβ=-0.0000
- **sem_axis_emotion** (full_ban): baseline β=-0.0033 (p=0.105) → exbantopic β=-0.0033 (p=0.108); Δβ=-0.0000
- **sem_axis_emotion** (post_first_2bd): baseline β=-0.0123 (p=0.0146) → exbantopic β=-0.0123 (p=0.0136); Δβ=-0.0001
- **sem_axis_ideology** (early_ban_7d): baseline β=-0.0081 (p=0.00287) → exbantopic β=-0.0083 (p=0.00291); Δβ=-0.0002
- **sem_axis_ideology** (full_ban): baseline β=-0.0038 (p=0.067) → exbantopic β=-0.0039 (p=0.0557); Δβ=-0.0002
- **sem_axis_ideology** (post_first_2bd): baseline β=-0.0065 (p=0.108) → exbantopic β=-0.0060 (p=0.149); Δβ=0.0006
- **style_index_llm** (early_ban_7d): baseline β=0.0303 (p=0.309) → exbantopic β=0.0278 (p=0.35); Δβ=-0.0025
- **style_index_llm** (full_ban): baseline β=0.0037 (p=0.772) → exbantopic β=0.0083 (p=0.445); Δβ=0.0046
- **style_index_llm** (post_first_2bd): baseline β=0.0506 (p=0.144) → exbantopic β=0.0523 (p=0.121); Δβ=0.0018

Full table: `results/tables/.../did/exbantopic_comparison.csv` (path resolved from study config).
