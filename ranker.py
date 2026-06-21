import pandas as pd
from pathlib import Path

DIR = Path(__file__).parent
CACHE_FILE = DIR / "features_cache.parquet"
OUTPUT_CSV = DIR / "CULT.csv"

def generate_reasoning(row):
    reasons = []
    
    reasons.append(f"Candidate has {row['yoe']} years of experience, currently working as {row['current_title']}.")
    
    if row['demo_fit'] > 0:
        reasons.append("Demonstrated experience shipping ranking/search systems at a product company.")
    else:
        reasons.append("Lacks clear evidence of shipping ranking/search systems in production.")
        
    if row['eval_sig'] > 0:
        reasons.append("Shows strong understanding of evaluation metrics (NDCG/MRR/MAP).")
        
    if row['avail_pen'] > 0:
        reasons.append(f"Noted availability concerns (Penalty: -{row['avail_pen']:.2f}).")
        
    if pd.notna(row['skills_str']) and row['skills_str']:
        reasons.append(f"Listed skills include: {row['skills_str']}.")

    return " ".join(reasons)

def run_ranking():
    print(f"Loading {CACHE_FILE}...")
    df = pd.read_parquet(CACHE_FILE)
    
    df_valid = df[df['is_valid'] == True].copy()
    
    print(f"Candidates after hard filters: {len(df_valid)}")
    
    df_valid['score'] = (
        0.30 * df_valid['similarity'] +
        0.25 * df_valid['demo_fit'] +
        0.15 * df_valid['exp_fit'] +
        0.10 * df_valid['eval_sig'] -
        0.20 * df_valid['avail_pen']
    )
    
    df_sorted = df_valid.sort_values(by=['score', 'candidate_id'], ascending=[False, True])
    
    top_100 = df_sorted.head(100).copy()
    
    top_100['rank'] = range(1, 101)
    
    top_100['reasoning'] = top_100.apply(generate_reasoning, axis=1)
    
    output_cols = ['candidate_id', 'rank', 'score', 'reasoning']
    top_100['score'] = top_100['score'].round(4)
    
    top_100[output_cols].to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"Successfully generated {OUTPUT_CSV}")

if __name__ == "__main__":
    run_ranking()
