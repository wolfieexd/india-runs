import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

DIR = Path(__file__).parent
INPUT_FILE = DIR / "candidates.jsonl"
OUTPUT_CACHE = DIR / "features_cache.parquet"

CONSULTING_FIRMS = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"}
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def is_honeypot(candidate):
    skills = candidate.get("skills", [])
    for s in skills:
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) <= 1:
            return True
            
    career = candidate.get("career_history", [])
    for c in career:
        sd = c.get("start_date")
        if sd:
            try:
                dt = datetime.strptime(sd, "%Y-%m-%d")
                if dt.year > 2024:
                    return True
            except:
                pass
    return False

def extract_features(candidate):
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    
    is_valid = True
    disqualify_reason = ""
    
    companies = [c.get("company", "").lower() for c in career]
    is_services_only = len(companies) > 0 and all(any(cf in comp for cf in CONSULTING_FIRMS) for comp in companies)
    if is_services_only:
        is_valid = False
        disqualify_reason = "Services only, no product"
        
    has_prod_eng = False
    has_research = False
    for c in career:
        title = c.get("title", "").lower()
        if "engineer" in title or "developer" in title:
            has_prod_eng = True
        if "research" in title or "academic" in title or "phd" in title:
            has_research = True
            
    if has_research and not has_prod_eng:
        is_valid = False
        disqualify_reason = "Pure research"
        
    total_months = sum(c.get("duration_months", 0) for c in career)
    if len(career) > 2 and total_months / len(career) < 18:
        is_valid = False
        disqualify_reason = "Job hopper"
        
    if len(career) > 0:
        recent_role = career[0]
        title = recent_role.get("title", "").lower()
        if ("architect" in title or "tech lead" in title) and recent_role.get("duration_months", 0) > 18:
            is_valid = False
            disqualify_reason = "No recent production code"
            
    if is_honeypot(candidate):
        is_valid = False
        disqualify_reason = "Honeypot"

    demo_fit = 0.0
    for c in career:
        desc = c.get("description", "").lower()
        comp = c.get("company", "").lower()
        if not any(cf in comp for cf in CONSULTING_FIRMS):
            if any(k in desc for k in ["ranking", "search", "retrieval", "recommendation"]):
                demo_fit = 1.0
                break

    exp_fit = 0.0
    yoe = profile.get("years_of_experience", 0)
    if 6 <= yoe <= 9:
        exp_fit += 0.5
    ml_months = sum(c.get("duration_months", 0) for c in career if "machine learning" in c.get("title", "").lower() or "ai" in c.get("title", "").lower())
    if ml_months >= 48:
        exp_fit += 0.5
        
    eval_sig = 0.0
    all_desc = " ".join([c.get("description", "").lower() for c in career])
    if any(k in all_desc for k in ["ndcg", "mrr", "map", "a/b test"]):
        eval_sig = 1.0
        
    avail_pen = 0.0
    la = signals.get("last_active_date")
    if la:
        try:
            la_dt = datetime.strptime(la, "%Y-%m-%d")
            if (datetime(2024, 5, 1) - la_dt).days > 90:
                avail_pen += 0.4
        except:
            pass
            
    if signals.get("recruiter_response_rate", 1.0) < 0.3:
        avail_pen += 0.3
    if signals.get("interview_completion_rate", 1.0) < 0.5:
        avail_pen += 0.2
    if not signals.get("open_to_work_flag", True):
        avail_pen += 0.5
        
    avail_pen = min(1.0, avail_pen)

    return {
        "candidate_id": candidate.get("candidate_id"),
        "is_valid": is_valid,
        "disqualify_reason": disqualify_reason,
        "demo_fit": demo_fit,
        "exp_fit": exp_fit,
        "eval_sig": eval_sig,
        "avail_pen": avail_pen,
        "yoe": yoe,
        "current_title": profile.get("current_title", ""),
        "open_to_work_flag": signals.get("open_to_work_flag", True),
        "recruiter_response_rate": signals.get("recruiter_response_rate", 1.0),
        "skills_str": ", ".join([s.get("name") for s in skills[:3]])
    }

def process_offline():
    print(f"Loading {INPUT_FILE}...")
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        if torch.version.hip:
            print("Initializing embedding model on AMD GPU (ROCm)")
        else:
            print("Initializing embedding model on NVIDIA GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Initializing embedding model on Apple Silicon (MPS)")
    else:
        print("Initializing embedding model on CPU")
    
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    
    jd_text = """
    Senior AI Engineer — Founding Team, Redrob AI.
    Applied ML/AI experience shipping end-to-end ranking, search, retrieval, or recommendation systems to real users at meaningful scale.
    Strong experience with vector databases, embeddings, hybrid retrieval, and evaluation metrics like NDCG, MRR, MAP.
    Must have hands-on production engineering experience, not just pure research or academic background.
    """
    jd_embedding = model.encode(jd_text, normalize_embeddings=True)
    
    features_list = []
    
    print("Processing candidates and computing embeddings...")
    count = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        batch_texts = []
        batch_feats = []
        
        for line in tqdm(f):
            count += 1
            candidate = json.loads(line)
            feats = extract_features(candidate)
            
            if not feats["is_valid"]:
                feats["similarity"] = 0.0
                features_list.append(feats)
                continue
                
            career = candidate.get("career_history", [])
            summary = candidate.get("profile", {}).get("summary", "")
            career_desc = " ".join([c.get("description", "") for c in career])
            text_to_embed = summary + " " + career_desc
            
            batch_texts.append(text_to_embed)
            batch_feats.append(feats)
            
            if len(batch_texts) >= 512:
                embs = model.encode(batch_texts, normalize_embeddings=True)
                for i, emb in enumerate(embs):
                    sim = (emb @ jd_embedding)
                    batch_feats[i]["similarity"] = float(sim)
                    features_list.append(batch_feats[i])
                batch_texts = []
                batch_feats = []
                
        if batch_texts:
            embs = model.encode(batch_texts, normalize_embeddings=True)
            for i, emb in enumerate(embs):
                sim = (emb @ jd_embedding)
                batch_feats[i]["similarity"] = float(sim)
                features_list.append(batch_feats[i])
                
    print(f"Total processed: {count}")
    df = pd.DataFrame(features_list)
    df.to_parquet(OUTPUT_CACHE)
    print(f"Saved cache to {OUTPUT_CACHE}")

if __name__ == "__main__":
    process_offline()
