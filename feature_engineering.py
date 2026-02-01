import pandas as pd
import numpy as np

def calculate_multi_factor_score(student_data, exam_type, params=None):
    """
    Calculates score based on the specific Step-by-Step Architecture.
    Allows for parameter tuning via the params dictionary.
    """
    if params is None:
        params = {
            'subject_threshold': 65,
            'high_avg_threshold': 70,
            'passing_avg_threshold': 55,
            'target_study_hours': 6,
            'min_sleep_hours': 6,
            'max_screen_time': 6,
            'continuation_score_threshold': 5
        }

    score = 0
    breakdown = {}
    recommendation = []
    
    # --- Step 3 & 4: Subject Analysis ---
    subjects = []
    if exam_type == 'JEE':
        subjects = [student_data.get('Physics_Marks', 0), student_data.get('Chemistry_Marks', 0), student_data.get('Maths_Marks', 0)]
    elif exam_type == 'NEET':
        subjects = [student_data.get('Physics_Marks', 0), student_data.get('Chemistry_Marks', 0), student_data.get('Biology_Marks', 0)]
    elif exam_type == 'Commerce' or exam_type == 'CA':
        subjects = [
            student_data.get('Accounts_Marks', 0), student_data.get('Law_Marks', 0), 
            student_data.get('Taxation_Marks', 0), student_data.get('Audit_Marks', 0),
            student_data.get('Fin_Mgmt_Marks', 0), student_data.get('Costing_Marks', 0)
        ]

    # 3.1 Avg calculation
    avg_mock_score = np.mean(subjects) if subjects else 0
    
    # 4. Subject-wise Strength
    high_subjects = sum(1 for s in subjects if s >= params['subject_threshold'])
    low_subjects = sum(1 for s in subjects if s < params['subject_threshold'])

    if high_subjects >= 2:
        score += 2
        breakdown['Subject Strength'] = f"+2 (>=2 Subjects ≥ {params['subject_threshold']})"
    elif low_subjects >= 2:
        score -= 2
        breakdown['Subject Weakness'] = f"-2 (>=2 Subjects < {params['subject_threshold']})"
        recommendation.append("Structured remediation plan needed for weak subjects.")

    # --- Step 5: Overall Academic Performance ---
    if avg_mock_score >= params['high_avg_threshold']:
        score += 3
        breakdown['Overall Academic'] = f"+3 (Avg >= {params['high_avg_threshold']})"
    elif avg_mock_score >= params['passing_avg_threshold']:
        score += 2
        breakdown['Overall Academic'] = f"+2 (Avg >= {params['passing_avg_threshold']})"
    else:
        score -= 2
        breakdown['Overall Academic'] = f"-2 (Avg < {params['passing_avg_threshold']})"

    # --- Step 6: Improvement Rate ---
    imp_rate = student_data.get('Improvement_Rate', 0)
    if imp_rate > 5:
        score += 2
        breakdown['Improvement'] = '+2 (> 5%)'
    elif imp_rate > 0:
        score += 1
        breakdown['Improvement'] = '+1 (> 0%)'
    elif imp_rate < 0:
        score -= 2
        breakdown['Improvement'] = '-2 (Negative)'

    # --- Step 7: Rank Trend ---
    rank_trend = student_data.get('Rank_Trend', 'Stable')
    if rank_trend == "Improving":
        score += 2
        breakdown['Rank Trend'] = '+2 (Improving)'
    elif rank_trend == "Declining":
        score -= 2
        breakdown['Rank Trend'] = '-2 (Declining)'

    # --- Step 8: Study Habits ---
    study_hours = student_data.get('Study_Hours', 0)
    if study_hours >= params['target_study_hours']:
        score += 2
        breakdown['Study Hours'] = f"+2 (>= {params['target_study_hours']} hrs)"
    elif study_hours < params['target_study_hours'] - 2:
        score -= 2
        breakdown['Study Hours'] = f"-2 (< {params['target_study_hours'] - 2} hrs)"

    consistency = student_data.get('Study_Consistency', 'Irregular')
    if consistency == "Regular":
        score += 2
        breakdown['Consistency'] = '+2 (Regular)'
    else:
        score -= 1
        breakdown['Consistency'] = '-1 (Irregular)'

    # --- Step 9: Lifestyle ---
    sleep = student_data.get('Sleep_Hours', 0)
    if sleep >= params['min_sleep_hours']:
        score += 1
        breakdown['Sleep'] = f"+1 (>= {params['min_sleep_hours']} hrs)"
    else:
        score -= 1
        breakdown['Sleep'] = f"-1 (< {params['min_sleep_hours']} hrs)"

    screen = student_data.get('Screen_Time', 0)
    if screen > params['max_screen_time']:
        score -= 2
        breakdown['Screen Time'] = f"-2 (> {params['max_screen_time']} hrs)"
    elif screen < params['max_screen_time'] - 2:
        score += 1
        breakdown['Screen Time'] = f"+1 (< {params['max_screen_time'] - 2} hrs)"

    # --- Step 10: Test Review & Coaching ---
    review = student_data.get('Test_Review_Behavior', 'Never')
    if review == "Always":
        score += 2
        breakdown['Test Review'] = '+2 (Always)'
    elif review == "Never":
        score -= 2
        breakdown['Test Review'] = '-2 (Never)'
    
    coaching = student_data.get('Coaching_Satisfaction', 3)
    if coaching >= 4:
        score += 1
        breakdown['Coaching'] = '+1 (High Quality)'
    elif coaching <= 2:
        score -= 1
        breakdown['Coaching'] = '-1 (Low Quality)'

    # --- Step 11: Psychological & Motivation (NEW) ---
    motivation = student_data.get('Motivation_Level', 3)
    if motivation >= 4:
        score += 1
        breakdown['Motivation'] = '+1 (High)'
    elif motivation <= 1:
        score -= 1
        breakdown['Motivation'] = '-1 (Low)'
    else:
        breakdown['Motivation'] = '0 (Neutral)'

    attempts = student_data.get('Attempts', 0)
    if attempts == 0:
        score += 2
        breakdown['Attempts'] = '+2 (Fresh Attempt)'
    elif attempts == 1:
        score += 1
        breakdown['Attempts'] = '+1 (1st Retake/Drop)'
    else:
        score -= 1
        breakdown['Attempts'] = '-1 (Multiple Retakes)'

    # --- Step 12: Final Prediction ---
    final_decision = "Continue Preparation" if score >= params['continuation_score_threshold'] else "Consider Dropping/Changing Strategy"
    
    # --- Step 13: Decision Support System (Advanced Recommendations) ---
    # We organize recommendations into categories with severity levels
    recommendations = {
        "Academic": [],
        "Lifestyle": [],
        "Strategy": []
    }

    # 13.1 Academic Triggers
    subject_names = []
    if exam_type == 'JEE':
        subject_names = ['Physics', 'Chemistry', 'Maths']
    elif exam_type == 'NEET':
        subject_names = ['Physics', 'Chemistry', 'Biology']
    
    # Specific Subject Advice
    if exam_type in ['JEE', 'NEET']:
        if subjects[0] < 60: # Physics
            recommendations["Academic"].append({"msg": "Physics Alert: Focus on concept visualization and numerical derivation.", "severity": "medium"})
        
        if exam_type == 'JEE' and subjects[2] < 60: # Maths
             recommendations["Academic"].append({"msg": "Maths Weakness: Increase daily numerical solving count by 50%. Focus on weak chapters.", "severity": "high"})
        
        if exam_type == 'NEET' and subjects[2] < 60: # Biology
             recommendations["Academic"].append({"msg": "Biology Lag: Increase NCERT reading frequency. Focus on memorization.", "severity": "high"})

    # "Weak two subjects + low sleep -> intervention plan"
    if low_subjects >= 2 and sleep < 6:
         recommendations["Strategy"].append({
             "msg": "CRISIS INTERVENTION PLAN: You are failing multiple subjects while sleep deprived. \n1. Stop learning new topics immediately.\n2. Sleep 7+ hours for 3 days.\n3. Revise only strong basics to regain confidence.", 
             "severity": "critical"
         })

    # "Weak Maths + good habits -> extra numericals" (Generic version for logic subjects)
    is_hard_working = (study_hours >= 6) and (consistency == "Regular")
    
    if is_hard_working and low_subjects > 0:
        recommendations["Academic"].append({
            "msg": "High Effort / Low Result: Your study technique might be ineffective. Switch from passive reading to active recall and timed practice.", 
            "severity": "medium"
        })

    # 13.2 Lifestyle Triggers
    # "High screen time -> digital detox recommendation"
    if screen > 6:
        recommendations["Lifestyle"].append({
            "msg": "Digital Detox Needed: Your screen time is critically high. Install app blockers and switch to physical books for 1 week.", 
            "severity": "high"
        })
    
    # Burnout handling
    burnout = student_data.get('Burnout_Symptoms', 'No')
    stress = student_data.get('Stress_Level', 2)

    if burnout == 'Yes' or stress >= 4:
         recommendations["Lifestyle"].append({
             "msg": "Burnout Protocol: Mandatory 2-day break. No studies. Nature walk or hobby time required.", 
             "severity": "high"
         })

    # 13.3 Strategic Triggers
    if avg_mock_score > 70 and student_data.get('Confidence_Level', 3) < 3:
        recommendations["Strategy"].append({
            "msg": "Imposter Syndrome Alert: Your scores are good, but confidence is low. Stop comparing with others. Trust your data.", 
            "severity": "medium"
        })
    
    if avg_mock_score < 40 and student_data.get('Confidence_Level', 3) > 4:
        recommendations["Strategy"].append({
            "msg": "Reality Check: Confidence is outpacing performance. Review your test papers to find actual gaps.", 
            "severity": "high"
        })

    # Default if empty
    if not any(recommendations.values()):
        recommendations["Strategy"].append({"msg": "Maintenance Mode: Continue with your current balanced routine.", "severity": "low"})

    return score, final_decision, recommendations, breakdown
