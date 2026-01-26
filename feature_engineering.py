import pandas as pd
import numpy as np

def calculate_multi_factor_score(student_data, exam_type):
    """
    Calculates score based on the specific Step-by-Step Architecture.
    """
    score = 0
    breakdown = {}
    recommendation = []
    
    # --- Step 3 & 4: Subject Analysis ---
    subjects = []
    if exam_type == 'JEE':
        subjects = [student_data.get('Physics_Marks', 0), student_data.get('Chemistry_Marks', 0), student_data.get('Maths_Marks', 0)]
    elif exam_type == 'NEET':
        subjects = [student_data.get('Physics_Marks', 0), student_data.get('Chemistry_Marks', 0), student_data.get('Biology_Marks', 0)]
    elif exam_type == 'Commerce':
        subjects = [
            student_data.get('Accounts_Marks', 0), student_data.get('Law_Marks', 0), 
            student_data.get('Taxation_Marks', 0), student_data.get('Audit_Marks', 0),
            student_data.get('Fin_Mgmt_Marks', 0), student_data.get('Costing_Marks', 0)
        ]

    # 3.1 Avg calculation
    avg_mock_score = np.mean(subjects) if subjects else 0
    
    # 4. Subject-wise Strength (Threshold 65)
    high_subjects = sum(1 for s in subjects if s >= 65)
    low_subjects = sum(1 for s in subjects if s < 65)

    if high_subjects >= 2:
        score += 2
        breakdown['Subject Strength'] = '+2 (>=2 Subjects ≥ 65)'
    elif low_subjects >= 2:
        score -= 2
        breakdown['Subject Weakness'] = '-2 (>=2 Subjects < 65)'
        recommendation.append("Structured remediation plan needed for weak subjects.")

    # --- Step 5: Overall Academic Performance ---
    if avg_mock_score >= 70:
        score += 3
        breakdown['Overall Academic'] = '+3 (Avg >= 70)'
    elif avg_mock_score >= 55:
        score += 2
        breakdown['Overall Academic'] = '+2 (Avg >= 55)'
    else:
        score -= 2
        breakdown['Overall Academic'] = '-2 (Avg < 55)'

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
    if study_hours >= 6:
        score += 2
        breakdown['Study Hours'] = '+2 (>= 6 hrs)'
    elif study_hours < 4:
        score -= 2
        breakdown['Study Hours'] = '-2 (< 4 hrs)'

    consistency = student_data.get('Study_Consistency', 'Irregular')
    if consistency == "Regular":
        score += 2
        breakdown['Consistency'] = '+2 (Regular)'
    else:
        score -= 1
        breakdown['Consistency'] = '-1 (Irregular)'

    # --- Step 9: Lifestyle ---
    sleep = student_data.get('Sleep_Hours', 0)
    if sleep >= 6.5:
        score += 1
        breakdown['Sleep'] = '+1 (>= 6.5 hrs)'
    else:
        score -= 1
        breakdown['Sleep'] = '-1 (< 6.5 hrs)'

    screen = student_data.get('Screen_Time', 0)
    if screen > 6:
        score -= 2
        breakdown['Screen Time'] = '-2 (> 6 hrs)'
    elif screen < 4:
        score += 1
        breakdown['Screen Time'] = '+1 (< 4 hrs)'

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

    # --- Step 11: Final Prediction ---
    final_decision = "Continue Preparation" if score >= 4 else "Consider Dropping/Changing Strategy"
    
    # --- Step 12: Scenario Recommendation ---
    # "1 weak subject -> Extra focus"
    if low_subjects == 1:
        recommendation.append("Extra focus required on the single weak subject.")
    
    # "2 weak subjects + positive habits -> Low chance / Remediation"
    is_positive_habits = (study_hours >= 6) and (consistency == "Regular")
    if low_subjects >= 2:
        if is_positive_habits:
            recommendation.append("Critical: structured remediation needed despite good habits.")
        else:
            recommendation.append("High Drop Risk: Poor academics combined with poor habits.")

    # "Average marks + Improving -> Continue"
    if 55 <= avg_mock_score < 70 and imp_rate > 0:
         recommendation.append("Good momentum. Maintain current strategy to improve marks further.")

    # "Strong academics + Poor lifestyle -> Lifestyle correction"
    is_poor_lifestyle = (sleep < 6.5) or (screen > 6)
    if avg_mock_score >= 70 and is_poor_lifestyle:
        recommendation.append("Risk of burnout. Urgent lifestyle correction (Sleep/Screen) needed.")

    if not recommendation:
        recommendation.append("Maintain consistency and focus on weak areas.")

    return score, final_decision, recommendation, breakdown
