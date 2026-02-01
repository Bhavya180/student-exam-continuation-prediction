import pandas as pd
import numpy as np
import random
import streamlit as st

@st.cache_data
def generate_jee_data(num_samples=20000):
    data = []
    for _ in range(num_samples):
        physics = np.random.randint(20, 100)
        chemistry = np.random.randint(20, 100)
        maths = np.random.randint(20, 100)
        practical_ability = np.random.choice(['Low', 'Medium', 'High'])
        
        improvement_rate = np.random.uniform(-5, 15)
        rank_trend = np.random.choice(['Improving', 'Stable', 'Declining'])
        study_hours = np.random.randint(2, 14)
        consistency = np.random.choice(['Irregular', 'Regular']) # Updated
        screen_time = np.random.uniform(1, 8)
        sleep_hours = np.random.uniform(4, 9)
        test_review = np.random.choice(['Always', 'Never']) # Updated
        coaching_satisfaction = np.random.randint(1, 6) # 1-5 scale
        
        # New Psychological Factors
        motivation = np.random.randint(1, 6)
        confidence = np.random.randint(1, 6)
        stress = np.random.randint(1, 6)
        burnout = np.random.choice(['Yes', 'No'])
        fear = np.random.randint(1, 6)
        goal_clarity = np.random.randint(1, 6)
        discipline = np.random.randint(1, 6)
        fatigue = np.random.randint(1, 6)
        attempts = np.random.randint(0, 4)

        # Calculate target based on logic
        avg_score = (physics + chemistry + maths) / 3
        score = 0
        if avg_score >= 55: score += 2
        if study_hours >= 6: score += 2
        if consistency == 'Regular': score += 2
        
        # New factors impact
        if motivation >= 4: score += 1
        elif motivation == 1: score -= 1
        
        if attempts == 0: score += 2
        elif attempts == 1: score += 1
        else: score -= 1
        
        target = 1 if score >= 5 else 0 
        
        data.append({
            'Physics_Marks': physics, 'Chemistry_Marks': chemistry, 'Maths_Marks': maths,
            'Practical_Ability': practical_ability,
            'Improvement_Rate': round(improvement_rate, 2),
            'Rank_Trend': rank_trend,
            'Study_Hours': study_hours,
            'Study_Consistency': consistency,
            'Screen_Time': round(screen_time, 1),
            'Sleep_Hours': round(sleep_hours, 1),
            'Test_Review_Behavior': test_review,
            'Coaching_Satisfaction': coaching_satisfaction,
            'Motivation_Level': motivation,
            'Confidence_Level': confidence,
            'Stress_Level': stress,
            'Burnout_Symptoms': burnout,
            'Fear_of_Failure': fear,
            'Goal_Clarity': goal_clarity,
            'Self_Discipline': discipline,
            'Mental_Fatigue': fatigue,
            'Attempts': attempts,
            'Target_Continuation': target
        })
    return pd.DataFrame(data)

@st.cache_data
def generate_neet_data(num_samples=20000):
    data = []
    for _ in range(num_samples):
        physics = np.random.randint(20, 100)
        chemistry = np.random.randint(20, 100)
        biology = np.random.randint(20, 100)
        theory_understanding = np.random.choice(['Low', 'Medium', 'High'])
        
        improvement_rate = np.random.uniform(-5, 15)
        rank_trend = np.random.choice(['Improving', 'Stable', 'Declining'])
        study_hours = np.random.randint(2, 14)
        consistency = np.random.choice(['Irregular', 'Regular'])
        screen_time = np.random.uniform(1, 8)
        sleep_hours = np.random.uniform(4, 9)
        test_review = np.random.choice(['Always', 'Never'])
        coaching_satisfaction = np.random.randint(1, 6)
        
        motivation = np.random.randint(1, 6)
        confidence = np.random.randint(1, 6)
        stress = np.random.randint(1, 6)
        burnout = np.random.choice(['Yes', 'No'])
        fear = np.random.randint(1, 6)
        goal_clarity = np.random.randint(1, 6)
        discipline = np.random.randint(1, 6)
        fatigue = np.random.randint(1, 6)
        attempts = np.random.randint(0, 4)

        avg_score = (physics + chemistry + biology) / 3
        score = 0
        if avg_score >= 55: score += 2
        if study_hours >= 6: score += 2
        
        if motivation >= 4: score += 1
        elif motivation == 1: score -= 1
        
        if attempts == 0: score += 2
        elif attempts == 1: score += 1
        else: score -= 1
        
        target = 1 if score >= 4 else 0

        data.append({
            'Physics_Marks': physics, 'Chemistry_Marks': chemistry, 'Biology_Marks': biology,
            'Theoretical_Understanding': theory_understanding,
            'Improvement_Rate': round(improvement_rate, 2),
            'Rank_Trend': rank_trend,
            'Study_Hours': study_hours,
            'Study_Consistency': consistency,
            'Screen_Time': round(screen_time, 1),
            'Sleep_Hours': round(sleep_hours, 1),
            'Test_Review_Behavior': test_review,
            'Coaching_Satisfaction': coaching_satisfaction,
            'Motivation_Level': motivation,
            'Confidence_Level': confidence,
            'Stress_Level': stress,
            'Burnout_Symptoms': burnout,
            'Fear_of_Failure': fear,
            'Goal_Clarity': goal_clarity,
            'Self_Discipline': discipline,
            'Mental_Fatigue': fatigue,
            'Attempts': attempts,
            'Target_Continuation': target
        })
    return pd.DataFrame(data)

@st.cache_data
def generate_commerce_data(num_samples=20000):
    data = []
    for _ in range(num_samples):
        accounts = np.random.randint(20, 100)
        law = np.random.randint(20, 100)
        taxation = np.random.randint(20, 100)
        audit = np.random.randint(20, 100)
        fin_mgmt = np.random.randint(20, 100)
        costing = np.random.randint(20, 100)
        
        concept_clarity = np.random.choice(['Low', 'Medium', 'High'])
        answer_writing = np.random.choice(['Poor', 'Average', 'Good'])
        syllabus_completion = np.random.choice(['On Time', 'Delayed'])
        
        improvement_rate = np.random.uniform(-5, 15)
        rank_trend = np.random.choice(['Improving', 'Stable', 'Declining'])
        study_hours = np.random.randint(2, 14)
        consistency = np.random.choice(['Irregular', 'Regular'])
        screen_time = np.random.uniform(1, 8)
        sleep_hours = np.random.uniform(4, 9)
        test_review = np.random.choice(['Always', 'Never'])
        coaching_satisfaction = np.random.randint(1, 6)
        
        motivation = np.random.randint(1, 6)
        confidence = np.random.randint(1, 6)
        stress = np.random.randint(1, 6)
        burnout = np.random.choice(['Yes', 'No'])
        fear = np.random.randint(1, 6)
        goal_clarity = np.random.randint(1, 6)
        discipline = np.random.randint(1, 6)
        fatigue = np.random.randint(1, 6)
        attempts = np.random.randint(0, 4)

        avg_score = (accounts + law + taxation + audit + fin_mgmt + costing) / 6
        score = 0
        if avg_score >= 55: score += 2
        if concept_clarity == 'High': score += 2
        if study_hours >= 6: score += 1
        if consistency == 'Regular': score += 1
        
        if motivation >= 4: score += 1
        elif motivation == 1: score -= 1
        
        if attempts == 0: score += 2
        elif attempts == 1: score += 1
        else: score -= 1
        
        target = 1 if score >= 5 else 0

        data.append({
            'Accounts_Marks': accounts, 'Law_Marks': law, 'Taxation_Marks': taxation,
            'Audit_Marks': audit, 'Fin_Mgmt_Marks': fin_mgmt, 'Costing_Marks': costing,
            'Concept_Clarity': concept_clarity,
            'Answer_Writing': answer_writing,
            'Syllabus_Completion': syllabus_completion,
            'Improvement_Rate': round(improvement_rate, 2),
            'Rank_Trend': rank_trend,
            'Study_Hours': study_hours,
            'Study_Consistency': consistency,
            'Screen_Time': round(screen_time, 1),
            'Sleep_Hours': round(sleep_hours, 1),
            'Test_Review_Behavior': test_review,
            'Coaching_Satisfaction': coaching_satisfaction,
            'Motivation_Level': motivation,
            'Confidence_Level': confidence,
            'Stress_Level': stress,
            'Burnout_Symptoms': burnout,
            'Fear_of_Failure': fear,
            'Goal_Clarity': goal_clarity,
            'Self_Discipline': discipline,
            'Mental_Fatigue': fatigue,
            'Attempts': attempts,
            'Target_Continuation': target
        })
    return pd.DataFrame(data)
