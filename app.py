import streamlit as st
import pandas as pd
import time

from feature_engineering import calculate_multi_factor_score
from model import ExamModel, get_trained_model, get_model_comparison, get_dataset_distribution
from utils import plot_radar_chart, plot_gauge_chart, plot_feature_importance, plot_confusion_matrix, plot_continuation_distribution, plot_exam_comparison

# Set Page Config
st.set_page_config(layout="wide", page_title="Competitive Exam Dropout Prediction System", page_icon="🎓")

st.title("🎓 Competitive Exam Dropout Prediction System")
st.markdown("---")

# --- Step 1: Exam Selection ---
# --- Navigation Logic ---
if 'page' not in st.session_state:
    st.session_state.page = "Prediction"

def go_to_comparison():
    st.session_state.page = "Comparison"

def go_to_prediction():
    st.session_state.page = "Prediction"

def go_to_exam_comparison():
    st.session_state.page = "Exam Comparison"

# --- Sidebar ---
st.sidebar.header("Select Examination")
# Only show exam selection in Prediction mode or if needed globally. 
# Keeping it global so state persists if we switch back.
exam_selection = st.sidebar.selectbox("Choose Your Target Exam", ["JEE(Joint Entrance Examination)", "NEET(National Eligibility cum Entrance Test)", "CA Foundation"])
exam_map = {
    "JEE(Joint Entrance Examination)": "JEE",
    "NEET(National Eligibility cum Entrance Test)": "NEET",
    "CA Foundation": "CA"
}
exam_type = exam_map.get(exam_selection, "JEE")
st.sidebar.markdown("---")
st.sidebar.info(f"Active Prediction Model: **{exam_type}**")

# Comparison Navigation
st.sidebar.markdown("---")
if st.sidebar.button("📊 Algorithm Performance Comparison"):
    go_to_comparison()
    st.rerun()

if st.sidebar.button("📈 Exam-wise Comparison Dashboard"):
    go_to_exam_comparison()
    st.rerun()

# --- Feature Parameter Tuning ---
st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Feature Threshold Tuning"):
    tuning_params = {
        'subject_threshold': st.slider("Subject Pass Mark", 33, 90, 65),
        'high_avg_threshold': st.slider("High Performance Avg", 60, 95, 70),
        'passing_avg_threshold': st.slider("Min Passing Avg", 33, 70, 55),
        'target_study_hours': st.slider("Target Study Hours", 4, 16, 6),
        'min_sleep_hours': st.slider("Min Sleep Hours", 4, 10, 6),
        'max_screen_time': st.slider("Max Screen Time", 1, 12, 6),
        'continuation_score_threshold': st.slider("Min Score to Continue", 0, 15, 5)
    }

# --- Main Content Router ---

if st.session_state.page == "Prediction":
    # --- Data Health Overview ---
    st.subheader("🌐 Dataset Context")
    # We can use a temporary model to get data distribution
    with st.expander("📊 View Dataset Distribution", expanded=True):
        df_dist = get_dataset_distribution(exam_type)
        st.plotly_chart(plot_continuation_distribution(df_dist), use_container_width=True)
        st.info("💡 **Why this matters:** Knowing the balance between continuation and dropout in the training data helps contextualize the AI's predictions.")
    
    st.markdown("---")

    # Function to render input forms
    def get_user_input(exam_type):
        st.subheader(f"Enter Academic, Psychological & Behavioral Details ⭐")
        
        col1, col2 = st.columns(2)
        
        inputs = {}
        
        with col1:
            st.markdown("#### 📚 Academic & Psychological Factors")
            if exam_type == 'JEE':
                inputs['Physics_Marks'] = st.slider("Physics Marks (Mock Avg)", 0, 100, 60)
                inputs['Chemistry_Marks'] = st.slider("Chemistry Marks (Mock Avg)", 0, 100, 60)
                inputs['Maths_Marks'] = st.slider("Maths Marks (Mock Avg)", 0, 100, 60)
                inputs['Practical_Ability'] = st.selectbox("Practical Problem Solving", ['Low', 'Medium', 'High'])

            elif exam_type == 'NEET':
                inputs['Physics_Marks'] = st.slider("Physics Marks (Mock Avg)", 0, 100, 60)
                inputs['Chemistry_Marks'] = st.slider("Chemistry Marks (Mock Avg)", 0, 100, 60)
                inputs['Biology_Marks'] = st.slider("Biology Marks (Mock Avg)", 0, 100, 60)
                inputs['Theoretical_Understanding'] = st.selectbox("Theoretical Understanding", ['Low', 'Medium', 'High'])

            elif exam_type == 'CA':
                inputs['Accounts_Marks'] = st.slider("Accounts Marks", 0, 100, 60)
                inputs['Law_Marks'] = st.slider("Law Marks", 0, 100, 60)
                inputs['Taxation_Marks'] = st.slider("Taxation Marks", 0, 100, 60)
                inputs['Audit_Marks'] = st.slider("Audit Marks", 0, 100, 60)
                inputs['Fin_Mgmt_Marks'] = st.slider("Fin. Mgmt Marks", 0, 100, 60)
                inputs['Costing_Marks'] = st.slider("Costing Marks", 0, 100, 60)
                inputs['Concept_Clarity'] = st.selectbox("Concept Clarity", ['Low', 'Medium', 'High'])
                inputs['Answer_Writing'] = st.selectbox("Answer Writing Quality", ['Poor', 'Average', 'Good'])
                inputs['Syllabus_Completion'] = st.selectbox("Syllabus Status", ['On Time', 'Delayed'])

            st.markdown("---")
            st.markdown("#### 🧠 Psychological State")
            inputs['Motivation_Level'] = st.select_slider("Motivation Level (1-5)", options=[1, 2, 3, 4, 5], value=4)
            inputs['Confidence_Level'] = st.select_slider("Confidence in clearing Exam", options=[1, 2, 3, 4, 5], value=3)
            inputs['Stress_Level'] = st.select_slider("Stress Level", options=[1, 2, 3, 4, 5], value=2)
            inputs['Burnout_Symptoms'] = st.radio("Burnout Symptoms?", ['No', 'Yes'], horizontal=True)
            inputs['Fear_of_Failure'] = st.select_slider("Fear of Failure (1-5)", options=[1, 2, 3, 4, 5], value=2)
            inputs['Goal_Clarity'] = st.select_slider("Goal Clarity (1-5)", options=[1, 2, 3, 4, 5], value=4)
            inputs['Self_Discipline'] = st.select_slider("Self-Discipline Rating", options=[1, 2, 3, 4, 5], value=4)
            inputs['Mental_Fatigue'] = st.select_slider("Mental Fatigue Level", options=[1, 2, 3, 4, 5], value=2)

        with col2:
            st.markdown("#### ⏳ Study Habits & Lifestyle")
            inputs['Improvement_Rate'] = st.slider("Improvement Rate (%)", -10.0, 20.0, 2.0)
            inputs['Rank_Trend'] = st.selectbox("Rank Trend", ['Improving', 'Stable', 'Declining'])
            inputs['Study_Hours'] = st.number_input("Study Hours/Day", 0, 18, 6)
            inputs['Study_Consistency'] = st.selectbox("Study Consistency", ['Regular', 'Irregular'])
            inputs['Screen_Time'] = st.number_input("Screen Time (Hrs)", 0, 12, 3, step=1)
            inputs['Sleep_Hours'] = st.number_input("Sleep Hours", 0, 12, 7, step=1)
            inputs['Coaching_Satisfaction'] = st.slider("Coaching Satisfaction (1-5)", 1, 5, 3)
            inputs['Test_Review_Behavior'] = st.selectbox("Review Tests?", ['Always', 'Never'])
            
            st.markdown("---")
            st.markdown("#### 🔄 Examination History")
            inputs['Attempts'] = st.selectbox("Number of Previous Attempts", [0, 1, 2, 3], index=0)
            
        return inputs

    inputs = get_user_input(exam_type)

    # --- Analysis Button ---
    if st.button("🚀 Analyze & Predict"):
        with st.spinner("Processing Factors..."):
            time.sleep(1) # UX Filler
            
            # 1. Rule-Based Scoring
            rb_score, rb_decision, rb_recs, rb_breakdown = calculate_multi_factor_score(inputs, exam_type, params=tuning_params)
            
            # 2. AI Model Prediction
            try:
                model, acc_rf, acc_ann = get_trained_model(exam_type)
                pred_rf, prob_rf, pred_ann, prob_ann = model.predict(inputs)
                
                rf_decision = "Continue" if pred_rf == 1 else "Drop/Change"
                rf_confidence = prob_rf if pred_rf == 1 else 1 - prob_rf

                ann_decision = "Continue" if pred_ann == 1 else "Drop/Change"
                ann_confidence = prob_ann if pred_ann == 1 else 1 - prob_ann
            
                # --- Display Results ---
                st.markdown("---")
                
                # --- Section 1: Prediction Summary ---
                st.subheader("🔮 Prediction Outcome Summary")
                
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.info("### 📐 Rule-Based")
                    st.metric("Score", rb_score)
                    st.write(f"**Decision:** {rb_decision}")
                    with st.expander("Why?"):
                        st.json(rb_breakdown)

                with res_col2:
                    st.success("### 🌲 Random Forest")
                    st.metric("Accuracy", f"{acc_rf*100:.1f}%")
                    st.write(f"**Decision:** {rf_decision}")
                    st.progress(rf_confidence)

                with res_col3:
                    st.warning("### 🧠 Neural Network")
                    st.metric("Accuracy", f"{acc_ann*100:.1f}%")
                    st.write(f"**Decision:** {ann_decision}")
                    st.progress(ann_confidence)

                st.markdown("---")

                # --- Section 2: Decision Support System ---
                st.subheader("🛡️ Decision Support System (Action Plan)")
                
                # Helper for severity colors
                def get_severity_icon(severity):
                    if severity == "critical": return "🚨"
                    if severity == "high": return "🔴"
                    if severity == "medium": return "🟠"
                    return "🟢"

                dss_col1, dss_col2, dss_col3 = st.columns(3)

                with dss_col1:
                    st.markdown("#### 🧠 Strategic Path")
                    if rb_recs.get("Strategy"):
                         for item in rb_recs["Strategy"]:
                            icon = get_severity_icon(item['severity'])
                            msg = f"{icon} {item['msg']}"
                            if item['severity'] in ['critical', 'high']:
                                st.error(msg)
                            else:
                                st.info(msg)
                    else:
                        st.write("No specific strategic interventions needed.")

                with dss_col2:
                    st.markdown("#### 📚 Academic Steps")
                    if rb_recs.get("Academic"):
                         for item in rb_recs["Academic"]:
                             icon = get_severity_icon(item['severity'])
                             st.warning(f"{icon} {item['msg']}")
                    else:
                        st.success("✅ Academic signals are stable.")

                with dss_col3:
                    st.markdown("#### 🌿 Lifestyle Fixes")
                    if rb_recs.get("Lifestyle"):
                         for item in rb_recs["Lifestyle"]:
                             icon = get_severity_icon(item['severity'])
                             st.error(f"{icon} {item['msg']}")
                    else:
                        st.success("✅ Lifestyle seems balanced.")
                    
                # --- Visualizations ---
                st.markdown("---")
                st.subheader("📈 Visual Insights")
                
                # --- Row 1: Success Probabilities ---
                st.subheader("🎯 Success Probability")
                kp1, kp2 = st.columns(2)
                with kp1:
                     st.plotly_chart(plot_gauge_chart(rf_confidence, "RF Success Probability"), use_container_width=True)
                with kp2:
                     st.plotly_chart(plot_gauge_chart(ann_confidence, "ANN Success Probability"), use_container_width=True)

                st.markdown("---")

                # --- Row 2: Deep Dive ---
                st.subheader("🔍 Deep Dive Analysis")
                g_col1, g_col2 = st.columns(2)
                
                with g_col1:
                    st.markdown("##### Subject Proficiency (Radar Chart)")
                    # Radar Chart Logic
                    if exam_type == 'JEE':
                        cats = ['Physics', 'Chemistry', 'Maths']
                        vals = [inputs['Physics_Marks'], inputs['Chemistry_Marks'], inputs['Maths_Marks']]
                    elif exam_type == 'NEET':
                        cats = ['Physics', 'Chemistry', 'Biology']
                        vals = [inputs['Physics_Marks'], inputs['Chemistry_Marks'], inputs['Biology_Marks']]
                    else:
                        cats = ['Accounts', 'Law', 'Tax', 'Audit', 'FM', 'Costing']
                        vals = [
                            inputs['Accounts_Marks'], inputs['Law_Marks'], 
                            inputs['Taxation_Marks'], inputs['Audit_Marks'],
                            inputs['Fin_Mgmt_Marks'], inputs['Costing_Marks']
                        ]
                    
                    # Add Psychological Factors (Scale to 0-100)
                    psy_cats = ['Motivation', 'Confidence', 'Goal Clarity', 'Discipline']
                    psy_vals = [
                        inputs['Motivation_Level'] * 20,
                        inputs['Confidence_Level'] * 20,
                        inputs['Goal_Clarity'] * 20,
                        inputs['Self_Discipline'] * 20
                    ]
                    
                    cats += psy_cats
                    vals += psy_vals
                    
                    st.plotly_chart(plot_radar_chart(cats, vals), use_container_width=True)
                    
                with g_col2:
                    st.markdown("##### Crucial Factors (Feature Importance)")
                    # Feature Importance
                    impt = model.get_feature_importance()
                    # Top 10
                    top_impt = dict(sorted(impt.items(), key=lambda x: x[1], reverse=True)[:10])
                    st.plotly_chart(plot_feature_importance(top_impt), use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

elif st.session_state.page == "Comparison":
    if st.button("← Back to Prediction"):
        go_to_prediction()
        st.rerun()

    st.header(f"⚖️ Algorithm Comparison: {exam_type}")
    st.write("Comparing Random Forest, Logistic Regression, Decision Tree, and ANN (Deep Learning).")

    with st.spinner("Training and Comparing Models (RF vs LR vs DT)..."):
        try:
            results = get_model_comparison(exam_type)
            
            # Metrics Table
            metrics_data = []
            for name, metrics in results.items():
                metrics_data.append({
                    "Algorithm": name,
                    "Accuracy": f"{metrics['Accuracy']*100:.2f}%",
                    "Precision": f"{metrics['Precision']*100:.2f}%",
                    "F1 Score": f"{metrics['F1 Score']*100:.2f}%"
                })
            
            st.subheader("Performance Metrics")
            st.table(pd.DataFrame(metrics_data).set_index("Algorithm"))
            
            # Confusion Matrices
            st.subheader("Confusion Matrices")
            
            # Row 1
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                st.write("**Random Forest**")
                st.plotly_chart(plot_confusion_matrix(results["Random Forest"]["Confusion Matrix"], "Random Forest"), use_container_width=True)
            with row1_col2:
                st.write("**Logistic Regression**")
                st.plotly_chart(plot_confusion_matrix(results["Logistic Regression"]["Confusion Matrix"], "Logistic Regression"), use_container_width=True)
            
            # Row 2
            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                st.write("**Decision Tree**")
                st.plotly_chart(plot_confusion_matrix(results["Decision Tree"]["Confusion Matrix"], "Decision Tree"), use_container_width=True)
            with row2_col2:
                st.write("**ANN (Deep Learning)**")
                st.plotly_chart(plot_confusion_matrix(results["ANN (Deep Learning)"]["Confusion Matrix"], "ANN"), use_container_width=True)
                
        except Exception as e:
            st.error(f"Error during comparison: {e}")

elif st.session_state.page == "Exam Comparison":
    if st.button("← Back to Prediction"):
        go_to_prediction()
        st.rerun()

    st.header("📈 Exam-wise Comparison Dashboard")
    st.write("Comparing continuation rates across JEE, NEET, and CA Foundation exams based on synthetic 10,000-record datasets.")

    with st.spinner("Calculating continuation rates for all exams..."):
        try:
            comparison_data = {}
            exams_to_compare = ["JEE", "NEET", "CA"]
            
            # Using progress bars for fetching data
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, exam in enumerate(exams_to_compare):
                status_text.text(f"Processing {exam} data...")
                df = get_dataset_distribution(exam)
                
                # Calculate continuation rate
                continuation_count = len(df[df['Target_Continuation'] == 1])
                total_count = len(df)
                rate = continuation_count / total_count if total_count > 0 else 0
                
                display_name = "JEE" if exam == "JEE" else "NEET" if exam == "NEET" else "CA Foundation"
                comparison_data[display_name] = rate
                
                progress_bar.progress((i + 1) / len(exams_to_compare))

            status_text.text("Logic complete!")
            
            # Display Chart
            st.plotly_chart(plot_exam_comparison(comparison_data), use_container_width=True)
            
            # Insights
            st.subheader("Key Takeaways")
            col1, col2, col3 = st.columns(3)
            for i, (exam, rate) in enumerate(comparison_data.items()):
                with [col1, col2, col3][i]:
                    st.metric(exam, f"{rate*100:.1f}%")
            
            st.info("💡 **Insight:** This dashboard demonstrates the system's ability to normalize and compare behavioral and academic trends across widely differing competitive fields.")

        except Exception as e:
            st.error(f"An error occurred while building the comparison dashboard: {e}")

