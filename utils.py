import plotly.graph_objects as go
import pandas as pd

def plot_radar_chart(categories, values, title="Subject Performance"):
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Student'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        title=title,
        showlegend=False
    )
    return fig

def plot_gauge_chart(value, title="Success Probability"):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "lightblue"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    return fig

def plot_feature_importance(importance_dict):
    df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
    df = df.sort_values(by='Importance', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df['Importance'],
        y=df['Feature'],
        orientation='h'
    ))
    fig.update_layout(title="Key Factors Influencing Prediction")
    return fig

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    # cm is a 2x2 numpy array
    labels = ['Drop/Change', 'Continue']
    
    # Annotations for the heatmap
    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                x=labels[j], 
                y=labels[i], 
                text=str(cm[i][j]),
                font=dict(color="white" if cm[i][j] > cm.max()/2 else "black"),
                showarrow=False
            ))

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        annotations=annotations
    )
    return fig

def plot_continuation_distribution(df):
    """
    Plots the distribution of continuation vs dropout in the dataset.
    """
    counts = df['Target_Continuation'].replace({1: 'Continue', 0: 'Drop/Change'}).value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=counts.index, 
        values=counts.values, 
        hole=.3,
        marker=dict(colors=['#1f77b4', '#ff7f0e'])
    )])
    
    fig.update_layout(
        title_text="Overall Continuation vs Dropout Distribution (%)",
        annotations=[dict(text='Students', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    return fig

def plot_exam_comparison(comparison_data):
    """
    Plots a comparison of continuation rates across different exams.
    comparison_data: dict {exam_name: continuation_rate}
    """
    exams = list(comparison_data.keys())
    rates = [r * 100 for r in comparison_data.values()]
    
    fig = go.Figure(data=[
        go.Bar(name='Continuation Rate', x=exams, y=rates, marker_color='rgb(55, 83, 109)')
    ])
    
    fig.update_layout(
        title='Continuation Rate Comparison across Examinations (%)',
        yaxis_title='Percentage (%)',
        xaxis_title='Examination',
        yaxis=dict(range=[0, 100]),
        template='plotly_white'
    )
    return fig
