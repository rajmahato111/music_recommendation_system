"""
Generate project visualization and metrics dashboard.
Similar to the sample project analysis visualization.
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path to import src modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from src.data_loader import DataLoader
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False
    # Don't print warning when imported from Streamlit
    if os.environ.get('STREAMLIT_SERVER') is None:
        print("Warning: Could not import DataLoader. Using sample data.")


def create_system_architecture_diagram(theme='dark'):
    """Create a system architecture diagram."""
    # Theme colors
    if theme == 'light':
        bg_color = '#FFFFFF'
        text_color = '#262730'
        box_line_color = 'RoyalBlue'
        box_fill_color = 'LightBlue'
        arrow_color = 'black'
    else:
        bg_color = '#0E1117'
        text_color = '#FAFAFA'
        box_line_color = '#4A90E2'
        box_fill_color = '#1E3A8A'
        arrow_color = 'white'

    fig = go.Figure()
    
    # Define components and their positions
    components = {
        'Data Sources': (0, 4),
        'Data Processing': (0, 3),
        'Embedding Layer': (0, 2),
        'Retrieval & Ranking': (0, 1),
        'Recommendation': (0, 0),
        'Application': (0, -1),
    }
    
    # Add boxes for each layer
    for name, (x, y) in components.items():
        fig.add_shape(
            type="rect",
            x0=x-1, y0=y-0.3,
            x1=x+1, y1=y+0.3,
            line=dict(color=box_line_color, width=2),
            fillcolor=box_fill_color,
        )
        fig.add_annotation(
            x=x, y=y,
            text=name,
            showarrow=False,
            font=dict(size=12, color=text_color)
        )
    
    # Add arrows between layers
    for i in range(len(components) - 1):
        y_start = list(components.values())[i][1] - 0.3
        y_end = list(components.values())[i+1][1] + 0.3
        fig.add_annotation(
            x=0, y=(y_start + y_end) / 2,
            ax=0, ay=10,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=arrow_color
        )
    
    fig.update_layout(
        title="System Architecture",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=800,
        height=600,
        showlegend=False,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color)
    )
    
    return fig


def create_data_flow_diagram(theme='dark'):
    """Create a data flow diagram."""
    # Theme colors
    if theme == 'light':
        bg_color = '#FFFFFF'
        text_color = '#262730'
        node_line_color = 'DarkGreen'
        node_fill_color = 'LightGreen'
        arrow_color = 'gray'
    else:
        bg_color = '#0E1117'
        text_color = '#FAFAFA'
        node_line_color = '#2E7D32'
        node_fill_color = '#14532D'
        arrow_color = 'lightgray'

    fig = go.Figure()
    
    # Define nodes
    nodes = {
        'Raw Data': (0, 3),
        'ETL': (0, 2),
        'Collaborative': (-2, 1),
        'Audio': (0, 1),
        'Features': (2, 1),
        'Hybrid': (0, 0),
        'ANN Index': (0, -1),
        'Ranking': (0, -2),
        'Recommendations': (0, -3),
    }
    
    # Add nodes
    for name, (x, y) in nodes.items():
        fig.add_shape(
            type="circle" if name in ['Raw Data', 'Recommendations'] else "rect",
            x0=x-0.5, y0=y-0.3,
            x1=x+0.5, y1=y+0.3,
            line=dict(color=node_line_color, width=2),
            fillcolor=node_fill_color,
        )
        fig.add_annotation(
            x=x, y=y,
            text=name,
            showarrow=False,
            font=dict(size=10, color=text_color)
        )
    
    # Add connections
    connections = [
        ('Raw Data', 'ETL'),
        ('ETL', 'Collaborative'),
        ('ETL', 'Audio'),
        ('ETL', 'Features'),
        ('Collaborative', 'Hybrid'),
        ('Audio', 'Hybrid'),
        ('Features', 'Hybrid'),
        ('Hybrid', 'ANN Index'),
        ('ANN Index', 'Ranking'),
        ('Ranking', 'Recommendations'),
    ]
    
    for start, end in connections:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        fig.add_annotation(
            x=x2, y=y2+0.3,
            ax=x1, ay=y1-0.3,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor=arrow_color
        )
    
    fig.update_layout(
        title="Data Flow Diagram",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3, 3]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-4, 4]),
        width=800,
        height=700,
        showlegend=False,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color)
    )
    
    return fig


def create_metrics_dashboard(theme='dark'):
    """Create a metrics dashboard with sample/placeholder data."""
    # Sample metrics data (replace with actual metrics when available)
    metrics_data = {
        'Model': ['Baseline', 'Item2Vec', 'Hybrid', 'Hybrid+Ranking'],
        'Recall@5': [0.25, 0.32, 0.38, 0.42],
        'Recall@10': [0.35, 0.45, 0.52, 0.58],
        'Precision@5': [0.30, 0.38, 0.45, 0.50],
        'Precision@10': [0.28, 0.35, 0.42, 0.48],
        'NDCG@10': [0.40, 0.52, 0.62, 0.68],
        'MRR': [0.35, 0.48, 0.58, 0.65],
    }
    
    df = pd.DataFrame(metrics_data)
    
    # Theme colors
    if theme == 'light':
        bg_color = '#FFFFFF'
        text_color = '#262730'
        grid_color = '#E0E0E0'
        template = 'plotly_white'
    else:
        bg_color = '#0E1117'
        text_color = '#FAFAFA'
        grid_color = '#4A4A4A'
        template = 'plotly_dark'
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Recall@K', 'Precision@K', 'NDCG@10', 'MRR'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Recall@K
    fig.add_trace(
        go.Bar(x=df['Model'], y=df['Recall@5'], name='Recall@5', marker_color='lightblue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=df['Model'], y=df['Recall@10'], name='Recall@10', marker_color='darkblue'),
        row=1, col=1
    )
    
    # Precision@K
    fig.add_trace(
        go.Bar(x=df['Model'], y=df['Precision@5'], name='Precision@5', marker_color='lightgreen'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=df['Model'], y=df['Precision@10'], name='Precision@10', marker_color='darkgreen'),
        row=1, col=2
    )
    
    # NDCG@10
    fig.add_trace(
        go.Bar(x=df['Model'], y=df['NDCG@10'], name='NDCG@10', marker_color='orange'),
        row=2, col=1
    )
    
    # MRR
    fig.add_trace(
        go.Bar(x=df['Model'], y=df['MRR'], name='MRR', marker_color='red'),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Model Performance Metrics Dashboard",
        height=800,
        showlegend=True,
        template=template,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color),
        yaxis=dict(gridcolor=grid_color)
    )
    
    fig.update_xaxes(title_text="Model", row=2, col=1, gridcolor=grid_color)
    fig.update_xaxes(title_text="Model", row=2, col=2, gridcolor=grid_color)
    fig.update_yaxes(title_text="Score", row=1, col=1, gridcolor=grid_color)
    fig.update_yaxes(title_text="Score", row=1, col=2, gridcolor=grid_color)
    fig.update_yaxes(title_text="Score", row=2, col=1, gridcolor=grid_color)
    fig.update_yaxes(title_text="Score", row=2, col=2, gridcolor=grid_color)
    
    # Update all axes with theme colors
    fig.update_xaxes(gridcolor=grid_color, title_font=dict(color=text_color), tickfont=dict(color=text_color))
    fig.update_yaxes(gridcolor=grid_color, title_font=dict(color=text_color), tickfont=dict(color=text_color))
    
    return fig


def create_training_progress_chart(theme='dark'):
    """Create a training progress chart (placeholder)."""
    # Sample training data
    epochs = np.arange(1, 11)
    train_loss = 0.8 * np.exp(-epochs/3) + 0.1 + np.random.normal(0, 0.02, 10)
    val_loss = 0.85 * np.exp(-epochs/3) + 0.12 + np.random.normal(0, 0.02, 10)
    
    # Theme colors
    if theme == 'light':
        bg_color = '#FFFFFF'
        text_color = '#262730'
        grid_color = '#E0E0E0'
        template = 'plotly_white'
    else:
        bg_color = '#0E1117'
        text_color = '#FAFAFA'
        grid_color = '#4A4A4A'
        template = 'plotly_dark'
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_loss,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_loss,
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="Training Progress (Item2Vec Model)",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        width=800,
        height=400,
        template=template,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color),
        yaxis=dict(gridcolor=grid_color)
    )
    
    return fig


def create_feature_importance_chart(theme='dark'):
    """Create a feature importance chart (placeholder)."""
    # Sample feature importance
    features = ['Similarity Score', 'Energy Diff', 'Danceability Diff', 
                'Popularity', 'Tempo Diff', 'Valence Diff', 'Year Diff']
    importance = [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07]
    
    # Theme colors
    if theme == 'light':
        bg_color = '#FFFFFF'
        text_color = '#262730'
        grid_color = '#E0E0E0'
        template = 'plotly_white'
    else:
        bg_color = '#0E1117'
        text_color = '#FAFAFA'
        grid_color = '#4A4A4A'
        template = 'plotly_dark'
    
    fig = go.Figure(data=[
        go.Bar(x=importance, y=features, orientation='h', marker_color='steelblue')
    ])
    
    fig.update_layout(
        title="Feature Importance (Ranking Model)",
        xaxis_title="Importance",
        yaxis_title="Feature",
        width=800,
        height=400,
        template=template,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color),
        yaxis=dict(gridcolor=grid_color)
    )
    
    return fig


def create_confusion_matrix_heatmap(theme='dark'):
    """Create a confusion matrix heatmap (placeholder)."""
    # Sample confusion matrix
    cm = np.array([[850, 150], [120, 880]])
    
    # Theme colors
    if theme == 'light':
        bg_color = '#FFFFFF'
        text_color = '#262730'
        template = 'plotly_white'
    else:
        bg_color = '#0E1117'
        text_color = '#FAFAFA'
        template = 'plotly_dark'
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Not Relevant', 'Relevant'],
        y=['Not Relevant', 'Relevant'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16, "color": text_color}
    ))
    
    fig.update_layout(
        title="Confusion Matrix (Ranking Model)",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=600,
        height=500,
        template=template,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color)
    )
    
    return fig


def create_roc_curve(theme='dark'):
    """Create ROC curve (placeholder)."""
    # Sample ROC curve data
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr)  # Sample ROC curve
    
    # Theme colors
    if theme == 'light':
        bg_color = '#FFFFFF'
        text_color = '#262730'
        grid_color = '#E0E0E0'
        template = 'plotly_white'
    else:
        bg_color = '#0E1117'
        text_color = '#FAFAFA'
        grid_color = '#4A4A4A'
        template = 'plotly_dark'
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name='ROC Curve',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="ROC Curve (Ranking Model)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=600,
        height=500,
        template=template,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color),
        yaxis=dict(gridcolor=grid_color)
    )
    
    return fig


def create_prediction_distribution(theme='dark'):
    """Create prediction distribution chart (placeholder)."""
    # Sample predictions
    predictions = np.random.beta(2, 5, 1000)
    
    # Theme colors
    if theme == 'light':
        bg_color = '#FFFFFF'
        text_color = '#262730'
        grid_color = '#E0E0E0'
        template = 'plotly_white'
    else:
        bg_color = '#0E1117'
        text_color = '#FAFAFA'
        grid_color = '#4A4A4A'
        template = 'plotly_dark'
    
    fig = go.Figure(data=[
        go.Histogram(x=predictions, nbinsx=30, marker_color='skyblue')
    ])
    
    fig.update_layout(
        title="Prediction Score Distribution",
        xaxis_title="Prediction Score",
        yaxis_title="Frequency",
        width=800,
        height=400,
        template=template,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color),
        yaxis=dict(gridcolor=grid_color)
    )
    
    return fig


def create_technology_stack_diagram(theme='dark'):
    """Create a technology stack visualization."""
    # Theme colors
    if theme == 'light':
        bg_color = '#FFFFFF'
        text_color = '#262730'
        box_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        line_color = 'black'
    else:
        bg_color = '#0E1117'
        text_color = '#FAFAFA'
        box_colors = ['#1E3A8A', '#14532D', '#713F12', '#7F1D1D']
        line_color = 'white'

    categories = {
        'Core': ['Python 3.9+', 'Pandas', 'NumPy', 'Scikit-learn'],
        'ML/Embeddings': ['Gensim', 'FAISS', 'HNSWlib', 'LightGBM'],
        'Web/Visualization': ['Streamlit', 'Plotly'],
        'Deployment': ['Docker', 'Docker Compose'],
    }
    
    fig = go.Figure()
    
    y_pos = 0
    
    for i, (category, techs) in enumerate(categories.items()):
        for j, tech in enumerate(techs):
            fig.add_shape(
                type="rect",
                x0=i, y0=y_pos,
                x1=i+1, y1=y_pos+0.8,
                line=dict(color=line_color, width=1),
                fillcolor=box_colors[i],
            )
            fig.add_annotation(
                x=i+0.5, y=y_pos+0.4,
                text=tech,
                showarrow=False,
                font=dict(size=10, color=text_color)
            )
            y_pos += 1
        y_pos += 0.5
    
    fig.update_layout(
        title="Technology Stack",
        xaxis=dict(
            tickmode='array',
            tickvals=[0.5, 1.5, 2.5, 3.5],
            ticktext=list(categories.keys()),
            showgrid=False,
            zeroline=False,
            tickfont=dict(color=text_color)
        ),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=1000,
        height=600,
        showlegend=False,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color)
    )
    
    return fig


def generate_all_visualizations():
    """Generate all visualizations and save them."""
    output_dir = Path(__file__).parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    print("Generating visualizations...")
    
    # Generate all charts
    charts = {
        'system_architecture': create_system_architecture_diagram(),
        'data_flow': create_data_flow_diagram(),
        'metrics_dashboard': create_metrics_dashboard(),
        'training_progress': create_training_progress_chart(),
        'feature_importance': create_feature_importance_chart(),
        'confusion_matrix': create_confusion_matrix_heatmap(),
        'roc_curve': create_roc_curve(),
        'prediction_distribution': create_prediction_distribution(),
        'technology_stack': create_technology_stack_diagram(),
    }
    
    # Save each chart
    for name, fig in charts.items():
        filepath = output_dir / f"{name}.html"
        fig.write_html(str(filepath))
        print(f"Saved: {filepath}")
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nTo view, open the HTML files in a web browser.")


if __name__ == "__main__":
    generate_all_visualizations()

