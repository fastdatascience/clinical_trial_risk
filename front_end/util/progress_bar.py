from plotly import graph_objects as go


def make_progress_graph(progress, total):
    progress_graph = (
        go.Figure(data=[go.Bar(x=[progress])])
            .update_xaxes(range=[0, total], showticklabels=False, visible=False)
            .update_yaxes(
            showticklabels=False, visible=False
        )
            .update_layout(height=20, margin=dict(t=0, b=0, l=0, r=0))
    )
    return progress_graph
