import re

import dash_html_components as html
import numpy as np
import pandas as pd
import pycountry
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from util.country_output import pretty_print_countries


def display_breakdown_graph_by_pages_in_document(what_to_display: str, tokenised_pages: list, condition_to_pages: dict,
                                                 country_to_pages: dict,
                                                 duration_to_pages: dict,
                                                 effect_estimate_to_pages: dict,
                                                 num_endpoints_to_pages: dict,
                                                 num_sites_to_pages: dict,
                                                 num_subjects_to_pages: dict,
                                                num_arms_to_pages: dict,
                                                 phase_to_pages: dict,
                                                 sap_to_pages: dict,
                                                 simulation_to_pages: dict) -> tuple:
    """
    Generate a graph according to the graph viewing option which is currently selected by the user in the UI

    :param what_to_display: What the user has selected
    :param tokenised_pages: The list of pages after tokenisation
    :param condition_to_pages: The AI model’s output and explanations for the condition field
    :param country_to_pages: The AI model’s output and explanations for the countries field
    :param duration_to_pages: The AI model’s output and explanations for the duration field
    :param effect_estimate_to_pages: The AI model’s output and explanations for the effect estimate field
    :param num_endpoints_to_pages: The AI model’s output and explanations for the num_endpoints field
    :param num_sites_to_pages: The AI model’s output and explanations for the num_sites field
    :param num_subjects_to_pages: The AI model’s output and explanations for the num_subjects field
    :param phase_to_pages: The AI model’s output and explanations for the phase field
    :param sap_to_pages: The AI model’s output and explanations for the SAP field
    :param simulation_to_pages: The AI model’s output and explanations for the simulation field
    :return: A Plotly graph object, the title of the graph (str), and and a human readable list of relevant content in the original document if applicable
    """
    if tokenised_pages is None or country_to_pages is None:
        layout = go.Layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig = go.Figure(layout=layout)
        fig.add_layout_image(
            dict(
                source="https://raw.githubusercontent.com/fastdatascience/logos/master/logo_transparent_background.png",
                xref="paper", yref="paper",
                x=0.4, y=0.75,
                sizex=0.4, sizey=0.4,
                xanchor="right", yanchor="bottom",
                opacity=0.5
            )
        )
        fig.add_annotation(x=0, y=1,
                           text="Drag and drop a PDF protocol into the box above.", showarrow=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return [
            fig, [], []
        ]

    page_nos = list(range(1, 1 + len(tokenised_pages)))

    if what_to_display == "overview" or what_to_display is None:
        word_counts = [len(tokens) for tokens in tokenised_pages]
        total_words = sum(word_counts)
        df = pd.DataFrame({"page": page_nos, "word count": word_counts})

        fig = px.bar(df, x="page", y="word count")
        layout = go.Layout(
            title=f"Word counts of each page. Page count: {len(tokenised_pages)}, word count: {total_words}",
            # margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            # plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_xaxes(title_text="Page number")
        fig.update_yaxes(title_text="Word count")
        fig.update_layout(layout)
        return [fig, [], []]

    lookup = {"condition": condition_to_pages,
              "country": country_to_pages,
              "duration": duration_to_pages,
              "effect_estimate": effect_estimate_to_pages,
              "num_endpoints": num_endpoints_to_pages,
              "num_sites": num_sites_to_pages,
              "num_subjects": num_subjects_to_pages,
              "num_arms": num_arms_to_pages,
              "phase": phase_to_pages,
              "sap": sap_to_pages,
              "simulation": simulation_to_pages}

    items_dict = lookup[what_to_display]

    terms_to_page_numbers = items_dict.get("pages", {})
    prediction = items_dict['prediction']
    probas = items_dict.get("page_scores", None)
    score = items_dict.get("score", 0)

    human_readable_prediction = prediction
    if what_to_display == "country":
        human_readable_prediction = pretty_print_countries(prediction)

    graph_surtitles_lookup = {
        "condition": [f"Condition identified: ", html.B(prediction), f". Confidence: {score * 100:.1f}%. The heat map below shows you key terms related to the condition and which pages they occurred on throughout the document."],
        "country": [f"Which countries were mentioned on which pages in the document? Estimated trial countries: ", html.B(human_readable_prediction), ". The AI looked at the countries which were mentioned more often and earlier on in the document than other countries. The graph below shows the candidate countries as a heat map throughout the pages of the document."],
        "duration": "",
        "effect_estimate": [f"Where was an effect estimate found in the document? The graph below shows some candidate effect estimates and a selection of key terms by page number, overlaid with page-level probabilities (in pink - click the legend to hide). The protocol is {score * 100:.1f}% likely to contain an effect estimate."],
        "num_endpoints": "",
        "num_sites": "",
        "num_subjects": [f"Which pages contained terms relating to the number of subjects? The sample size appears to be {prediction} with confidence {score * 100:.1f}%."],
        "num_arms": [f"Which pages contained terms relating to the number of arms? The trial appears to have {prediction} arm(s)."],
        "phase": [f"Where was the phase mentioned in the document? The graph below shows possible phases and which pages they were mentioned on. The document is most likely to be ", html.B(f"Phase {prediction}"), "."],
        "sap": [f"Which pages contained highly statistical content and were likely to be part of the SAP? Graph of a selection of key statistical terms by page number, overlaid with page-level probabilities (in pink - click the legend to hide). The protocol is {score * 100:.1f}% likely to contain an SAP."],
        "simulation": [f"Which pages mentioned words related to simulation? The graph below shows a selection of simulation-related terms by page number. The protocol is {score * 100:.1f}% likely to involve simulation for sample size."]}

    graph_surtitle = graph_surtitles_lookup[what_to_display]

    human_readable_what_to_display = re.sub('_', ' ', what_to_display.upper())
    if what_to_display == "num_subjects":
        human_readable_what_to_display = "NUMBER OF SUBJECTS"
    if what_to_display == "condition" and prediction in ("HIV", "TB"):
        human_readable_what_to_display = prediction
    graph_title = f"Graph of key {human_readable_what_to_display} related terms by page number in document"

    total_terms_found = 0
    for v in terms_to_page_numbers.values():
        total_terms_found += len(v)
    if total_terms_found == 0:
        graph_surtitle.append(f" No terms relating to {human_readable_what_to_display} were found in the document.")

    page_matrix = np.zeros((len(terms_to_page_numbers), len(tokenised_pages)), dtype=np.int32)

    most_popular_countries = sorted(terms_to_page_numbers.items(), key=lambda a: len(a[1]))
    country_names = []
    hovertext = []
    for country_idx, (country_code, page_nos_containing_occurrence) in enumerate(most_popular_countries):
        for page_no in page_nos_containing_occurrence:
            page_matrix[country_idx, page_no] += 1

        if what_to_display == "country":
            country = pycountry.countries.lookup(country_code)
            name_for_y_axis = country.flag + country_code
            country_name = country.name
        else:
            name_for_y_axis = country_code
            country_name = country_code

        country_names.append(name_for_y_axis)
        this_row_hovertext = []
        for page_no in range(len(tokenised_pages)):
            num_mentions = page_matrix[country_idx, page_no]
            if num_mentions == 0:
                text = f"{country_name} was not mentioned on page {page_no + 1}"
            elif num_mentions == 1:
                text = f"{country_name} was mentioned once on page {page_no + 1}"
            else:
                text = f"{country_name} was mentioned {num_mentions} times on page {page_no + 1}"
        this_row_hovertext.append(text)
        hovertext.append(this_row_hovertext)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Heatmap(
        z=page_matrix,
        x=page_nos,
        y=country_names,
        text=hovertext,
        showscale=False,
        hoverinfo="text",
        colorscale='Blues',
        textfont={"size": 20}, colorbar=None), secondary_y=False)

    contexts = []
    if "context" in items_dict and type(items_dict["context"]) is dict:
        if len(items_dict["context"]) > 0:
            contexts.extend([html.H3(f"Possible mentions of {human_readable_what_to_display} in the document")])
        for value, context in items_dict["context"].items():
            contexts.extend([html.B(value), " ", context, html.Br()])

    if probas is not None:
        fig.add_trace(go.Bar(x=list(range(1, len(probas) + 1)), y=probas, marker=dict(
            color='rgba(246, 78, 139, 0.4)',
        ), name="probability of each page", showlegend=True), secondary_y=True)
        fig.update_yaxes(title_text=f"Probability that {human_readable_what_to_display} is on this page",
                         secondary_y=True, visible=True)

        layout = go.Layout(
            yaxis2=dict(
                range=[0, 1]
            )
        )
        fig.update_layout(layout)

    fig.update_xaxes(title_text="Page number")
    fig.update_yaxes(title_text="Word mentioned in document", secondary_y=False)

    # fig = go.Figure(data=go.Heatmap(
    #     z=page_matrix,
    #     x=page_nos,
    #     y=country_names,
    #     text=hovertext,
    #     hoverinfo="text",
    #     colorscale='Blues',
    #     textfont={"size": 20}))
    layout = dict(hovermode='y')
    fig.update_layout(layout)

    layout = go.Layout(
        title=graph_title,
        # margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_layout(layout)
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
    ))

    return [fig, graph_surtitle, contexts]
