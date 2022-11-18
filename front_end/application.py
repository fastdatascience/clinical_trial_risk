import io
import os
import re
import time
import traceback

import dash
import dash_auth
import dash_html_components as html
import numpy as np
import pandas as pd
from dash import ctx
from dash import dcc, html, Input, Output
from dash.dependencies import State

from layout.body import get_body, file_to_text
from tika import parser
from util import graph_callbacks
from util.clientside_callbacks import add_clientside_callbacks
from util.pdf_parser import parse_pdf
from util.pdf_report_generator import generate_pdf
from util.progress_bar import make_progress_graph
from util.protocol_master_processor import MasterProcessor
from util.risk_assessor import calculate_risk_level
from util.score_to_risk_level_converter import get_risk_level_and_traffic_light
from util.word_cloud_generator import WordCloudGenerator

COMMIT_ID = os.environ.get('COMMIT_ID', "not found")

# cache = diskcache.Cache("./cache")
# long_callback_manager = DiskcacheLongCallbackManager(cache)

VALID_USERNAME_PASSWORD_PAIRS = {
    'admin': 'DsRNJmZ'
}

word_cloud_generator = WordCloudGenerator("models/idfs_for_word_cloud.pkl.bz2")
master_processor = MasterProcessor("models/condition_classifier.pkl.bz2", "models/phase_rf_classifier.pkl.bz2",
                                   "models/spacy-textcat-phase-04-model-best",
                                   "models/sap_classifier.pkl.bz2",
                                   "models/effect_estimate_classifier.pkl.bz2",
                                   "models/num_subjects_classifier.pkl.bz2",
                                   "models/spacy-textcat-international-11-model-best",
                                   "models/simulation_classifier.pkl.bz2",
                                   "models/phase_arm_num_subjects_model_11_keras.keras", )

dash_app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"},
                         {"name": "description",
                          "content": "Analyse your Clinical Trial protocols and identify risk factors using Natural Language Processing, from Fast Data Science."}],
)
auth = dash_auth.BasicAuth(
    dash_app,
    VALID_USERNAME_PASSWORD_PAIRS
)

dash_app.title = "Clinical Trial Risk Tool"
server = dash_app.server  # For Google Cloud Platform
app = dash_app.server  # For Azure

# Create app layout
dash_app.layout = get_body()


@dash_app.callback(
    output=[Output("log_tika", "children")],
    inputs=[Input("location", "href")]
)
def wake_up_tika_web_app_on_page_load(location):
    """
    Wake up the Tika web app the first time the page is loaded.

    This is to ensure that there is not a huge turnaround time the first time the user uploads a PDF.
    :param location: a dummy trigger just to ensure that this function is called as soon as the browser requests the URL of the app.
    :return: Some human-readable description to be displayed in a log view for diagnostics
    """
    print(f"Initialising Tika server")
    start_time = time.time()
    response = parser.from_buffer(io.BytesIO(b""), xmlContent=True)
    end_time = time.time()
    print("Initialised Tika server")
    return [[f"Version: {COMMIT_ID}.", html.Br(), f"Initialised server for parsing text from PDFs at {time.ctime()}.",
             html.Br(),
             f"Response was {len(str(response))} characters received in {end_time - start_time:.2f} seconds."]]


@dash_app.callback(
    output=[Output("progress_bar_container", "style"),
            Output("risk_container", "style"),
            Output("interval", "disabled"),
            Output("interval", "n_intervals"),
            Output("progress_bar", "max"),
            Output("progress_bar", "label")],
    inputs=[Input("dataset", "value"),
            Input('upload-data', 'contents'),
            Input("score", "data")]
)
def display_progress_bar(dataset, data, score):
    triggered_id = ctx.triggered_id
    if (dataset is None and data is None) or triggered_id == "score":
        return [{"display": "none"}, {"display": "block"}, True, 0, 10, ""]
    else:
        if data:
            num_chars_in_file = len(data)
            file_size_string = f"Analysing your {0.8 * num_chars_in_file / 1024 / 1024:.1f} MB PDF file..."
        else:
            num_chars_in_file = 1000000
            file_size_string = f"Analysing your pre-parsed file"
        num_blocks_in_progress_bar = int(np.round(num_chars_in_file / 1000000 * 2))
        return [{"display": "block"}, {"display": "none"}, False, 0, num_blocks_in_progress_bar, file_size_string]


@dash_app.callback(
    output=[  # Output("paragraph_id", "children"),
        Output("file_name", "data"),
        Output("file_date", "data"),
        Output("protocol_status", "children"),
        Output("page_count", "children"),
        Output("tokenised_pages", "data"),
        Output("condition", "value"),
        Output("condition_to_pages", "data"),
        Output("phase", "value"),
        Output("phase_to_pages", "data"),
        Output("sap", "value"),
        Output("sap_to_pages", "data"),
        Output("effect_estimate", "value"),
        Output("effect_estimate_to_pages", "data"),
        Output("num_subjects", "value"),
        Output("num_subjects_to_pages", "data"),
        Output("num_arms", "value"),
        Output("num_arms_to_pages", "data"),
        Output("countries", "value"),
        Output("country_to_pages", "data"),
        Output("simulation", "value"),
        Output("simulation_to_pages", "data"),
        Output("num_subjects_explanation", "children"),
        Output("log", "children"),
    ]
    ,
    inputs=[Input("dataset", "value"),
            Input('upload-data', 'contents'),
            State('upload-data', 'filename'),
            State('upload-data', 'last_modified')],
    prevent_initial_call=True
)
def user_uploaded_file(  # set_progress,
        dataset, contents, file_name, file_date):
    tasks_completed = []

    # Disabled progress bar
    def set_progress(text):
        pass

    # A function which updates the progress bar and also logs an event to a text log.
    def report_progress(task):
        print("Progress report:", task.strip())
        tasks_completed.append(task.strip())
        if task.endswith("\n"):
            tasks_completed.append(html.Br())
        else:
            tasks_completed.append(" ")
        set_progress([make_progress_graph(len(tasks_completed), 20), tasks_completed])

    if file_name is None and dataset is None:
        raise Exception()

    report_progress("Converting PDF to text...")

    start_time = time.time()

    if dataset is not None:
        pages = file_to_text[dataset]
        file_name = dataset
        report_progress(f"Taking file from dropdown {dataset}.")
    else:
        report_progress(f"Parsing file {file_name} ({int(np.round(len(contents) / 1000))} KB).")

        pages = parse_pdf(contents)

    end_time = time.time()
    report_progress(
        f"The PDF document was converted to text in {end_time - start_time:.2f} seconds and contained {len(pages)} pages.\n")
    desc = f"Protocol: {file_name}"

    start_time = time.time()
    tokenised_pages, condition_to_pages, phase_to_pages, sap_to_pages, effect_estimate_to_pages, num_subjects_to_pages, num_arms_to_pages, country_to_pages, simulation_to_pages = master_processor.process_protocol(
        pages, report_progress)
    end_time = time.time()

    report_progress(
        f"The NLP analysis ran in {end_time - start_time:.2f} seconds.\n")

    page_count = f"({str(len(pages))} pages)"

    condition = condition_to_pages['prediction']
    phase = phase_to_pages['prediction']
    sap = sap_to_pages['prediction']
    effect_estimate = effect_estimate_to_pages['prediction']
    num_subjects = num_subjects_to_pages['prediction']
    num_arms = num_arms_to_pages['prediction']
    countries = country_to_pages['prediction']
    simulation = simulation_to_pages['prediction']

    num_subjects_explanation = num_subjects_to_pages.get("comment", "")

    return [file_name, str(file_date), desc, page_count, tokenised_pages, condition, condition_to_pages,
            phase, phase_to_pages, sap, sap_to_pages, effect_estimate, effect_estimate_to_pages,
            num_subjects, num_subjects_to_pages, num_arms, num_arms_to_pages, countries, country_to_pages, simulation,
            simulation_to_pages,
            num_subjects_explanation,
            tasks_completed
            ]


@dash_app.callback(
    [

        Output("calculation_data_table", "data"),
        Output("calculation_data_table", "columns"),
        Output("score", "data"),
        Output("log_scoring", "children"),
    ],
    [
        State("file_name", "data"),
        Input('condition', 'value'),
        Input('condition_to_pages', 'data'),
        Input('phase', 'value'),
        Input('sap', 'value'),
        Input('effect_estimate', 'value'),
        Input('num_subjects_and_tertile', 'data'),
        Input('num_arms', 'value'),
        Input('num_sites', 'value'),
        Input('num_endpoints', 'value'),
        Input('duration', 'value'),
        Input('countries', 'value'),
        Input('simulation', 'value'),
    ]
)
def fill_table(
        file_name,
        condition,
        condition_to_pages,
        phase,
        sap,
        effect_estimate,
        num_subjects_and_tertile,
        num_arms,
        num_sites,
        num_endpoints,
        duration,
        countries,
        simulation, ):
    try:
        if file_name is None:
            df = pd.DataFrame()
            df["Parameter"] = ["File not loaded"]
            total_score = None
            description = []
        elif condition == "Error":
            df = pd.DataFrame()
            df["Parameter"] = ["Your file could not be processed."]
            total_score = "Error"
            if "error" in condition_to_pages:
                total_score += ": " + condition_to_pages["error"]
            description = ["The pathology could not be identified."]
        elif condition not in ["HIV", "TB"]:
            df = pd.DataFrame()
            df["Parameter"] = ["Please choose an HIV or TB trial"]
            total_score = "ONLY HIV AND TB TRIALS ARE SUPPORTED"
            description = ["The trial is not an HIV or TB trial, so a score could not be calculated."]
        else:
            is_international = int(len(countries) > 1)
            total_score, df, description = calculate_risk_level(file_name, condition, phase, sap, effect_estimate,
                                                                num_subjects_and_tertile,
                                                                num_arms,
                                                                is_international, simulation)

        table_data = list([dict(d) for _, d in df.iterrows()])
        table_columns = [{"name": i, "id": i} for i in df.columns]
    except:
        print("Error calculating risk\n", traceback.format_exc())
        return [[], [], None, []]
    html_description = []
    for d in description:
        if len(html_description) > 0:
            html_description.append(html.Br())
        html_description.append(d)
    return table_data, table_columns, total_score, html_description


@dash_app.callback(
    [
        Output("download_pdf", "data"),
    ],
    [
        Input('export_pdf', 'n_clicks'),
        State("calculation_data_table", "data"),
        State("calculation_data_table", "columns"),
        State("tertiles_table", "data"),
        State("tertiles_table", "columns"),
        State("score", "data"),
        State("tokenised_pages", "data"),
        State("condition", "value"),
        State("condition_to_pages", "data"),
        State("phase", "value"),
        State("phase_to_pages", "data"),
        State("sap", "value"),
        State("sap_to_pages", "data"),
        State("effect_estimate", "value"),
        State("effect_estimate_to_pages", "data"),
        State("num_subjects", "value"),
        State("num_subjects_to_pages", "data"),
        State("num_subjects", "value"),
        State("num_arms_to_pages", "data"),
        State("countries", "value"),
        State("country_to_pages", "data"),
        State("simulation", "value"),
        State("simulation_to_pages", "data"),
        State("word_cloud", "src"),
        State("log", "children"),
        State("log_scoring", "children"),
    ]
)
def export_pdf(*args):
    return generate_pdf(*args)


@dash_app.callback(
    [
        Output("download_excel", "data"),
    ],
    [
        Input('btn_download', 'n_clicks'),
        State("calculation_data_table", "data"),
        State("calculation_data_table", "columns"),
    ]
)
def download_table(download_button_clicks, data, columns):
    if download_button_clicks == 0 or columns is None or data is None:
        return [None]
    df = pd.DataFrame()
    for col in columns:
        col_name = col['name']
        if col_name != "Score":
            column_data = []
            for r in data:
                cell_data = r[col['id']]
                if cell_data is not None and col_name == "Excel Formula" and "IF" not in cell_data:
                    cell_data += "+RAND()*0"  # Force the cell to recalculate
                column_data.append(cell_data)
            if col_name == "Excel Formula":
                col_name = "Score"
            df[col_name] = column_data

    file_name = df["Parameter"].iloc[0]
    excel_file_name = re.sub(".pdf", "", file_name) + ".xlsx"

    return [dcc.send_data_frame(df.to_excel, excel_file_name, sheet_name=f"Risk score calculation", index=False)]


@dash_app.callback(
    [
        Output("risk-indicator", "color"),
        Output("risk-indicator", "label"),
    ],
    [
        Input("score", "data"),
    ]
)
def show_gauge(
        score):
    """
    Once the score has been calculated a risk level can be displayed as a traffic light
    :param score:
    :return: A Dash object to display the traffic light and a human readable description of the risk
    """
    if score is None:
        return ["#999999", {"label": "FILE NOT LOADED", "style": {"font-size": "18pt"}}]

    if type(score) is str:
        return ["#999999", {"label": score, "style": {"font-size": "18pt"}}]

    risk_level, traffic_light = get_risk_level_and_traffic_light(score)

    return [traffic_light, {"label": risk_level, "style": {"font-size": "18pt"}}]


@dash_app.callback(
    [

        Output("time_series_graph", "figure"),
        Output("graph_surtitle", "children"),
        Output("context_display", "children")
    ],
    [
        Input("which_graph_to_display", "value"),
        Input("tokenised_pages", "data"),
        Input("condition_to_pages", "data"),
        Input("country_to_pages", "data"),
        Input("duration_to_pages", "data"),
        Input("effect_estimate_to_pages", "data"),
        Input("num_endpoints_to_pages", "data"),
        Input("num_sites_to_pages", "data"),
        Input("num_subjects_to_pages", "data"),
        Input("num_arms_to_pages", "data"),
        Input("phase_to_pages", "data"),
        Input("sap_to_pages", "data"),
        Input("simulation_to_pages", "data"),
    ]
)
def display_breakdown_graph_by_pages_in_document(*args):
    """
    When the user has selected a particular parameters explanation which should be displayed, this function generate the graph.
    :param args:
    :return:
    """
    return graph_callbacks.display_breakdown_graph_by_pages_in_document(*args)


@dash_app.callback(
    [

        Output("word_cloud", "src"),
        Output("log_word_cloud", "children"),
    ],
    [
        Input("tokenised_pages", "data"),
        Input("condition_to_pages", "data")
    ]
)
def update_wordcloud(tokenised_pages, condition_to_pages):
    """
    The word cloud is generated using Matplotlib instead of Plotly and so it is slower and therefore it is generated on a separate call back which is expected to run after the main callbacks.
    :param tokenised_pages:
    :param condition_to_pages:
    :return: The word cloud graph as base64 encoded image
    """
    if tokenised_pages is None or condition_to_pages is None:
        return [None, None]
    return word_cloud_generator.generate_word_cloud(tokenised_pages, condition_to_pages)


# Make sure the Javascript callbacks are added too.
add_clientside_callbacks(dash_app)

# Main
if __name__ == "__main__":
    port = os.environ.get('dash_port', 8050)
    debug = os.environ.get('dash_debug') == "True"
    dash_app.run_server(debug=debug, host="0.0.0.0", port=port)
