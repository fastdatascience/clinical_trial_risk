import os
import pickle as pkl
import re

import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import pycountry
from dash import dash_table

from util.constants import EXPLANATION_OPTIONS
from util.tertile_provider import DefaultSampleSizeTertileProvider

tertile_finder = DefaultSampleSizeTertileProvider("sample_size_tertiles.csv")

# Google Tag Manager
google_tag_manager_iframe = html.Iframe(src="https://www.googletagmanager.com/ns.html?id=GTM-PNPSD9B", width=0,
                                        height=0, style={"display": "None", "visibility": "hidden"})

input_folder = "../data/preprocessed_tika/"

file_to_text = {}
dataset_selector_style = None
try:
    if os.stat(input_folder):
        for root, folder, files in os.walk(input_folder):
            for file_name in files:
                if not file_name.endswith("pkl"):
                    continue
                pdf_file = re.sub(".pkl", "", file_name)

                full_file = input_folder + "/" + file_name
                #         print (full_file)
                with open(full_file, 'rb') as f:
                    text = pkl.load(f)
                file_to_text[pdf_file] = text
except:
    print("Could not find any preprocessed protocols")
    print("The protocol selector menu will be hidden.")
    dataset_selector_style = {"display": "none"}

# The options to display in the dropdown for each y and x_i.
dataset_options = [
    {"label": dataset, "value": dataset}
    for dataset in file_to_text
]

condition_options = [
    {"label": "HIV", "value": "HIV"},
    {"label": "TB", "value": "TB"},
    {"label": "Other - not supported in prototype 🔴", "value": "Other"},
    {"label": "Error - cannot process PDF file 🔴", "value": "Error"}
]

phase_options = [
    {"label": "1 🔴", "value": 1.0},
    {"label": "1/2 🔴", "value": 1.5},
    {"label": "2 🟡", "value": 2.0},
    {"label": "2/3 🟢", "value": 2.5},
    {"label": "3 🟢", "value": 3.0},
    {"label": "Other/unknown", "value": 0},
]

countries_options = [
    {"label": country.flag + country.name, "value": country.alpha_2}
    for country in pycountry.countries
]
countries_options.append({"label": "unnamed countries", "value": "XX"})

yes_no_options = [
    {"label": x, "value": y}
    for x, y in [("Yes 🟢", 1), ("No 🔴", 0), ("Error processing document", -1)]
]

explanation_options = [
    {"label": explanation_option_desc, "value": explanation_option}
    for explanation_option_desc, explanation_option in EXPLANATION_OPTIONS
]

# Construct the HTML elements of the UI.
rows = [
    # Google Tag Manager
    google_tag_manager_iframe,
    # empty Div to trigger javascript file for graph resizing
    dcc.Interval(id="interval", interval=1 * 1000, n_intervals=0, disabled=True),
    dcc.Location(id="location"),
    dcc.Store(id="pages"),
    dcc.Store(id="file_name"),
    dcc.Store(id="file_date"),
    dcc.Store(id="tokenised_pages"),
    dcc.Store(id="condition_to_pages"),
    dcc.Store(id="country_to_pages"),
    dcc.Store(id="duration_to_pages"),
    dcc.Store(id="effect_estimate_to_pages"),
    dcc.Store(id="num_endpoints_to_pages"),
    dcc.Store(id="num_sites_to_pages"),
    dcc.Store(id="num_subjects_to_pages"),
    dcc.Store(id="num_arms_to_pages"),
    dcc.Store(id="phase_to_pages"),
    dcc.Store(id="sap_to_pages"),
    dcc.Store(id="simulation_to_pages"),
    dcc.Store(id="score"),
    dcc.Store(id="stage_1_complete"),
    dcc.Store(id="stage_2_complete"),
    dcc.Store(id="stage_3_complete"),
    dcc.Store(id="stage_4_complete"),
    dcc.Store(id="stage_5_complete"),
    dcc.Store(id="stage_6_complete"),
    dcc.Store(id="stage_7_complete"),
    dcc.Store(id="stage_8_complete"),
    dcc.Store(id="num_subjects_and_tertile"),
    dcc.Download("download_excel"),
    dcc.Download("download_pdf"),
    html.Div(id="output-clientside"),
    html.Div(
        [
            html.Div(
                [
                    # html.Img(
                    #     # src=app.get_asset_url("logolg.svg"),
                    #     id="plotly-image",
                    #     style={
                    #         "height": "60px",
                    #         "width": "auto",
                    #         "margin-bottom": "25px",
                    #     },
                    # )
                    #
                ],
                className="one-third column",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(
                                "Clinical Trial Risk Tool",
                                style={"margin-bottom": "0px"},
                            ),
                            html.Div(
                                "Analyse your HIV and TB Clinical Trial Protocols and identify risk factors using Natural Language Processing.", ),
                            html.Div(
                                "Upload a Clinical Trial Protocol in PDF format, and the tool will generate a risk assessment of the trial.", ),
                            html.Div(
                                ["You can find example protocols by searching on ", html.A(["ClinicalTrials.gov"],
                                                                                           href="https://clinicaltrials.gov/ct2/results?cond=Hiv&age_v=&gndr=&type=&rslt=&u_prot=Y&Search=Apply",
                                                                                           target="ct.gov"
                                                                                           ), "."]),
                        ]
                    )
                ],
                className="two-thirds column",
                id="title",
            ),
            html.Div(
                [
                    # "Prototype for use on HIV and TB trials in LMICs"
                    "Prototype for use on HIV and TB trials in ",
                    html.A(["LMICs"], href="https://data.worldbank.org/country/XO", target="wb"),
                    html.Br(),
                    "Single-document protocols only.",
                    html.Br(),
                    "Protocols with SAP as a separate PDF are not supported."
                ],
                className="one-third column",
                id="button",
            ),
        ],
        id="header",
        className="row flex-display",
        style={"margin-bottom": "25px"},
    ),
]

rows.append(

    html.Div(
        [
            html.Div(
                [
                    html.P("Choose a protocol", className="control_label", style=dataset_selector_style),
                    dcc.Dropdown(
                        id="dataset",
                        options=dataset_options,
                        multi=False,
                        value=None,
                        className="dcc_control",
                        style=dataset_selector_style
                    ),
                    dcc.Upload(id='upload-data',
                               children=html.Div([
                                   'Drag and Drop Protocol PDF', html.Br(), ' or ', html.Br(),
                                   html.A('Select File from your Computer')
                               ]),
                               style={
                                   # 'width': '100%',
                                   'height': '120px',
                                   'lineHeight': '40px',
                                   'borderWidth': '1px',
                                   'borderStyle': 'dashed',
                                   'borderRadius': '5px',
                                   'textAlign': 'center',
                                   'margin': '10px'
                               },
                               # Allow multiple files to be uploaded
                               #         multiple=True,
                               accept="application/pdf"),

                ],
                className="pretty_container four columns",
            ),

            html.Div(
                [
                    html.Span([
                        # html.P(id="paragraph_id", children=["Button not clicked"]),
                        html.P("⌛ Processing your protocol. Please wait...", className="control_label"),
                        daq.GraduatedBar(
                            id='progress_bar',
                            value=0,
                            max=10,
                        ),
                        # html.Div([""], id="progress_bar", className="control_label", style={"width": "60px", "background-color": "red"}),
                        # dbc.Progress(value=0, id="progress"),
                        # dcc.Graph(id="progress_bar_graph", figure=make_progress_graph(0, 10)),
                        html.P("", className="control_label",
                               id="progress_bar_message", style={"margin": 0, "padding": 0, "font-size": "smaller"})

                    ], id="progress_bar_container", style={"display": "none"}
                    ),
                    html.Span([
                        html.Div("Risk of uninformativeness", style={"text-align": "center"}),
                        daq.Indicator(
                            id='risk-indicator',
                            label={"label": "FILE NOT LOADED", "style": {"font-size": "18pt"}},
                            size=40,
                            color='#999999'
                        ),

                        html.Div([html.Span("Please upload a protocol on the left", id="protocol_status"),
                                  ],
                                 style={"text-align": "center"}),
                        html.Div([html.A(id="page_count",  # style={"font-size": "smaller"}
                                         )],
                                 style={"text-align": "center"}),
                        html.Div([html.A("Export report as PDF", id="export_pdf")],
                                 style={"text-align": "center"}),
                        html.Div([html.A("View log of the analysis", id="view_log")],
                                 style={  # "font-size": "smaller",
                                     "text-align": "center"}),

                    ], id="risk_container", style={"display": "inline"}),

                ],
                className="pretty_container four columns",
            ),

            html.Div(
                [
                    html.Span([
                        html.Div("Word cloud generated from this document"),
                        html.Img(id='word_cloud', style={"max-width": "100%", "max-height": "400px"}),
                        html.Span(
                            "This panel shows a word cloud with some of the most important words from this document. Words which indicate the pathology are shown in red.",
                            className="tooltiptextbottom"
                        )
                    ],
                        className="tooltip"
                    )
                ],
                className="pretty_container four columns",
            )
        ],
        className="row flex-display", style={"align": "center"}
    )

)

rows.append(

    html.H3(
        ["Explanation of analysis ",
         html.Span(" Move the mouse over an item or click 'explain' for more information",
                   style={"font-size": "12pt", "font-weight": "normal"})]
    ),
)

rows.append(

    html.Div(
        [

            html.Div(
                [
                    html.Span(
                        [
                            html.P(["Trial is for condition ",
                                    html.A(html.Sup("explain"), id="explain_condition")
                                    ]
                                   , className="control_label"),
                            dcc.Dropdown(
                                id="condition",
                                options=condition_options,
                                multi=False,
                                className="dcc_control",
                            ),
                            html.Span(
                                "The AI identified the protocol as being a trial for a particular condition. Click on 'explain' to find out which words on which pages led the AI to this decision.",
                                className="tooltiptext"
                            )
                        ],
                        className="tooltip",

                    ),

                    html.Span(
                        [
                            html.P(["Trial phase ",
                                    html.A(html.Sup("explain"), id="explain_phase", style={"font-size": "smaller"})],
                                   className="control_label"),
                            dcc.Dropdown(
                                id="phase",
                                options=phase_options,
                                multi=False,
                                className="dcc_control",
                            ),
                            html.Span(
                                "The AI identified the protocol as being a trial for a particular phase. All other things being equal, later phase trials are slightly lower risk. Click 'explain' to find out which words on which pages led the AI to this decision.",
                                className="tooltiptext"
                            )
                        ],
                        className="tooltip",
                    ),
                    html.Span(
                        [
                            html.P(["Has the Statistical Analysis Plan been completed? ",
                                    html.A(html.Sup("explain"), id="explain_sap")],
                                   className="control_label"),
                            dcc.Dropdown(
                                id="sap",
                                options=yes_no_options,
                                multi=False,
                                className="dcc_control",
                            ),
                            html.Span(
                                "The AI tried to find evidence of the SAP being included in the protocol. Trials including a completed SAP are much more likely to be informative. Click 'explain' to find out which words on which pages led the AI to this decision. Please note that some protocols supply the SAP as a separate document, which you can select manually here.",
                                className="tooltiptext"
                            )
                        ],
                        className="tooltip",
                    ),

                    html.Span(
                        [
                            html.P(["Has the Effect Estimate been disclosed? ",
                                    html.A(html.Sup("explain"), id="explain_effect_estimate",
                                           )],
                                   className="control_label"),
                            dcc.Dropdown(
                                id="effect_estimate",
                                options=yes_no_options,
                                multi=False,
                                className="dcc_control",
                            ),
                            html.Span(
                                "The AI tried to find evidence of the effect estimate being included in the protocol. Trials including a valid effect estimate are much more likely to be informative. Click 'explain' to find out which words on which pages led the AI to this decision.",
                                className="tooltiptext"
                            )
                        ],
                        className="tooltip",

                    ),
                    html.Br(),

                    html.Span(
                        [
                            html.P(["Number of subjects ", html.Span("", id="subjects_traffic_light"), " ",
                                    html.A(html.Sup("explain"), id="explain_num_subjects",
                                           )],
                                   className="control_label"),
                            daq.NumericInput(
                                id="num_subjects",
                                value=100,
                                min=10,
                                max=100000,
                                className="dcc_control",
                            ),
                            html.P("", id="num_subjects_explanation", className="control_label"),
                            html.P(["Sample size tertile: ", html.Span([], id="sample_size_tertile"), " ",
                                    html.A(html.Sup("set values of tertiles"), id="explain_tertiles",
                                           )
                                    ], className="control_label"),
                            html.Span(
                                "The AI attempted to extract the sample size from the protocol. Trials with an adequate sample size are more likely to be informative. Sample sizes are converted from raw numbers to a tertile (0, 1, 2) indicating a small, medium or large trial for this phase and pathology. Click 'explain' to find out which words on which pages led the AI to this decision.",
                                className="tooltiptext"
                            )
                        ],
                        className="tooltip",

                    ),
                    html.Br(),
                    html.Span(
                        [
                            html.P(["Number of arms ", html.Span("", id="arms_traffic_light"), " ",
                                    html.A(html.Sup("explain"), id="explain_num_arms",
                                           )],
                                   className="control_label"),
                            dcc.Dropdown(
                                id="num_arms",
                                value=2,
                                options=[{"label": "1", "value": 1},
                                         {"label": "2", "value": 2},
                                         {"label": "3", "value": 3},
                                         {"label": "4", "value": 4},
                                         {"label": "5+", "value": 5}, ],
                                className="dcc_control",
                            ),
                            html.P("", id="num_arms_explanation", className="control_label"),
                            html.Span(
                                "The AI attempted to extract the number of arms from the protocol. Click 'explain' to find out which words on which pages led the AI to this decision.",
                                className="tooltiptext"
                            )
                        ],
                        className="tooltip",

                    ),
                    html.Br(),
                    html.P("Number of sites", className="control_label", style={"display": "none"}),
                    daq.NumericInput(
                        id="num_sites",
                        value=1,
                        min=1,
                        max=1000,
                        className="dcc_control", style={"display": "none"}
                    ),

                    html.P("Number of primary endpoints", className="control_label", style={"display": "none"}),
                    daq.NumericInput(
                        id="num_endpoints",
                        value=1,
                        min=1,
                        max=10,
                        className="dcc_control", style={"display": "none"}
                    ),

                    html.P("Primary duration of trial (months)", className="control_label", style={"display": "none"}),
                    daq.NumericInput(
                        id="duration",
                        value=12,
                        min=1,
                        max=1000,
                        className="dcc_control",
                        style={"display": "none"}
                    ),

                    html.Span(
                        [
                            html.P(["Countries of investigation ", html.Span("", id="countries_traffic_light"), " ",
                                    html.A(html.Sup("explain"), id="explain_country")],
                                   className="control_label"),
                            dcc.Dropdown(
                                id="countries",
                                options=countries_options,
                                multi=True,
                                className="dcc_control",
                            ),
                            html.Span(
                                "The AI attempted to identify the country that the trial takes place in. Trials which take place in multiple countries are more likely to be informative. Click 'explain' to find out which countries were mentioned on which pages of the document.",
                                className="tooltiptext"
                            )

                        ],
                        className="tooltip"

                    ),
                    html.Br(),
                    html.Span(
                        [
                            html.P(["Trial uses simulation for sample size? ",
                                    html.A(html.Sup("explain"), id="explain_simulation",
                                           )],
                                   className="control_label",
                                   ),
                            dcc.Dropdown(
                                id="simulation",
                                options=yes_no_options,
                                multi=False,
                                className="dcc_control",
                            ), html.Span(
                            "The AI attempted to identify clues about whether the authors used simulation to determine their sample size. Click 'explain' to find out which words on which pages led the AI to this decision.",
                            className="tooltiptext"
                        )
                        ], className="tooltip"

                    ),

                ],
                className="pretty_container three columns",
            ),

            html.Div([dcc.Tabs(
                [dcc.Tab([

                    html.Div(
                        [
                            html.Div([
                                html.Span("Display explanation of AI decision for:", style={
                                    "display": "inline-block", "vertical-align": "center"
                                }),
                                html.Span([
                                    dcc.Dropdown(
                                        id="which_graph_to_display",
                                        options=explanation_options,
                                        multi=False,
                                        value="condition",
                                        className="dcc_control",
                                    )], style={
                                    "display": "inline-block", "width": "50%"
                                }),
                                html.Span(
                                    [html.Div(
                                        "Since protocols can be very long documents, this panel allows you to understand where the meaningful content was in the document which led the AI to making certain decisions."),
                                        html.Div(
                                            "You can choose a parameter, such as 'condition', to see the AI's explanation for how and where it found that information in the protocol"),
                                        html.Div(
                                            "The x-axis shows page number and the y-axis shows the key words or phrases found in the text, so the heatmap displayed allows you to see which 'hot' terms were present at which points in the document.")],
                                    className="tooltiptextbottom"
                                )
                            ], className="tooltip", style={
                                "vertical-align": "center", "width": "100%"
                            }),
                            html.Div(
                                [
                                    html.Div(id="graph_surtitle"),
                                    dcc.Graph(id="time_series_graph", style={'height': '70vh'}),
                                    html.Div(id="context_display")
                                ],
                                className="pretty_container",
                            ),
                        ],
                        className="pretty_container",
                        style={"vertical-align": "center"}
                    ),

                ], label="Breakdown by page number", value="tab_graph"),

                    dcc.Tab([
                        html.Div([
                            "The table below shows how the risk of this protocol was calculated. Protocols are scored according to a simple linear formula between 0 and 100, where 100 would be a perfect low-risk protocol and 0 would be a high-risk protocol. Each parameter extracted in the table on the left is entered into a spreadsheet and contributes to the scoring with an associated weight. For example, by far the strongest indicator that a protocol is low-risk is the presence of a statistical analysis plan. You can change the weights in the dropdowns on the left, or you can download the risk calculations as a spreadsheet and experiment with the parameters in Excel."]),
                        html.Button("Download to Excel", id="btn_download"),
                        html.Div(
                            [
                                dash_table.DataTable(
                                    id="calculation_data_table",
                                    editable=True,
                                    row_deletable=True,
                                    page_size=20),
                            ],
                        ),

                    ], label="Risk calculation spreadsheet", value="tab_calc"),
                    dcc.Tab([
                        html.P(
                            "This is a log of the analysis of the text which was carried out by the protocol analysis tool, with an explanation of the scoring."
                        ),
                        html.Div(
                            id="log_tika"
                        ),
                        html.Div(
                            id="log"
                        ),
                        html.Div(
                            id="log_scoring"
                        ),
                        html.Div(
                            id="log_word_cloud"
                        ),

                    ], label="How the protocol was analysed", value="tab_log"),
                    dcc.Tab([

                        html.Div(
                            [
                                html.P(
                                    "The model characterises trials as small, medium and large according to the number of participants. Since early phase trials are smaller than later trials, tertiles are used to define what counts as e.g. a small HIV Phase I trial. The table of tertiles is given below. The tertiles were derived from a sample of 21 trials in LMICs, but have been rounded and manually adjusted based on statistics from ClinicalTrials.gov data."),
                                dash_table.DataTable(
                                    id="tertiles_table",
                                    editable=True,
                                    row_deletable=False,
                                    export_format="xlsx", page_size=20,
                                    data=tertile_finder.DF_TERTILES_DATA_FOR_DASH,
                                    columns=tertile_finder.DF_TERTILES_COLUMNS_FOR_DASH

                                ),
                            ],
                        ),

                    ], label="Sample size tertiles", value="tab_tertiles"),

                ], id="tabs"

            )],
                className="pretty_container nine columns")

        ],
        className="row flex-display",
    )
)

rows.append(

    html.Div(
        [

            "Dashboard by ",
            html.A(["Thomas Wood"], href="https://freelancedatascientist.net", target="freelance"),
            " at ",
            html.A(["Fast Data Science"], href="https://fastdatascience.com", target="fds"),
            ". ",
            html.A(["View source code on Github"], href="https://github.com/fastdatascience/clinical_trial_risk",
                   target="github")
        ],
        className="attribution", style={"text-align": "center"}
    ),

)


def get_body():
    return html.Div(
        rows,
        id="mainContainer",
        style={"display": "flex", "flex-direction": "column"},
    )
