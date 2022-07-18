from dash.dependencies import Input, Output

JAVASCRIPT_FUNCTION_TO_CALCULATE_SAMPLE_SIZE_TERTILE = """function(num_subjects, condition, phase, data, columns) {
            if (num_subjects == null || (condition != "HIV" && condition != "TB") || phase === 0 || phase == null) {
                return ["", "", [num_subjects, 0, 0, 0]];
            } 
            if (phase === 1.5) {
                lower = (data[0][condition + " lower tertile"] + data[1][condition + " lower tertile"]) / 2;
                upper = (data[0][condition + " upper tertile"] + data[1][condition + " upper tertile"]) / 2;
            } else if (phase === 2.5) {
                lower = (data[1][condition + " lower tertile"] + data[2][condition + " lower tertile"]) / 2;
                upper = (data[1][condition + " upper tertile"] + data[2][condition + " upper tertile"]) / 2;
            } else {
                lower = data[phase - 1][condition + " lower tertile"];
                upper = data[phase - 1][condition + " upper tertile"];
            }
            if (num_subjects < lower) {
               return ["0 (small trial)", "🔴", [num_subjects, 0, lower, upper]];
            }
            if (num_subjects < upper) {
               return ["1 (medium trial)", "🟡", [num_subjects, 1, lower, upper]];
            }
            return ["2 (large trial)", "🟢", [num_subjects, 2, lower, upper]];
        }"""


def add_clientside_callbacks(dash_app):
    """
    There are a number of client side callbacks which reduce the need for so many server side requests. These are normally callbacks which update the colour of flags or where the values in a dropdown depend on the values in another.
    The client side callbacks are all written in Javascript which must be supplied as a single Python string. Of course the majority of the business logic of this application is written in Python, however a small amount of the processing is done in Javascript and the JavaScript is all in this file.

    :param dash_app: The application object used by Dash
    """
    # Depending on how many countries are in the dropdown, this Javascript snippet displays a traffic light of the right colour to indicate whether this boosts the score or not.
    dash_app.clientside_callback(
        """function(countries) {
            if (countries == null || countries.length == 0) {
                return [""]
            }
            if (countries.length > 1) {
                return ["🟢"]
            }
            return ["🔴"]
        }""",
        Output("countries_traffic_light", "children"), [Input("countries", "value")])

    # When the user clicks "View log" or "Explain", the tab view is put to the correct tab.
    # If they click a control with ID e.g. explain_condition, then graph is set to condition.
    dash_app.clientside_callback(
        """function(display_log, page_count, condition, phase, sap, effect_estimate, num_subjects, country, simulation, tertiles) {
            var triggered = dash_clientside.callback_context.triggered.map(t => t.prop_id);
            if (triggered === null) {
                return ["tab_graph", "condition"];
            }
            triggered = String(triggered);
            if (triggered === null || triggered === '') {
                return ["tab_graph", "condition"];
            } 
            //console.log("triggered is" + triggered + "andincludes" + triggered.includes("log") + "type" + typeof(triggered));
            if (triggered.includes("log")) {
                return ["tab_log", "condition"];
            }
            if (triggered.includes("page_count")) {
                return ["tab_graph", "overview"];
            }
            if (triggered.includes("tertile")) {
                return ["tab_tertiles", "num_subjects"];
            }
            var what_to_select = triggered.replace(/explain_/, '');
            what_to_select = what_to_select.replace(/.n_clicks/, '');
            //console.log("what_to_select is" + what_to_select);
            return ["tab_graph", what_to_select];
        }""",
        [Output("tabs", "value"), Output("which_graph_to_display", "value")],
        [Input("view_log", "n_clicks"), Input("page_count", "n_clicks"), Input("explain_condition", "n_clicks"),
         Input("explain_phase", "n_clicks"), Input("explain_sap", "n_clicks"),
         Input("explain_effect_estimate", "n_clicks"),
         Input("explain_num_subjects", "n_clicks"), Input("explain_country", "n_clicks"),
         Input("explain_simulation", "n_clicks"), Input("explain_tertiles", "n_clicks")

         ])

    """
    Return the lower and upper tertile of the sample size for a given pathology and phase.

    :param condition: "HIV" or "TB"
    :param phase: the phase of the trial (1.0, 1.5, 2.0, 2.5, 3.0).
    :return:
    """
    dash_app.clientside_callback(
        JAVASCRIPT_FUNCTION_TO_CALCULATE_SAMPLE_SIZE_TERTILE,
        [Output("sample_size_tertile", "children"),
         Output("subjects_traffic_light", "children"),
         Output("num_subjects_and_tertile", "data")],
        [Input("num_subjects", "value"),
         Input("condition", "value"),
         Input("phase", "value"),
         Input("tertiles_table", "data"),
         Input("tertiles_table", "columns")
         ])

    dash_app.clientside_callback(
        """function(n_intervals) {
            return [n_intervals];
        }
        """,
        [Output("progress_bar", "value")],
        [Input("interval", "n_intervals")]
    )
