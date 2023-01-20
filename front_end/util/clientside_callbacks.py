from dash.dependencies import Input, Output, State

JAVASCRIPT_FUNCTION_TO_CALCULATE_SAMPLE_SIZE_TERTILE = """function(num_subjects, condition, phase, data, columns) {
            if (num_subjects == null || (condition != "HIV" && condition != "TB") || phase === 0 || phase == null) {
                return ["", "", [num_subjects, 0, 0, 0]];
            }
            var lookup;
            if (phase === 4) {
                lookup = 7;
            } else {
                lookup = phase * 2;
            }
            lower = data[lookup][condition + " lower tertile"];
            upper = data[lookup][condition + " upper tertile"];
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
        """function(display_log, page_count, condition, phase, sap, effect_estimate, num_subjects, num_arms, country, simulation, tertiles) {
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
         Input("explain_num_subjects", "n_clicks"), Input("explain_num_arms", "n_clicks"),
         Input("explain_country", "n_clicks"),
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




    # When the user clicks "Send feedback" we auto-populate a Google form.
    dash_app.clientside_callback(
        """function(send_feedback, file_name,    condition, condition_to_pages,
                 phase, phase_to_pages,
                 sap, sap_to_pages,
                 effect_estimate,
                 effect_estimate_to_pages,
                 num_subjects,
                 num_subjects_to_pages,
                 num_arms,
                 num_arms_to_pages,
                 countries,
                 country_to_pages,
                 simulation,
                 simulation_to_pages) {
            if (send_feedback > 0) {
                try {
                    if (condition_to_pages["prediction"].toString() !== condition.toString()) {
                        condition = condition_to_pages["prediction"] + " -> " + condition;
                    }
                    if (phase_to_pages["prediction"].toString() !== phase.toString()) {
                        phase = phase_to_pages["prediction"] + " -> " + phase;
                    }
                    if (sap_to_pages["prediction"].toString() !== sap.toString()) {
                        sap = sap_to_pages["prediction"] + " -> " + sap;
                    }
                    if (effect_estimate_to_pages["prediction"].toString() !== condition.toString()) {
                        effect_estimate = effect_estimate_to_pages["prediction"] + " -> " + effect_estimate;
                    }
                    if (num_subjects_to_pages["prediction"].toString() !== num_subjects.toString()) {
                        num_subjects = num_subjects_to_pages["prediction"] + " -> " + num_subjects;
                    }
                    if (num_arms_to_pages["prediction"].toString() !== num_arms.toString()) {
                        num_arms = num_arms_to_pages["prediction"] + " -> " + num_arms;
                    }
                    if (country_to_pages["prediction"].toString() !== countries.toString()) {
                        countries = country_to_pages["prediction"] + " -> " + countries;
                    }
                    if (simulation_to_pages["prediction"].toString() !== condition.toString()) {
                        simulation = simulation_to_pages["prediction"] + " -> " + simulation;
                    }
                } catch (err) {
                    console.log(err);
                }
                window.open("https://docs.google.com/forms/d/e/1FAIpQLSclA0WkZOlG6oy4xBzwUgJVoOoCwiXxUEdYK5ntKFdkhNCx1w/viewform?usp=pp_url&entry.44156177=" +encodeURIComponent(file_name) + "&entry.1222736710=" +encodeURIComponent(condition) + "&entry.1475388048=" +encodeURIComponent(phase) + "&entry.995144902=" +encodeURIComponent(sap) + "&entry.2035138595=" +encodeURIComponent(effect_estimate) + "&entry.1962015267=" +encodeURIComponent(num_subjects) + "&entry.2032818806=" +encodeURIComponent(num_arms) + "&entry.1125470565=" +encodeURIComponent(countries) + "&entry.934539296=" +encodeURIComponent(simulation));
            }

            return False;
        }""",
        [Output("dummy2", "value")],
        [Input("send_feedback", "n_clicks"),  State("file_name", "data"), State("condition", "value"),
        State("condition_to_pages", "data"),
        State("phase", "value"),
        State("phase_to_pages", "data"),
        State("sap", "value"),
        State("sap_to_pages", "data"),
        State("effect_estimate", "value"),
        State("effect_estimate_to_pages", "data"),
        State("num_subjects", "value"),
        State("num_subjects_to_pages", "data"),
        State("num_arms", "value"),
        State("num_arms_to_pages", "data"),
        State("countries", "value"),
        State("country_to_pages", "data"),
        State("simulation", "value"),
        State("simulation_to_pages", "data"),
         ],
            prevent_initial_call=False)